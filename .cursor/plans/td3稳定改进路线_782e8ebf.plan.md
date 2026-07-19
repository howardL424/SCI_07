---
name: TD3稳定改进路线
overview: 现有库对齐代码与轻量验证已完成，下一步只做一次全量迁移；随后把新专家生成改成「物理 transition 出库即对齐 + stage3 sidecar + 可版本化奖励投影」；再进入采样、纯 BC 与 TD3 精调。暂不切换到 GAIL。
todos:
  - id: data-alignment
    content: 完成现有 expert_lib 对齐代码、子集验证、观测归一化与一致性检查（已完成，勿重做）
    status: completed
  - id: legacy-full-rebuild
    content: 对现有 0714 专家库执行一次全量 stage2 重建、stage3 缓存（dt=1e-3）和一致性检查
    status: completed
  - id: transition-core-and-schema
    content: 抽取可预初始化的一步动力学核心；定义并写入新 transition schema（含 s'/action_norm/终止事实）
    status: completed
  - id: generation-pipeline-fix
    content: 改造 extract/generate，使新专家库出库即保存权威物理 transition（无需 rebuild）
    status: completed
  - id: stage3-sidecar
    content: 扩展 stage3 预计算为可恢复 sidecar；新库直接读 terminal_abs，不重放 stage2
    status: completed
  - id: reward-projection
    content: 新增集中式 reward_projection，并让 load_expert_buffer 兼容新旧库、廉价投影奖励
    status: completed
  - id: pipeline-validation
    content: 5–10 条小批量验收 + 扩展测试；通过后才进入采样/BC
    status: completed
  - id: expert-sampling
    content: 以 per-trajectory 为准构建可复现、轨迹均衡的约 99k 专家集
    status: completed
  - id: controlled-bc
    content: 固定训练随机性并运行纯 BC，验证状态到动作映射
    status: completed
  - id: td3bc-refine
    content: 用 BC Actor 初始化和 Q 归一化 BC 约束进行 TD3 精调
    status: pending
  - id: conditional-ablations
    content: 仅按诊断结果进行状态、Q-filter、奖励或动作空间消融
    status: pending
isProject: false
---

# TD3 专家热启动与退化治理

## 目标与边界

- 目标：先确认专家映射可学，再让 TD3 在不坍塌为固定边界动作的前提下超过专家。
- 不重跑 IPOPT。现有 `expert_lib` 只做**一次性**派生重建；**长期**靠改生成管线消除「每换库就全量 stage2 重放」。
- `ok/debug` 只表示 IPOPT 收敛状态；`debug` 轨迹不自动删除。任务质量由与在线环境对齐后的标签另行标记。
- 暂不切换 GAIL，不恢复旧版每步 stage2 ZEM 奖励，不先加任意动作边界惩罚。
- Python 解释器固定：`E:\Python\venvs\rl_pytorch\Scripts\python.exe`。
- 说明/纪要文档统一引用 `[rl_td3/md document/](rl_td3/md%20document/)`；不要在 `rl_td3/` 根目录或 `历次对话md` 新建/更新同名文档。相关上下文：`[SESSION_SUMMARY_20260718_EXPERT_DATA_AND_TD3_PLAN.md](rl_td3/md%20document/SESSION_SUMMARY_20260718_EXPERT_DATA_AND_TD3_PLAN.md)`、`[TD3专家轨迹改进对话补充.md](rl_td3/md%20document/TD3专家轨迹改进对话补充.md)`、`[BC_TD3_USAGE.md](rl_td3/md%20document/BC_TD3_USAGE.md)`。

## Agent 执行顺序（必须遵守）

```text
legacy-full-rebuild
  → transition-core-and-schema
  → generation-pipeline-fix
  → stage3-sidecar
  → reward-projection
  → pipeline-validation
  → expert-sampling → controlled-bc → td3bc-refine → (按需) conditional-ablations
```

- `data-alignment` **已完成**：代码、轻量单测、子集 `dt` 对比、观测归一化已落地。后续 Agent **不要重新设计或重写该步**，除非一致性检查失败且能指出具体缺陷。
- 任一停止条件触发时，停止进入下一节，先报告再改。

---

## 1. 专家数据策略：旧库一次迁移 + 新库分层生成

### 1.0 数据分层（原意不变，执行时以此为准）


| 层             | 权威内容                                                           | 何时失效                               |
| ------------- | -------------------------------------------------------------- | ---------------------------------- |
| 物理 transition | `s_raw`、`a_phys`、`a_norm`、`s'_raw`、绝对态、`done/intercepted`、距离事实 | stage2 动力学 / 动作映射盒 / 积分步长 / 终止定义变化 |
| stage3 结果     | `r_minT`、`reached/diverged`、`n_steps`、`stage3_dt`、`r_min1/2`   | stage3 PN / 积分 / 终止判据变化            |
| 奖励投影          | 过程奖、`-50`、`R1`、`bonus`、`both_ok`、`reward_version`              | 奖励公式/阈值变化（只重投影，不重放动力学）             |


普通奖励实验 **不得** 触发全量 stage2 rebuild。

---

### 1.1 `legacy-full-rebuild` — 现有 0714 库一次性全量迁移

**状态**：`data-alignment` 已完成；本步只执行全量跑数与验收，不改算法设计。

#### 前置检查（必须先做）

1. 源库存在：`E:\matlab\result_0714\expert_lib_1000.mat`。
2. 检查目标产物是否已存在且完整：
  - `rl_td3/data/expert_transitions_aligned.mat`
  - `rl_td3/data/expert_terminal_cache.npz`
3. 若产物已存在：先跑一致性检查；**已通过且源库路径匹配时，禁止无条件覆盖**。仅当检查失败、源库路径不匹配、或用户明确要求重跑时才重建。
4. 正式 stage3 使用已验证的 `**dt=1e-3**`（与在线一致）。若执行前发现子集对比结论或配置已变，**停止并报告**，不擅自改精度。

#### 可复制命令（按序）

工作目录：`D:\LTY\yan\yanerxia\GAIL`

```powershell
# A. 全量 stage2 重建（可能数小时；当前脚本无断点续跑）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/rebuild_expert_data.py `
  --expert E:\matlab\result_0714\expert_lib_1000.mat `
  --out rl_td3/data/expert_transitions_aligned.mat

# B. stage3 终端缓存（分钟级外推；正式 dt=1e-3）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/precompute_expert_terminals.py `
  --expert E:\matlab\result_0714\expert_lib_1000.mat `
  --aligned rl_td3/data/expert_transitions_aligned.mat `
  --dt 1e-3 `
  --out rl_td3/data/expert_terminal_cache.npz

# C. 一致性检查
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/check_expert_alignment.py `
  --expert E:\matlab\result_0714\expert_lib_1000.mat `
  --aligned rl_td3/data/expert_transitions_aligned.mat `
  --cache rl_td3/data/expert_terminal_cache.npz
```

相关实现（只调用，不重写）：`[rl_td3/tools/rebuild_expert_data.py](rl_td3/tools/rebuild_expert_data.py)`、`[rl_td3/matlab/rebuild_expert_transitions.m](rl_td3/matlab/rebuild_expert_transitions.m)`、`[rl_td3/tools/precompute_expert_terminals.py](rl_td3/tools/precompute_expert_terminals.py)`、`[rl_td3/tools/check_expert_alignment.py](rl_td3/tools/check_expert_alignment.py)`。

#### 运行约束

- 全量 stage2 重建可能需 **数小时（注：后续实际跑证明仅几分钟，agent意见有时实在不靠谱）**；属旧格式库的一次性迁移，**不重跑 IPOPT**。
- 当前 `rebuild_expert_data.py` / `rebuild_expert_transitions.m` **不支持断点续跑**。执行时：后台跑、写日志、防止机器睡眠。**不要虚构 checkpoint。**
- 异常中断后：先检查产物是否完整、再决定是否从头重跑；不要在半成品上继续训练。

#### 成功标准

- 对齐文件与 cache 均写出，且 `check_expert_alignment` 硬检查全部通过。
- 轨迹数/步数一致、无 NaN、动作 ∈ `[-1,1]`、每条轨迹仅末步 `done`、拦截轨迹无 `R1/bonus`、cache/aligned 与源库路径匹配。
- 观测归一化仍用首版 scale：`[3e4,1.2e3,π/2,π, 3e4,1.2e3,π/2,π, 1.5e4,π/2,π,1e3,1]`（只除不裁剪，统计越界比例）。

#### 停止条件

- 源库缺失 / 路径与产物声明不一致。
- 子集对比结论与 `dt=1e-3` 正式约定冲突且未裁决。
- 一致性检查硬失败。

---

### 1.2 `transition-core-and-schema` — 共享动力学核心 + 新 schema

**目标**：生成侧与在线侧共用同一套一步动力学，避免两套公式漂移；同时定义新库 transition 字段。

#### 要改的文件

- 新增或抽取：可预初始化 helper/config 的一步动力学核心（建议放在 `rl_td3/matlab/`，由 `[env_step.m](rl_td3/matlab/env_step.m)` 与专家抽取共用）。
- `[expert_trajectory/extract_expert_transitions.m](expert_trajectory/extract_expert_transitions.m)`：在现有 RK4 循环上**补存字段**（当前已逐步积分，补 `s'`/终止标记**不应**再整段重积一遍 stage2）。

#### 兼容字段（保留）

- `state`、`action`（物理）、`state_abs`、`t` — 旧工具可读。

#### 新增权威字段（至少）

每步：

- `action_norm`（或等价 `action_clipped`）：映射并裁剪到 TD3 `[-1,1]^3` 后的训练动作；
- `next_state`：用**裁剪后动作**执行一步后的真实 `s'`；
- `next_state_abs`：对应绝对态；
- `done`、`intercepted`、`stage2_done`；
- 可选保留 `action_phys` 作审计（可与现有 `action` 同义）。

轨迹级：

- `r_min1`、`r_min2`、真实 stage2 末态 `terminal_abs`、`t_end`；
- 版本元数据：`schema_version`、`stage2_dynamics_version`、`action_mapping_version`、`obs_norm_version`。

#### 硬性要求

1. **先映射/裁剪动作，再用同一动作积分**；禁止「buffer 动作已裁剪、`s'` 来自未裁剪动作」。
2. 末步 `next_state` = 执行最后一拍动作后的真实末态（修掉旧「先记再推」末态早半拍）。
3. 拦截立刻截断；物理层只记 `intercepted=true`，**不在物理层写死 `-50` 常数**。
4. 观测存 raw；训练侧统一 `[rl_td3/obs_norm.py](rl_td3/obs_norm.py)`。
5. **禁止**在大循环里直接反复调用「每次重建 helper/config」的现版 `env_step`，除非基准证明开销可接受；默认走预初始化核心。
6. 共享实现，或对固定样例做逐步等价测试；不得长期维护两套未校验公式。

#### 成功标准

- 核心函数可被 `env_step` 与 `extract` 共用（或等价测试通过）。
- schema 字段清单写入代码注释/`md document` 用法说明（只更新 `[rl_td3/md document/BC_TD3_USAGE.md](rl_td3/md%20document/BC_TD3_USAGE.md)`）。

---

### 1.3 `generation-pipeline-fix` — 新库出库即对齐

#### 要改的文件

- `[expert_trajectory/extract_expert_transitions.m](expert_trajectory/extract_expert_transitions.m)`（接上节 schema）
- `[expert_trajectory/generate_ps_expert_library.m](expert_trajectory/generate_ps_expert_library.m)`：汇总新字段；`expert_sa` **仅兼容/QA**；训练权威源 = per-trajectory 物理 transition。

#### 执行要点

- 仅在 `ok_use` 轨迹上写完整 transition（与现逻辑一致）。
- 批处理主文件先安全落盘（含物理 transition）；**不要**把 stage3 绑死在「主文件未保存就不能中断」的路径上（见 1.4）。
- 新库生成后，`load_expert_buffer` 应能**不经** `rebuild_`* 直接读物理 transition。

#### 成功标准（本步）

- 小批量 5–10 条生成成功，transitions 含 `next_state`/`action_norm`/`done`/`intercepted`。
- 扁平 `expert_sa` 若保留，仅作校验，不再作为训练切轨迹依据。

---

### 1.4 `stage3-sidecar` — 独立可恢复终端结果

#### 要改的文件

- 扩展 `[rl_td3/tools/precompute_expert_terminals.py](rl_td3/tools/precompute_expert_terminals.py)`：
  - **旧库**：继续读 `expert_transitions_aligned.mat`；
  - **新库**：直接读生成时保存的 `terminal_abs` / `t_end` / `r_min1` / `r_min2`，**无需** stage2 rebuild。

#### 存储约定

- 权威：`r_minT`、`reached`、`diverged`、`n_steps`、`stage3_dt`、以及 `r_min1/r_min2`。
- 派生（带 `reward_version`）：`R1`、`bonus`、`both_ok` — 可缓存，**不能代替**原始 `r_minT`。
- 拦截轨迹跳过 stage3；不得写入正终端奖励。

#### 流程约束

- 生成主库先落盘 → stage3 作为**第二阶段/sidecar**运行（可断点续跑更佳）。
- stage3 放在生成时或事后，**积分成本本质相同**；并入管线的价值是自动化与版本一致，不是省掉 stage3 计算。
- 仅改 stage3 PN/步长/终止：只重算 sidecar，不重放 stage2。

#### 成功标准

- sidecar 可追溯：源库路径/hash、`stage3_dt`、动力学版本。
- 旧库路径与新库路径均能独立跑通。

---

### 1.5 `reward-projection` — 可版本化廉价投影

#### 要新增/修改的文件

- **新增** `[rl_td3/reward_projection.py](rl_td3/reward_projection.py)`：过程奖、拦截惩罚、`R1`、联合 `bonus`、`reward_version` 及阈值/系数配置。
- 修改 `[rl_td3/load_expert_buffer.py](rl_td3/load_expert_buffer.py)`：
  - 优先读新库直接 transition；
  - 兼容现有 aligned mat；
  - 奖励由已存物理事实投影，不把缓存 `reward` 当唯一真值。
- `[rl_td3/evasion_env.py](rl_td3/evasion_env.py)` 与 MATLAB 在线奖励保持现有语义；增加固定/随机样例测试，约束 Python 投影 ↔ MATLAB 返回一致。

#### 投影规则（原意）

- 过程奖：由 `next_state` 距离 + `intercepted` 等事实计算。
- 终端：`R1/bonus/both_ok` 由 `r_minT/r_min1/r_min2` + 当前阈值/系数计算。
- 可保留默认 `reward_cached` 方便加载，但非物理权威。
- 换奖励：只重投影列；**不调 MATLAB、不重放 stage2/stage3**（只要新公式能用已存物理量表达）。若依赖未保存路径量：先评估补存；无法恢复才重建对应阶段。

#### 失效边界（执行时强制）


| 变更                           | 动作                   |
| ---------------------------- | -------------------- |
| 网络 / lr / BC 系数 / seed / 采样  | 不重算专家物理数据            |
| 可由已存物理量表达的奖励公式/阈值            | 只重算奖励投影              |
| stage3 PN / 积分 / 终止          | 只重算 stage3 sidecar   |
| stage2 动力学 / 动作映射盒 / 步长 / 终止 | 重建 stage2 transition |
| IPOPT 目标/约束 / 专家参数化 / 初态库    | 重新生成专家库              |


---

### 1.6 `pipeline-validation` — 小批量验收与停止条件

#### 测试与验收动作

1. 扩展 `[rl_td3/tests/test_data_alignment.py](rl_td3/tests/test_data_alignment.py)`：新旧 schema、末步 `s'`、拦截截断、奖励重投影、版本字段。
2. 生成 5–10 条新专家（不调用 `rebuild_*`）→ `load_expert_buffer` 可加载。
3. 随机抽轨迹，逐步与在线 `env_step` 对照：`action_norm`、`s'`、`done`、`intercepted` 在约定容差内一致。
4. 末步 `s'` 为真实执行末态；拦截仅一次末步 done，且无 stage3 正奖励。
5. 改一次奖励系数/bonus → **不调 MATLAB** 即得新奖励列，并与在线一致性测试通过。
6. 产物记录：`schema_version`、`stage2_dynamics_version`、`action_mapping_version`、`obs_norm_version`、`stage3_version/dt`、`reward_version`。

#### 全局停止条件（任一即停，不得进入 §2）

- 源库 / aligned / cache 路径不匹配。
- stage3 正式配置与约定冲突未裁决。
- 逐步对齐超容差。
- 拦截轨迹仍含终端正奖励。
- 修改奖励时仍被迫调用 MATLAB 重放动力学。

#### 本阶段完成后

- 才可宣称：新库训练前**不再**需要 `rebuild_`* 全量 stage2 重放；普通奖励实验只做廉价投影。
- 需要同步用法时，**只更新** `[rl_td3/md document/BC_TD3_USAGE.md](rl_td3/md%20document/BC_TD3_USAGE.md)`。
- 然后进入 §2 `expert-sampling`。

---

## 2. 构建专家集与可复现实验

**依赖**：§1 全部相关 todo 完成（旧库迁移 + 新管线验收，或至少旧库迁移完成且训练暂用 aligned 派生数据）。

- 保持约 99k 的对照预算，但改为按轨迹均衡、按 `tgo` 分层、强制包含终端步的可复现采样；不能再直接取 `expert_sa[:99000]`，也不能让长轨迹支配样本。
- 按轨迹划分 BC train/validation，避免同一轨迹的相邻步泄漏到两侧。
- Replay 预填充与 Actor BC 使用同一轨迹映射；`bc_max_steps` 改由实际 `args.buf × bc_ratio` 计算。
- 保存质量元数据（`ok/debug`、规避、打击、联合成功、拦截）。初始纯 BC 基线先对所有 `ok_use` 轨迹报告整体和分层结果；不要在验证前硬删除 debug 或仅保留 `both_ok`。
- “所有 `ok_use`”仅作为诊断对照；正式 Actor BC 默认使用在线语义重放后未拦截的轨迹，拦截轨迹及其 `-50` transition 仍可保留给 Critic replay，但不得作为等权 Actor 专家标签。受控比较 Actor BC 中拦截轨迹占比（至少 `0%` 与全量约 `24%`）；若需利用失败轨迹，只单独测试安全前缀或显式降权，不与默认方案混用。
- 在 `[rl_td3/train_td3.py](rl_td3/train_td3.py)` 增加统一 seed，覆盖 SB3、NumPy、PyTorch、动作噪声及 train/eval env。外部 `[rl_td3/evaluate_td3.py](rl_td3/evaluate_td3.py)` 已可复现，无需重复修复；训练内 best-model 评估仍需固定初态序列。
- 快速开发可用单 seed；候选方案结论至少基于 3 个配对训练 seed，并对所有模型使用相同评估 seed/episodes。

## 3. 纯 Actor BC 诊断

- 新增独立 BC-only 训练/验证入口；当前代码只有 Replay 预填充 + TD3 同步衰减 BC 正则，并没有真正先训练 Actor。
- 先用归一化后的现有 13 维状态训练 Actor，记录 train/validation MSE、逐动作 MAE、动作贴边率和闭环联合成功率；同时与“直接播放原专家动作”的同一派生环境结果对照。
- 判断门：
  - 若 BC 验证误差持续较高或闭环明显低于专家，先进入“状态/动作可辨识性消融”，不要直接调 TD3、奖励或 GAIL。
  - 若 BC 能接近专家且动作不坍塌，再进入 TD3 精调。

### 3.1 执行结果（2026-07-19，seed=0）

入口：`rl_td3/train_bc.py`；产物：`rl_td3/checkpoints_bc/bc_diag_seed0/`。

| 项 | 结果 |
|----|------|
| 数据 | train 73721 / val 11501（`expert_set_99k`，Actor 排除拦截） |
| best val MSE | **0.360**（epoch 3；early stop@18） |
| val MAE | 0.472；γ/wT/wE ≈ 0.55 / 0.35 / 0.52 |
| BC 贴边率 | ≈0%（未坍塌到边界） |
| 专家标签贴边（val） | γ 31% / wT 4% / wE 27% |
| MC both_ok（20 ep） | 50% |
| 专家动作回放 both_ok（20 轨） | 90% |
| 同 IC 上 BC both_ok | 75% |
| 判断门 | **`fail_high_val_error`** → **先做 §5 状态/动作可辨识性消融**，不进 TD3 精调 |

解读：映射误差大（尤其 γ），策略偏内点、学不到专家贴边动作；闭环同 IC 尚可但离线门未过。下一步按计划做状态消融 / 动作可辨识性，而非调 λ 或进 TD3。

## 4. BC 初始化后的 TD3 精调

- 用最佳 BC checkpoint 初始化 TD3 Actor，避免随机 Actor 从第 0 步同时承受 Q 与 BC 梯度。
- 先实现量级可比的 TD3+BC：对 Q 项做 `mean(|Q|)` 归一化，BC 系数衰减到小的正下限而不是 150k 后归零；分别记录 Q 项、BC 项和二者梯度/损失量级。
- Actor 继续使用现有独立专家集；Critic 若需长期保留示范，再使用独立 demo buffer 或固定混采比例，避免预填数据被在线 buffer 覆盖。
- 固定场景选择 best，主指标为联合成功率；同时约束动作贴边率、实际 `a_M` 饱和率，并报告规避率/打击率。先做短程 checkpoint 对比，再决定是否运行完整 300k。

## 5. 仅在诊断触发时做消融

- 状态消融：比较归一化的现有 13 维、增加 LOS 角速度/必要姿态的相对状态、目标中心绝对/笛卡尔状态。现有 `state_abs` 可用于派生，不重跑 IPOPT；不预设哪种必然更优。
- 动作可辨识性：测量权重变化对 `a_M` 的敏感度和 8g 饱和率。若存在大面积等价/饱和区，再测试固定低敏感维度、按专家 K 段降低更新频率或改为权重比参数化。
- 若标准 TD3+BC仍受次优专家限制，再把 Q-filter 作为下一项单独对照，不与首轮方案同时引入。
- 只有在数据、BC 和 TD3 均已对齐后才重标定奖励；任何目标稠密项必须先证明与真实 stage3 `r_minT` 相关。stage2 ZEM 已知不可靠，不直接用于逐步惩罚。

## 验证与交付

- 为数据重建、归一化、轨迹采样、seed 和 BC-only 增加轻量测试/统计；运行相关 lint 和 smoke test。
- 每阶段保存配置、seed、数据摘要、动作分布与评估结果，避免只看 `mean_reward`。
- 不自动重跑 IPOPT；长时间 300k/多 seed 训练前先给出短程结果和是否通过上述判断门。
- 文档分别写清「物理 transition」「stage3 结果」「奖励投影」的版本和失效条件，不再笼统地把所有奖励修改都归为全量重放；用法说明只更新 `[rl_td3/md document/BC_TD3_USAGE.md](rl_td3/md%20document/BC_TD3_USAGE.md)`。
- 生成管线改造验收通过后，文档写明：新 `expert_lib` 训练前**不再**需要 `rebuild_`* 全量 stage2 重放；只有 stage2 动力学/动作/终止语义变化才重建 stage2，普通奖励调整只做廉价投影。

