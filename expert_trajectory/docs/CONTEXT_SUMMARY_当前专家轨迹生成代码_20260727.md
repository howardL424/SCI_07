# 上下文纪要：当前专家轨迹生成代码（物理 transition 版）

> 本文件基于 2026-07-27 工作区中的当前代码整理，用于替代早期纪要
> `CONTEXT_SUMMARY_专家轨迹批量生成.md` 中已经过时的生成、落盘和训练接口说明。
> 早期单打靶建模过程仍可参考 `CONTEXT_SUMMARY_伪谱专家轨迹.md`。
>
> 本文只记录当前实现、已经完成的演进、历史实跑结果和已知问题，不包含下一步计划。

---

## 0. 核心设定（当前权威约定）

### 0.1 生成范围

- 输入仍为项目根目录下的 `samples_clean.mat`，只处理 `samples_clean` 中非全零的有效行。
- 14 列样本只用于重建攻击弹 M、防御弹 D1/D2 的 18 维绝对初态，不直接作为策略观测。
- 专家优化与训练 transition 只覆盖 **stage2 规避段**：
  - stage1：攻击弹 PN 打目标；
  - stage2：式（47）双防规避与目标打击一体化制导；
  - stage3：PN 收尾，不在主专家库生成循环中运行。
- stage3 的真实打击结果通过
  `rl_td3/tools/precompute_expert_terminals.py` 事后计算并保存为 sidecar `.npz`。

### 0.2 优化变量与固定量

- stage2 单打靶决策变量：
  `theta(t) = [gamma, wT, wE]`，默认分成 `K=10` 段，每段常值。
- 固定量：
  - `w = 1e4`；
  - `dstar = 20 m`；
  - 攻击弹过载上限 `8g`；
  - 防御弹过载上限 `10g`。
- 默认优化盒：
  - `gamma ∈ [-3, 3]`；
  - `wT ∈ [1, 1e5]`；
  - `wE ∈ [10, 1e4]`。
- 默认安全和目标门槛：
  - `miss1 >= 6 m`；
  - `miss2 >= 6 m`；
  - stage2 末端目标 ZEM `missT <= 30 m`。

### 0.3 状态、动作与训练权威源

13 维原始物理观测保持不变：

| 列 | 含义 |
|---|---|
| 1 | `r_MD1` |
| 2 | `rdot_MD1` |
| 3 | `qy_D1` |
| 4 | `qz_D1` |
| 5 | `r_MD2` |
| 6 | `rdot_MD2` |
| 7 | `qy_D2` |
| 8 | `qz_D2` |
| 9 | `r_MT` |
| 10 | `qy_MT` |
| 11 | `qz_MT` |
| 12 | `V_M` |
| 13 | `tgo_norm` |

当前必须区分三类动作表示：

1. `transitions.action` / `action_phys`：`[w, gamma, wT, wE]`，用于物理参数审计。
2. `transitions.action_norm`：真正传给共享 stage2 动力学、也真正用于训练的
   3 维归一化动作。
3. `results.expert_sa`：`[state, action]` 的 17 列兼容/质检表，不再是训练权威源。

当前训练权威源为：

```text
results.trajectories(k).transitions
```

即按轨迹保存的 `phys_transition_v1`。它直接提供 `(s, a, s', done)`、轨迹边界、
拦截事实和真实 stage2 末态，不再依赖扁平大表反推边界或重建下一状态。

---

## 1. 已完成的代码修改与功能

### 1.1 文件清单与当前职责

| 文件 | 当前职责 |
|---|---|
| `generate_ps_expert_library.m` | 批处理主入口；支持样本子集、断点续跑、checkpoint、物理 transition 校验与版本落盘 |
| `reconstruct_geometry.m` | 将 `samples_clean` 的 14 列参数重建为 M/D1/D2 的 18 维绝对初态 |
| `solve_warmstart.m` | 扫描固定 `wE` 候选，选出 stage1-stage3 可跑通或评分最高的阶段 A 初值 |
| `simulate_warmstart_trajectory.m` | 以 `dt=1e-3 s` 运行 PN → 规避 → PN/optimal 的三阶段 warm-start |
| `solve_ss_trajectory.m` | 对 stage2 做单打靶 NLP，优化 `3×K` 的 `theta_K` |
| `casadi_dynamics.m` | 式（47）+ 三飞行器运动学的 CasADi 符号动力学 |
| `casadi_guidance_duo_target.m` | 数值式（47）的可微、光滑、无分支符号实现 |
| `casadi_target_zem.m` | 计算 stage2 末端对静止目标的 ZEM |
| `extract_expert_transitions.m` | 用在线环境共享核心完整重放 stage2，直接生成物理 transition |
| `setup_expert_path.m` | 加入专家模块、项目根目录和 `rl_td3/matlab` 共享核心路径 |
| `rl_td3/matlab/stage2_core_init.m` | 一次性初始化共享动力学 helper、参数和版本元数据 |
| `rl_td3/matlab/stage2_step.m` | 专家抽取与在线环境共用的一步式（47）+ RK4 物理推进 |
| `rl_td3/matlab/phys_to_action_norm.m` | 物理权重到归一化动作的映射与裁剪 |
| `rl_td3/matlab/transition_schema_meta.m` | 集中声明 schema、动力学、动作映射和观测归一化版本 |
| `rl_td3/tools/precompute_expert_terminals.py` | 从真实 stage2 末态运行 stage3，生成终端事实和奖励派生量 sidecar |

虽然入口名仍含 `ps`，当前主优化器实际是
`solve_ss_trajectory` 的 **single shooting（单打靶）**，并不是早期的伪谱配点法。
`solve_ps_trajectory.m`、`build_ps_nlp.m` 主要作为旧方案代码保留。

### 1.2 当前完整调用链

```text
samples_clean.mat
  → 有效行筛选 / valid_pos 子集选择
  → reconstruct_geometry
  → solve_warmstart
      → simulate_warmstart_trajectory
      → 扫 wE=[1440,700,400,250,150,100,60]
  → solve_ss_trajectory
      → 取 warm-start 的 stage2 初态与时长
      → CasADi 符号 RK4 滚动
      → IPOPT + limited-memory Hessian
      → theta_K[3×K]、miss、J
  → ok/debug/fail 入库判定
  → extract_expert_transitions
      → 物理权重映射并裁剪为 action_norm
      → 共享 stage2_step 完整重放
      → 生成 s/a/s'/done/terminal_abs
  → generate_ps_expert_library 汇总
  → expert_lib*.mat
  → precompute_expert_terminals.py
  → expert_terminal_cache*.npz
```

### 1.3 阶段 A：warm-start 已形成完整三阶段仿真

`simulate_warmstart_trajectory` 当前包含：

- stage1：攻击弹以 `KM=3` 的 PN 飞向原点目标；
- 当任一防御弹距离攻击弹不超过 `r_safe=8000 m` 时进入 stage2；
- stage2：
  - 两弹距离差超过 3000 m 时，只对近弹调用单弹规避律；
  - 否则调用式（47）双防+目标一体化制导；
- stage3：默认恢复 PN，也可通过 `stage3_guidance='optimal'` 切换为式（47）；
- 记录拦截、双弹转为远离、到达目标、持续发散和各阶段能耗。

`solve_warmstart` 默认按 `wE` 从大到小扫描，即从弱机动到强机动。选择优先级为：

```text
规避达标且到达目标的最大 wE
  > 规避达标的最大 wE
  > 综合评分最高的尽力解
```

阶段 A 的 `tau.success` 表示两枚防御弹均已转为远离且未发生 6 m 内拦截；
它不等价于 stage2 NLP 已成功，也不等价于最终 stage3 命中目标。

### 1.4 阶段 B：单打靶 NLP 与目标函数

当前单打靶默认 `K=10`、每段 `nsub=12` 个符号 RK4 子步，仅优化 30 个变量。
状态由闭环动力学滚动得到，不再引入配点状态和缺陷约束。

目标函数为：

```text
J = term_avoid + term_energy + term_target

term_avoid  = 0.25 * s_w  * [(miss1-dstar)^2 + (miss2-dstar)^2]
term_energy = 0.5  * s_wE * integral(||a_M||^2 dt)
term_target = 0.5  * s_wT * missT^2
```

当前固定评分系数：

```text
s_w=2e3, s_wE=55, s_wT=22, Jscale=1e5
```

这里的 `s_w/s_wE/s_wT` 是目标函数固定评分系数；决策变量
`gamma/wT/wE` 是式（47）的时变控制权重，两者不能混淆。

IPOPT 默认：

```text
max_iter=300
tol=1e-3
hessian_approximation='limited-memory'
```

### 1.5 入库判定比旧纪要更严格

当前逻辑为：

```text
IPOPT 正式求解成功
  → ok_use=true, ok_mode='ok'

IPOPT 未正式收敛，但 opti.debug 可提取 theta_K 和 miss
  → miss1>=6 且 miss2>=6 且 missT<=30
  → ok_use=true, ok_mode='debug'

其余
  → ok_use=false, ok_mode='fail'
```

旧纪要只强调 debug 轨迹的双防 `miss>=6`，已不完整；当前 debug 入库还必须满足
`missT_ceil`。`debug` 表示优化器未完全收敛但中间解通过硬门槛，不等同于物理失败。

### 1.6 transition 改为“生成时即对齐”

旧实现只保存每步当前 `state/action/state_abs/t`，训练侧再按相邻行推断
`next_state`，最后一步还缺少执行最后一个动作后的真实末态。

当前 `extract_expert_transitions` 每一步执行顺序为：

```text
读取当前 theta_K
  → phys_to_action_norm
  → 裁剪到 [-1,1]^3
  → 用裁剪后的同一个 action_norm 调 stage2_step
  → 保存当前 s、实际动作 a、真实 s' 和终止事实
```

因此：

- 标签动作与产生 `next_state` 的动作一致；
- `terminal_abs` 是执行最后一拍后的真实 18 维 stage2 末态；
- 发生 6 m 内拦截时立即截断；
- 正常走到 `Tev` 时末步 `stage2_done=true`；
- 生成侧与在线 `env_step` 共用 `stage2_core_init + stage2_step`。

`rl_td3/matlab/test_stage2_core_equiv.m` 对共享核心与在线包装进行逐步等价检查，
同时检查动作映射往返和新旧动作坐标对同一物理参数的控制等价性。

### 1.7 动作空间改为相对权重比值

当前默认版本为：

```text
action_mapping_version = ratio_log_box_v1
```

归一化动作顺序不是旧版 `[gamma_n,wT_n,wE_n]`，而是：

```text
[rhoE_n, rhoT_n, gamma_n]
```

其中：

```text
rhoE = w/wE       ∈ [1, 100]
rhoT = wT/wE      ∈ [1e-2, 1e3]

rhoE_n  = log10(rhoE) - 1
rhoT_n  = (log10(rhoT) - 0.5) / 2.5
gamma_n = gamma / 3
```

采用比值的原因是式（47）实际依赖
`w1/wE、w2/wE、wT/wE`，绝对权重存在尺度冗余。共享动力学将归一化动作解码成
相对权重，并以 `wE=1` 代入式（47），避免先生成任意绝对权重组合再裁旧盒。

---

## 2. 核心架构与关键设计决策

1. **单打靶只负责寻找控制参数，数值共享核心负责定义训练物理事实。**  
   CasADi 优化使用光滑式（47）、光滑过载饱和和符号 RK4，以保证可微与求解稳定；
   入库前再用与在线环境完全相同的数值 helper、硬过载饱和和 RK4 重放。
   因而 NLP 的 `miss` 是筛选与诊断量，`transitions` 中的物理轨迹才是训练权威。

2. **按轨迹保存 transition，不再把 `expert_sa` 当经验回放。**  
   每条轨迹明确保存边界、`s'`、`done`、拦截和末态，消除了扁平表拼接处把下一条
   轨迹首状态误当作上一条 `s'` 的风险，也不再需要新库执行 `rebuild_*`。

3. **先裁动作，再用同一个动作产生下一状态。**  
   debug 解或优化盒可能超出训练动作盒。如果只裁标签、不重放状态，监督动作和动力学
   后果就会错位。当前实现以裁剪后的 `action_norm` 推进，并用
   `n_actions_clipped` 记录裁剪次数。

4. **stage3 与主库解耦。**  
   主库保存真实 stage2 末态和最小弹间距；stage3 终端事实单独缓存。这样普通奖励公式
   调整只需重新投影，不必重跑 IPOPT 或 stage2；stage3 积分步长、版本和专家库指纹也能
   独立核对。

5. **物理事实与奖励语义分离。**  
   `transitions.intercepted` 只记录是否被拦截，不在专家主库中硬写 `-50`；
   过程奖励、拦截惩罚和 stage3 终端奖励由训练侧统一投影，避免奖励改动污染物理数据。

6. **版本戳集中管理。**  
   当前代码声明：

   ```text
   schema_version          = phys_transition_v1
   stage2_dynamics_version = stage2_rk4_eq47_v1
   action_mapping_version  = ratio_log_box_v1
   obs_norm_version        = obs_scale_0714_v1
   ```

   schema、动力学公式、动作映射或观测归一化任一发生变化时，不能只凭文件名判断兼容性。

7. **网络观测归一化不在 MATLAB 主库中预先写死。**  
   主库保存 13 维原始物理观测；Python 训练侧按 `obs_scale_0714_v1` 逐维缩放。
   `state` 与 `next_state` 必须使用同一归一化版本。

8. **目标质量分为不同层次。**  
   - `ok/debug`：优化器状态与 stage2 NLP 门槛；
   - `intercepted/r_min1/r_min2`：数值重放后的 stage2 物理安全性；
   - `missT`：stage2 末端符号 ZEM；
   - sidecar 的 `r_minT/reached/diverged`：stage3 后真实打击事实；
   - `R1/bonus/both_ok`：带奖励版本的派生标签。  
   这些量不能互相替代。

---

## 3. 当前批处理、数据格式与历史实跑结果

### 3.1 `generate_ps_expert_library` 主要选项

| 选项 | 默认值 | 含义 |
|---|---:|---|
| `run_stageB` | `false` | 是否运行单打靶 NLP |
| `plot` | `true` | 是否绘制第一条阶段 A 轨迹 |
| `save` | `false` | 是否保存主库 |
| `save_file` | `expert_trajectory/expert_lib.mat` | 输出路径；纯文件名自动放到专家目录 |
| `checkpoint_every` | `50` | 每多少个所选样本保存 checkpoint；0 为禁用 |
| `resume_ckpt` | `''` | 从含 `results_ckpt` 的 checkpoint 续跑 |
| `valid_pos` | `[]` | 在“有效行序号”上的子集，用于分批 |
| `cfg` | `struct()` | 阶段 A 配置 |
| `ss_cfg` | `struct()` | 单打靶配置 |
| `ex_cfg` | `struct()` | transition 抽取配置 |

`N=Inf` 表示使用全部有效样本。当前 `scripts/P3_generate1000.m` 虽然文件名保留
“1000”，实际调用 `generate_ps_expert_library(Inf, opts)`，并不限制为 1000 条。

### 3.2 分批与断点恢复

- `P3_generate1000_half1.m` 和 `P3_generate1000_half2.m` 按有效样本序号一分为二；
- `RUN_MODE='fresh'` 从头运行；
- `RUN_MODE='resume'` 按修改时间自动选择对应半批的最新 checkpoint；
- checkpoint 文件名为 `*_ckptXXXX.mat`；
- checkpoint 内变量名是 `results_ckpt`，最终文件内变量名是 `results`。

`valid_pos` 是在“去掉全零行之后”的序号，`valid_idx` 才是原始
`samples_clean` 行号。两者不能混用。

### 3.3 当前主库结构

库级主要字段：

```text
results.trajectories
results.N
results.valid_pos
results.valid_idx
results.stats
results.expert_sa
results.expert_sa_role       = 'qa_compat_only'
results.train_authority      = 'per_trajectory_phys_transition'
results.stage3_in_library    = false
results.schema_version
results.stage2_dynamics_version
results.action_mapping_version
results.obs_norm_version
```

每个 `trajectories(k)` 保存：

```text
idx / valid_pos / X / geo
tau / wE_used
stageB / ok_use
transitions
```

每条 `transitions` 的当前权威字段：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `state` | `N×13` | 当前原始物理观测 |
| `action_norm` | `N×3` | 实际执行并用于训练的归一化动作 |
| `next_state` | `N×13` | 执行动作后的真实下一观测 |
| `done` | `N×1 logical` | stage2 episode 是否终止 |
| `intercepted` | `N×1 logical` | 是否因 6 m 内拦截终止 |
| `stage2_done` | `N×1 logical` | 是否正常到达 `Tev` |
| `state_abs` | `N×18` | 当前绝对态 |
| `next_state_abs` | `N×18` | 下一绝对态 |
| `action/action_phys` | `N×4` | 原始 `[w,gamma,wT,wE]` 审计值 |
| `t` | `N×1` | 当前 stage2 相对时间 |
| `r_min1/r_min2` | 标量 | 数值重放全程最小弹间距 |
| `terminal_abs` | `1×18` | 执行最后一拍后的真实 stage2 末态 |
| `t_end` | 标量 | stage2 实际终止时刻 |
| `n_actions_clipped` | 标量 | 本轨迹发生动作裁剪的步数 |
| 四类版本字段 | 字符串 | schema、动力学、动作映射、观测归一化版本 |

`results.stats` 除阶段 A/B 统计外，还汇总：

```text
n_phys_traj
n_phys_steps
n_phys_intercepted
```

### 3.4 `expert_sa` 的保留用途

`results.expert_sa` 仍是 `N_total_steps×17`：

```text
第 1~13 列  = state
第 14~17 列 = [w,gamma,wT,wE]
```

它只用于旧工具兼容、范围检查和物理参数审计。当前训练加载器优先读取
per-trajectory `state/action_norm/next_state/done`；不得再把 `expert_sa` 直接当作
已对齐的 replay transition。

### 3.5 已归档 P3 批处理的历史事实

现有日志记录的一次双半批正式运行处理了 `2351` 个有效样本，而不是 1000 个：

| 指标 | half1 | half2 | 合计 |
|---|---:|---:|---:|
| 处理样本 | 1175 | 1176 | 2351 |
| 阶段 A 成功 | 1175 | 1176 | 2351 |
| stageB ok | 447 | 462 | 909 |
| stageB debug | 191 | 187 | 378 |
| stageB fail | 537 | 527 | 1064 |
| 可用轨迹 | 638 | 649 | 1287 |
| `expert_sa` 步数 | 70374 | 74144 | 144518 |
| 日志累计用时 | 452.5 min | 448.0 min | 900.5 min |

由此可得：

- 阶段 A 成功率：`100%`；
- stageB 可用率：`1287/2351 ≈ 54.7%`；
- 每条可用轨迹平均约 `112.3` 步；
- ok/debug/fail 合计与全部有效样本数一致。

历史 stage3 缓存日志对 1287 条可用轨迹给出的统计为：

```text
r_minT 中位数 = 1.324 m
r_minT <= 10 m 比例 = 92.9%
R1 > 0 比例 = 92.9%
bonus > 0 比例 = 69.8%
```

以上 P3 与 stage3 数字属于现有归档库和当时缓存流程的历史实跑结果，
不能据此声称当前 `ratio_log_box_v1 + phys_transition_v1` 代码已经重新全量生成。

---

## 4. 避坑指南 / 已知问题

### 4.1 当前代码与现有归档库不是同一代数据

`expert_lib_1000_file.m` 当前固定指向：

```text
E:\matlab\result_0628\reault_p3_1000\expert_lib_1000.mat
```

现有 1287 条归档库和其 144518 步统计早于当前“生成时直接写
`phys_transition_v1`”及默认“`ratio_log_box_v1`”的最终实现。
读取数据时应检查库内版本字段/实际 transition 字段，不能仅看文件名。

### 4.2 `action` 不等于实际训练动作

- `action/action_phys` 保存优化器给出的原始物理权重；
- `action_norm` 才是裁剪后实际推进 `next_state` 的动作；
- 若 `n_actions_clipped>0`，原始物理动作与共享核心实际执行的等效动作可能不同。

训练和 BC 标签必须使用 `action_norm`。`action_phys` 只能用于审计、画图和回查
`theta_K`。

### 4.3 动作坐标顺序已变化

旧版：

```text
td3_box_log_v1 = [gamma_n, wT_n, wE_n]
```

当前默认：

```text
ratio_log_box_v1 = [rhoE_n, rhoT_n, gamma_n]
```

两者都是 3 维且范围都为 `[-1,1]`，仅凭形状无法发现用错顺序。必须核对
`action_mapping_version`。

### 4.4 优化盒与训练比值盒不完全重合

单打靶允许 `wE` 最低到 10、`wT` 最低到 1；当前比值动作盒限制
`rhoE<=100`、`rhoT>=1e-2`。因此部分优化结果会在抽取阶段被裁剪。
这正是必须完整重放并记录 `n_actions_clipped` 的原因。

### 4.5 `missT`、真实 `r_minT` 与奖励标签不是一回事

- `stageB.miss(3)`：CasADi 符号积分末端 ZEM；
- sidecar `r_minT`：真实 stage3 PN rollout 的最小弹目距离；
- `R1/bonus/both_ok`：按特定 `reward_version` 从物理事实投影得到。

用 `missT<=30` 入库不保证最终命中；反过来，较大的 stage2 ZEM 也不必然意味着
stage3 无法回收。

### 4.6 主库不含 stage3，也不直接保存训练奖励

`results.stage3_in_library=false` 是有意设计，不是字段缺失。主库负责 stage2 物理
transition；stage3 终端事实由带专家库指纹、`stage3_version` 和 `stage3_dt` 的
sidecar 保存；奖励由 Python 侧按版本投影。

### 4.7 `P3_merge_halves.m` 仍按旧库级结构手工组装

当前合并脚本只合并 `trajectories/expert_sa/valid_pos/valid_idx/N` 和旧版阶段统计，
没有复制或重建：

```text
expert_sa_role / train_authority / stage3_in_library
schema_version / stage2_dynamics_version
action_mapping_version / obs_norm_version
n_phys_traj / n_phys_steps / n_phys_intercepted
```

半库中的每条 `transitions` 字段仍在，但合并后的库级版本和统计会缺失。
因此不能把“半库生成函数具备新 schema”直接等同于“旧合并脚本产物也完整具备新 schema”。

### 4.8 checkpoint 变量名与累计用时

- checkpoint 读取变量名是 `results_ckpt`；
- 最终库读取变量名是 `results`；
- 当前 checkpoint 保存处写入的是本次进程的 `toc(t0_total)`，没有把
  `elapsed_prev` 加入 checkpoint。一次 resume 后直接跑完时最终用时正确；若多次从
  “续跑过程中产生的新 checkpoint”再次恢复，累计用时可能少计更早的运行段。

### 4.9 P3 文件名中的“1000”已不代表样本上限

`P3_generate1000.m` 和两个 half 脚本都根据当前有效样本总数运行；当
`samples_clean.mat` 有 2351 个有效行时，实际就是 2351 个待处理样本。
脚本名、输出名和真实 `results.N` 必须分开理解。

### 4.10 CasADi 路径仍含机器相关回退值

主入口优先检查 MATLAB toolbox 下的 CasADi，否则回退到：

```text
E:\matlab\casadi-3.7.2-windows64-matlab2018b
```

该路径是当前机器约定，不是可移植的项目相对路径。

### 4.11 `pipeline_validation_report.json` 是旧动作映射下的 smoke 记录

现有报告显示：

```text
schema_version=phys_transition_v1
action_mapping_version=td3_box_log_v1
status=passed
```

而当前默认元数据已经是 `ratio_log_box_v1`。因此该报告可证明物理 schema 管线曾在
旧动作映射下通过 smoke，不能作为当前默认比值动作全链路重新验收的证据。

### 4.12 旧脚本和注释中仍有历史命名

- `generate_ps_*`、`solve_ps_*` 中的 `ps` 来自早期伪谱方案；
- 部分注释仍写“阶段B伪谱 NLP”；
- `zhijian*.m` 主要围绕旧 `expert_sa` 和归档库；
- `plot_expert_traj_3d.m` 读取 `transitions.state_abs`，显示的是 stage2 规避段，
  不是包含 stage1/stage3 的完整任务轨迹。

阅读代码时应以实际调用链和版本字段为准，不以历史文件名推断当前算法。

---

## 5. 当前代码状态摘要

- 当前生成器已经从“输出扁平 `(s,a)` 表”演进为“按轨迹直接输出可训练物理
  transition”。
- 专家抽取与在线环境共用同一个 stage2 数值动力学核心，动作裁剪、`s'`、终止和真实
  末态在生成时完成对齐。
- 当前默认动作空间已由绝对 `[gamma,wT,wE]` 坐标改为式（47）更直接的
  `[w/wE,wT/wE,gamma]` 对数比值坐标。
- 主库只负责 stage2 物理事实，stage3 和奖励投影已经解耦为带版本的 sidecar 流程。
- 现有 2351 条输入、1287 条可用轨迹、144518 步的 P3 归档结果属于旧一代全量数据；
  它可用于历史对照，但不能替代对当前 schema 与动作映射的版本核对。
- 当前最需要警惕的结构性遗留是：旧 P3 合并脚本不完整保留新库级元数据，以及多个旧
  文件名/质检脚本仍把 `expert_sa` 放在比当前训练接口更显眼的位置。
