# 会话纪要：纯 Actor BC 诊断 + 比值动作空间对照（2026-07-19）

> **用途**：新对话接续本上下文，继续「状态空间消融」等改进。  
> **相关计划**：`.cursor/plans/td3稳定改进路线_782e8ebf.plan.md`（§3 controlled-bc 已完成；§5 conditional-ablations 待做）  
> **用法说明**：`rl_td3/md document/BC_TD3_USAGE.md`（已同步动作映射与 BC 诊断结果）  
> **Python**：`E:\Python\venvs\rl_pytorch\Scripts\python.exe`  
> **工作目录**：`D:\LTY\yan\yanerxia\GAIL`

---

## 0. 一句话结论

1. **纯 Actor BC（旧动作坐标）**：可学性不足（val MSE≈0.36），闭环同 IC 75% vs 专家回放 90%，判断门 `fail_high_val_error`，**禁止直接进 TD3**。  
2. **比值动作空间对照**：物理上合法、旧→新标签动力学等价（`Δa_M~1e-8`），但 BC **无实质改善**（γ MAE 仍≈0.55；同 IC both_ok 仍 75%/90%）。  
3. **下一步**：按计划进入 **状态空间消融**（§5），不要再调 λ/奖励/GAIL，也不要先做完整 TD3 精调。

---

## 1. 对话起点与前置进度

本对话开始时，改进计划中以下 todo **已完成**（勿重做）：

| id | 内容 |
|----|------|
| data-alignment | 专家库对齐、obs 归一化、一致性检查 |
| legacy-full-rebuild | 0714 全量 stage2 + stage3（dt=1e-3） |
| transition-core-and-schema | 共享 `stage2_core` + phys_transition schema |
| generation-pipeline-fix | 新库出库即对齐 |
| stage3-sidecar | 可恢复 sidecar |
| reward-projection | 廉价奖励投影 |
| pipeline-validation | 小批量验收通过 |
| expert-sampling | 轨迹均衡约 99k + train/val 按轨划分 |

**本对话完成**：

| id | 内容 |
|----|------|
| controlled-bc | 独立纯 Actor BC 入口 + 首轮诊断 |
| （计划外对照） | `ratio_log_box_v1` 动作空间实现 + 同划分 BC/闭环对照 |

**仍 pending**：

| id | 内容 |
|----|------|
| td3bc-refine | BC 初始化 + Q 归一化 TD3+BC（**判断门未过，暂不执行**） |
| conditional-ablations | 状态 / 动作可辨识性 / Q-filter / 奖励（**下一步重点：状态消融**） |

专家集摘要（`rl_td3/data/expert_set_99k.json`，seed=0）：

- Replay 94753 / Actor 73721（排除拦截）
- train 轨 1158 / val 轨 129
- Actor 拦截步占比 0%；全库拦截率约 24%
- expert both_ok_val ≈ 0.705；未拦截子集 ≈ 0.929

---

## 2. 本对话做了什么（按时间线）

### 2.1 纯 Actor BC 诊断入口（controlled-bc）

**动机（计划原文）**：现有 `train_td3.py` 只有 Replay 预填充 + TD3 同步衰减 BC 正则，**没有**真正先把 Actor 训成专家；随机 Actor 从第 0 步同时吃 Q 与弱 BC，历史上易贴边坍塌。

**新增/修改**：

| 文件 | 作用 |
|------|------|
| `rl_td3/train_bc.py` | 独立 BC-only 训练/验证/可选闭环 |
| `rl_td3/bc_actor.py` | 与 SB3 TD3 Actor 同构 MLP `[256,256]+tanh`；可 `copy_bc_weights_to_td3_actor` |
| `rl_td3/expert_sampling.py` | 增加 `val_actor_step_indices`（val 轨全步）；`load_sample_plan` / `bc_arrays_from_plan` |
| `rl_td3/evasion_env.py` | `reset_from_abs`（专家 IC 回放对照） |
| `rl_td3/tests/test_bc_actor.py` | 合成数据单测 |
| `rl_td3/tools/run_bc_closed_loop.py` | 对已有 ckpt 补跑闭环 |

**训练约定**：

- 观测：归一化 13 维（`obs_norm`，`OBS_SCALE` 固定）
- 划分：与 `expert_set_99k.npz` 同一 train/val 轨；Actor 默认排除拦截
- 指标：train/val MSE、逐动作 MAE、贴边率（\|a\|≥0.99）、闭环联合成功率
- 对照：同 IC 上「专家动作开环回放」vs「BC 策略」
- 判断门写入 `*_report.json`，**不自动进 TD3**

**首轮旧坐标结果**（`rl_td3/checkpoints_bc/bc_diag_seed0/`，seed=0）：

| 项 | 数值 |
|----|------|
| train / val 步数 | 73721 / 11501 |
| best epoch | 3（early stop @18，patience=15） |
| best val MSE | **0.360** |
| val MAE（γ / wT / wE） | 0.546 / 0.354 / 0.516 |
| BC 贴边率 | ≈0%（未坍到边界） |
| 专家标签贴边（val） | γ 31% / wT 4% / wE 27% |
| MC both_ok（20 ep, samples_clean） | 50% |
| 专家回放 both_ok（20 轨） | 90% |
| 同 IC 上 BC both_ok | 75% |
| 判断门 | **`fail_high_val_error`** |

解读：策略偏内点、学不到专家贴边动作；映射误差大（尤其 γ）；闭环尚可但离线门未过。

### 2.2 比值动作空间对照（用户提议）

用户提出将动作改为：

\[
\left(\frac{w}{w_E},\ \frac{w_T}{w_E},\ \gamma\right)
\]

理由：式(47) 实际通过相对权重 `w1/wE, w2/wE, wT/wE` 起作用；专家是打靶/伪谱解出的四参数，改比值不要求重跑 IPOPT。

**事前判断（对话中共识）**：

- 物理上合法：四参数在 `w=1e4` 固定时可无损压成三维比值。
- **并非降维消歧**：新旧三维在固定 `w` 下可逆同构；**γ 完全不变**，故不一定改善 BC。
- 重标必须用「已裁剪后」权威 `action_norm` 反解物理量再算比值，避免标签与 `next_state` 不一致。
- 适合作为状态消融前的低成本对照；失败则进状态消融。

用户选定映射方式：**对数盒 `log_box`**（按当前权重盒推导范围）。

### 2.3 比值映射实现细节

**版本名**：`ratio_log_box_v1`（现为默认）；旧版保留为 `td3_box_log_v1` 作 A/B。

**归一化（动作顺序 `[ρE_n, ρT_n, γ_n]`）**：

```text
ρE = w/wE ∈ [1, 100]     →  ρE_n = log10(ρE) - 1
ρT = wT/wE ∈ [1e-2, 1e3] →  ρT_n = (log10(ρT) - 0.5) / 2.5
γ  ∈ [-3, 3]             →  γ_n  = γ / 3
```

（由 `w=1e4` 与旧盒 `wE∈[1e2,1e4]`、`wT∈[1e2,1e5]` 推出。）

**核心文件**：

| 文件 | 变更 |
|------|------|
| `rl_td3/action_mapping.py` | Python 权威：双版本 phys↔norm、相对权重、`remap_action_norm` |
| `rl_td3/matlab/phys_to_action_norm.m` | 支持 version 参数 |
| `rl_td3/matlab/action_norm_to_phys.m` | 同上 |
| `rl_td3/matlab/stage2_step.m` | 按 version 解码；式(47) 直接传相对权重且 `wE=1`；记录 `a_M` / 8g 饱和 / 旧盒可行域 |
| `rl_td3/matlab/stage2_core_init.m` | `cfg.action_mapping_version` 可覆盖 |
| `rl_td3/matlab/transition_schema_meta.m` | 默认 `action_mapping_version=ratio_log_box_v1` |
| `rl_td3/matlab/env_step.m` | cache key 含 amap |
| `rl_td3/evasion_env.py` | `action_mapping=` 传入 MATLAB cfg；norm/denorm 委托 `action_mapping.py` |
| `rl_td3/load_expert_buffer.py` | `apply_action_mapping`；加载时旧→新确定性重标 |
| `rl_td3/train_bc.py` | `--action-mapping`；判断门关注 γ MAE |
| `rl_td3/train_td3.py` | `--action-mapping`；env 与 buffer 一致 |
| `rl_td3/tools/check_action_mapping_equiv.py` | MATLAB 抽检旧/新 `a_M` 与 `s'` |
| `rl_td3/tests/test_action_mapping.py` | 往返 / 相对权重 / remap 单测 |

**专家数据策略**：不重跑 IPOPT、不重放全量 stage2/stage3；只对已有 `action_norm` 做确定性重标。

**等价抽检**（`check_action_mapping_equiv.py`，30 步）：

```text
max|Δa_M| ≈ 1e-8
max|Δs'|  ≈ 5e-11
[OK] 专家标签旧→新 a_M / next_state 等价
```

### 2.4 比值 BC 实验结果

产物：`rl_td3/checkpoints_bc/bc_ratio_seed0/`  
命令要点：同 `expert_set_99k.npz`、seed=0、`--action-mapping ratio_log_box_v1`、`--closed-loop`（20 MC + 20 回放轨）。

| 项 | 旧 `td3_box_log_v1` | 比值 `ratio_log_box_v1` |
|----|---------------------|-------------------------|
| best val MSE | 0.360 | **0.321**（略降；部分因 ρT 对数压缩，**不可单凭 MSE 下结论**） |
| γ MAE | 0.546 | **0.549**（未改善） |
| ρE / ρT（或旧 wT/wE）MAE | wT 0.35 / wE 0.52 | ρE 0.515 / **ρT 0.185** |
| BC 贴边 | ≈0% | ≈0% |
| 同 IC both_ok（BC / 专家） | 75% / 90% | **75% / 90%** |
| MC both_ok | 50% | 50% |
| 旧权重盒可行域（val 标签） | — | ≈99.8% |

报告内决策字段（已写入 JSON）：

```text
decision.next_step = state_ablation
decision.enter_td3 = false
判断门 verdict = fail_high_val_error
```

**结论**：比值坐标实现正确且动力学等价，但对「状态→动作可学性」帮助有限；瓶颈更像是 **13 维观测对专家动作（尤其 γ）辨识不足**，而非单纯 `(wT,wE)` 参数化。

---

## 3. 关键产物路径

| 产物 | 路径 |
|------|------|
| 旧坐标 BC best | `rl_td3/checkpoints_bc/bc_diag_seed0/bc_actor_best.pt` |
| 旧坐标报告 | `rl_td3/checkpoints_bc/bc_diag_seed0/bc_actor_report.json` |
| 比值 BC best | `rl_td3/checkpoints_bc/bc_ratio_seed0/bc_ratio_best.pt` |
| 比值报告（含 closed_loop + decision） | `rl_td3/checkpoints_bc/bc_ratio_seed0/bc_ratio_report.json` |
| 99k 采样计划 | `rl_td3/data/expert_set_99k.npz` (+ `.json`) |
| 对齐 transition | `rl_td3/data/expert_transitions_aligned.mat` |
| stage3 sidecar | `rl_td3/data/expert_terminal_cache.npz` |
| 专家库 | `E:\matlab\result_0714\expert_lib_1000.mat` |
| BC 日志示例 | `rl_td3/logs/bc_ratio_20260719_203818.log` |

---

## 4. 复现命令（新对话可直接用）

```powershell
cd D:\LTY\yan\yanerxia\GAIL
$PY = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"

# 映射单测
& $PY rl_td3/tests/test_action_mapping.py
& $PY rl_td3/tests/test_bc_actor.py

# 旧→新 a_M/s' 等价（需 MATLAB Engine）
& $PY rl_td3/tools/check_action_mapping_equiv.py --n 30

# 比值 BC + 闭环（已跑过，勿无故覆盖）
& $PY rl_td3/train_bc.py `
  --sample-plan rl_td3/data/expert_set_99k.npz `
  --save-dir rl_td3/checkpoints_bc/bc_ratio_seed0 `
  --name bc_ratio --seed 0 `
  --action-mapping ratio_log_box_v1 `
  --closed-loop --eval-episodes 20 --replay-traj 20
```

---

## 5. 新对话应做什么 / 不应做什么

### 5.1 应做（优先：状态空间消融）

计划 §5 原文方向：

1. **状态消融**（首选）：在**不重跑 IPOPT** 前提下，用现有 `state_abs` 派生候选观测，与当前归一化 13 维对照纯 BC：
   - 增加 LOS 角速度 / 必要姿态的相对状态
   - 目标中心绝对 / 笛卡尔状态
   - **不预设哪种更好**；仍用同一 `expert_set_99k` 划分与 seed
2. 主指标仍用：val MSE（尤其 **γ MAE**）、贴边率、同 IC both_ok vs 专家回放；**不要只看新坐标 MSE**。
3. 若某状态表示使 BC 闭环明显接近专家且 γ 可学，再考虑用该表示 +（可选）比值动作进入 `td3bc-refine`。

### 5.2 暂缓 / 禁止

- **不要**因比值 MSE 略降就宣布动作空间成功并进 TD3。
- **不要**先加动作边界惩罚、切回 GAIL、恢复旧 stage2 ZEM 逐步奖励。
- **不要**无故全量 rebuild stage2 / 重跑 IPOPT / 重算 stage3（除非状态派生证明缺字段且无法从 `state_abs` 恢复）。
- 动作可辨识性（`a_M` 对权重敏感度、8g 饱和）可作为状态消融的**并行诊断**，但首轮消融以状态为主。

### 5.3 技术锚点（避免新对话踩坑）

- 默认动作映射已是 **`ratio_log_box_v1`**：`train_td3` / `EvasionEnv` / `load_expert_buffer` 需一致；旧 aligned 标签会自动重标。
- 旧库 `action_mapping_version` 字段多为 `td3_box_log_v1`（或缺失时按旧处理）。
- 专家回放依赖 `state_abs` 首步 + 重标后的 `action_norm`；`reset_from_abs` 已实现。
- BC Actor 权重与 SB3 `mu.*` 对齐，备后续 TD3 初始化（但当前未过门）。
- 闭环里「专家回放 r_minT mean」可能被少数离群拉高，看 **both_ok 率** 比看 mean r_minT 更稳。

---

## 6. 建议的新对话开场提示（可复制）

```text
请阅读：
  rl_td3/md document/SESSION_SUMMARY_20260719_BC_AND_RATIO_ACTION.md
  rl_td3/md document/BC_TD3_USAGE.md（§2.1 动作、§3.8 BC）
  .cursor/plans/td3稳定改进路线_782e8ebf.plan.md（§5 消融）

背景：纯 BC 与比值动作对照均未过判断门（γ 不可学、同 IC 75% vs 专家 90%）。
任务：开始状态空间消融——用 state_abs 派生候选观测，同一 expert_set_99k 划分重跑 train_bc，
比较 γ MAE 与同 IC 闭环；通过判断门前不要进 TD3。
```

---

## 7. 版本戳（本对话结束时）

```text
schema_version          = phys_transition_v1
stage2_dynamics_version = stage2_rk4_eq47_v1
action_mapping_version  = ratio_log_box_v1   # 默认；旧标签可重标自 td3_box_log_v1
obs_norm_version        = obs_scale_0714_v1
stage3_version / dt     = pn_rollout_v1 / 1e-3
reward_version          = r1_bonus_v1
```
