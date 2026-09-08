# 纯 TD3 对照基线使用说明

> **生成时间**：2026-07-26  
> **更新**：2026-09-08（默认噪声改为标定 ×30；显式 `rls_eta`；验证仍未通过，勿重训）  
> **用途**：论文 / 正式实验中相对 BC-TD3 的无专家对照  
> **共享环境与协议**：与 [`rl_td3/md document/BC_TD3_USAGE.md`](../../rl_td3/md%20document/BC_TD3_USAGE.md) 一致  
> **改动过程 / 验证待办（跨对话上下文）**：[`PLAN1_IDENT_CHANGELOG_AND_TODO.md`](PLAN1_IDENT_CHANGELOG_AND_TODO.md)

---

## 0. 对照定义

| 项 | 纯 TD3（本目录） | BC-TD3（`rl_td3`） |
|----|------------------|-------------------|
| 算法类 | SB3 原生 `TD3` | `td3_bc.TD3BC` |
| Replay 专家预填 | 无 | 有（约 `buf×0.33`） |
| Actor BC 正则 | 无 | `λ(t)‖π(s_E)-a_E‖²` 衰减 |
| `learning_starts` | 1000 | 有 BC 时 0 |
| 环境 / 奖励 / 观测 / 动作 | 复用 `rl_td3` | 同左 |
| 主超参（steps/lr/net/noise/…） | 与 `train_td3.py` 对齐 | 默认见 BC 文档 §4.2 |
| 产物目录 | `td3_baseline/checkpoints*` | `rl_td3/checkpoints*` |
| 参数辨识 CLI（Plan 1） | **有**（见 §7） | **不暴露**；默认 `analytic_truth` 行为不变 |

**不要**用 `rl_td3/train_td3.py --no-bc` 充当本正式基线：该入口仍挂在 `TD3BC` 上，且 CLI 保留专家相关参数，易混淆。

---

## 1. 目录

```text
td3_baseline/
├── train_td3_pure.py
├── evaluate_td3_pure.py          # 转发 rl_td3/evaluate_td3.py（含辨识评估参数）
├── run_ident_ablation_pure.py    # 同 IC / 同噪声种子的三模式配对消融
├── md document/
│   └── TD3_BASELINE_USAGE.md
├── checkpoints/
├── checkpoints_smoke/
└── logs/
```

复用（不复制）：`rl_td3/evasion_env.py`、`action_mapping.py`、`obs_norm.py`、`ident_cfg.py`、`matlab/`、`evaluate_td3.py`。

Plan 1 辨识相关 MATLAB（共享，默认关闭）：

| 文件 | 作用 |
|------|------|
| `rl_td3/matlab/stage2_ident_init.m` | 三通道 Kalman + FF-RLS 状态初始化 |
| `rl_td3/matlab/stage2_ident_step.m` | 测量噪声 → KF → 因果 sparse FF-RLS |
| `rl_td3/matlab/stage2_ident_commit.m` | 提交本拍饱和 `a_M`，供下一拍回归 |
| `rl_td3/matlab/stage2_core_init.m` | `coeff_mode` / 噪声 / KF / RLS 默认配置 |
| `rl_td3/matlab/stage2_step.m` | 可选 `ident_state`；系数注入式(47) |
| `rl_td3/matlab/env_step.m` | 持久化辨识状态；`t=0` / 配置变化时重置 |
| `evasion_helpers.m` | `computeOptimalCmd_duo_target_coeffs` |
| `rl_td3/matlab/test_stage2_ident.m` | 单元 / 回归测试 |
| `rl_td3/ident_cfg.py` | CLI → `matlab_cfg` 构建与落盘 |

---

## 2. 依赖与工作目录

| 项 | 约定 |
|----|------|
| Python | `E:\Python\venvs\rl_pytorch\Scripts\python.exe` |
| 工作目录 | `D:\LTY\yan\yanerxia\GAIL` |
| MATLAB Engine | 与 BC-TD3 相同（训练初态 `samples_clean.mat`；EvalCallback 默认 `samples_clean_val.mat` 中固定 150 个初态） |
| 专家库 / terminal cache | **不需要** |

---

## 3. 训练

### 3.1 命令

```powershell
cd D:\LTY\yan\yanerxia\GAIL
$PY = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"

# 冒烟（5k 步 → td3_baseline/checkpoints_smoke）
& $PY td3_baseline/train_td3_pure.py --smoke
注：想指定非默认冒烟测试步数，在最后加--steps 20000（想要的步数）即可

# 正式 300k（务必独立 save-dir）
& $PY td3_baseline/train_td3_pure.py `
  --save-dir td3_baseline/checkpoints/training_300k_$(Get-Date -Format 'yyyyMMdd_HHmm') `
  --name td3_pure --seed 0

# 正式非默认步数（例：260k）——必须同时改 --steps 与独立 save-dir/--name
& $PY td3_baseline/train_td3_pure.py `
  --steps 260000 `
  --save-dir td3_baseline/checkpoints/training_260k_$(Get-Date -Format 'yyyyMMdd_HHmm') `
  --name td3_pure_260k `
  --seed 0
```

默认 **不开启** 参数辨识（`coeff_mode=analytic_truth`）。带辨识训练示例见 §7.3。

### 3.2 默认超参

| 参数 | 默认值 | CLI |
|------|--------|-----|
| `total_steps` | 300,000 | `--steps` |
| `learning_rate` | `1e-4` | `--lr` |
| `batch_size` | 256 | `--batch` |
| `buffer_size` | 300,000 | `--buf` |
| `learning_starts` | 1,000 | （代码内） |
| `gamma` / `tau` | 0.99 / 0.005 | （代码内） |
| `policy_delay` | 3 | （代码内） |
| `net_arch` | [256, 256] | （代码内） |
| `noise_sigma` | 0.1 | `--noise` |
| `eval_freq` | 25,000 | `--eval-freq` |
| `n_eval_episodes` | 150（固定 val 初态子集；`0`=整池 300） | `--n-eval` |
| 评估初态取法 | 定序置换取前 N 个，种子 0；清单落 `{save_dir}/eval_logs/eval_ic_pool.json` | `--eval-ic-seed`；`--eval-random-ic` 回退随机抽 |
| `train_samples` | `samples_clean.mat` | `--train-samples` |
| `eval_samples` | `samples_clean_val.mat` | `--eval-samples` |
| `grad_clip` | ±10.0 | `--grad-clip`（0=关） |
| `seed` | 0 | `--seed` |
| `action_mapping` | `ratio_log_box_v1` | `--action-mapping` |
| checkpoint | 每 50,000 步 | |

`--smoke`：`steps=5000`，`save_dir=td3_baseline/checkpoints_smoke`，`name=td3_pure_smoke`，且 `eval_freq=1000` / `n_eval=5`（否则冒烟步数内不会触发评估）。

训练内评估为**固定初态协议**（`rl_td3/fixed_ic_eval.py`）：每次评估遍历同一组 val 初态，配合 `deterministic=True` 使评估无采样噪声，`best_model` 不再由抽样运气决定。

### 3.3 产物

| 产物 | 路径 |
|------|------|
| 最终模型 | `{save_dir}/{name}.zip` |
| 最优模型 | `{save_dir}/best/best_model.zip` |
| 周期 checkpoint | `{save_dir}/{name}_*_steps.zip` |
| EvalCallback | `{save_dir}/eval_logs/`（每 `--eval-freq` 步 × `--n-eval` 局，默认 25k×150 固定 val-IC；同目录 `eval_ic_pool.json` 记录初态清单） |
| TensorBoard | `{save_dir}/tb_logs/` |
| 终端 tee | `td3_baseline/logs/{name}_{时间戳}.log` |
| 辨识运行配置（Plan 1） | `{save_dir}/ident_run_config.json` |

---

## 4. 蒙特卡洛评估

与 BC-TD3 **同一协议**（默认全量 `samples_clean_test`）：

```powershell
& $PY td3_baseline/evaluate_td3_pure.py `
  --model td3_baseline/checkpoints/<你的目录>/best/best_model
```

也可直接调用 `rl_td3/evaluate_td3.py`（封装仅转发）。旧随机协议：`--episodes 100 --seed 0`。

日志默认写入模型所在 training 目录，例如：
`{save_dir}/monte_carlo_eval_best_seed{seed}_{时间戳}.log`（可用 `--log-file` / `--no-log`）。

成功判据：双防 `r_MD≥6`、打击 `r_minT≤10`、both_ok、拦截 —— 与 BC 文档 §5.3 相同。

评估时也可覆盖 `--coeff-mode` 等辨识参数（见 §7），用于**同一 checkpoint、不同系数路径**的配对对比；不改模型权重。

---

## 5. 实验建议

1. 与 BC-TD3 主跑使用相同 `steps`、`lr`、`action-mapping`；训后 MC 用同一 `--samples`（默认 test）。
2. 下结论建议 ≥3 个训练 `--seed`；全量 test 评估无需再对齐 MC 抽样种子；非全量评估要对比同一 `--seed`
3. 产物目录与 `rl_td3/checkpoints` 隔离，避免覆盖。
4. **参数辨识**：2026-09-08 的 30 IC 配对消融**未通过**（见 changelog §8.4）。在用户拍板前**不要**用 `sparse_ffrls` 重训纯 TD3，也勿改专家库 / BC-TD3（Plan 2）。

---

## 6. 关键键文件

| 需求 | 文件 |
|------|------|
| 纯 TD3 训练 | `td3_baseline/train_td3_pure.py` |
| 评估封装 | `td3_baseline/evaluate_td3_pure.py` |
| 辨识配对消融 | `td3_baseline/run_ident_ablation_pure.py` |
| 辨识 CLI 公共模块 | `rl_td3/ident_cfg.py` |
| 环境 / 奖励 | `rl_td3/evasion_env.py` + `rl_td3/matlab/` |
| BC-TD3 主线 | `rl_td3/train_td3.py` / `rl_td3/td3_bc.py` |
| BC-TD3 用法 | `rl_td3/md document/BC_TD3_USAGE.md` |

---

## 7. 参数辨识（Plan 1）

### 7.1 设计摘要

- **目的**：攻击方制导式(47) 的 `Q/b/v_r` 不再默认只用真值解析；可在带噪测量下用三状态 Kalman + 因果 FF-RLS（可选 L1）在线估计，验证是否降低系数抖动 / 改善闭环。
- **不动**：13 维 policy 观测仍为干净状态；防御弹 PN 与 RK4 交战动力学用真值；式(12) 不注入过程噪声；不改专家库 / BC-TD3。
- **模式**（`coeff_mode`）：

| 模式 | 含义 |
|------|------|
| `analytic_truth` | 现有解析真值路径（**默认**，与旧结果一致） |
| `analytic_noisy` | 带噪测量代入解析 `computeBC`（带噪基线） |
| `sparse_ffrls` | Kalman 预处理 + 固定遗忘因子 RLS + L1 soft-threshold |

- **影子模式**：`--ident-shadow` → `ident_apply_cmd=false`：仍跑 KF/RLS 诊断，但指令用解析真值（只记录不控制）。
- **因果时序**：本拍 RLS 用上一拍**量测** `r` 与上一拍饱和 `a_M`；首拍仅 PN 解析 warm-start（`K_D=4`，不用真值 `a_D`），不做 RLS；episode `t=0` 与配置变化时重置辨识状态。

### 7.2 CLI 参数（训练 / 评估共用）

由 `rl_td3/ident_cfg.py` 注入；`train_td3_pure.py` 与 `evaluate_td3.py`（含 `evaluate_td3_pure.py`）均支持。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--coeff-mode` | `analytic_truth` | 上表三选一 |
| `--ident-shadow` | off | 影子：估参但不改指令 |
| `--sigma-scale` | `30` | 噪声标定倍率元数据；`sigma-*` 存的是已缩放最终值 |
| `--sigma-q` | `3°`（rad） | LOS 角度测量 σ（0.1° × 30） |
| `--sigma-qdot` | `0.03` rad/s | LOS 角速率 σ（0.001 × 30） |
| `--sigma-theta` / `--sigma-psi` | 同 `σ_q` | 姿态角测量 σ |
| `--sigma-r` | `0` m | 距离 σ（第2节约定精确；不注入量测） |
| `--sigma-rdot` | `0` m/s | 距离率 σ（同上） |
| `--kf-qj` | `1e4` | Kalman 白 jerk 强度（滤波器参数，非植物过程噪声） |
| `--kf-sigma-a0` | `50` | 加速度初协方差尺度 |
| `--kf-sigma-r` / `--kf-sigma-rdot` | `10` / `0.5` | KF **设计**标准差，与实际 σ 解耦 |
| `--rls-lambda` | `0.98` | 遗忘因子 |
| `--rls-mu` | `1e-3` | 式(31)稀疏度权重（记录用） |
| `--rls-eta` | `0.1` | 缩放域软阈值；`0`=关闭收缩（20 IC 标定候选） |
| `--rls-P0` | `1e4` | RLS 初协方差 |
| `--rls-r0` / `--rls-a0` | `1e4` / `50` | 回归量缩放 |
| `--noise-seed` | `0` | 测量噪声可复现种子（hash RNG，不污染 MATLAB 全局 RNG） |

训练时完整配置写入 `{save_dir}/ident_run_config.json`。

### 7.3 带辨识训练 / 评估示例

```powershell
$PY = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"
cd D:\LTY\yan\yanerxia\GAIL

# 仅评估：同一 best_model，换系数路径（不重训）
& $PY td3_baseline/evaluate_td3_pure.py `
  --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model `
  --coeff-mode sparse_ffrls --rls-eta 0.1 --noise-seed 0

& $PY td3_baseline/evaluate_td3_pure.py `
  --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model `
  --coeff-mode analytic_noisy --noise-seed 0

# 确认默认路径未改（应与历史 MC 一致量级）
& $PY td3_baseline/evaluate_td3_pure.py `
  --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model `
  --coeff-mode analytic_truth

# 带辨识闭环训练（2026-09-08 验证未通过，禁止执行，仅留语法）
& $PY td3_baseline/train_td3_pure.py `
  --save-dir td3_baseline/checkpoints/training_300k_ffrls_$(Get-Date -Format 'yyyyMMdd_HHmm') `
  --name td3_pure_ffrls --seed 0 `
  --coeff-mode sparse_ffrls --rls-eta 0.1 --noise-seed 0
```

### 7.4 同 IC / 同噪声配对消融

脚本：`td3_baseline/run_ident_ablation_pure.py`  
对同一 checkpoint、前 N 个 test IC、每局 `noise_seed + local_idx`，依次跑：

`analytic_truth` → `analytic_noisy` → `sparse_ffrls(η)` →（可选）`sparse_ffrls(η=0)`

```powershell
& $PY td3_baseline/run_ident_ablation_pure.py `
  --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model `
  --episodes 30 --noise-seed 0 --also-eta0 `
  --out-dir td3_baseline/checkpoints/training_300k_20260730_1917/ident_ablation_N30_20260908
```

产物：`summary.json`、`metrics_{mode}.npz`。

**首轮 5-IC 结果摘要**（checkpoint `training_300k_20260730_1917/best`，当时噪声默认仍为 `σ_r/σ_ṙ=10/1`）：

| 模式 | both_ok | 平均 r_MD1 / r_MD2 |
|------|---------|-------------------|
| analytic_truth | 100% | 37.6 / 34.9 |
| analytic_noisy | 80%（IC4 拦截） | 35.0 / 31.3 |
| sparse_ffrls (μ=1e-3) | 100% | 314 / 116 |
| sparse_ffrls (μ=0) | 100% | 370 / 185 |

解读：相对 `analytic_noisy`，稀疏 FF-RLS 恢复了 both_ok，但脱靶距离明显偏大（更保守）。μ>0 与 μ=0 接近。是否“改善”需看更大样本与参数 RMSE / 指令平滑度，不能只看 both_ok。

**2026-09-08 正式 30 IC**（`sigma_scale=30`，`η=0.1`，产物 `ident_ablation_N30_20260908/`）：

| 模式 | both_ok | 平均 r_MD1 / r_MD2 | 平均 r_minT |
|------|---------|---------------------|-------------|
| analytic_truth | 96.7% | 34.6 / 34.3 | 0.10 |
| analytic_noisy | 90.0% | 24.0 / 105.3 | 0.13 |
| sparse_ffrls (η=0.1) | 93.3% | 148.9 / 132.4 | 0.13 |
| sparse_ffrls (η=0) | 93.3% | 184.3 / 190.0 | 20.3 |

结论：**未通过**重训门槛。both_ok 略升，但 miss 系统性偏大；影子 θ RMSE 未收敛。详见 `PLAN1_IDENT_CHANGELOG_AND_TODO.md` §8。

### 7.4.1 影子逐步诊断

脚本：`td3_baseline/run_ident_shadow_diag_pure.py`  
固定 checkpoint，`--ident-shadow` 下逐步落盘 KF / θ / 指令 RMSE。

```powershell
& $PY td3_baseline/run_ident_shadow_diag_pure.py `
  --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model `
  --episodes 5 --noise-seed 0 `
  --out-dir td3_baseline/checkpoints/training_300k_20260730_1917/ident_shadow_diag_plan1
```

产物：`summary.json`、`shadow_diag_series.npz`、`ident_run_config.json`。

**首轮 5-IC 影子摘要**（`σ_r=σ_ṙ=0`，`μ=1e-3`）：KF `r/ṙ` ≈0；θ RMSE 末段均值 ~2e6（未收敛）；指令预测 RMSE ~103。详见 `PLAN1_IDENT_CHANGELOG_AND_TODO.md` §5.1。

### 7.4.2 超参网格（影子）

脚本：`td3_baseline/run_ident_hyperparam_scan_pure.py`

```powershell
& $PY td3_baseline/run_ident_hyperparam_scan_pure.py `
  --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model `
  --episodes 5 --noise-seed 0 `
  --out-dir td3_baseline/checkpoints/training_300k_20260730_1917/ident_hyperparam_scan_plan1
```

⚠ **首轮 48 点扫描的结论已作废**：`σ_r=σ_ṙ=0` 使 `R≈0`，卡尔曼增益对 `kf_qj` 的缩放不变，
`μ≤1e-2` 对 `θ_sc` 的软阈值也可忽略——两个旋钮在该配置下均不起作用（48 个配置 `med_rddot` 全等于 494 即为证据）。
重跑前请先处理 changelog §5.2.2 的遗留项。

### 7.5 单元测试

```powershell
& "D:\Program Files\MATLAB\R2024a\bin\matlab.exe" -batch `
  "addpath('rl_td3/matlab'); addpath('.'); addpath('expert_trajectory'); ok=test_stage2_ident(); exit(~ok);"
```

覆盖：`analytic_truth` 回归、无噪 KF、合成 FF-RLS、因果时序、episode 重置、coeffs 数值入口等价、式(13)/(15) 恒等式、PN warm-start 无真值 `a_D`、KF 设计协方差灵敏度、显式 η 与 A21 投影。

### 7.6 设计边界（首版故意不做）

- 不采用 PE/Gram 门控、dead-zone、协方差 trace-limit、自适应遗忘因子  
- 不给式(12) 注入过程噪声（`kf_qj` 仅为 KF 模型参数）  
- 不改 13D obs / 3D action；噪声只走制导系数路径  
- 不读取、不重放、不修改专家库；BC-TD3 入口保持默认 `analytic_truth`
