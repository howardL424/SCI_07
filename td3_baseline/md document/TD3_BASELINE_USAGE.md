# 纯 TD3 对照基线使用说明

> **生成时间**：2026-07-26  
> **用途**：论文 / 正式实验中相对 BC-TD3 的无专家对照  
> **共享环境与协议**：与 [`rl_td3/md document/BC_TD3_USAGE.md`](../../rl_td3/md%20document/BC_TD3_USAGE.md) 一致

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

**不要**用 `rl_td3/train_td3.py --no-bc` 充当本正式基线：该入口仍挂在 `TD3BC` 上，且 CLI 保留专家相关参数，易混淆。

---

## 1. 目录

```text
td3_baseline/
├── train_td3_pure.py
├── evaluate_td3_pure.py   # 转发 rl_td3/evaluate_td3.py
├── md document/
│   └── TD3_BASELINE_USAGE.md
├── checkpoints/
├── checkpoints_smoke/
└── logs/
```

复用（不复制）：`rl_td3/evasion_env.py`、`action_mapping.py`、`obs_norm.py`、`matlab/`、`evaluate_td3.py`。

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

---

## 5. 实验建议

1. 与 BC-TD3 主跑使用相同 `steps`、`lr`、`action-mapping`；训后 MC 用同一 `--samples`（默认 test）。
2. 下结论建议 ≥3 个训练 `--seed`；全量 test 评估无需再对齐 MC 抽样种子；非全量评估要对比同一 `--seed`
3. 产物目录与 `rl_td3/checkpoints` 隔离，避免覆盖。

---

## 6. 关键键文件

| 需求 | 文件 |
|------|------|
| 纯 TD3 训练 | `td3_baseline/train_td3_pure.py` |
| 评估封装 | `td3_baseline/evaluate_td3_pure.py` |
| 环境 / 奖励 | `rl_td3/evasion_env.py` + `rl_td3/matlab/` |
| BC-TD3 主线 | `rl_td3/train_td3.py` / `rl_td3/td3_bc.py` |
| BC-TD3 用法 | `rl_td3/md document/BC_TD3_USAGE.md` |
