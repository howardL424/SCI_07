# 会话摘要：测量噪声倍率定标（2026-08-06）

目标：在固定 100 IC 的 MC 上，找一个统一 `sigma_scale`，使 BC-TD3 与纯 TD3 的**联合成功率略低于 90%**，且最好纯 TD3 < BC-TD3。

## 协议与模型

- MC：`run_mc_fixed100.py`，共用 `ic_pool_n100_seed0.json`，`coeff_mode=analytic_noisy`
- 每局 `noise_seed = seed0 + local_idx`（与 `mc_noisy_300` 对齐）
- BC-TD3：`rl_td3/checkpoints/training_300k_20260730_1917/best/best_model`
- 纯 TD3：`td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model`
- 通道基准（×1）：`σ_q/θ/ψ=0.1°`，`σ_q̇=0.001 rad/s`，`σ_r/ṙ=0`

## 扫描结果（联合成功率，N=100）

| σ 倍率 | BC-TD3 | 纯 TD3 |
|--------|--------|--------|
| ×10 | 93% | 95% |
| ×20 | 96% | 91% |
| ×30 | **89%** | **88%** |

结论：**采用 `sigma_scale = 30`**（两者略低于 90%，且纯 TD3 略低于 BC-TD3）。

结果目录：`results/sigma_x10/`、`sigma_x20/`、`sigma_x30/`。

## 代码变更记录（默认噪声改为 30×）

基准 σ 数值本身未改；只把统一倍率默认值从 **1 → 30**，原 `=1` 行注释保留、不删除。

| 文件 | 变更 |
|------|------|
| `table2_noisy_ident/noisy_ident_cfg.m` | 缺省 `sigma_scale`：1→30 |
| `table2_noisy_ident/main_table2_noisy_ident.m` | `opts.sigma_scale`：1→30 |
| `table2_noisy_ident/run_table2_noisy_ident.m` | `local_default(..., 30)` |
| `mc_noisy_300/main_mc_four_noisy_300.m` | `addParameter` 默认 30 |
| `mc_noisy_300/run_mc_rl_noisy.py` | `--sigma-scale` 默认 30 |
| `mc_baseline_300/main_mc_pn_sine_300.m` | `sigma_scale` 默认 30（仍需 `enable_noise=true`） |
| `mc_fixed100/run_mc_fixed100.py` | `--sigma-scale` 默认 30 |

`rl_td3/ident_cfg.py` 中的 1× 基准 σ **未改**，仍作为倍率单位。
