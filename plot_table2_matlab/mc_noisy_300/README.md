# 四方法带测量噪声蒙特卡洛（不带辨识）

对 **纯 PN / 正弦机动 / 纯 TD3 / BC-TD3** 在同一 test-IC 池上做带测量噪声的 300 局评估。  
**不跑参数辨识**（无 sparse FF-RLS）：TD3/BC-TD3 使用 `coeff_mode=analytic_noisy`（带噪测量代入解析系数）。

初态池与判据与无噪 MC 一致：`samples_clean_test.mat`，规避 `r_MD≥6 m`，打击 `r_minT≤10 m`。

## 快速运行

```matlab
cd('D:\LTY\yan\yanerxia\GAIL\plot_table2_matlab\mc_noisy_300')

% 烟雾（推荐先跑）
main_mc_four_noisy_300('N', 2)

% 正式 300 局，默认 30× 噪声
main_mc_four_noisy_300

% 显式 1× / 3×
main_mc_four_noisy_300('sigma_scale', 1)
main_mc_four_noisy_300('sigma_scale', 3)

% 只跑基线 / 只跑 RL
main_mc_four_noisy_300('methods', {'pn','sine'})
main_mc_four_noisy_300('methods', {'td3','bctd3'})
```

纯 Python（仅 TD3/BC-TD3）：

```text
E:\Python\venvs\rl_pytorch\Scripts\python.exe ^
  plot_table2_matlab/mc_noisy_300/run_mc_rl_noisy.py ^
  --algo both --episodes 300
```

也可在原 PN/正弦脚本上单独开噪声接口（默认仍关闭，旧结果可复现）：

```matlab
cd('..\mc_baseline_300')
main_mc_pn_sine_300('enable_noise', true, 'sigma_scale', 30, 'N', 300)
```

## 噪声约定

通道基准（`sigma_scale=1`，与 `rl_td3/ident_cfg.py` 一致）：

| 通道 | σ |
|------|---|
| `σ_q` / `σ_θ` / `σ_ψ` | 0.1° |
| `σ_q̇` | 0.001 rad/s |
| `σ_r` / `σ_ṙ` | 0 |

- **现行默认 `sigma_scale=30`**（2026-08-06 由 1 改为 30；原值在代码中注释保留）。见 `../mc_fixed100/SESSION_SUMMARY_noise_scale_20260806.md`。
- `sigma_scale` 统一乘到所有 σ。
- 每局 `noise_seed = noise_seed0 + (i-1)`，四方法同 IC 对齐。
- **防御弹 PN 始终真值几何**；攻击方制导用测量加噪。
- PN/正弦：噪声按 `noise_hold_dt=0.1 s` 采样并保持（对齐 stage2 周期，避免 dt=1 ms 过强）。
- 成功判据 / 拦截判定 / `r_safe` 触发一律用真值。

## 输出

`results/sigma_x<scale>/`

| 文件 | 内容 |
|------|------|
| `mc_pn_N*_noisy_x*.mat` / `mc_sine_...` | PN、正弦逐局指标 |
| `mc_td3_N*_noisy_x*.npz` / `mc_bctd3_...` | TD3、BC-TD3 逐局指标 |
| `summary_*_x*.json` | RL 方法汇总 |
| `mc_four_noisy_*.log` | 终端日志 |

## 文件

| 文件 | 说明 |
|------|------|
| `main_mc_four_noisy_300.m` | 四方法总入口 |
| `run_mc_rl_noisy.py` | TD3/BC-TD3 `analytic_noisy` MC |
| `../mc_baseline_300/simulate_baseline_episode.m` | 已加 `enable_noise` 接口 |
| `../mc_baseline_300/main_mc_pn_sine_300.m` | 已加噪声 CLI 参数 |

## 依赖

- MATLAB R2024a；PN/正弦路径需 `expert_trajectory/`
- Python：`E:\Python\venvs\rl_pytorch\Scripts\python.exe`（MATLAB Engine + 训练环境）
- 模型默认路径与 Table-2 轨迹脚本一致（`td3_baseline/.../best_model`、`rl_td3/.../best_model`）
