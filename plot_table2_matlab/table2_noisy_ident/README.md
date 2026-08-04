# Table-2 带测量噪声轨迹复算

在原有纯 TD3 / BC-TD3 三段轨迹求解上，仅对 stage2 注入参数辨识用的测量噪声，输出三维轨迹与 D1/D2 最近接近放大图。原 `plot_table2_matlab` 文件不修改。

## 快速运行

```matlab
cd('D:\LTY\yan\yanerxia\GAIL\plot_table2_matlab\table2_noisy_ident')
main_table2_noisy_ident      % 1× 默认噪声 → figures/、results/
main_table2_noisy_ident_x3   % 3× 噪声 → figures/sigma_x3/、results/sigma_x3/
```

## 改噪声大小

默认基准（`preset='plan1'`，`sigma_scale=1`）：

| 通道 | σ |
|------|---|
| `σ_q` / `σ_θ` / `σ_ψ` | 0.1° |
| `σ_q̇` | 0.001 rad/s |
| `σ_r` / `σ_ṙ` | 0 |

`sigma_scale` 会把上表所有 σ 统一乘倍率（KF 的 R 同步按新 σ 重建）：

```matlab
opts = struct();
opts.preset      = 'plan1';          % 'plan1' | 'range' | 'clean'
opts.sigma_scale = 5;                % ← 改这里：5 倍噪声
opts.noise_seed  = 0;
opts.coeff_mode  = 'analytic_noisy';  % 或 'sparse_ffrls'
opts.obs_noise   = false;            % true=策略 13 维观测也加噪
% opts.out_tag  可不写：scale≠1 时自动为 'sigma_x5'

run_table2_noisy_ident(opts);
```

- `preset='range'`：在 plan1 基础上加距离噪声 `σ_r=10 m`、`σ_ṙ=1 m/s`（再乘 `sigma_scale`）。跑 `sparse_ffrls` 时建议用此预设，避免 `σ_r=0` 导致 KF 退化。
- `preset='clean'`：全部 σ=0，用于核对管线是否退化为解析真值。

## 目录结构

| 路径 | 说明 |
|------|------|
| `run_table2_noisy_ident.m` | 求解 + 出图入口（可调参） |
| `noisy_ident_cfg.m` | 噪声 / KF / RLS 配置 |
| `simulate_table2_noisy_episode.m` | 单局三段仿真 |
| `plot_noisy_vs_clean.m` | 真值系数 vs 带噪对照图 |
| `figures/`、`figures/<out_tag>/` | 轨迹图与放大图 |
| `results/`、`results/<out_tag>/` | `record_table2_*_noisy.mat` |

## 依赖

- MATLAB R2024a + Python 接口 → `E:\Python\venvs\rl_pytorch\Scripts\python.exe`
- 策略与共享桥接：`table2_bctd3/`、`rl_td3/matlab/`、`expert_trajectory/`
- 对照图需要已有无噪 record：`table2_td3_pure/record_table2_td3_pure.mat`、`table2_bctd3/record_table2_bctd3.mat`
