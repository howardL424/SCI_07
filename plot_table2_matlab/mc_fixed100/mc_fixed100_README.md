# 固定 100 初态蒙特卡洛（多模型共用清单）（用于定噪声大小）

从 `samples_clean_test.mat` 有效行做 **定序置换**（`order_seed`），取前 `n_ic=100`，无放回。  
约 10 组模型对比时，**全部复用同一份 `ic_pool_*.json`**，保证配对可比。

不是旧协议的「每局有放回随机抽」。

## 快速运行

```powershell
cd D:\LTY\yan\yanerxia\GAIL

# 1) 只生成 IC 清单（推荐先做一次，之后各组共用）
E:\Python\venvs\rl_pytorch\Scripts\python.exe `
  plot_table2_matlab/mc_fixed100/run_mc_fixed100.py --dump-ic-only

# 2) 单模型（name=path 写在同一个参数里）
E:\Python\venvs\rl_pytorch\Scripts\python.exe `
  plot_table2_matlab/mc_fixed100/run_mc_fixed100.py `
  --model bctd3_best=rl_td3/checkpoints/.../best/best_model `
  --prefer-td3bc `
  --ic-pool plot_table2_matlab/mc_fixed100/results/ic_pool_n100_seed0.json

# 3) 多模型：复制 models.example.json → models.json，改路径后
E:\Python\venvs\rl_pytorch\Scripts\python.exe `
  plot_table2_matlab/mc_fixed100/run_mc_fixed100.py `
  --models-file plot_table2_matlab/mc_fixed100/models.json `
  --ic-pool plot_table2_matlab/mc_fixed100/results/ic_pool_n100_seed0.json

# 烟雾（清单前 2 局）
... --smoke 2
```

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--n-ic` | 100 | 固定初态个数 |
| `--order-seed` | 0 | 置换种子；改了会得到另一批 100 个 |
| `--ic-pool` | （无） | 已有清单 JSON；指定后不再重新抽样 |
| `--samples` | `samples_clean_test.mat` | 初态池 |
| `--coeff-mode` | `analytic_truth` | 与 `evaluate_td3` 一致；带噪可改 `analytic_noisy` |
| `--sigma-scale` | **30** | 统一乘到各通道 σ（原默认 1，2026-08-06 改为 30） |

## 输出（`results/`）

| 文件 | 内容 |
|------|------|
| `ic_pool_n100_seed0.json` | 共用初态清单（`local_idx` / `row_idx`） |
| `mc_<name>_N*_*.npz` | 逐局指标 |
| `summary_<name>_*.json` | 单模型汇总 |
| `batch_summary_*.json` | 本批全部模型对照表 |

## 文件

| 文件 | 说明 |
|------|------|
| `run_mc_fixed100.py` | 主脚本 |
| `models.example.json` | 多模型配置示例 |
| `models_sigma_scan.json` | 噪声定标用模型清单（BC/TD3 300k） |
| `SESSION_SUMMARY_noise_scale_20260806.md` | 噪声倍率扫描与默认改为 30× 的记录 |
