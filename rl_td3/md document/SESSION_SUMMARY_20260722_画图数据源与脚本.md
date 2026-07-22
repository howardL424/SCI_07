# 本次对话画图内容总结（2026-07-20 ~ 07-22）

> 对应对话主题：导师工作总结用图 — 训练曲线、0711 动作分布、seed=1 全量 MC 动作直方图。  
> 画图脚本均在 `rl_td3/plots/`；**默认输出目录均为 `rl_td3/out_pictures/`**（或其子目录）。

---

## 0. 总览

| 批次 | 图件 | 画图脚本 | 默认输出 |
|------|------|----------|----------|
| A. 两跑对比训练曲线 | 叠加图 + 并排图 | `plots/plot_training_curves_0711_0715.py` | `out_pictures/training_curves_*.png` |
| B. 0711 单跑曲线 + 历史动作分布 | 曲线 1 张 + 条形/直方图 2 张 | `plots/plot_0711_curve_and_actions.py` | 曲线 → `out_pictures/`；动作图 → `out_pictures/0711_300k_action_distribution_old/` |
| C. seed=1 的 100 轮 MC 全量动作分布 | 综合图 + 物理直方图 + 摘要/缓存 | `plots/plot_mc_action_distribution.py` | `out_pictures/0711_300k_action_distribution_new/` |

**未改动的已有训练/环境代码**：本对话画图批次只新增上述脚本；未修改既有训练/环境 `.py`（用户确认过 C 批次）。

---

## 1. 批次 A — 0711 vs 0715 训练曲线

### 1.1 产出文件

| 文件（当前路径） | 内容 |
|------------------|------|
| `rl_td3/out_pictures/training_curves_0711_vs_0715.png` | 两跑 eval mean±std 叠加；星号标 best；灰虚线 λ→0 @150k |
| `rl_td3/out_pictures/training_curves_0711_0715_side_by_side.png` | 左右分图 |

### 1.2 数据源

| 运行 | EvalCallback 日志 |
|------|-------------------|
| 0711 | `rl_td3/checkpoints/training_300k_td3bc_0711/eval_logs/evaluations.npz` |
| 0715 | `rl_td3/checkpoints/training_300k_20260715_1008/eval_logs/evaluations.npz` |

字段：`timesteps`（每 10k 一步）、`results`（shape `(30, 20)`，20 个 eval episode 的回报）。

关键数字（对话中读出）：

- 0711 best @ **150000**，mean≈**101.08**；final@300k≈56.35  
- 0715 best @ **100000**，mean≈**101.06**；final@300k≈76.20  

### 1.3 画图代码

- 脚本：[`rl_td3/plots/plot_training_curves_0711_0715.py`](../plots/plot_training_curves_0711_0715.py)
- 默认 `OUT_DIR = rl_td3/out_pictures/`
- 重跑（仓库根目录）：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/plots/plot_training_curves_0711_0715.py
```

---

## 2. 批次 B — 0711 单独训练曲线 + 历史动作分布

### 2.1 产出文件

| 文件（当前路径） | 内容 |
|------------------|------|
| `rl_td3/out_pictures/training_curve_0711.png` | 0711 单跑 eval 曲线 |
| `rl_td3/out_pictures/0711_300k_action_distribution_old/action_distribution_0711.png` | 上：全段 20 ep 物理角点条；下：MC 初态 20 次物理角点条 |
| `rl_td3/out_pictures/0711_300k_action_distribution_old/action_histogram_0711_normalized.png` | 归一化动作 `[-1,1]` 直方图（由全段探测汇总计数重建） |

### 2.2 数据源

| 子图 | 数据源 | 说明 |
|------|--------|------|
| 训练曲线 | `checkpoints/training_300k_td3bc_0711/eval_logs/evaluations.npz` | 同批次 A 的 0711 |
| full-traj 条形图 / 归一化直方图 | `rl_td3/logs/0711_300k/full_traj_action_probe_20ep.log` | **独立**再跑的 20 条完整 episode，共 **1106** 步；**不是**从 100 轮 MC 里抽前 20 条 |
| MC 初态条形图（图 bottom） | `rl_td3/logs/0711_300k/monte_carlo_eval_300k_td3bc_0711_20260711_1431.log` 末尾「抽样 20 个初态」 | 只看 reset 后**第一步**动作 |

全段探测全局角点计数（写入脚本常量，非逐步数组）：

| 量 | 分布（1106 步） |
|----|-----------------|
| γ | 100% = +3 |
| wT | 7.2% = 1e2；92.8% = 1e5 |
| wE | 99.3% = 1e2；0.7% = 1e4 |

### 2.3 对话中澄清的抽样语义（写进总结用）

1. **full-traj 20 条 ≠ 100 轮 MC 的子集**；是另起的一次性探针（脚本已删，只留 log）。  
2. **MC 末尾 20 个初态**也是另抽序列（`evaluate_td3.py` 用同一 seed 重启 RNG），只探针第一步。  
3. **当时（0711 历史评估）**环境未必已滤 `samples_clean` 全零行；**当前代码**在 `EvasionEnv` 加载时用 `‖row‖>1e-12` 过滤，MC 与末尾 20 初态共用该池。

### 2.4 画图代码

- 脚本：[`rl_td3/plots/plot_0711_curve_and_actions.py`](../plots/plot_0711_curve_and_actions.py)
- 默认输出：
  - 训练曲线 → `rl_td3/out_pictures/`
  - 动作图 → `rl_td3/out_pictures/0711_300k_action_distribution_old/`
- 重跑：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/plots/plot_0711_curve_and_actions.py
```

---

## 3. 批次 C — seed=1、100 轮 MC 全量动作分布（推荐作「新」主图）

### 3.1 产出文件

| 文件（当前路径） | 内容 |
|------------------|------|
| `rl_td3/out_pictures/0711_300k_action_distribution_new/action_distribution_mc_seed1_100ep.png` | 物理角点条 + 归一化直方图 |
| `rl_td3/out_pictures/0711_300k_action_distribution_new/action_histogram_mc_seed1_100ep_physical.png` | γ / log10(wT) / log10(wE) 直方图 |
| `rl_td3/out_pictures/0711_300k_action_distribution_new/action_distribution_mc_seed1_100ep_summary.json` | 占比摘要 |
| `rl_td3/out_pictures/0711_300k_action_distribution_new/mc_actions_seed1_100ep.npz` | 逐步动作缓存（可复用，免重跑 MATLAB） |

### 3.2 数据源

| 项 | 值 |
|----|-----|
| 模型 | `rl_td3/checkpoints/training_300k_td3bc_0711/best/best_model` |
| 动作映射 | `td3_box_log_v1`（0711 旧坐标） |
| 观测 | `raw`（0711 未用 `OBS_SCALE`） |
| seed / episodes | **1** / **100** |
| 统计范围 | 100 轮全部 **stage2** 步，合计 **8630** 步 |
| 一致性核验 | `steps_per_ep` 与 `checkpoints/.../rep_trajs_seed1/mc_episode_index.npz` 的 `steps` **完全一致** |

> `rep_trajs_seed1/selected_trajs.npz` **只有代表轨**逐步动作，不足以画 100 轮全量分布；故脚本会重放 MC（或读已有 `mc_actions_*.npz` 缓存）。

结果（摘要）：全程钉死单角点  

- γ ≡ +3（100%）  
- wT ≡ 1e5（100%）  
- wE ≡ 1e2（100%）  
- 三维同时 `|a_n|≥0.99`：100%  

与同次 MC 的 `mc_summary.json` 中 `type_a: 1.0` 一致。

### 3.3 画图代码

- 脚本：[`rl_td3/plots/plot_mc_action_distribution.py`](../plots/plot_mc_action_distribution.py)
- 默认 `--out-dir`：`rl_td3/out_pictures/0711_300k_action_distribution_new`
- 重跑 MC 并出图（仓库根目录；可不显式传 `--out-dir`）：

```powershell
# 若已有缓存且要强制重跑：先删掉对应 mc_actions_seed1_100ep.npz
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/plots/plot_mc_action_distribution.py `
  --seed 1 --episodes 100 `
  --model rl_td3/checkpoints/training_300k_td3bc_0711/best/best_model `
  --action-mapping td3_box_log_v1 --obs-mode raw
```

- 仅用已有缓存出图（跳过 MATLAB）：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/plots/plot_mc_action_distribution.py `
  --seed 1 --episodes 100 `
  --cache rl_td3/out_pictures/0711_300k_action_distribution_new/mc_actions_seed1_100ep.npz
```

缓存 NPZ 字段：`action_norm`、`gamma`、`wT`、`wE`、`ep_id`、`steps_per_ep`、`seed`、`episodes`。

---

## 4. 与「代表轨落盘绘图」的边界（同日相关、但非本对话主产出）

同日另有工具链（交接见 `SESSION_SUMMARY_20260720_MC选轨与绘图.md`）：

| 脚本 | 作用 |
|------|------|
| `rl_td3/tools/dump_mc_representative_trajs.py` | seed 固定 MC 落盘 + 选代表轨 |
| `rl_td3/plots/plot_representative_trajs.py` | 代表轨三维 / 加速度 / 动作时序图 |

产物目录示例：`checkpoints/training_300k_td3bc_0711/rep_trajs_seed1/`。  
**本对话批次 C 的动作直方图不依赖代表轨 NPZ**，而依赖全量 100 轮逐步动作缓存。

---

## 5. 汇报选用建议（简）

| 用途 | 建议图 |
|------|--------|
| 两跑训练过程对照 | 批次 A 叠加图 |
| 0711 训练过程单页 | `training_curve_0711.png` |
| 动作坍塌（旧诊断，20 ep 探针） | `0711_300k_action_distribution_old/` |
| 动作坍塌（正式 seed=1、100 轮全量，优先） | `0711_300k_action_distribution_new/` |

---

## 6. 文件路径速查

```text
rl_td3/
├── plots/
│   ├── plot_training_curves_0711_0715.py      # 批次 A → out_pictures/
│   ├── plot_0711_curve_and_actions.py         # 批次 B → out_pictures/ (+ ..._old/)
│   └── plot_mc_action_distribution.py         # 批次 C → out_pictures/..._new/
├── out_pictures/
│   ├── training_curves_0711_vs_0715.png
│   ├── training_curves_0711_0715_side_by_side.png
│   ├── training_curve_0711.png
│   ├── 0711_300k_action_distribution_old/
│   │   ├── action_distribution_0711.png
│   │   └── action_histogram_0711_normalized.png
│   └── 0711_300k_action_distribution_new/
│       ├── action_distribution_mc_seed1_100ep.png
│       ├── action_histogram_mc_seed1_100ep_physical.png
│       ├── action_distribution_mc_seed1_100ep_summary.json
│       └── mc_actions_seed1_100ep.npz
├── logs/0711_300k/
│   ├── full_traj_action_probe_20ep.log        # 批次 B full-traj
│   └── monte_carlo_eval_300k_td3bc_0711_*.log # 批次 B MC 初态
└── checkpoints/
    ├── training_300k_td3bc_0711/eval_logs/evaluations.npz
    ├── training_300k_20260715_1008/eval_logs/evaluations.npz
    └── training_300k_td3bc_0711/rep_trajs_seed1/mc_episode_index.npz
```
