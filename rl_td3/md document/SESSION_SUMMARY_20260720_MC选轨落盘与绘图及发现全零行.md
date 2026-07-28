# 会话交接：0711 TD3+BC 的 MC 落盘、选轨与绘图及发现全零行

> 日期：2026-07-20（后续补充：全零行根因定位与过滤、seed=1 重跑）  
> 目的：为后续新对话继续做 Monte Carlo（MC）评估、代表轨选择、数据落盘和绘图提供准确上下文。  
> 本文对应模型：`rl_td3/checkpoints/training_300k_td3bc_0711/best/best_model.zip`  
> 文件名：`SESSION_SUMMARY_20260720_MC选轨与绘图及发现全零行.md`

---

## 1. 任务背景与当前结论

目标是对旧的 0711 TD3+BC 模型执行可复现 MC，自动选择代表轨，并保存后续作图所需的：

- 攻击弹 M、两个防御弹 D1/D2 的 stage2 三维位置；
- 攻击弹二维法向加速度与还原的三维惯性系加速度；
- 每步物理动作参数 `gamma`、`wT`、`wE`；
- 规避/打击结果、最小距离、动作模式等元数据。

### 0711 模型的兼容性约束（非常重要）

0711 是旧 checkpoint，必须使用以下组合：

| 项 | 值 | 原因 |
|---|---|---|
| 模型 | `.../training_300k_td3bc_0711/best/best_model` | EvalCallback 的 best 模型 |
| 动作映射 | `td3_box_log_v1` | 网络输出为旧坐标 `[gamma_n, wT_n, wE_n]` |
| 观测模式 | `raw` | 0711 用未做 `OBS_SCALE` 的物理观测训练 |

**不能**用当前新模型默认的 `ratio_log_box_v1 + scaled` 直接加载 0711；否则动作解释或 SB3 observation space 检查会不匹配。

### 会话后期关键结论（全零行）

seed=1 过滤前选中的 ep002「初态共点」**不是** 8 km 规避触发逻辑失效，而是抽到了 `samples_clean.mat` 中的**全零填充行**。  
已在 `EvasionEnv` 加载时过滤全零行；过滤后 seed=1 重跑联合成功率升至 **94%**，且 **100% A 型**（此前大量 B 型短轨来自全零行假几何）。

---

## 2. 新增/修改的代码

### 2.1 MC 落盘与选轨

新增：

```text
rl_td3/tools/dump_mc_representative_trajs.py
```

功能：

1. 固定 seed 重放 MC；首轮 `env.reset(seed=seed)`，后续连续抽样；
2. 每条 episode 保存 stage2 全程数据到内存；
3. 按规则选出代表轨；
4. 保存全量 episode 索引、被选轨的逐步数组和 JSON 元数据；**不画图**。

参数：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe `
  rl_td3/tools/dump_mc_representative_trajs.py `
  --model rl_td3/checkpoints/training_300k_td3bc_0711/best/best_model `
  --action-mapping td3_box_log_v1 `
  --obs-mode raw `
  --episodes 100 `
  --seed <SEED>
```

默认输出：

```text
rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed<SEED>/
├── mc_summary.json
├── mc_episode_index.npz
├── selected_meta.json
└── selected_trajs.npz
```

### 2.2 加速度回传与旧观测兼容

修改：

```text
rl_td3/matlab/env_step.m
rl_td3/evasion_env.py
```

`env_step.m` 的 `info_vec` 从原来的 5 项扩展为 9 项：

```text
[r_MD1; r_MD2; r_MT; intercepted; stage2_done;
 a_M_sat_y; a_M_sat_z; a_M_raw_y; a_M_raw_z]
```

前 5 项保持不变，因此旧调用者仍兼容。Python `info` 新增：

```text
info["a_M_sat"]  # [a_y, a_z]，饱和后、实际传入 updateState 的法向指令
info["a_M_raw"]  # [a_y, a_z]，饱和前诊断指令
```

`EvasionEnv` 新增 `obs_normalize: bool = True`：

- `True`：当前默认新管线，返回 `obs / OBS_SCALE`；
- `False`：兼容 0711，返回 raw 物理 13 维观测，Box 使用旧的 `LEGACY_OBS_HIGH`。

### 2.3 绘图

新增：

```text
rl_td3/plots/plot_representative_trajs.py
```

默认读 `rep_trajs_seed1`。对每条有效入选轨输出三张 PNG：

1. `*_3d_trajectory.png`：M/D1/D2 stage2 三维轨迹；
2. `*_inertial_acceleration_3x1.png`：攻击弹惯性系 `[a_x,a_y,a_z]`，3×1；
3. `*_action_parameters_3x1.png`：`gamma`、`wT`、`wE`，3×1。

运行：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe `
  rl_td3/plots/plot_representative_trajs.py `
  --run-dir rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed1
```

输出目录为 `{run-dir}/figures/`。

可视化已实现：

- 轨迹 / 加速度 / 动作曲线均标出 `dt=0.1 s` 离散采样点；
- 动作用零阶保持阶梯线 + 圆点，避免跳变被误读为线性平滑；
- 非共线成功轨增加 M–D2 最近交会局部放大；
- 共线退化轨在图注中明确说明「无法靠视角分离」。

### 2.4 samples_clean 全零行过滤（已落地）

修改：`rl_td3/evasion_env.py`

加载 `samples_clean` 后：

```text
valid_mask = ||row|| > 1e-12
保留 valid 行；记录 _sample_row_ids（过滤后局部下标 → 原始 mat 行号）
```

- 实测：`3400` 行 → 保留 `2351`，滤除全零 `1049` 行（约从 index `2351` 起的填充块）。
- `reset` 返回的 `info["row_idx"]` 仍是**原始 mat 行号**；另有 `info["local_idx"]` 为过滤后库内下标。
- 训练 / 评估 / MC 落盘凡经 `EvasionEnv` 都会自动受益。

---

## 3. 数据字段与物理含义

`selected_trajs.npz` 的 key 以：

```text
{tier}_ep{episode:03d}/...
```

为前缀，例如 `success_median_ep013/pos_M`。

| 字段 | 形状 | 含义 |
|---|---:|---|
| `t` | N+1 | stage2 时刻，含初始 `t=0` |
| `t_step` | N | 与动作、加速度一一对应 |
| `pos_M`, `pos_D1`, `pos_D2` | (N+1)×3 | 绝对位置 `[x,y,z]` |
| `att_M` | (N+1)×3 | M 的 `[theta, psi, V]` |
| `action_norm` | N×3 | 网络归一化动作 |
| `gamma`, `wT`, `wE` | N | 反归一化后的旧物理权重 |
| `a_M_body` | N×2 | 实际饱和法向指令 `[a_y,a_z]` |
| `a_M_body_raw` | N×2 | 饱和前指令 |
| `a_M_inertial` | N×3 | 从 `a_M_body + theta + psi + V` 还原的惯性系加速度 `[a_x,a_y,a_z]` |
| `r_MD1`, `r_MD2`, `r_MT` | N | 每步 step 后距离 |

三维航迹沿用专家图的坐标约定：

```matlab
plot3(x, z, y)
```

注意：

- 所有保存的三维位置和加速度都仅覆盖 **stage2（RL 规避段）**；
- stage3 使用一次性 PN rollout，只返回 `r_minT/reached/diverged/n_steps`，当前未保存 stage3 逐步轨迹；
- 因此“3D trajectory 图”不能被解读为完整 PN → stage2 → PN 的端到端轨迹。

---

## 4. 当前自动选轨规则

`dump_mc_representative_trajs.py` 的分层规则：

1. **`success_median`**  
   从 `both_ok=True` 的轨迹中选 `r_minT` 最接近该成功组中位数的一条；

2. **`evade_fail_strike`**  
   从 `evade_ok=True`、`strike_ok=False`、`intercepted=False` 中优先选 `r_minT ∈ [15,50] m` 的一条，取最接近区间中点者；  
   若区间没有样本，则 fallback 为所有该类失败轨中最接近区间中点者；  
   若该类候选为空，则本层不写入；

3. **`type_b`**  
   只有当前已选轨全部为 A 型时，才额外选一条 B 型。若已有非 A，则跳过。

动作类型定义：

| 类型 | 判据 |
|---|---|
| A 型 | 三维归一化动作的段内标准差均 `< 1e-3`，即整段常数角点 |
| B 型 | 非 A，且所有维度近边界或至少一维有明显段内切换（`max std_n >= 0.1`） |

---

## 5. 已跑 MC 结果

### 5.1 seed=0（过滤全零行**之前**）

目录：

```text
rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed0/
```

| 指标 | 值 |
|---|---:|
| 规避成功 | 94% |
| 打击成功 | 64% |
| 联合成功 | 64% |
| 被拦截 | 6% |
| A 型 | 70% |
| B 型 | 30% |

选中：

| 分层 | ep | 类型 | 结果 |
|---|---:|---|---|
| 成功中位 | 20 | A | `r_minT=0.1319 m`，80 步 |
| 规避成功/打击失败 | 0 | B | fallback，`r_minT=550 m`，11 步 |

> 说明：seed=0 目录仍是过滤前产物；若需与过滤后口径对齐，应重跑并另存或覆盖。

### 5.2 seed=1 过滤前（历史，含全零行污染）

当时结果：

| 指标 | 值 |
|---|---:|
| 规避成功 | 95% |
| 打击成功 | 62% |
| 联合成功 | 62% |
| 被拦截 | 5% |
| A 型 | 67% |
| B 型 | 33% |

选中：成功中位 ep69（A）；规避失败打击 ep2（B，`r_minT=550`，11 步）——后者后证实为全零行。

### 5.3 seed=1 过滤后（当前权威）

目录（已覆盖写入）：

```text
rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed1/
```

| 指标 | 过滤前 | 过滤后 |
|------|--------|--------|
| 规避成功 | 95% | **94%** |
| 打击成功 | 62% | **94%** |
| 联合成功 | 62% | **94%** |
| 被拦截 | 5% | **6%** |
| A 型 | 67% | **100%** |
| B 型 | 33% | **0%** |

选中：

| 分层 | ep | 类型 | 结果 |
|---|---:|---|---|
| 成功中位 | 13 | A | `r_minT≈0.117 m`，75 步 |
| 规避成功/打击失败 | — | — | **无候选**（规避成功者全部也打击成功） |
| B 型 | — | — | **无** |

解读：

- 过滤前大量「B 型短轨 + `r_minT=550`」主要来自全零行假几何，不是真实策略双模态；
- 过滤后本 seed 下策略表现为全程 A 型角点 + 高联合成功率；
- `figures/` 中旧的 ep069/ep002 图对应过滤前选轨，**未自动重画**；需要时对当前 `selected_trajs.npz` 再跑绘图脚本。

### 5.4 与历史日志的差异

历史日志 `logs/0711_300k/monte_carlo_eval_300k_td3bc_0711_20260711_1431.log` 记录过 69% 联合成功。  
过滤前当前代码重放 seed=0/1 约 62–64%；过滤后 seed=1 达 94%。

差异来源可能叠加：环境/奖励/stage3/观测路径变更、是否抽到全零行、seed 不同。论文/报告必须写明「代码版本、是否过滤全零行、模型、seed、episodes」，不要混用不同时间点数字。

---

## 6. ep002 / 全零行根因（必须保留）

### 6.1 现象

过滤前 seed=1 的 ep002：

```text
steps = 11             -> stage2 仅 1.1 s
M/D1/D2/Target 起点均在原点
M 沿 -x，D1/D2 重合沿 +x（严格共线）
r_minT = 550 m
仍被标 evade_ok（r_min 只统计 step 后距离）
```

用户疑问：规避不是在 `min(r_MD1,r_MD2)≤8 km` 启动吗？为何会共点？

### 6.2 根因链

1. `samples_clean[2567]`（及同批全零行，如 3231）**整行全 0**；
2. `reconstruct_geometry(zeros)` → M0/D10/D20 都在原点（仅速度姿态取默认）；
3. warm-start：`min(r_MD)≈0 ≤ r_safe=8000` → **t=0 立刻标 stage=2**（仅 1 帧）再进 stage3；
4. `env_reset`：只要 `find(stage==2)` 非空就 `ok=true`，`Tev=max(Δt,1)=1`，把共点态交给 RL；
5. **8 km 触发逻辑本身没错**——距离已是 0，条件一开始就成立。

统计：`samples_clean` 共 3400 行，**1049 行全零**。

### 6.3 结论

- ep002 **不是**物理上有代表性的「规避成功但打击失败」；
- 过滤全零行后，该类假 B 型短轨从 MC 池中消失；
- 仍建议后续补强：`env_reset` 对「stage2 仅 1 帧 / 初态距离过小」直接 `ok=false`，作为第二道保险。

---

## 7. 过滤前 seed=1 图的状态（存档说明）

过滤前曾生成：

```text
rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed1/figures/
  ep069_success_median_*.png
  ep002_evade_fail_strike_*.png
```

其中 ep069 最近点 `min(M,D2)=22.67 m > 6 m`，说明当时成功轨并非被 D2 拦截，只是视角上像相接。  
ep002 图仅作「全零行假几何」存档参考，**勿作论文失败对照**。

过滤后权威选轨为 ep13；若要新图，请对当前 `rep_trajs_seed1` 重跑 `plot_representative_trajs.py`。

---

## 8. 后续建议（优先级）

### 8.1 已完成

- [x] `EvasionEnv` 加载时过滤 `samples_clean` 全零行；
- [x] seed=1、100 轮 MC 在过滤后重跑并覆盖 `rep_trajs_seed1/`。

### 8.2 仍建议

1. **seed=0 亦按过滤后口径重跑**（或明确标注旧目录为「过滤前」）；
2. **落盘 `row_idx` / `diverged` / `reached_pn` / `n_steps_stage3`** 到 `mc_episode_index`；
3. **`env_reset` 几何健全性**：初态 `min(r_MD1,r_MD2,r_MT)` 过小或 stage2 时长过短 → `ok=false`；
4. **失败分层**：近失 / 远失 / stage3 发散 / 非法初态；
5. **若要端到端三维航迹**：扩展 `stage3_rollout` 输出逐步位置与加速度，再与 stage2 拼接。

### 8.3 代表轨主图建议（当前）

- 主成功案例用过滤后 seed=1 的 **ep13**（或重跑后的新中位轨）；
- 不要使用过滤前的 ep002；
- 动作坍塌仍可用全段贴边统计 / A 型常数角点说明，不必依赖 B 型短轨。

---

## 9. 相关旧背景

- 0711 的历史全段动作探针：`rl_td3/logs/0711_300k/full_traj_action_probe_20ep.log`
  - 三维动作全程贴边；
  - A 型：`gamma=3, wT=1e5, wE=1e2` 常数；
  - B 型：短轨先 `[1,1,1]`，再切至 `[1,-1,-1]`（过滤前 MC 中大量此类实为全零行）。
- 诊断总结：`rl_td3/md document/SESSION_SUMMARY_20260711_FULL_TRAJ_ACTION_COLLAPSE.md`
- 用法总览：`rl_td3/md document/BC_TD3_USAGE.md`
