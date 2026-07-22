# 会话交接：0711 TD3+BC 的 MC 落盘、代表轨选取与绘图

> 日期：2026-07-20  
> 目的：为后续新对话继续做 Monte Carlo（MC）评估、代表轨选择、数据落盘和绘图提供准确上下文。  
> 本文对应模型：`rl_td3/checkpoints/training_300k_td3bc_0711/best/best_model.zip`

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

---

## 3. 数据字段与物理含义

`selected_trajs.npz` 的 key 以：

```text
{tier}_ep{episode:03d}/...
```

为前缀，例如 `success_median_ep069/pos_M`。

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

3. **`type_b`**  
   只有当前两条都为 A 型时，才额外选一条 B 型。若第 2 条已是 B 型，则不重复保存。

动作类型定义：

| 类型 | 判据 |
|---|---|
| A 型 | 三维归一化动作的段内标准差均 `< 1e-3`，即整段常数角点 |
| B 型 | 非 A，且所有维度近边界或至少一维有明显段内切换（`max std_n >= 0.1`） |

---

## 5. 已跑 MC 结果

### 5.1 seed=0

目录：

```text
rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed0/
```

本次**当前代码版本**重放结果：

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

### 5.2 seed=1

目录：

```text
rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed1/
```

结果：

| 指标 | 值 |
|---|---:|
| 规避成功 | 95% |
| 打击成功 | 62% |
| 联合成功 | 62% |
| 被拦截 | 5% |
| A 型 | 67% |
| B 型 | 33% |

选中：

| 分层 | ep | 类型 | 结果 |
|---|---:|---|
| 成功中位 | 69 | A | `r_minT=0.1240 m`，89 步 |
| 规避成功/打击失败 | 2 | B | fallback，`r_minT=550 m`，11 步 |

### 5.3 与历史日志的差异

历史日志 `logs/0711_300k/monte_carlo_eval_300k_td3bc_0711_20260711_1431.log` 记录过 69% 联合成功；本次当前代码重放 seed=0 是 64%。

不要直接把两者解释为模型退化或随机差异。0711 之后环境、奖励、stage3、观测规范和数据管线均经历过修改；当前重放使用的是为兼容 checkpoint 加入的 raw 观测路径。论文/报告应明确“代码版本、模型、seed、评估样本数”，不要混合不同时间点的数字。

---

## 6. ep002 的关键诊断（必须保留）

用户询问 ep002 为什么仅约 1 秒，以及为什么三维图中 Target/M start/轨迹重叠。

检查 `selected_trajs.npz` 后发现：

```text
ep002:
  steps = 11             -> stage2 仅 1.1 s
  M start  = [0, 0, 0]
  D1 start = [0, 0, 0]
  D2 start = [0, 0, 0]
  Target   = [0, 0, 0]
  M        沿 x 负方向离开
  D1/D2    完全重合，沿 x 正方向离开
  r_minT   = 550 m
```

因此：

1. **11 步不是完整端到端任务时长**，只是 RL stage2；随后确实调用 stage3 PN，但落盘未存该段位置与时长；
2. `r_minT=550 m` 表明 stage3 未命中；不能说它“完成了完整打击过程”；
3. 三弹和目标在 stage2 起点共点、且后续严格共线，真实几何上无法通过任何三维相机视角分开；
4. `r_MD1/r_MD2` 的 MC 统计从 step 后才开始更新，故这条轨迹仍被标为 `evade_ok`，尽管保存的初态显示共点。

**结论：ep002 不是适合作为物理代表性“规避成功但打击失败”案例的轨迹。**  
它更像 warm-start / 初态构造中的退化样本，应在后续 MC 选轨时显式过滤，而不应放进论文主图。

当前 30 多条 `evade_ok & !strike_ok` 样本均为 `r_minT=550 m`，没有用户指定的 `15–50 m` 软失败样本；当前 fallback 虽然按规则保存，但没有论文代表性。

---

## 7. 当前图的状态（seed=1）

图目录：

```text
rl_td3/checkpoints/training_300k_td3bc_0711/rep_trajs_seed1/figures/
```

共 6 张：

```text
ep069_success_median_3d_trajectory.png
ep069_success_median_inertial_acceleration_3x1.png
ep069_success_median_action_parameters_3x1.png
ep002_evade_fail_strike_3d_trajectory.png
ep002_evade_fail_strike_inertial_acceleration_3x1.png
ep002_evade_fail_strike_action_parameters_3x1.png
```

已实现的可视化调整：

- 所有轨迹、加速度、动作曲线均显示每个 `dt=0.1 s` 的离散数据点；
- `gamma/wT/wE` 用阶梯线（zero-order hold）+ 圆点，不再让动作跳变看起来被线性平滑；
- ep069 三维图增加 M–D2 最近交会局部放大：
  - `min(M,D2)=22.67 m`；
  - 高于拦截阈值 `6 m`，故未被 D2 拦截；
  - 图内以绿色/黄色点及虚线连线标记最近点对；
- ep002 图明确写出“初态共点、沿 x 严格共线，不能靠相机视角分离”，用 Target、M/D1/D2 的不同空心起点符号表达共点。

---

## 8. 后续新 MC/选轨建议（优先级）

### 8.1 先修复选轨有效性过滤

在 `dump_mc_representative_trajs.py` 中，给每条轨加入并过滤以下情况：

1. **初态相撞/共点过滤**  
   reset 后即计算：

   ```text
   ||M0-D1_0|| > KILL_R
   ||M0-D2_0|| > KILL_R
   ||M0-T||   > 合理下限
   ```

   不满足的样本标为 `invalid_initial_geometry`，不能进入成功率或代表轨候选；

2. **落盘 `row_idx`**  
   当前 `env.reset()` 信息中有 `row_idx`，但 `run_episode_record` 尚未写入 summary；应记录，以便复现具体 `samples_clean` 初态；

3. **落盘 stage3 状态**  
   记录 `diverged`、`reached_pn`、`n_steps_stage3`。当前 `EvasionEnv.step()` 的终止 info 有这些字段，但 MC 落盘脚本没有保存；

4. **区别有效打击失败与 PN 发散**  
   将失败分层至少拆为：
   - `valid_evade_fail_near_miss`：合法初态且 `10 < r_minT <= 50`；
   - `valid_evade_fail_far_miss`：合法初态但远脱靶；
   - `stage3_diverged`；
   - `invalid_initial_geometry`。

只有第一类适合论文中作为“近失失败”对照。

### 8.2 若要端到端三维轨迹，必须扩展 stage3

当前 `stage3_rollout.m` 只返回标量。若要画“规避 + 末制导”的完整三维航迹，需要新增一个轨迹记录版本（或给原函数加可选输出），至少返回：

```text
t_stage3, pos_M_stage3, pos_T_stage3(optional), a_M_stage3
```

随后在 Python 中将 stage2 与 stage3 轨迹拼接，并在图例标注阶段切换点。

### 8.3 代表轨主图建议

在未完成上述过滤前：

- 可保留 ep069 作为当前代码下的成功动作坍塌诊断案例；
- 不要把 ep002 当作物理失败案例写入论文结论；
- 更稳妥的主结果仍是合法筛选后的 MC 统计；
- 若要展示动作坍塌，可单独用 A/B 全段动作分布，而非使用退化 B 型初态论证控制性能。

---

## 9. 相关旧背景

- 0711 的历史全段动作探针：`rl_td3/logs/0711_300k/full_traj_action_probe_20ep.log`
  - 三维动作全程贴边；
  - A 型：`gamma=3, wT=1e5, wE=1e2` 常数；
  - B 型：短轨先 `[1,1,1]`，再切至 `[1,-1,-1]`。
- 诊断总结：`rl_td3/md document/SESSION_SUMMARY_20260711_FULL_TRAJ_ACTION_COLLAPSE.md`
- 用法总览：`rl_td3/md document/BC_TD3_USAGE.md`

