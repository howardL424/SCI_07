# BC-TD3 训练与蒙特卡洛评估使用说明

> **生成时间**：2026-07-16  
> **与当前代码对齐**；旧版设计对话摘要见 `CONTEXT_TD3_TRAINING.md`  
> **历史实验和分析改进记录**：同目录 `SESSION_SUMMARY_*.md`

---

## 0. 速览

| 项 | 当前约定 |
|----|----------|
| Python 解释器 | `E:\Python\venvs\rl_pytorch\Scripts\python.exe` |
| 工作目录 | `D:\LTY\yan\yanerxia\GAIL` |
| 默认训练步数 | 300,000 |
| 默认学习率 | `1e-4` |
| 算法类 | `td3_bc.TD3BC`（非裸 `TD3`） |
| 默认专家库 | `E:\matlab\result_0714\expert_lib_1000.mat` |
| 对齐 transition | 旧库：`rl_td3/data/expert_transitions_aligned.mat`；**新库无需** |
| 终端奖励缓存 | `rl_td3/data/expert_terminal_cache.npz`（stage3 sidecar；训练前必须有） |
| 观测归一化 | 固定 `OBS_SCALE`（`obs_norm.py`），专家与在线共用 |
| 训练随机种子 | **当前无 CLI `--seed`**（训练默认不可复现） |
| 评估随机种子 | `--seed`（默认 `0`），控制 `samples_clean` 初态抽样序列 |

---

## 1. 目录与关键文件

```
rl_td3/
├── train_td3.py / evaluate_td3.py / td3_bc.py   # 训练与评估入口
├── evasion_env.py / obs_norm.py                  # 在线环境与观测归一化
├── load_expert_buffer.py / reward_projection.py  # 专家加载与奖励投影
├── data/                      # 派生数据产物（aligned / sidecar / smoke）
│   ├── expert_transitions_aligned.mat
│   ├── expert_terminal_cache.npz
│   └── …
├── tools/                     # 数据管线 CLI（非训练热路径）
│   ├── rebuild_expert_data.py
│   ├── precompute_expert_terminals.py
│   ├── check_expert_alignment.py
│   ├── compare_stage3_dt.py
│   └── validate_zem_vs_rminT.py
├── tests/                     # 轻量验收与单测
│   ├── test_data_alignment.py
│   ├── test_reward_projection.py
│   └── test_pipeline_validation.py
├── matlab/                    # stage2/stage3 MATLAB 核心
├── logs/                      # 终端 tee / rebuild 日志
├── md document/               # 用法说明与会话纪要
├── checkpoints*/              # 训练产物
└── plots/
```

### 1.1 物理 transition schema（`phys_transition_v1`）

生成侧（`extract_expert_transitions`）与在线侧（`env_step`）共用 `stage2_core_init` + `stage2_step`。

**硬性约定**

1. 先 `phys_to_action_norm` 映射/裁剪，再用同一 `action_norm` 积分（禁止「标签已裁剪、`s'` 来自未裁剪动作」）。
2. 末步 `next_state` / `terminal_abs` = 执行最后一拍后的真实末态。
3. 拦截立刻截断；物理层只记 `intercepted=true`，**不写死 −50**（奖励由 `stage2_process_reward` / 后续投影层写入）。
4. 观测存 raw；训练侧统一 `obs_norm.py`（`OBS_NORM_VERSION=obs_scale_0714_v1`）。
5. 大循环只 `stage2_core_init` 一次，禁止逐步重建 helper。

**兼容字段（每步，旧工具可读）**

| 字段 | 形状 | 含义 |
|------|------|------|
| `state` | N×13 | 相对态势 raw |
| `action` | N×4 | `[w,γ,wT,wE]` 原始分段 theta（审计） |
| `state_abs` | N×18 | 绝对态 |
| `t` | N×1 | 时刻 (s) |

**新增权威字段（每步）**

| 字段 | 形状 | 含义 |
|------|------|------|
| `action_norm` | N×3 | 裁剪后 TD3 动作 ∈[-1,1] |
| `next_state` | N×13 | 裁剪动作积分后的真实 `s'` |
| `next_state_abs` | N×18 | 对应绝对态 |
| `done` | N×1 logical | `intercepted \|\| stage2_done` |
| `intercepted` | N×1 logical | `min(r_MD)≤6` |
| `stage2_done` | N×1 logical | 按时长正常结束（拦截时为 false） |
| `action_phys` | N×4 | 与 `action` 同义（审计别名） |

**轨迹级**

| 字段 | 含义 |
|------|------|
| `r_min1` / `r_min2` | stage2 全程最小弹间距离 |
| `terminal_abs` | 真实 stage2 末态（18） |
| `t_end` | 末时刻 |
| `schema_version` | 当前 `phys_transition_v1` |
| `stage2_dynamics_version` | 当前 `stage2_rk4_eq47_v1` |
| `action_mapping_version` | 当前 `td3_box_log_v1` |
| `obs_norm_version` | 当前 `obs_scale_0714_v1` |

版本常量见 `rl_td3/matlab/transition_schema_meta.m`。

---

## 2. 问题设定（与代码一致）

### 2.1 状态 / 动作

- **观测**：13 维物理态势，经 `obs_norm.OBS_SCALE` 逐维相除后输入网络（不裁剪）
- **动作**：归一化 `[γ_n, wT_n, wE_n] ∈ [-1,1]³`
- **固定**：`w=1e4`，`d*=20 m`
- **反归一化**（Python / MATLAB 必须同步）：

```text
γ  = γ_n × 3                         → [-3, 3]
wT = 10^(wT_n × 1.5 + 3.5)           → [1e2, 1e5]
wE = 10^(wE_n + 3)                   → [1e2, 1e4]
```

`OBS_SCALE`（0714 分位数首版，非旧 `obs_high`）：

```text
[3e4, 1.2e3, π/2, π, 3e4, 1.2e3, π/2, π, 1.5e4, π/2, π, 1e3, 1]
```
### 2.2 Episode 结构

1. **stage2（RL）**：从规避段起点起，每 `dt=0.1 s` 输出权重，式(47) 制导  
2. **终止**：被拦截（`min(r_MD)≤6 m`）或规避时长用尽（`t≥Tev`）  
3. **stage3（非 RL）**：规避正常结束后，`stage3_rollout.m` 用 PN 一次积完，得到真实终端脱靶量 `r_minT`

### 2.3 奖励（当前）

**过程奖励**（每步，`env_step.m`）：

```text
R2 = log10(min(r_MD1, d*)-5)  if r_MD1>6 else 0
R3 = log10(min(r_MD2, d*)-5)  if r_MD2>6 else 0
reward = 0.05 * (R2 + R3)
拦截：reward = -50，terminated
```

**终端奖励**（仅规避正常结束且未拦截，Python 追加）：

```text
R1 = 80/(5+r_minT)  if r_minT≤25 else 0
bonus = +100        if 双防全程 min≥6m 且 r_minT≤10m else 0
总终端增量 = R1 + bonus
```

> 旧版「每步 ZEM 目标项 / 能量惩罚 / 规避段内 zT≤25 判到达」已废弃。

奖励由 [`rl_td3/reward_projection.py`](../reward_projection.py) 集中投影：

- 过程项：`next_state` 距离 + `intercepted` → `project_process_rewards`
- 终端项：sidecar 的 `r_minT` + `r_min1/r_min2` → `project_terminal`（`reward_version=r1_bonus_v1`）
- 缓存 `reward` / `R1` / `bonus` 列为派生；**换系数只重投影，不重放 stage2/stage3**
- `load_expert_buffer(..., reward_kwargs={proc_coef:0.1})` 可覆盖阈值

### 2.4 TD3BC

```text
actor_loss = -Q(s, π(s)) + λ(t) · ‖π(s_E) - a_E‖²
λ(t) = λ0 · max(0, 1 - t / T_bc)
```

默认：`λ0=1.0`，`T_bc=150_000`（线性衰减到 0）。

---

## 3. 训练前准备

### 3.1 依赖

- 已配置的 `rl_pytorch` 环境（Gymnasium、SB3、PyTorch、MATLAB Engine、`mat73` 等）
- 本机 MATLAB 可被 Engine 启动
- `samples_clean.mat` 在仓库根目录
- 专家库 `.mat` 路径正确（默认见上表）

### 3.2 旧库：重建对齐 transition（一次性迁移）

以 `trajectories(k).transitions` 为权威源，用 MATLAB 共享核心重放生成 `(s,a,s',r,done)`：

```powershell
cd D:\LTY\yan\yanerxia\GAIL

# 全量（较久）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/rebuild_expert_data.py

# 冒烟子集
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/rebuild_expert_data.py --max-traj 20 `
  --out rl_td3/data/expert_transitions_aligned_smoke.mat
```

也可在 MATLAB 中直接调用 `rebuild_expert_transitions(expert_path, out_path)`。

**新库无需本步**：`generate_ps_expert_library` 出库的 `transitions` 已含 `action_norm`/`next_state`/`done`/`intercepted`/`terminal_abs`（见 §1.1）。`expert_sa` 仅 QA。

### 3.3 子集对比 stage3 dt（建议先做）

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/compare_stage3_dt.py `
  --aligned rl_td3/data/expert_transitions_aligned.mat --n-sample 40
```

正式 stage3 与在线一致，使用 **`dt=1e-3`**。

### 3.4 生成专家终端 sidecar（必做；可断点续跑）

stage3 为**第二阶段**，与主库落盘解耦：

```powershell
# 旧库：读 aligned
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/precompute_expert_terminals.py `
  --expert E:\matlab\result_0714\expert_lib_1000.mat `
  --aligned rl_td3/data/expert_transitions_aligned.mat `
  --dt 1e-3 `
  --out rl_td3/data/expert_terminal_cache.npz

# 新库：直接读 transitions.terminal_abs（可省略 --aligned）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/precompute_expert_terminals.py `
  --expert path\to\new_expert_lib.mat `
  --dt 1e-3 `
  --out rl_td3/data/expert_terminal_cache_new.npz `
  --resume
```

- 权威：`r_minT` / `reached` / `diverged` / `n_steps_stage3` / `stage3_dt` / `r_min1,2`
- 派生（`reward_version`）：`R1` / `bonus` / `both_ok` — 不可代替 `r_minT`
- 拦截跳过 stage3；sidecar 含 `expert_fingerprint` / `stage3_version` 可追溯
- 换成新专家库或重算末态后必须**重新**预计算；`--resume` 可按指纹续跑

### 3.5 一致性检查

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tools/check_expert_alignment.py
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tests/test_data_alignment.py
```

### 3.6 新管线小批量验收（pipeline-validation）

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/tests/test_pipeline_validation.py
# 报告 → rl_td3/data/pipeline_validation_report.json
```

**已通过（2026-07-19）**：5 条 phys smoke 库（与 `extract` 共用 stage2 核心，不经 `rebuild_*`）

| 检查 | 结果 |
|------|------|
| 新 schema / 版本字段 | OK |
| `load_expert_buffer` 跳过 rebuild | OK |
| 逐步 ↔ `env_step`（action_norm / s' / done / intercepted） | OK |
| 末步 done 唯一；拦截无正终端奖 | OK |
| 改 `proc_coef` 重投影（不调 MATLAB） | OK |
| 旧 aligned 路径仍可用 | OK |
| 正式 `stage3_dt` | `1e-3` |

当前版本戳：

```text
schema_version          = phys_transition_v1
stage2_dynamics_version = stage2_rk4_eq47_v1
action_mapping_version  = td3_box_log_v1
obs_norm_version        = obs_scale_0714_v1
stage3_version / dt     = pn_rollout_v1 / 1e-3
reward_version          = r1_bonus_v1
```

**结论**：新 `expert_lib` 训练前**不再**需要 `rebuild_*` 全量 stage2 重放；普通奖励实验只做 `reward_projection` 廉价投影。

---

## 4. 训练使用说明

### 4.1 推荐命令

在仓库根目录执行：

```powershell
# 冒烟（5k 步，写入 checkpoints_smoke，不覆盖正式产物）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py --smoke

# 正式 300k（默认超参）——务必指定独立 save-dir，避免覆盖
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py `
  --save-dir rl_td3/checkpoints/training_300k_YYYYMMDD_HHMM `
  --name td3_300k

# 自定义专家库 / 终端缓存
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py `
  --expert E:\matlab\result_0714\expert_lib_1000.mat `
  --terminal-cache rl_td3/data/expert_terminal_cache.npz `
  --save-dir rl_td3/checkpoints/my_run `
  --name td3_run1

# 关闭 BC 预填充（纯在线 TD3；若 λ0>0 且无专家集，BC 正则不生效）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py --no-bc --steps 300000 `
  --save-dir rl_td3/checkpoints/no_bc --name td3_nobc
```

### 4.2 默认超参（`train_td3.py` → `DEFAULTS`）

| 参数 | 默认值 | CLI |
|------|--------|-----|
| `total_steps` | 300,000 | `--steps` |
| `learning_rate` | 1e-4 | `--lr` |
| `batch_size` | 256 | `--batch` |
| `buffer_size` | 300,000 | `--buf` |
| `gamma` | 0.99 | （代码内） |
| `tau` | 0.005 | （代码内） |
| `policy_delay` | 3 | （代码内） |
| `net_arch` | [256, 256] | （代码内） |
| `noise_sigma` | 0.1 | `--noise` |
| `learning_starts` | 有 BC 时为 0；`--no-bc` 时为 1000 | |
| `eval_freq` | 10,000 | （代码内） |
| `n_eval_episodes` | 20 | `--n-eval` |
| `grad_clip` | ±10.0（元素级 clamp） | `--grad-clip`（0=关） |
| `bc_ratio` | 0.33（最多写 99,000 条） | `--bc-ratio` |
| `bc_lambda0` | 1.0 | `--bc-lambda0` |
| `bc_decay_steps` | 150,000 | `--bc-decay-steps` |
| checkpoint | 每 50,000 步 | |

`--smoke` 预设：`steps=5000`，`save_dir=rl_td3/checkpoints_smoke`，`name=td3_smoke`。

### 4.3 训练时实际流程

1. 启动两套 MATLAB Engine（train env + eval env）  
2. 构建 `TD3BC`，探索噪声 `Normal(0, σ)`  
3. 除非 `--no-bc`：从专家库写入最多 `buffer_size × bc_ratio` 条，并 `set_expert_dataset`  
4. `learn`：`CheckpointCallback` + `EvalCallback`（deterministic）+ 可选梯度裁剪  
5. 保存最终 `{save_dir}/{name}.zip`；最优策略在 `{save_dir}/best/best_model.zip`

### 4.4 产物路径

| 产物 | 路径 |
|------|------|
| 最终模型 | `{save_dir}/{name}.zip` |
| 最优模型 | `{save_dir}/best/best_model.zip` |
| 周期 checkpoint | `{save_dir}/{name}_*_steps.zip` |
| EvalCallback 日志 | `{save_dir}/eval_logs/` |
| TensorBoard | `{save_dir}/tb_logs/` |
| 终端 tee 日志 | `rl_td3/logs/{name}_{时间戳}.log`（可用 `--log-file` / `--no-log`） |

### 4.5 训练随机性说明

- **`train_td3.py` 当前没有 `--seed`**，不固定 NumPy / PyTorch / CUDA / SB3 / 环境抽样。
- 两次「同命令」训练**不保证**数值可复现；对比实验请用相同命令行 + 独立 `--save-dir`，并用蒙特卡洛（固定 `--seed`）比模型质量。

### 4.6 训练注意点

- 非默认 `--steps` 却仍用默认 `save_dir`/`name` 时，脚本会警告可能覆盖正式产物；并行冒烟请用 `--smoke` 或显式改目录。
- 专家库若为 MATLAB v7.3，由 `mat73` 读取；**不要改** `.mat` 本体。
- 部分 `samples_clean` 行 warm-start 后无 stage2，`reset` 最多重试 10 次。

---

## 5. 蒙特卡洛评估使用说明

### 5.1 命令

```powershell
cd D:\LTY\yan\yanerxia\GAIL

# 推荐：评估 best_model，100 轮，固定种子 0（默认）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/evaluate_td3.py `
  --model rl_td3/checkpoints/training_300k_20260715_1008/best/best_model `
  --episodes 100 `
  --seed 0

# 换种子做另一组可复现抽样
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/evaluate_td3.py `
  --model rl_td3/checkpoints/training_300k_20260715_1008/best/best_model `
  --episodes 100 `
  --seed 42

# 调试：打印第 1 轮每步信息
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/evaluate_td3.py `
  --model path/to/model --episodes 5 --seed 0 --render
```

`--model` 写不带或带 `.zip` 均可。优先 `TD3BC.load`，失败再回退 `TD3.load`。

### 5.2 固定种子行为（`--seed`）

蒙特卡洛每轮会从 `samples_clean.mat` 里**随机抽一行**当初态。能不能复现，取决于这个「抽哪一行」的随机数是否被固定。

环境里真正用来抽行的是 Gymnasium 的 `env.np_random`，不是全局的 `np.random`。要固定它，必须调用：

```python
env.reset(seed=某个整数)
```

当前 `evaluate_td3.py` 的做法：

| 行为 | 说明 |
|------|------|
| `--seed` 默认 | `0` |
| MC 第 1 轮 | `env.reset(seed=args.seed)`，把 RNG 钉死 |
| MC 第 2～N 轮 | `env.reset()`（不传 seed），沿同一随机序列继续往下抽 |
| 策略 | `predict(..., deterministic=True)`，无探索噪声 |
| 末尾动作探针 | MC 结束后再用同一 seed 重启 RNG，另抽 20 个初态打印 `(γ,wT,wE)` |

因此：**相同模型 + 相同 `--seed` + 相同 `--episodes` → 同一组初态序列与同一套统计**（MATLAB/硬件无异常时）。

对比不同模型时，务必使用**相同 `--seed` 与 `--episodes`**。

### 5.3 成功判据（与训练环境一致）

| 指标 | 条件 |
|------|------|
| 双防规避成功 | 全程 `r_MD1_min≥6` 且 `r_MD2_min≥6` |
| 打击成功 | stage3 后 `r_minT≤10 m` |
| 双防+打击 | 上两者同时成立 |
| 被拦截 | 任一步 `r_MD<6` |

### 5.4 建议评估对象

| 用途 | 模型路径 |
|------|----------|
| 主结果 | `{save_dir}/best/best_model` |
| 最终步对照 | `{save_dir}/{name}` |
| 某中间步 | `{save_dir}/{name}_150000_steps` 等 |

---

## 6. 端到端推荐流程

```powershell
cd D:\LTY\yan\yanerxia\GAIL
$PY = "E:\Python\venvs\rl_pytorch\Scripts\python.exe"

# 1) 终端缓存（专家库变更后必做）
& $PY rl_td3/tools/precompute_expert_terminals.py

# 2) 冒烟
& $PY rl_td3/train_td3.py --smoke

# 3) 正式训练（示例）
& $PY rl_td3/train_td3.py `
  --save-dir rl_td3/checkpoints/training_300k_$(Get-Date -Format 'yyyyMMdd_HHmm') `
  --name td3_300k

# 4) 蒙特卡洛（固定种子，便于写论文/对比）
& $PY rl_td3/evaluate_td3.py `
  --model rl_td3/checkpoints/<你的目录>/best/best_model `
  --episodes 100 `
  --seed 0
```

---

## 7. 关键键文件速查

| 需求 | 文件 |
|------|------|
| 式47 | `evasion_helpers.m` → `computeOptimalCmd_duo_target` |
| 共享一步动力学 | `rl_td3/matlab/stage2_core_init.m` + `stage2_step.m` |
| 物理 transition schema | `expert_trajectory/extract_expert_transitions.m`；见上文 §1.1 |
| 训练入口 | `rl_td3/train_td3.py` |
| TD3+BC | `rl_td3/td3_bc.py` |
| 环境 / 终端 R1 | `rl_td3/evasion_env.py` |
| stage2 奖励 | `rl_td3/matlab/stage2_process_reward.m`（`env_step` 调用） |
| stage3 脱靶 | `rl_td3/matlab/stage3_rollout.m` |
| BC 预填充 | `rl_td3/load_expert_buffer.py` |
| 奖励投影 | `rl_td3/reward_projection.py` |
| 数据管线 CLI | `rl_td3/tools/`（rebuild / precompute / check） |
| 验收单测 | `rl_td3/tests/` |
| 派生数据 | `rl_td3/data/` |
| 评估 | `rl_td3/evaluate_td3.py` |
| 旧设计对话摘要 | `rl_td3/md document/CONTEXT_TD3_TRAINING.md` |
| 历史实验结论 | `rl_td3/md document/SESSION_SUMMARY_*.md` |
