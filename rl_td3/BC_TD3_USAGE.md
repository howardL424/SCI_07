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
| 终端奖励缓存 | `rl_td3/expert_terminal_cache.npz`（**训练前必须有**） |
| 训练随机种子 | **当前无 CLI `--seed`**（训练默认不可复现） |
| 评估随机种子 | `--seed`（默认 `0`），控制 `samples_clean` 初态抽样序列 |

---

## 1. 目录与关键文件

```
rl_td3/
├── train_td3.py                 # 训练入口
├── td3_bc.py                    # TD3 + 衰减 BC 正则
├── evaluate_td3.py              # 蒙特卡洛评估
├── evasion_env.py               # Gymnasium 环境（MATLAB Engine）
├── load_expert_buffer.py        # 专家库 → ReplayBuffer + 专家 (s,a) 注入
├── precompute_expert_terminals.py  # 专家末态跑 stage3，生成终端奖励缓存
├── expert_terminal_cache.npz    # 终端 R1/+100 缓存（需先生成）
├── matlab/
│   ├── env_reset.m              # 抽初态 → warm-start → stage2 起点
│   ├── env_step.m               # stage2：式47 + RK4 + 过程奖励
│   ├── stage3_rollout.m         # stage2 结束后一次 PN 积分 → r_minT
│   └── target_zem.m             # ZEM（reset 仍返回；新奖励不再用）
└── checkpoints/                 # 正式训练产物（建议每次用独立子目录）
```

---

## 2. 问题设定（与代码一致）

### 2.1 状态 / 动作

- **观测**：13 维（与 `extract_expert_transitions.m` 一致）
- **动作**：归一化 `[γ_n, wT_n, wE_n] ∈ [-1,1]³`
- **固定**：`w=1e4`，`d*=20 m`
- **反归一化**（Python / MATLAB 必须同步）：

```text
γ  = γ_n × 3                         → [-3, 3]
wT = 10^(wT_n × 1.5 + 3.5)           → [1e2, 1e5]
wE = 10^(wE_n + 3)                   → [1e2, 1e4]
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

### 3.2 生成专家终端奖励缓存（必做）

BC 末步要与在线环境一样追加 `R1+bonus`，需先缓存：

```powershell
cd D:\LTY\yan\yanerxia\GAIL

E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/precompute_expert_terminals.py
# 或指定专家库 / 输出：
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/precompute_expert_terminals.py `
  --expert E:\matlab\result_0714\expert_lib_1000.mat `
  --out rl_td3/expert_terminal_cache.npz
```

说明：

- 默认 stage3 步长 `--dt 1e-2`（比在线 `1e-3` 快）；要更高保真可 `--dt 1e-3`
- 换成新专家库后必须**重新**跑本脚本
- 缺缓存时 `load_expert_buffer` 会直接报错并提示上述命令

### 3.3（可选）检查专家库格式

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/load_expert_buffer.py E:\matlab\result_0714\expert_lib_1000.mat
```

不启动训练；会打印 `expert_sa` 形状、动作范围、过程奖励与缓存摘要。

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
  --terminal-cache rl_td3/expert_terminal_cache.npz `
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
| 终端 tee 日志 | `rl_td3/logs（终端输出记录）/{name}_{时间戳}.log`（可用 `--log-file` / `--no-log`） |

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
& $PY rl_td3/precompute_expert_terminals.py

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
| state/action 定义 | `expert_trajectory/extract_expert_transitions.m` |
| 训练入口 | `rl_td3/train_td3.py` |
| TD3+BC | `rl_td3/td3_bc.py` |
| 环境 / 终端 R1 | `rl_td3/evasion_env.py` |
| stage2 奖励 | `rl_td3/matlab/env_step.m` |
| stage3 脱靶 | `rl_td3/matlab/stage3_rollout.m` |
| BC 预填充 | `rl_td3/load_expert_buffer.py` |
| 评估 | `rl_td3/evaluate_td3.py` |
| 旧设计对话摘要 | `rl_td3/CONTEXT_TD3_TRAINING.md` |
| 历史实验结论 | `rl_td3/SESSION_SUMMARY_*.md` |
