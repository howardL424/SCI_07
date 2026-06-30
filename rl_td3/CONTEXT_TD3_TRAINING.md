# TD3 制导律权重训练 — 对话上下文摘要

> **适用范围**：本文件仅覆盖「用 TD3 直接训练论文式(47)制导律权重系数」的代码生成与设计决策。  
> **论文锚点**：`新论文_0625.pdf`（双防规避 + 目标打击一体化最优制导）；**不**包含后续 Python 环境/依赖安装排错内容。  
> **生成时间**：2026-06-30

---

## 0. 任务背景（对话起点）

**原论文在线流程**：状态 `X` → BP 代理模型 → LM 迭代求参 `p=[d*, γ]` → 代入式(47)得加速度指令。

**本项目改造目标**：用强化学习替代 BP+LM，学习映射  
`π(a|s) → a=[γ, w_T, w_E]`（`w` 固定、`d*` 固定），每控制步更新一次，经式(47)生成攻击弹指令。

**训练范式**：**BC 热启动 + TD3 精调**  
- 用伪谱专家库 `expert_lib_1000.mat` 预填充 ReplayBuffer（行为克隆初始化）  
- 再在线与环境交互，用 Stable-Baselines3 TD3 精调 Actor  
- 与模板中「GAIL 预训练 + DDPG 精调」结构对等，但实现更轻量

**技术栈（设计选型）**：
- Python：`Gymnasium` 环境 + `Stable-Baselines3` TD3  
- 仿真：`MATLAB Engine` 调用现有 `evasion_helpers.m`（式47 数值版）  
- 专家数据：伪谱批处理产物 `expert_lib_1000.mat`（与 `extract_expert_transitions.m` 对齐）

---

## 1. 已完成的代码与功能

### 1.1 新增目录结构

```
GAIL/
├── evasion_helpers.m              # 已有：式47 + RK4 + PN（环境与专家提取共用）
├── samples_clean.mat              # 已有：初态采样（reset 随机抽行）
├── expert_trajectory/             # 已有：伪谱专家轨迹生成
│   ├── reconstruct_geometry.m
│   ├── simulate_warmstart_trajectory.m
│   ├── extract_expert_transitions.m
│   └── expert_lib_1000_file.m     # 默认归档路径指针
└── rl_td3/                        # ★ 本次新增
    ├── matlab/
    │   ├── env_reset.m            # episode 初始化
    │   └── env_step.m             # 单步仿真 + reward
    ├── evasion_env.py             # Gymnasium 环境（Python 侧）
    ├── train_td3.py               # BC + TD3 训练入口
    ├── load_expert_buffer.py      # 专家库 → ReplayBuffer 预填充
    ├── evaluate_td3.py            # 蒙特卡洛评估
    └── CONTEXT_TD3_TRAINING.md    # 本文件
```

### 1.2 MATLAB 包装层

| 文件 | 功能 |
|------|------|
| `env_reset.m` | `samples_clean` 一行 → `reconstruct_geometry` → `simulate_warmstart_trajectory` → 定位规避段 `stage==2` 起点 → 输出 **13维 obs**、**18维绝对状态**、`Tev`、`r_MT0` |
| `env_step.m` | 接收归一化动作 `[γ_n,wT_n,wE_n]`，反归一化后调用 `computeOptimalCmd_duo_target`（式47），RK4 推进 `dt=0.1s`，防御弹 PN，计算 reward 与 done |

### 1.3 Python 层

| 文件 | 功能 |
|------|------|
| `evasion_env.py` | `gymnasium.Env`：`reset`/`step` 经 MATLAB Engine 调上述 `.m`；`norm_action` / `denorm_action` 静态方法供 BC 与评估复用 |
| `train_td3.py` | SB3 `TD3` + `NormalActionNoise`；默认 BC 预填充后 `learn(500_000)`；支持 `--no-bc`、`--steps`、`--expert` |
| `load_expert_buffer.py` | 读 `results.expert_sa [N×17]`，丢弃第14列 `w`，归一化 `[γ,wT,wE]` 写入 ReplayBuffer；支持 v7 / v7.3 `.mat` |
| `evaluate_td3.py` | 加载 `.zip` 模型，蒙特卡洛统计规避率/到达率/脱靶距离/动作分布 |

### 1.4 依赖的既有工程资产（未改动物理核心）

- `evasion_helpers.m` → `computeOptimalCmd_duo_target`（式47 三通道 D1/D2/目标耦合）
- `expert_trajectory/extract_expert_transitions.m` → **state/action 字段定义权威来源**
- 专家库默认路径：`E:\matlab\result_0628\reault_p3_1000\expert_lib_1000.mat`（见 `expert_lib_1000_file.m`）

---

## 2. 核心架构与设计决策

### 2.1 总体数据流

```mermaid
flowchart TD
    subgraph train [训练侧 Python]
        TD3[SB3 TD3 Actor-Critic]
        RB[ReplayBuffer]
        ENV[EvasionEnv Gymnasium]
    end
    subgraph matlab [仿真侧 MATLAB]
        RESET[env_reset.m]
        STEP[env_step.m]
        EH[evasion_helpers.m 式47]
    end
    EXPERT[expert_lib_1000.mat] -->|BC 预填充| RB
    RB --> TD3
    TD3 -->|a=[γ_n,wT_n,wE_n]| ENV
    ENV --> RESET
    ENV --> STEP
    STEP --> EH
    EH -->|obs 13维 reward done| ENV
    ENV --> TD3
```

### 2.2 状态空间 `s`（13 维，与 `extract_expert_transitions.m` 一致）

| 列 | 符号 | 含义 |
|----|------|------|
| 1 | `r_MD1` | 攻–D1 距离 (m) |
| 2 | `rdot_MD1` | 攻–D1 接近率（正=远离） |
| 3–4 | `qy_D1M, qz_D1M` | D1→M 视线角 (rad) |
| 5 | `r_MD2` | 攻–D2 距离 |
| 6 | `rdot_MD2` | 攻–D2 接近率 |
| 7–8 | `qy_D2M, qz_D2M` | D2→M 视线角 |
| 9 | `r_MT` | 攻–目标距离 |
| 10–11 | `qy_MT, qz_MT` | M→T 视线角 |
| 12 | `V_M` | 攻弹速度 (m/s) |
| 13 | `tgo_norm` | `(Tev - t) / Tev`，规避段归一化剩余时间 |

**episode 范围**：仅 **规避段**（warm-start 进入 `stage==2` 之后），不是全段 PN→规避→PN。  
**绝对状态**（内部用，不直接进网络）：`state_abs [18]` = `[M(6); D1(6); D2(6)]`。

### 2.3 动作空间 `a`（3 维，用户确认的设计）

| 决策 | 内容 |
|------|------|
| **输出时机** | **每控制步**（`dt=0.1s`）更新，与专家 `(s,a)` 逐步提取一致 |
| **网络输出** | `[γ_n, wT_n, wE_n] ∈ [-1,1]³`（SB3 `Box` 动作空间） |
| **固定参数** | `w = 1e4`；`d* = 20 m`（`cfg.dstar`） |
| **反归一化** | `γ = γ_n×3`；`wT = 10^(wT_n+3)`；`wE = 10^(wE_n+3)` |
| **分配式(23)** | `w1 = w/(1+e^{-γ})`，`w2 = w·e^{-γ}/(1+e^{-γ})` |
| **执行** | `(w1,w2,wT,wE,d*)` → `computeOptimalCmd_duo_target` → `a_M` → `updateState` |

**不在动作空间**：`w`（固定）、`d*`（固定）。原 BP+LM 中的 `d*` 不再在线优化。

### 2.4 奖励函数（`env_step.m`）

每步稠密奖励：

```
R = k1·[log10(r_MD1) + log10(r_MD2)]     # 规避距离（用 step 后值，下限 1m）
  - kT·(r_MT / r_MT0)                      # 目标距离项（归一化，鼓励逼近目标）
  - kE·||a_M||² · dt                       # 能量惩罚
```

默认系数：`k1=1.0`，`kT=0.5`，`kE=1e-6`。

终端：
- 被拦截（`min(r_MD1,r_MD2) ≤ 6 m`）：`R -= 100`，`terminated=True`
- 到达目标（`r_MT ≤ 50 m` 且 `t > 0.5·Tev`）：`R += 100`，`terminated=True`
- 超时（`t ≥ 1.5·Tev`）：`terminated=True`（无额外终端奖励）

### 2.5 BC 预填充规则（`load_expert_buffer.py`）

- 数据源：`results.expert_sa`，shape `[N_total, 17]`
- 列映射：`0:13` state，`13` w（**丢弃**），`14:17` → γ, wT, wE
- 动作归一化与 `EvasionEnv.norm_action` / `env_step.m` 反归一化 **必须成对一致**
- `next_state`：同轨迹相邻步；episode 末步 `done=True`，`next_state` 用自身填充
- `reward`：按与 `env_step.m` 相同的公式在 Python 侧重算（BC 阶段无 `a_M` 幅值时用近似）

### 2.6 TD3 默认超参（`train_td3.py`）

| 参数 | 默认值 |
|------|--------|
| `total_timesteps` | 500,000 |
| `learning_rate` | 1e-3 |
| `batch_size` | 256 |
| `buffer_size` | 300,000 |
| `gamma` | 0.99 |
| `tau` | 0.005 |
| `policy_delay` | 2 |
| `net_arch` | [256, 256] |
| `action_noise` | `Normal(0, 0.1)`，3 维 |
| `learning_starts` | 1,000（有 BC 时可视为已有数据） |

### 2.7 与论文 / 模板定位

| 对比项 | 原 BP+LM | 本方案 |
|--------|----------|--------|
| 在线求参 | LM 迭代 | TD3 Actor 前向一次 |
| 预训练 | BP 监督学习 | 专家库 ReplayBuffer 预填充（BC） |
| 动作 | 主要为 `d*, γ`（规避段初值） | 每步 `γ, wT, wE`（`w,d*` 固定） |
| 专家来源 | BP 训练样本分布 | 伪谱 `expert_lib_1000.mat` |

---

## 3. 强化学习训练与验证使用方法

**工作目录**：`D:\LTY\yan\yanerxia\GAIL`  
**命令位置**：Cursor 底部终端（PowerShell），在项目根目录执行。

### 3.1 推荐流程

```powershell
# 0) 确认 Python 解释器指向已配置 RL 依赖的虚拟环境（如 rl_pytorch）

# 1) 专家库格式检查（不启动 MATLAB，不训练）
python rl_td3/load_expert_buffer.py E:\matlab\result_0628\reault_p3_1000\expert_lib_1000.mat

# 2) BC + TD3 正式训练（会启动 MATLAB Engine，首启较慢）
python rl_td3/train_td3.py

# 3) 冒烟训练（先验证链路）
python rl_td3/train_td3.py --steps 5000

# 4) 纯 TD3 对比（不做 BC）
python rl_td3/train_td3.py --no-bc --steps 500000

# 5) 指定专家库路径
python rl_td3/train_td3.py --expert E:\path\to\expert_lib_1000.mat

# 6) 评估已训练模型
python rl_td3/evaluate_td3.py --model rl_td3/checkpoints/best/best_model --episodes 100
```

### 3.2 训练时实际发生的事

1. `EvasionEnv.__init__` 启动 MATLAB Engine，`addpath`：`rl_td3/matlab`、`GAIL/`、`expert_trajectory/`
2. `load_expert_buffer`（除非 `--no-bc`）将专家 transition 写入 ReplayBuffer
3. 每个 rollout：`reset` 从 `samples_clean` 随机抽初态 → warm-start 到规避段起点
4. 每 `step`：TD3 输出 3 维动作 → `env_step.m` 仿真 0.1s → 返回 13 维 obs 与 reward
5.  checkpoint 保存至 `rl_td3/checkpoints/`；TensorBoard 日志在 `rl_td3/checkpoints/tb_logs/`

### 3.3 产物路径

| 产物 | 路径 |
|------|------|
| 最终模型 | `rl_td3/checkpoints/td3_evasion_bc.zip` |
| 最优模型 | `rl_td3/checkpoints/best/best_model.zip` |
| 周期性 checkpoint | `rl_td3/checkpoints/td3_evasion_bc_*_steps.zip` |
| 评估日志 | `rl_td3/checkpoints/eval_logs/` |

---

## 4. 约束条件与避坑指南

### 4.1 维度与数据对齐

- **state 必须是 13 维**，顺序严格按 `extract_expert_transitions.m`；与 `samples_clean` 的 12/14 维初态向量不是同一套定义。
- **action 是 3 维**（不是最初方案的 4 维）；专家 `expert_sa` 第 14 列 `w≈1e4` 在 BC 加载时**必须丢弃**。
- **归一化/反归一化**在 Python（`norm_action`）与 MATLAB（`env_step.m`）两处实现，改一处必须同步改另一处。

### 4.2 专家库 `.mat` 格式

- 大文件常为 **MATLAB v7.3 (HDF5)**，`scipy.io.loadmat` 会报 `NotImplementedError`。
- `load_expert_buffer.py` 已做 fallback：`mat73.loadmat`。**不要修改** `expert_lib_1000.mat` 本体。
- `expert_sa` 是**所有轨迹步拼接**的大矩阵；`done` 边界靠 `results.trajectories(k).transitions.t` 长度推断。

### 4.3 MATLAB Engine 与路径

- 首次 `train_td3.py` / `EvasionEnv()` 会启动 MATLAB，耗时数十秒属正常。
- `env_reset.m` / `env_step.m` 依赖：`evasion_helpers.m`、`reconstruct_geometry.m`、`simulate_warmstart_trajectory.m`；路径由 `addpath` 注入。
- 每步一次 Python↔MATLAB 往返，**单步约 1–5 ms 量级**；全训练偏慢，属架构取舍（保证与 MATLAB 式47 完全一致）。

### 4.4 reset 失败重采样

- 部分 `samples_clean` 初态 warm-start 后**无规避段**（`stage==2` 为空），`env_reset` 返回 `ok=false`。
- `EvasionEnv.reset` 最多重试 `max_retries=10` 次；连续失败会 `RuntimeError`。

### 4.5 仿真与论文差异点（已知简化）

- `env_step.m` 规避段**始终**用式(47)双防+目标一体化制导；未复现 `simulate_warmstart_trajectory.m` 中「两防距离差>3000m 时改单弹规避」的分支逻辑。
- `reset` 用 warm-start 定位规避段起点，但 **在线 step 不从 warm-start 中间状态接续全段**，而是从规避段初态开始独立 rollout。
- `Tev` 取 warm-start 轨迹中规避段时长估计，用于 `tgo_norm` 与超时判断。

### 4.6 奖励与量级

- `wT`/`wE` 物理范围跨 **2 个数量级**，必须用 **对数归一化** `10^(a+3)`，否则 Actor 梯度失衡。
- `kT·r_MT/r_MT0` 已做归一化；若规避项过弱/过强，优先调 `k1`/`kT`，而不是改 state 定义。
- BC 预填充的 reward 为 Python 重算近似，与在线 `env_step` 可能有微小偏差；在线训练后以环境 reward 为准。

### 4.7 与伪谱专家生成流水线的关系

```
samples_clean → 阶段A warm-start → 阶段B 单打靶 NLP → extract_expert_transitions
    → expert_lib_1000.mat → load_expert_buffer (BC) → TD3 在线精调
```

- 专家生成入口：`expert_trajectory/generate_ps_expert_library.m`
- 若专家库未生成或路径错误，BC 步骤会 `FileNotFoundError`；可加 `--no-bc` 先跑通环境。

### 4.8 尚未实现 / 待验证项（代码生成时状态）

- [ ] 端到端冒烟：`reset → step` 链路在真实 MATLAB 环境下的完整跑通
- [ ] `samples_clean.mat` 用 `scipy.io.loadmat` 读取（若也为 v7.3，需与专家库同样处理）
- [ ] 与 BP+LM 基线 (`main_changjing3_0421_LM.m`) 的定量对比脚本
- [ ] 奖励系数 `k1,kT,kE` 的敏感性标定

---

## 5. 关键文件速查

| 需求 | 文件 |
|------|------|
| 式47 数值实现 | `evasion_helpers.m` → `computeOptimalCmd_duo_target` |
| state/action 定义 | `expert_trajectory/extract_expert_transitions.m` |
| 专家库格式说明 | `expert_trajectory/docs/Pseudospectral_Expert_Trajectory_Spec.md` |
| 项目总改造路线 | `GAIL-DDPG_3D_Extension_Template.md` §5–§8 |
| Gym 环境 | `rl_td3/evasion_env.py` |
| 单步仿真+奖励 | `rl_td3/matlab/env_step.m` |
| 训练入口 | `rl_td3/train_td3.py` |
| BC 预填充 | `rl_td3/load_expert_buffer.py` |
| 评估 | `rl_td3/evaluate_td3.py` |

---

## 6. 新对话续接建议

开启新对话时可附带：

```
请阅读 GAIL/rl_td3/CONTEXT_TD3_TRAINING.md，继续 TD3 制导权重训练相关工作。
当前状态：[填写 smoke test / 训练进度 / 遇到的问题]
```

并说明是调试环境、改 reward、扩展 state，还是对接论文实验对比。
