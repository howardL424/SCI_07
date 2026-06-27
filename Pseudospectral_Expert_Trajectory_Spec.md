# 伪谱专家轨迹改造规格（精简版）



> **目标**：改造 `main_jiaBP3_0421_duo.m` → Radau 伪谱法生成 **1000 条**专家轨迹 → GAIL 预训练。  

> **参考代码**：

> - 双防规避 + 样本循环：`main_jiaBP3_0421_duo.m`

> - **攻–目标相对运动 + 三阶段流程**：`main_changjing3_0421_LM.m`

> - 理论锚点：本人论文 `新论文_0625.pdf`、Hui 2025 伪谱框架

> - **旧版备份**：`Pseudospectral_Expert_Trajectory_Spec_v1.1_backup.md`



---



## 1. 改造流水线（两阶段 warm-start）



```

samples_clean 初态 X_k (1000组)

  → 【阶段 A】固定权重 RK4 全段仿真 → 可行轨迹 τ_feas^(k)

  → 【阶段 B】τ_feas^(k) 作初值 → Radau 伪谱 NLP 放开权重优化 → τ_opt^(k)

  → 逐步提取 (s_t, a_t)

  → 保存 expert_lib_1000.mat

```



**GAIL 要学**：$\pi(\mathbf{a}|\mathbf{s})$，$\mathbf{a}=[w,\gamma,w_T,w_E]$，$d^*=20$ m 固定。



**轨迹范围**：**全段仿真**（PN 打目标 → 双防规避 → PN 恢复打目标），含 $w_T$ 目标打击项。



> 两阶段是**每条轨迹的数值求解策略**，不是“1000 条里 $w,w_E$ 永远固定”。阶段 A 临时固定权重只为生成初值；阶段 B 输出的 $\{w^*,\gamma^*,w_T^*,w_E^*\}$ 才是专家动作标签。



---



## 2. 代码分工（改造时直接对照）



| 功能 | 主参考文件 | 关键函数/逻辑 |

| :--- | :--- | :--- |

| 双防相对运动、最优规避指令 | `main_jiaBP3_0421_duo.m` | `computeLOS`, `computeMDDot`, `computeBC`, `calc_Phi12`, `computeOptimalCmd_duo` |

| **攻–目标相对运动** | **`main_changjing3_0421_LM.m`** | `computeMTDot`（式2）、`computeLOS(pos_M,T)` |

| 三阶段仿真（PN→规避→PN） | **`main_changjing3_0421_LM.m`** | `r_safe=8000` 触发；stage 1/2/3 主循环 |

| 初态重建、批量样本 | `main_jiaBP3_0421_duo.m` | 420–441 行几何重建 |

| 初态采样分布 | 论文 §5.1 + `samples_clean` | 见 §4 |

| **阶段 A warm-start** | **`main_changjing3_0421_LM.m`** | RK4 正向积分，权重全固定 |



> `main_changjing3_0421_LM.m` 中 `computeOptimalCmd_duo` 仍为**双防耦合**（式34），规避前/后用 PN 打目标；伪谱阶段 B 需在 NLP 中编码**式47完整目标项 $w_T$**（RK4 脚本中尚未编码，改造时需新增）。



---



## 3. 状态 / 动作定义



### 3.1 动作 $\mathbf{a}$（4维，GAIL输出）



$$\mathbf{a} = [w,\,\gamma,\,w_T,\,w_E]^T,\quad d^* = 20\ \mathrm{m}\ \text{（固定）}$$



| 参数 | 典型值/边界 | 代码变量 | 阶段 A | 阶段 B |

| :--- | :--- | :--- | :---: | :---: |

| $w$ | $[10^3,\,10^5]$，典型 $10^4$ | `a` | 固定 | 决策变量 |

| $\gamma$ | $[-3,3]$ | `gamma` → `a1,a2` via sigmoid | 固定 | 决策变量 |

| $w_T$ | $>0$，典型 $10^3$ | 待加 | 固定 | 决策变量 |

| $w_E$ | $[10^2,\,10^4]$，典型 $1.44\times10^3$ | `wE` | 固定 | 决策变量 |



权重分配：$w_1 = w/(1+e^{-\gamma})$，$w_2 = w e^{-\gamma}/(1+e^{-\gamma})$。



**阶段 A 默认固定值**（warm-start 用，不要求最优）：



| 参数 | 默认值 |

| :--- | :--- |

| $w$ | $10^4$ |

| $\gamma$ | $0$（$w_1=w_2$） |

| $w_T$ | $10^3$（初猜，阶段 B 再优化） |

| $w_E$ | $1.44\times10^3$ |



### 3.2 状态变量的区分（GAIL状态 vs 伪谱动力学状态）



**1. 伪谱法(CasADi)的内部动力学状态 $\mathbf{x}(t)$**：

- Radau 伪谱 NLP 的连续状态变量为各飞行器运动学状态：`[pos1, pos2, pos3, theta, psi, V]`。

- 缺陷约束、过载约束、目标函数均基于 $\mathbf{x}(t)$ 演化计算。



**2. GAIL 策略网络的状态输入 $\mathbf{s}$（对齐 `samples_clean` 初态）**：

- GAIL 根据**初始几何态势**输出权重 $\mathbf{a}$，故 $\mathbf{s}$ 为初态特征 $\mathbf{X}$。

- 论文定义：$\mathbf{X}=[\mathbf{X}_1,\mathbf{X}_2,r_{AD0},\beta]$，$\mathbf{X}_i=[q_{yi0},\eta^A_{yi0},\eta^A_{zi0},\eta^D_{yi0},\eta^D_{zi0}]$。



**`samples_clean` 列映射**（14 列，仅初态；不含 $d^*$、不含 $\gamma$）：



| 列 | 字段 | 备注 |

| :---: | :--- | :--- |

| 1–4 | $\eta^A_{y10},\eta^A_{z10},\eta^A_{y20},\eta^A_{z20}$ | 攻方初始视线角误差 |

| 5 | $q_{y10}$ | 初始高低视线角 |

| 6–9 | $\eta^D_{y10},\eta^D_{z10},\eta^D_{y20},\eta^D_{z20}$ | 防方初始视线角误差 |

| 10 | $r_{AD0}$ | 攻–防初始距离基准 |

| 11 | $\beta$ | 距离差参数 |

| 12–14 | $q_{z20},q_{z10},q_{y20}$ | 补充视线角 |



距离：$r_{AD10}=r_{AD0}+\beta$，$r_{AD20}=r_{AD0}-\beta$；$r_{AD0}$ 来自 `samples_clean` 第 10 列。$\gamma$ 及全部权重量由**阶段 B 伪谱法求解**，不作为采样输入。



### 3.3 攻–目标通道（来自 `main_changjing3_0421_LM.m`）



```matlab

T = [0, 0, 0];

[q_theta, q_psi, r_MT] = computeLOS(pos_M, T);           % 攻看目标

[q_dot_theta, q_dot_psi] = computeMTDot(M, r_MT, q_theta, q_psi);  % 式(2)

a_M_cmd = computePN(KM, V_M, q_dot_MT(1), q_dot_MT(2), theta_M);   % 规避前/后

```



终端时刻：$t_{fT} = t - r_{AT}/\dot{r}_{AT}$（式49）。



---



## 4. 仿真常数（论文 §5.1）



| 参数 | 值 |

| :--- | :--- |

| $V_A, V_D$ | 500 / 600 m/s |

| $K_A, K_{Di}$ | 3 / 4 |

| 过载上限 | 8g / 10g |

| $r_{safe}$ | 8 km |

| 拦截半径 | 6 m |

| $d^*$ | 20 m（固定，不进 `samples_clean`） |



**采样分布**：$q_{yi0}\in[20°,70°]$；$\eta^A_{yi0},\eta^D_{yi0}\in[-20°,20°]$；$\eta^A_{zi0},\eta^D_{zi0}\in[-10°,10°]$；$\beta\in[-1.5,1.5]$ km；$r_{AD0}\in[6.5,13.5]$ km；$\eta^A_{yi0},\eta^A_{zi0}$ 正态，其余均匀。



---



## 5. 两阶段伪谱求解方案



### 5.1 阶段 A：固定权重 RK4 可行轨迹（warm-start）



**目的**：为阶段 B 提供物理可行、三阶段可跑通的初值 $\tau_{\text{feas}}$，不要求 $J$ 最优。



**输入**：`samples_clean` 第 $k$ 组初态 → 几何重建（参考 `main_jiaBP3_0421_duo.m` 420–441 行）→ 各飞行器 `[pos, theta, psi, V]`。



**固定权重**（见 §3.1 默认值）：$w,\gamma,w_T,w_E$ 全部常数，$d^*=20$。



**仿真逻辑**（沿用 `main_changjing3_0421_LM.m`）：



| Stage | 条件 | 攻弹指令 |

| :---: | :--- | :--- |

| 1 | $r_{MDi}>r_{safe}$ | PN 打目标 |

| 2 | 任一 $r_{MDi}\le r_{safe}$ 且规避未结束 | `computeOptimalCmd_duo`（双防 OBSM） |

| 3 | 规避结束 | PN 恢复打目标 |



**输出** $\tau_{\text{feas}}^{(k)}$：

- 时间序列 $t_k$

- 各飞行器状态 $\mathbf{x}_M(t_k),\,\mathbf{x}_{D1}(t_k),\,\mathbf{x}_{D2}(t_k)$

- 记录 stage 切换时刻（供阶段 B 分段参考）



**验收**：动力学积分无发散；过载未持续饱和；双防最小距离 $\ge 6$ m；目标通道 $r_{MT}$ 合理下降。



### 5.2 阶段 B：Radau 伪谱 NLP（放开权重优化）



**目的**：以 $\tau_{\text{feas}}$ 为初值，求解最优轨迹 $\tau_{\text{opt}}$ 及专家动作序列 $\mathbf{a}^*(t_k)$。



**初值构造**：

- 将 $\tau_{\text{feas}}$ 插值到 Radau 配点网格（$N=20\sim40$）→ 状态初值 $\mathbf{x}_k^{(0)}$

- 各配点权重初值填阶段 A 固定常数 → $\{w_k^{(0)},\gamma_k^{(0)},w_{T,k}^{(0)},w_{E,k}^{(0)}\}$



**决策变量**（推荐，按收敛难度递进）：



| 子方案 | 放开变量 | 适用 |

| :--- | :--- | :--- |

| B1（首版） | $\gamma_k,\,w_{T,k}$ | 阶段 B 首次尝试 |

| B2（完整） | $w_k,\,\gamma_k,\,w_{T,k},\,w_{E,k}$ | B1 收敛后再放开 |



权重经 sigmoid 映射为 $a_{1k},a_{2k}$，嵌入 `computeOptimalCmd_duo` 闭环导引律 → 攻弹加速度 $\mathbf{u}_{A,k}$。



**目标函数**（式22离散，**含 $w_T$ 目标打击项**）：

$$J \approx \sum_i \frac{w_i}{2}[z_i(t_{fi})-d^*]^2 + \frac{w_E}{2}\sum_k \|\mathbf{u}_{A,k}\|^2\Delta t_k + \frac{w_T}{2}z_T^2(t_{fT})$$



**约束**：

- Radau 缺陷（式5 运动学 + 式1/2 相对运动）

- $\|a_A\|\le 8g$，$\|a_D\|\le 10g$

- 终端脱靶 $\ge 6$ m

- 三阶段切换逻辑在 NLP 中表达（stage 边界与 $r_{safe}$ 触发）



**输出**：

- 最优轨迹 $\tau_{\text{opt}}^{(k)}$

- 各配点专家动作 $\mathbf{a}_k^*=[w_k^*,\gamma_k^*,w_{T,k}^*,w_{E,k}^*]$

- 可插值到 $\Delta t=0.01\sim0.05$ s 供 GAIL 训练



**不收敛降级策略**：

1. B1 仅放开 $\gamma,w_T$，$w,w_E$ 保持阶段 A 常数

2. 减小配点数 $N$ 或缩短时间窗

3. 丢弃该初态，补采新初态重试



### 5.3 两阶段关系示意



```

samples_clean X_k

       │

       ▼

┌──────────────────────────────────┐

│  阶段 A：RK4，权重全固定          │

│  w=1e4, γ=0, wT=1e3, wE=1.44e3  │

│  → τ_feas (状态轨迹 + stage边界)  │

└──────────────────────────────────┘

       │ 插值到 Radau 配点

       ▼

┌──────────────────────────────────┐

│  阶段 B：伪谱 NLP，放开权重       │

│  min J，Radau缺陷 + 过载 + 脱靶   │

│  → τ_opt, a*(t_k)                │

└──────────────────────────────────┘

       │

       ▼

  extract (s_t, a_t) → expert_lib

```



---



## 6. 新增文件（建议）



| 文件 | 作用 |

| :--- | :--- |

| `reconstruct_geometry.m` | 从 `samples_clean` 14 列重建各飞行器初态 |

| `simulate_warmstart_trajectory.m` | **阶段 A**：固定权重 RK4 全段仿真，输出 $\tau_{\text{feas}}$ |

| `build_ps_nlp.m` | **阶段 B**：CasADi 构造 Radau 伪谱 NLP（含 $w_T$、三阶段逻辑） |

| `solve_ps_trajectory.m` | **阶段 B**：以 $\tau_{\text{feas}}$ 为初值调用 IPOPT 求解 |

| `extract_expert_transitions.m` | 从 $\tau_{\text{opt}}$ 提取 $(s_t, a_t)$ 序列 |

| `generate_ps_expert_library.m` | 批处理主入口：A → B → extract，循环 1000 组 |



**输出** `expert_lib_1000.mat`：



```matlab

trajectories(k).t          % 时间

trajectories(k).state      % 动力学状态 x(t)

trajectories(k).action     % 专家动作 [w, gamma, w_T, w_E](t)

trajectories(k).accel        % 攻弹加速度

trajectories(k).meta       % 初态 X_k, stage边界, 求解器状态, J 终值

```



---



## 7. 实施阶段



| 阶段 | 内容 | 规模 |

| :--- | :--- | :--- |

| P0 | 合并两脚本函数；阶段 A RK4 全段回放验证 | 10 组 |

| P1 | 阶段 B 伪谱 NLP（B1：仅 $\gamma,w_T$ 放开，含 $w_T$ 目标项） | 10 组 |

| P2 | 两阶段贯通；输出 $(s,a)$；$d^*=20$ 固定 | 100 条 |

| P3 | B2 四权重全放开；扩至 1000 条 + RK4 质检 | 1000 条 |

| P4 | （可选）配点权重分段常值 → 时变权重 | 1000 条 |



---



## 8. 原待确认事项（已确认）



| # | 事项 | 当前倾向 |

| :---: | :--- | :--- |

| 1 | 伪谱求解器 | CasADi + IPOPT |

| 2 | $r_{AD0}$ 来源 | `samples_clean` 第 10 列，$\mathcal{U}[6.5,13.5]$ km |

| 3 | 轨迹范围 | **全段**（PN + 规避 + PN），含 $w_T$ |

| 4 | 失败轨迹 | 丢弃并补采 |

| 5 | 阶段 B 首版放开变量 | B1：$\gamma,w_T$；收敛后 B2 四权重全放开 |

| 6 | `samples_clean` 生成脚本 | 见`samples_clean.mat` |



---



*版本 v1.2 | 两阶段 warm-start；旧版见 `Pseudospectral_Expert_Trajectory_Spec_v1.1_backup.md`*

