# Paper Digest: GAIL-DDPG for 3D Endo-Atmospheric Penetration

> **Base Reference**: Hui et al. (2025) — *Endo-atmospheric maneuver penetration strategy based on generative adversarial reinforcement learning*  
> [Chinese Journal of Aeronautics, 38(4): 103200](https://doi.org/10.1016/j.cja.2024.08.031)  
> **My Goal**: Extend to full 3D motion, integrate my existing **Optimal Guidance Law** derivation, and refine state/action spaces.  
> **Status**: Sections 1–5 document the base paper faithfully; Sections 6–8 are placeholders for **my paper** (to be filled when manuscript is provided).

---

## 1. Problem Overview (Hui 2025)

### 1.1 Adversarial Setting (TAD Three-Body Game)

- **A**: Attack missile (HGV), **D**: Interceptor, **T**: Target.
- OODA loop: Observe (detection) → Orient (capability assessment) → Decide (maneuver timing/command) → Act (penetration + handover to mid-terminal guidance).
- Interceptor adopts **proportional guidance**; attack missile learns penetration maneuver online.

### 1.2 Optimal Control Formulation

**Performance index** (Eqs. 15–19):

$$
J = k_1 J_1 + k_2 J_2,\quad J_1 = -R_{AD},\quad J_2 = \|\mathbf{s}_{Af} - \mathbf{s}^*_{Af}\|_2 + (-V_{Af})
$$

| Term | Meaning |
| :--- | :--- |
| $J_1$ | Maximize miss distance to interceptor during penetration |
| $J_2$ | Minimize handover position error + maximize terminal velocity (energy retention) |

**Constraints**:
- **Initial**: $\mathbf{x}_{A0} = \mathbf{x}^*_{A0}$, $\mathbf{x}_{D0} = \mathbf{x}^*_{D0}$ (with Monte Carlo perturbations in data generation).
- **Terminal**: $\|\mathbf{s}_{Af} - \mathbf{s}^*_{Af}\|_2 \le d$; $V_{Af} \ge V^*_{Af}$.
- **Path** (Eq. 14): overload $\|\mathbf{n}\|_2 \le n_{\max}$, dynamic pressure $q_A \le q_{\max}$, heat flux $\dot{Q}_A \le \dot{Q}_{\max}$.

**MDP mapping** (Sec. 2.4): discretized dynamics $\mathbf{x}(k+1)=f(\mathbf{x}(k),\mathbf{u}(k))$ → Markov decision process $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$.

---

## 2. Base Paper: Vehicle & Interceptor Models

### 2.1 HGV Aerodynamics (Eqs. 1–2)

$$
C_L = \ell_0 + \ell_1\alpha + \ell_2\alpha^2 + \ell_3 Ma + \ell_4 e^{\ell_5 Ma},\quad
C_D = d_0 + d_1\alpha + d_2\alpha^2 + d_3 Ma + d_4 e^{d_5 Ma}
$$

Fitted coefficients: Tables 1–2 in paper. HGV: $L/D_{\max}=3.5$, $Ma_{\max}=13$, glide range 800–1000 km.

### 2.2 Attack Missile Kinematics (Eqs. 3–5)

Full 3D state in paper: $\mathbf{x}=[x,y,z,V,\theta,\psi]^T$, control $\mathbf{u}=[\alpha,\gamma]^T$ ($\alpha$: AOA, $\gamma$: inclination/bank-related angle).

$$
\begin{cases}
\dot{x} = V\cos\theta\cos\psi \\
\dot{y} = V\sin\theta \\
\dot{z} = V\cos\theta\sin\psi \\
\dot{V} = -\dfrac{D}{m} - g\sin\theta \\
\dot{\theta} = \dfrac{L - mg\cos\theta}{mV} \\
\dot{\psi} = -\dfrac{Z}{mV\cos\theta}
\end{cases}
\qquad
L=\tfrac{1}{2}\rho V^2 S C_L,\;
D=\tfrac{1}{2}\rho V^2 S C_D
$$

> **Important**: Paper explicitly states *"Lateral plane motion is not considered"* — simulation reduces to **longitudinal-plane (2D) penetration** even though Eq. (4) is 3D-capable. Pseudospectral & RL both use the 2D-reduced formulation.

### 2.3 Interceptor Model (Eqs. 6–10)

6-DOF with thrust $P$, sideslip $\beta_D$; guided by proportional navigation:

$$
\dot{\theta}_D = \frac{k_D\left[V_D\sin(\lambda_{AD}-\theta_D) - V\sin(\lambda_{AD}-\theta)\right]}{R_{AD}},\quad
R_{AD}=\|\mathbf{s}_A-\mathbf{s}_D\|_2
$$

### 2.4 Pseudospectral State Selection (Eq. 40)

For Radau discretization, paper selects:

$$
\mathbf{x}_A = [x,\, y,\, V,\, \theta,\, \alpha]^T,\quad
\mathbf{x}_D = [x_D,\, y_D,\, V_D,\, \theta_D,\, \alpha_D]^T
$$

Both attack and interceptor trajectories are solved **jointly** in the NLP (no separate numerical integration of interceptor path).

---

## 3. Expert Data: Radau Pseudospectral Method (Sec. 3.1–3.2)

### 3.1 Pipeline

```
Optimal control (Eq. 20)
    → time-domain map s ∈ [-1,1] per segment (Eqs. 28–29)
    → LGR collocation + Lagrange interpolation of x(s), u(s) (Eqs. 30–33)
    → defect constraints at nodes (Eq. 35)
    → boundary & path constraints (Eqs. 36–37)
    → segment continuity (Eq. 38)
    → NLP solve → trajectory library Γ = {τ^(1), …, τ^(l)}
```

Each expert trajectory: $\tau^{(l)} = \langle (\mathbf{s}_1,\mathbf{a}_1), \ldots, (\mathbf{s}_n,\mathbf{a}_n) \rangle$.

### 3.2 Random Error Model (Table 3)

| Parameter | HGV | Interceptor | Distribution |
| :--- | :--- | :--- | :--- |
| Detection distance (km) | $40\pm10$ | $20\pm2$ | Uniform |
| Blind spot (m) | 200–500 | 200–500 | Uniform |
| Angle / LOS rate / range / velocity error | Zero mean | Zero mean | Gaussian |
| 1st-order control delay (s) | 0.1–1 | 0.1–1 | Uniform |
| Initial position deviation (km) | $\pm3$ | $\pm2$ | Uniform |
| Initial velocity deviation (m/s) | $\pm600$ | $\pm300$ | Uniform |

**Sample count**: 1000 Monte Carlo runs → expert library for GAIL warm-start.

---

## 4. GAIL-DDPG Training Framework (Sec. 3.3)

### 4.1 Two-Stage Architecture (Fig. 3, Fig. 5)

| Stage | Algorithm | Purpose | Transfer |
| :--- | :--- | :--- | :--- |
| **I — Imitation** | GAIL (generator + discriminator $D(\mathbf{s},\mathbf{a};\phi)$) | Learn expert policy from pseudospectral trajectories | Generator $\pi(\cdot\|\mathbf{s};\theta)$ → Actor online network when converged or $N_{\mathrm{GAIL}}$ reached |
| **II — Fine-tune** | DDPG (Actor–Critic, experience replay) | Exploration beyond expert; improve generalization | Hot-start from GAIL weights + exploration noise |

**Networks**:
- **Actor**: online $\mu(\mathbf{s}\|\theta^\mu)$ + target $\mu'(\mathbf{s}\|\theta^{\mu'})$ (soft update, Eq. 57).
- **Critic**: online $Q(\mathbf{s},\mathbf{a}\|\theta^Q)$ + target $Q'(\mathbf{s},\mathbf{a}\|\theta^{Q'})$ (soft update, Eq. 56).
- **Discriminator**: MLP, input $(\mathbf{s},\mathbf{a})$, output ∈ [0,1] (closer to 1 ⇒ more expert-like).

### 4.2 Base Paper RL Definition (Eq. 44–46)

**State** (11-D):

$$
\mathbf{s} = \left[x,\, y,\, V,\, \theta,\, \alpha,\, n_{\max},\, x^*_{Af},\, y^*_{Af},\, x_D,\, y_D,\, V_D\right]^T
$$

**Action** (1-D): AOA $\alpha \in [-4^\circ,\, 12^\circ]$.

**Reward** (dense process reward to mitigate sparse terminal reward):

$$
R_1 = \frac{5}{0.5 + \|\mathbf{s}_{Af}-\mathbf{s}^*_{Af}\|_2^2},\quad
R_2 = 0.003\,(V_{Af}-2000),\quad
R_3 = \log_{10}(R_{AD}-149)
$$

$$
R = \begin{cases}
0 & \text{penetration failed} \\
R_1 + R_2 + R_3 & \text{otherwise}
\end{cases}
$$

### 4.3 Training Steps (Summary)

**GAIL phase** (Steps 1–6):
1. Sample expert trajectory $\tau_{\mathrm{real}}$ from library.
2. Roll out $\tau_{\mathrm{fake}}$ with current generator.
3. Discriminator reward $u_t = \ln D(\mathbf{s}^{\mathrm{fake}}_t, \mathbf{a}^{\mathrm{fake}}_t; \phi)$.
4. Update generator via **TRPO** (trust region, Eqs. 50–51).
5. Update discriminator via gradient ascent on Eq. (53).
6. Repeat until convergence or $N_{\mathrm{GAIL}}$ → assign $\theta \to \theta^\mu$.

**DDPG phase** (Steps 7–11):
7. Collect $(\mathbf{s}_t, \mathbf{a}_t, r_t, \mathbf{s}_{t+1})$ with exploration noise.
8. Critic loss: $L = \frac{1}{N}\sum_i (y_i - Q(\mathbf{s}_i,\mathbf{a}_i))^2$, $y_i = r_i + \gamma Q'(\mathbf{s}_{i+1}, \mu'(\mathbf{s}_{i+1}))$.
9. Actor: $\nabla_{\theta^\mu} J \approx \frac{1}{N}\sum_i \nabla_{\mathbf{a}} Q(\mathbf{s},\mathbf{a})|_{\mathbf{a}=\mu(\mathbf{s})}\nabla_{\theta^\mu}\mu(\mathbf{s})$.
10. Soft-update target networks ($\eta$: soft coefficient).
11. Repeat until convergence.

### 4.4 Reported Performance (Sec. 4)

| Metric | DDPG | GAIL-DDPG |
| :--- | :--- | :--- |
| Convergence episodes | ~800 | ~500 (GAIL ~200 + DDPG ~400) |
| Mean reward | ~19.5 | ~20.3 (+2.8%) |
| Training speedup | — | ~37.5% |

**Monte Carlo (1000 runs, Table 5)**:

| Strategy | Penetration (<300 m miss) | Handover error (<1 km) | $V_{Af}>2000$ m/s | All met |
| :--- | :--- | :--- | :--- | :--- |
| No maneuver | 0.7% | 0.1% | 100% | 0.1% |
| Random maneuver | 1.7% | 0% | 100% | 0% |
| DDPG | 94.6% | 91.4% | 92.3% | 79.8% |
| **GAIL-DDPG** | **96.2%** | **94.1%** | **94.5%** | **83.3%** |

**Simulation setup (Table 4)**: Interceptor mass 75 kg, HGV 500 kg; kill radius 150 m; handover center $(40, 28)$ km; typical detection 50 km (HGV) / 20 km (interceptor).

---

## 5. 按我论文落地的改造路线（最终版）

| 模块 | Hui 2025 基线 | 我论文改造后（本项目采用） |
| :--- | :--- | :--- |
| 对抗对象 | 单防御者 | **双防御者 + 目标四体对抗** |
| 模型维度 | 2D为主（虽给3D式） | **完整3D运动学/相对运动** |
| 理论主线 | 伪谱 + GAIL-DDPG | **先用我论文最优制导律形成可计算模型，再用伪谱生成专家轨迹，再用GAIL-DDPG替代BP在线参数拟合** |
| 专家数据来源 | Radau样本库 | **按我论文方程 + BP原输入状态随机分布，生成1000条轨迹** |
| 学习对象 | 直接学动作策略 | **学“状态→制导律参数”映射** $\pi(\mathbf{a}|\mathbf{s})$，$\mathbf{a}=[w,\gamma,w_T,w_E]$ |
| 状态定义 | 10维（文中Eq.44） | **沿用我论文BP输入态势定义（12维）** |
| 动作定义 | 1维AOA | **4维制导律参数** $\mathbf{a}=[w,\,\gamma,\,w_T,\,w_E]^T$；$d^*$ 为人为固定（如15 m），**不在动作空间** |
| 贡献定位 | 知识-数据协同训练 | **“最优制导律+伪谱专家库+GAIL-DDPG”统一框架** |

---

## 6. 我论文模型到伪谱专家库的映射

### 6.1 运动学与相对运动方程（按我论文）

- 采用我论文中的四体关系：攻击者A、目标T、防御者$D_1,D_2$。
- 直接使用文中方程组作为伪谱离散对象：
  - 攻防相对运动：式(1)
  - 攻目标相对运动：式(2)
  - 视线方向相对加速度：式(3)、式(4)
  - 飞行器运动方程：式(5)
  - 状态空间表达：式(11)、式(14)
  - 零控脱靶量及其导数：式(16)–(21)
  - 二次型性能指标：式(22)–(23)
  - 最优指令：式(47)

> 说明：您已指出原论文部分公式经改造后结果不再符合实际，后续工程实现以“当前有效的运动学方程与约束”为准。模板中不再绑定旧数值结果，只保留可复现流程。

### 6.2 专家样本生成规则（替换旧BP训练样本）

目标：构建 `1000` 条专家轨迹库 $\Gamma=\{\tau^{(1)},...,\tau^{(1000)}\}$，供GAIL预训练。

1. **采样空间沿用BP输入状态分布**（按您论文第4.1/5.1节）：
   - 初始中值距离 $r_{AD0}\in[6.5,13.5]$ km；
   - 距离差调节量 $\beta\in[-1.5,1.5]$ km，且
     $r_{AD10}=r_{AD0}+\beta,\; r_{AD20}=r_{AD0}-\beta$；
   - 视线角 $q_{yi0}\in[20^\circ,70^\circ]$；
   - 速度前置角 $\eta^A_{yi0},\eta^D_{yi0}\in[-20^\circ,20^\circ]$；
   - 速度前置角 $\eta^A_{zi0},\eta^D_{zi0}\in[-10^\circ,10^\circ]$；
   - 其余初态量按论文设定采样。
2. **采样分布策略**：
   - 除 $\eta^A_{yi0},\eta^A_{zi0}$ 使用正态分布外，其余参数使用均匀分布。
3. **离散求解方式**：
   - 用Radau伪谱法对上述状态空间系统进行离散并求解最优控制轨迹。
4. **输出格式**：
   - 每条轨迹存储为 $\tau=\langle(\mathbf{s}_1,\mathbf{a}_1),\ldots,(\mathbf{s}_n,\mathbf{a}_n)\rangle$，其中 $\mathbf{a}_k=[w,\gamma,w_T,w_E]^T$（伪谱最优解对应的制导律参数序列）。
   - 附终端指标：两防御者脱靶量、能耗、终端偏差。

---

## 7. 用GAIL-DDPG替代BP-LM的实施方案

### 7.1 替代关系（核心创新点）

- 原流程：`状态X -> BP代理模型 f(X,p) -> LM求参数 p=[d*,γ] -> 代入式(47)得指令`
- 新流程：`状态 s -> GAIL-DDPG 策略网络 π(a|s) -> 输出制导律参数 a=[w,γ,w_T,w_E] -> 代入式(47)得加速度指令`

即：**BP做参数拟合 + LM迭代求参** 被 **GAIL预训练 + DDPG微调** 统一替代。改造后 $d^*$ 不再由网络/LM 优化，而是任务前给定常数（如 $d^*=15\,\mathrm{m}$）。

### 7.1.1 动作空间（4维）

$$
\mathbf{a} = \left[w,\;\gamma,\;w_T,\;w_E\right]^T
$$

| 参数 | 含义 | 来源 |
| :--- | :--- | :--- |
| $w$ | 双防御者规避权重总和（$w_1+w_2=w$，见式(23)） | 式(22)(23) |
| $\gamma$ | 两防御者规避脱靶量权衡系数 | 式(23) |
| $w_T$ | 目标到达脱靶量权重 | 式(22) |
| $w_E$ | 控制能量权重 | 式(22) |

**不在动作空间**：$d^*$（期望规避脱靶量）——人为设定、全程不变，例如 $d^*=15\,\mathrm{m}$。原 BP-LM 中 $d^*$ 与 $\gamma$ 同为待优化参数；改造后仅 $\gamma$ 等四参数由策略网络输出。

**执行链路**：$\mathbf{s}_t \xrightarrow{\pi(\cdot|\mathbf{s}_t)} \mathbf{a}_t \xrightarrow{\text{式(47)}} \mathbf{u}_A(t)$（攻击者法向加速度指令）。

### 7.2 状态、动作、奖励定义（与我论文一致）

1. **状态空间 $s$（建议沿用BP输入12维）**
   - 由我论文中初始态势变量构成，核心包括：
     - 两枚防御者相对距离与差异项（含 $\beta$）；
     - 视线角与速度前置角组合（式(50)、式(51)相关变量）；
     - 必要的飞行状态量与约束余量（过载/剩余时间等）。
2. **动作空间 $\mathbf{a}$（4维制导律参数）**
   - $\mathbf{a}=[w,\,\gamma,\,w_T,\,w_E]^T$，经式(47)映射为攻击者加速度指令 $\mathbf{u}_A$。
   - $d^*$ 为固定任务常数（如15 m），不随时间变化，**不纳入** $\mathcal{A}$。
   - 建议动作边界（可与伪谱专家采样一致）：$w$、$w_T$、$w_E$ 取正值；$\gamma\in[-3,3]$（参考原 BP 训练采样）。
3. **奖励函数 $R$（建议多目标一致）**
   - 规避收益：增大 $d_{1,\min}, d_{2,\min}$；
   - 能耗惩罚：最小化机动能量积分项；
   - 终端任务约束：保持目标终端偏差小；
   - 失败惩罚：任一防御者安全距离违规给予大惩罚。

### 7.3 训练流程

1. 用第6节生成的1000条伪谱轨迹训练GAIL（判别器+生成器）。
2. 将生成器参数迁移到DDPG Actor进行热启动。
3. 在在线交互环境中用DDPG继续优化，提升泛化与鲁棒性。
4. 对比基线：纯PN、正弦机动、单防御者最优法、原BP-LM法、本方法（GAIL-DDPG）。

---

## 8. 实验与复现配置（按当前创新思路）

### 8.1 数据与网络配置建议

| 项 | 配置 |
| :--- | :--- |
| 专家数据规模 | 1000条伪谱轨迹（按第6.2节采样） |
| GAIL输入 | $(\mathbf{s},\mathbf{a})$，$\mathbf{s}$ 为12维态势，$\mathbf{a}=[w,\gamma,w_T,w_E]^T$ |
| Actor输出维 | **4**（制导律参数）；$d^*$ 固定，由环境/任务配置传入式(47) |
| 环境执行 | 每步用 $\mathbf{a}_t$ + 固定 $d^*$ 计算式(47)指令，再推进仿真 |
| DDPG初始化 | Actor由GAIL生成器权重热启动 |
| 经验回放 | 优先保留高质量近端规避样本，增强稀有高威胁场景学习 |

### 8.2 对比实验组

1. 纯PN；
2. 正弦机动；
3. 单防御者最优规避法；
4. 原BP-LM参数优化法；
5. **本文方法：伪谱 + GAIL-DDPG**。

### 8.3 评价指标

- 两防御者最小脱靶量：$d_{1,\min}, d_{2,\min}$；
- 单位质量机动能耗；
- 终端目标偏差（保持论文中的高精度要求）；
- 成功规避率与任务完成率（规避+到达目标）。

### 8.4 项目目录（2025-06 整理）

| 目录/文件 | 用途 |
| :--- | :--- |
| `expert_trajectory/` | 伪谱专家轨迹生成（阶段A/B、CasADi、批处理、质检脚本） |
| `expert_trajectory/setup_expert_path.m` | 一键添加本模块 + 项目根路径 |
| `evasion_helpers.m` | **共享**：数值版式47制导 + 动力学（TD3 环境与专家提取共用） |
| `main_changjing3_0421_LM.m` | 参考仿真（三阶段 PN→规避→PN） |
| `verify_rl_env.py` | Python RL 环境检查 |
| `samples_clean.mat` | 初态样本（项目根目录，专家生成与 TD3 共用） |

专家库归档路径（读取/质检用）：`E:\matlab\result_0628\reault_p3_1000\expert_lib_1000.mat`（P3 生成后手动移入）。批处理生成入口：

```matlab
addpath('expert_trajectory'); setup_expert_path();
generate_ps_expert_library(100, struct('run_stageB', true, 'save', true));
```

### 8.5 当前版本待办

- [x] 动作空间已定为 $\mathbf{a}=[w,\gamma,w_T,w_E]$，$d^*$ 固定（如15 m），经式(47)映射为指令；
- [ ] 给出12维状态在代码中的字段顺序，确保与训练数据一致；
- [ ] 固化1000条轨迹的数据格式（建议 `npz/mat`: `state`, `action`, `next_state`, `reward`, `done`）；
- [ ] 先做小样本冒烟训练（如100条）验证维度与奖励，再扩展到1000条正式训练。

---

## Appendix A — 我论文关键符号对照

| 符号 | 含义 | 来源 |
| :--- | :--- | :--- |
| $r_{ADi}, r_{AT}$ | 攻击者与第$i$防御者/目标距离 | 式(1)(2) |
| $q_{yi},q_{zi},q_\theta,q_\psi$ | 视线角变量 | 式(1)(2) |
| $\eta^A_{yi},\eta^D_{yi},\eta^A_{zi},\eta^D_{zi}$ | 速度前置角 | 式(50)(51) |
| $z_i(t), z_T(t)$ | 零控脱靶量 | 式(16)(20) |
| $w_1,w_2,w_E,w_T,\gamma$ | 制导权重（动作空间：$w,\gamma,w_T,w_E$） | 式(22)(23)(47) |
| $d^*$ | 期望规避脱靶量 | 式(22)(47) |
| $\mathbf{u}_A$ | 最优规避制导加速度指令（由 $\mathbf{a}$ + $d^*$ 经式(47)计算） | 式(47) |

## Appendix B — 与本项目直接相关的我论文锚点

| 锚点 | 用途 |
| :--- | :--- |
| 式(1)–(5) | 3D动力学/相对运动建模 |
| 式(11)(14) | 伪谱离散的状态空间形式 |
| 式(16)–(23) | 零控脱靶量与性能指标 |
| 式(47) | 最优规避制导指令表达 |
| 第4.1–5.1节 | BP输入状态与随机采样分布（迁移为伪谱专家数据采样规则） |
