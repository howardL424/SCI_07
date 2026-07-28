# Hui et al. (2025) CJA — GAIL-DDPG 大气层内机动突防策略（Agent 参考摘要）

> **用途**：供本仓库后续 agent 快速理解该文方法、状态/动作/奖励设定与结论，避免重复通读 PDF。  
> **原文**：Chinese Journal of Aeronautics, 2025, 38(4): 103200  
> **DOI**：https://doi.org/10.1016/j.cja.2024.08.031  
> **本地 PDF**：`c:\Users\LTY\Zotero\storage\EFGSMV3R\Hui 等 - 2025 - Endo-atmospheric maneuver penetration strategy based on generative adversarial reinforcement learnin.pdf`  
> **抽取说明**：关键公式（状态/动作/奖励、Table 3–5）已对照原文 PDF 核对；其余细节仍以原文图/式为准。

---

## 1. 文献元数据

| 项 | 内容 |
|---|---|
| 标题 | Endo-atmospheric maneuver penetration strategy based on generative adversarial reinforcement learning |
| 作者 | Yaoluo HUI*, Xiumin LI, Chen LIANG, Junzheng SUN, Zheng DU |
| 单位 | 北京机电工程系统总体部；北京控制与电子技术研究所 |
| 通讯 | huiyaoluo91@163.com |
| 收稿/修回/录用/上线 | 2024-03-06 / 2024-04-15 / 2024-06-02 / 2024-08-30 |
| 关键词 | Hypersonic glide vehicle; Endo-atmospheric penetration strategy; Deep reinforcement learning; Guidance; **GAIL-DDPG** |
| 开放获取 | CC BY-NC-ND 4.0 |

---

## 2. 一句话结论

用 **Radau 伪谱法离线生成带随机扰动的“专家”突防轨迹库 → GAIL 热启动策略 → DDPG 继续探索优化**，在稀疏奖励的大气层内滑翔突防对抗中，比纯 DDPG **更快收敛、略优成功指标**，并学到可解释的机动样式（跃升 / 先诱骗下压再拉起）。

---

## 3. 问题设定（与本仓库相关）

### 3.1 对抗场景

- **三体几何**：攻击弹 A（HGV）、拦截弹 D、目标 T（文中将末端命中需求等价为 **中末制导交班** 位置/速度约束）。
- 单枚攻击弹 vs 单枚拦截弹；过程按 **OODA** 描述：观测相对运动 → 判断是否需对抗 → 决定机动时机/指令 → 执行并进入中末交班区。
- 拦截弹采用 **比例导引** 追击；攻击弹根据拦截弹状态生成突防指令。
- 本质：**Target–Attack–Defend (TAD)** 型对抗，转化为 **最优控制 / MDP**。

### 3.2 飞行器与气动

- 研究对象：**HGV**，最大升阻比约 3.5，最大速度约 13 Ma，滑翔段射程约 800–1000 km。
- 升/阻力系数用关于 **攻角 α、马赫 Ma** 的多项式拟合（多元非线性最小二乘辨识），系数见表 1–2（原文）。
- 拦截弹升阻模型引用文献 [20]（Jiang 等，Aerospace 2022）。

### 3.3 约束与性能指标

**终端（中末交班）约束（概念）**

- 位置：\(\|s_{Af} - s_{Af}^*\|_2 \le d\)
- 速度：\(V_{Af} \ge V_{Af}^*\)（保留末端能量）

**过程约束**

- 过载、动压、热流密度上限（原文 Eq.(14) 三项最大值依次为：\(n_{\max}=30\)、\(q_{\max}=500\,\mathrm{kPa}\)、\(\dot{Q}_{\max}=6000\,\mathrm{kW/m^2}\)；PDF 行文易把 “30, 500 kPa” 误读成 30500）。

**优化指标（最优控制层）**

- 突防：\(J_1 = -R_{AD}\)（拉大 A–D 距离）
- 精确打击等价：\(J_{21}=\|s_{Af}-s_{Af}^*\|_2\)，\(J_{22}=-V_{Af}\)，\(J_2=J_{21}+J_{22}\)（原文 Eq.(16)–(18)；\(J_{21}\) **无平方**）
- 综合：\(J = k_1 J_1 + k_2 J_2\)

**与 RL 的桥接（文中论证）**

- 离散动力学 \(x(k+1)=f[x(k),u(k)]\) 满足马尔可夫性 → 可用 \(\langle S,A,T,R,\gamma\rangle\) 求策略。

---

## 4. 方法流水线（GAIL-DDPG）

整体见图 3（原文）：

```
最优控制建模
    → Radau 伪谱法离线求解（含初始偏差/传感/控制延迟等随机误差）
    → 生成专家轨迹样本库（约 1000 条）
    → GAIL：生成器模仿专家 + 判别器区分真伪（TRPO 更新生成器）
    → 将生成器参数热启动赋给 DDPG Actor
    → DDPG Actor–Critic + 探索噪声 继续优化
    → 在线突防策略（部署时通常只用策略网络）
```

### 4.1 Radau 伪谱专家数据与扰动处理

- 将连续最优控制在 LGR 配点上离散 → NLP。
- 攻击/拦截相关状态取：  
  \(x_A=[x,y,V,\theta,\alpha]\)，\(x_D=[x_D,y_D,V_D,\theta_D,\alpha_D]\)（符号以原文为准）。

**扰动加在哪、加什么**

- 扰动主要加在 **专家轨迹库生成阶段**（§3.2）：对每组 Monte Carlo 抽样条件，用 Radau **离线重解** 一条最优突防轨迹，再汇总成库；不是训练时再单独做滤波/补偿模块。
- 文中明确覆盖三类过程误差（Table 3）：
  1. **探测/导引头（或第三方制导）误差**：探测距离、盲区、测角、视线角速度、测距、测速；
  2. **姿控一阶延迟**（攻防双方均采样）；
  3. **初始位置/速度偏差**（双方不同量级）。
- 另在贡献点中强调：同时纳入攻防双方 **探测–制导–控制** 链路误差，并保留突防过程中的现实约束后再离线寻优。

| 参数 | HGV | Interceptor | 分布 |
|---|---|---|---|
| 探测距离 (km) | 40±10 | 20±2 | Uniform |
| 盲区 (m) | 200–500 | 200–500 | Uniform |
| 测角/视线角速度/测距/测速误差 | 零均值 | 零均值 | Gaussian |
| 一阶延迟 | 0.1–1 | 0.1–1 | Uniform |
| 初始位置偏差 (km) | ±3 | ±2 | Uniform |
| 初始速度偏差 (m/s) | ±600 | ±300 | Uniform |

**如何“处理”扰动（方法层面）**

- **样本层**：约 **1000** 次带扰仿真 → 轨迹库 \(\nu=\{f^{(l)}\}\)，每条为状态–动作序列；扰动通过多样化专家分布进入 GAIL，使模仿策略在传感/初始不确定性下仍可用。
- **学习层**：GAIL 先拟合该带扰专家分布；随后 DDPG 再加 **探索噪声** 做热启动后精调，弥补纯模仿对分布外扰动的不足（文中对比基线即体现这一点）。
- **评估层**：Table 5 的 1000 次 MC 在同一套随机对抗条件下对比策略，检验带扰专家热启动后的泛化，而非仅看无扰标称弹道。

### 4.2 GAIL-DDPG 网络结构（概念）

| 模块 | 作用 |
|---|---|
| Discriminator \(D(s,a;\phi)\) | 判别 \((s,a)\) 是否像专家；接近 1 表示更像专家 |
| Generator / Actor \(\pi(\cdot\|s;\theta)\) | 策略；GAIL 阶段用 TRPO，后转 DDPG online policy |
| Critic \(Q(s,a)\) | 评估动作价值；训练后可丢弃，推理只用 Actor |
| Target nets | soft update |

**两阶段动机（文中强调）**

1. **GAIL**：用专家库加速收敛、缩短训练。  
2. **DDPG**：在热启动后加探索噪声，提升泛化与最终性能（避免纯模仿上限）。

---

## 5. MDP 定义（最重要，供对齐本仓库）

### 5.1 状态空间

假设攻击弹带主动雷达，拦截弹进入探测视场后可获其位置/速度信息，状态构造为：

\[
s = \big[x,\, y,\, V,\, \theta,\, \alpha,\, n_{\max},\, x_{Af}^*,\, y_{Af}^*,\, x_D,\, y_D,\, V_D\big]
\]

（原文 Eq.(44)；\(\theta\) 为弹道倾角类量，\(\alpha\) 为攻角。）

### 5.2 动作空间

- **连续动作**：攻击弹 **攻角 AOA**  
- 取值范围：**\([-4^\circ,\, 12^\circ]\)**

### 5.3 奖励函数（稀疏 + 过程激励）

三部分过程/终端奖励；**突防失败则总奖励为 0**（原文 Eq.(45)–(46)）：

\[
\begin{aligned}
R_1 &= \frac{5}{0.5 + \|s_{Af}-s_{Af}^*\|_2}
\quad\text{（交班位置：误差越大奖励越小）}\\
R_2 &= 0.003\,(V_{Af}-2000)
\quad\text{（交班速度，鼓励 \(>2000\,\mathrm{m/s}\)）}\\
R_3 &= \log_{10}(R_{AD}-149)
\quad\text{（与拦截脱靶相关；杀伤半径 150 m）}
\end{aligned}
\]

\[
R =
\begin{cases}
0, & \text{突防失败}\\
R_1+R_2+R_3, & \text{否则}
\end{cases}
\]

**设计意图**：缓解纯终端稀疏奖励；同时兼顾突防脱靶、交班精度与剩余速度（能量）。


### 5.4 训练步骤（摘要）

**GAIL 阶段（Steps 1–6）**

1. 从专家集均匀采样真实轨迹 \(f_{real}\)  
2. 当前策略与环境交互得 \(f_{fake}\)  
3. 判别器给伪轨迹逐步打分 \(u_t=\ln D(s_t^{fake},a_t^{fake};\phi)\)  
4. **TRPO** 更新生成器（带 KL/距离约束）  
5. 用 \(f_{real},f_{fake}\) **梯度下降**更新判别器（标准 GAIL 形式的 \(F\)）  
6. 收敛或达 \(N_{GAIL}\) 次后，**把生成器参数赋给 DDPG Actor**

**DDPG 阶段（Steps 7–11）**

7. 加探索噪声生成动作，存 \((s_t,a_t,r,s_{t+1})\) 到 replay  
8. 采样 batch，最小化 Critic TD 损失  
9. 策略梯度更新 Actor  
10. soft update 目标网络  
11. 直至收敛或最大迭代

文中训练曲线描述：约 **episode 200** 从 GAIL 切到 DDPG；GAIL-DDPG 约 **500 episodes** 收敛，纯 DDPG 约 **800**；声称训练速度约 **+37.5%**，最终平均回报约 **+2.8%**（相对 DDPG）。

---

## 6. 仿真条件与结果

### 6.1 平台与参数（Table 4）

| 参数 | Interceptor | Attack |
|---|---|---|
| Mass (kg) | 75 | 500 |
| Reference area (m²) | 0.3 | 0.579 |
| Min speed (m/s) | 400 | 1000 |
| AOA (°) | −20–20 | **−4–12** |
| Typical detection range (km) | 20 | 50 |
| Detector off-axis (°) | 45 | 10 |
| Attitude control time constant | 0.1 | 1 |
| Initial position x,y (km) | 50, 27 | 0, 25 |
| Initial velocity (m/s) | 2000 | 4000 |
| Initial path angle (°) | 0 | 0 |
| Mid-terminal handover center (km) | — | **(40, 28)** |
| Interceptor kill radius (m) | **150** | — |

硬件：AMD Ryzen 5 3600 @4.2 GHz，8 GB×2 DDR4。

### 6.2 蒙特卡洛对比（Table 5，各 1000 次）

判定口径（文中）：

- 突防成功：脱靶距离 **>300 m**（相对拦截）  
- 交班位置误差 **<1 km**  
- 剩余速度 **>2000 m/s**  
- “All conditions meet”：上述同时满足

| 方法 | 突防成功率 % | 交班误差达标 % | 剩余速度达标 % | 全条件满足 % |
|---|---:|---:|---:|---:|
| No maneuver | 0.7 | 0.1 | 100 | 0.1 |
| Random maneuver | 1.7 | 0 | 100 | 0 |
| DDPG | 94.6 | 91.4 | 92.3 | 79.8 |
| **GAIL-DDPG** | **96.2** | **94.1** | **94.5** | **83.3** |

### 6.3 学到的典型机动（定性）

1. **平飞后突然跃升突防**：拦截弹大过载饱和，脱靶约 **480 m**；攻击弹交班速度约 **3500 m/s**。  
2. **先下压诱骗再拉起**：先降高诱使拦截跟随，再反向拉起指向交班点；拦截能量/过载受限，脱靶约 **200 m**。

文中称在非优势空域仍能灵活用机动，具有一定“智能涌现”。

---

## 7. 作者结论（三条）

1. 相对传统纯 RL：**微 GAIL 提供专家样本热启动**，再叠加探索，训练更短、性能更好。  
2. **过程奖励**缓解长程稀疏奖励，并在突防与中末交班精度约束下尽量保能量。  
3. GAIL-DDPG 能适应多种对抗态势，在非占优条件下仍可机动突防。

---

## 8. 对本仓库（GAIL / TD3 / 突防）的可迁移要点

| 维度 | 本文做法 | 对本仓库可能的启示 |
|---|---|---|
| 专家数据来源 | **Radau 伪谱最优控制** + Monte Carlo 误差，非人工示教 | 专家集质量/扰动建模直接影响 GAIL 上限 |
| 算法组合 | **GAIL(TRPO) → DDPG 热启动** | 与本仓 **BC/GAIL + TD3** 同属“模仿热启动 + off-policy 精调”范式 |
| 动作 | 单通道连续 **AOA** | 本仓若用过载/倾侧等，维度不同但“连续控制 + 稀疏突防奖励”同构 |
| 奖励 | 失败置零 + \(R_1=5/(0.5+\|\cdot\|_2)\) + 速度 + \(\log\) 脱靶 | 可对照本仓稀疏奖励与过程 shaping；\(R_1\) 为误差倒数奖励 |
| 成功判据 | 脱靶、交班误差、剩余速度 **多目标同时满足** | 评估不要只看单一 return |
| 基线 | 无机动 / 随机机动 / 纯 DDPG | 证明“纯随机探索难、专家热启动有用” |
| 局限（推断） | 单对单、拦截策略固定为比例导引；GAIL 阶段用 TRPO 而非现代 off-policy IL | 换拦截策略/多智能体时需重做专家库与状态定义 |

### 8.1 与本仓库关键词对齐

- 文中算法名：**GAIL-DDPG**（不是 GAIL-TD3）。  
- 专家生成：**伪谱法最优控制**，不是 BC 回归拟合演示数据 alone。  
- 判别器奖励形式：标准 GAIL \(\ln D\) / 真伪轨迹对抗。  
- 工程卖点：**收敛速度 + 全条件成功率**，而非仅最终平均回报。

---

## 9. 关键参考文献（文中直接相关）

- [20] Jiang L, et al. Anti-interception guidance for HGV: DRL. *Aerospace* 2022.（文中拦截模型与 RBT-DDPG 相关）  
- [22] Jiang L, et al. Midcourse penetration with DRL. *IEEE Access* 2021.  
- [24] Wang XF, Gu KR. 深度强化学习与模仿学习结合的突防策略. *宇航学报* 2023.（同方向：OC + GAIL 类）  
- [25] Gong XP, et al. 三体对抗与 DRL. *战术导弹技术* 2022.  
- [26] Xue H, et al. HGV 再入气动系数拟合. *宇航学报* 2021.

---

## 10. 原文结构速查

| 节 | 内容 |
|---|---|
| 1 Introduction | 突防规划 vs 机动对抗；纯 DRL 慢收敛 → 知识+数据协作 |
| 2 Problem | HGV/拦截模型；约束；J；MDP 桥接 |
| 3 Method | Radau；样本；GAIL-DDPG 网络与训练；奖励 |
| 4 Simulation | Table 4–5；训练曲线；两例机动分析 |
| 5 Conclusions | 三条结论 |

---

## 11. 引用建议（给后续文档）

```text
Hui Y, Li X, Liang C, Sun J, Du Z. Endo-atmospheric maneuver penetration
strategy based on generative adversarial reinforcement learning.
Chinese Journal of Aeronautics, 2025, 38(4): 103200.
doi:10.1016/j.cja.2024.08.031
```

**Agent 使用建议**：实现或对比实验前，优先核对原文 **Eq.(44)–(46)、Fig.5–7、Table 3–5**；本摘要不替代公式级复现。
