# 伪谱专家轨迹改造规格（精简版）

> **目标**：改造 `main_jiaBP3_0421_duo.m` → Radau 伪谱法生成 **1000 条**专家轨迹 → GAIL 预训练。  
> **参考代码**：
> - 双防规避 + 样本循环：`main_jiaBP3_0421_duo.m`
> - **攻–目标相对运动 + 三阶段流程**：`main_changjing3_0421_LM.m`
> - 理论锚点：本人论文 `新论文_0625.pdf`、Hui 2025 伪谱框架

---

## 1. 改造流水线

```
初态采样 (1000组, 论文§5.1分布)
  → Radau伪谱 NLP（式22目标 + 式11/14缺陷约束 + 过载约束）
  → 最优轨迹 τ^(k)
  → 逐步提取 (s_t, a_t)
  → 保存 expert_lib_1000.mat
```

**GAIL 要学**：$\pi(\mathbf{a}|\mathbf{s})$，$\mathbf{a}=[w,\gamma,w_T,w_E]$，$d^*=15$ m 固定。

---

## 2. 代码分工（改造时直接对照）

| 功能 | 主参考文件 | 关键函数/逻辑 |
| :--- | :--- | :--- |
| 双防相对运动、最优规避指令 | `main_jiaBP3_0421_duo.m` | `computeLOS`, `computeMDDot`, `computeBC`, `calc_Phi12`, `computeOptimalCmd_duo` |
| **攻–目标相对运动** | **`main_changjing3_0421_LM.m`** | `computeMTDot`（式2）、`computeLOS(pos_M,T)` |
| 三阶段仿真（PN→规避→PN） | **`main_changjing3_0421_LM.m`** | `r_safe=8000` 触发；stage 1/2/3 主循环 |
| 初态重建、批量样本 | `main_jiaBP3_0421_duo.m` | 420–441 行几何重建 |
| 初态采样分布 | 论文 §5.1 + `samples_clean` | 见 §4 |

> `main_changjing3_0421_LM.m` 中 `computeOptimalCmd_duo` 仍为**双防耦合**（式34），规避前/后用 PN 打目标；**式47完整目标项 $w_T$ 尚未编码**，伪谱首版可沿用此结构。

---

## 3. 状态 / 动作定义

### 3.1 动作 $\mathbf{a}$（4维，GAIL输出）

$$\mathbf{a} = [w,\,\gamma,\,w_T,\,w_E]^T,\quad d^* = 15\ \mathrm{m}\ \text{（固定）}$$

| 参数 | 典型值/边界 | 代码变量 |
| :--- | :--- | :--- |
| $w$ | $10^4$ | `a` |
| $\gamma$ | $[-3,3]$ | `gamma` → `a1,a2` via sigmoid |
| $w_T$ | $>0$（首版可设0） | 待加 |
| $w_E$ | $1.44\times10^3$ | `wE` |

权重分配：$w_1 = w/(1+e^{-\gamma})$，$w_2 = w e^{-\gamma}/(1+e^{-\gamma})$。

### 3.2 状态 $\mathbf{s}$（12维初态，对齐 BP 输入）

论文：$\mathbf{X}=[\mathbf{X}_1,\mathbf{X}_2,r_{AD0},\beta]$，$\mathbf{X}_i=[q_{yi0},\eta^A_{yi0},\eta^A_{zi0},\eta^D_{yi0},\eta^D_{zi0}]$。

`samples_clean` 15列映射（伪谱采样用前12维角度 + 单独采 $r_{AD0}$）：

| 列 | 字段 |
| :---: | :--- |
| 1–4 | $\eta^A_{y10},\eta^A_{z10},\eta^A_{y20},\eta^A_{z20}$ |
| 5 | $q_{y10}$ |
| 6–9 | $\eta^D_{y10},\eta^D_{z10},\eta^D_{y20},\eta^D_{z20}$ |
| 10–11 | $d^*,\gamma$（**伪谱中为决策变量，不进状态**） |
| 12 | $\beta$ |
| 13–15 | $q_{z20},q_{z10},q_{y20}$ |

距离：$r_{AD10}=r_{AD0}+\beta$，$r_{AD20}=r_{AD0}-\beta$，$r_{AD0}\sim\mathcal{U}[6.5,13.5]$ km。

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
| $w, w_E$ | $10^4$, $1.44\times10^3$ |

**采样分布**：$q_{yi0}\in[20°,70°]$；$\eta^A_{yi0},\eta^D_{yi0}\in[-20°,20°]$；$\eta^A_{zi0},\eta^D_{zi0}\in[-10°,10°]$；$\beta\in[-1.5,1.5]$ km；$\eta^A_{yi0},\eta^A_{zi0}$ 正态，其余均匀。

---

## 5. 伪谱 NLP 要点

**决策变量（推荐）**：配点上的 $\{w,\gamma,w_T,w_E\}$ 或首版固定后仅优化 $u_A=[a_{yA},a_{zA}]$。

**目标函数**（式22离散）：
$$J \approx \sum_i \frac{w_i}{2}[z_i(t_{fi})-d^*]^2 + \frac{w_E}{2}\sum_k \|\mathbf{u}_{A,k}\|^2\Delta t_k + \frac{w_T}{2}z_T^2(t_{fT})$$

**约束**：Radau 缺陷（式5运动学 + 式1/2相对运动）；$\|a_A\|\le8g$，$\|a_D\|\le10g$；终端脱靶 $\ge 6$ m。

**配点**：$N=20\sim40$；输出可插值到 $\Delta t=0.01\sim0.05$ s。

**首版简化**：$w_T=0$，三阶段用 `main_changjing3_0421_LM.m` 逻辑，伪谱只优化规避段。

---

## 6. 新增文件（建议）

| 文件 | 作用 |
| :--- | :--- |
| `sample_initial_conditions.m` | 1000组初态采样 |
| `build_ps_nlp.m` / `solve_ps_trajectory.m` | 伪谱 NLP 构造与求解 |
| `extract_expert_transitions.m` | 提取 $(s,a)$ 序列 |
| `generate_ps_expert_library.m` | 批处理主入口 |

**输出** `expert_lib_1000.mat`：`trajectories(k).t/state/action/accel/meta`。

---

## 7. 实施阶段

| 阶段 | 内容 | 规模 |
| :--- | :--- | :--- |
| P0 | 合并两脚本函数；RK4 回放验证三阶段 | 10组 |
| P1 | 规避段伪谱 NLP（双防，$w_T=0$） | 100条 |
| P2 | 输出 $(s,a)$，$d^*=15$ 固定 | 100条 |
| P3 | 扩至1000条 + RK4质检 | 1000条 |
| P4 | （可选）加入 $w_T$ 目标耦合 | 1000条 |

---

## 8. 待你确认（改造前）

1. **伪谱求解器**：`fmincon` / IPOPT / GPOPS-II？
2. **$r_{AD0}$**：用论文分布 $[6.5,13.5]$ km，还是保持代码 8 km？
3. **轨迹范围**：仅规避段，还是 `main_changjing3_0421_LM.m` 全流程（PN+规避+PN）？
4. **失败轨迹**：丢弃补采 or 保留？
5. **`samples_clean` 生成脚本**能否提供？

---

*版本 v1.1 | 精简自 v1.0，攻–目标参考 `main_changjing3_0421_LM.m`*
