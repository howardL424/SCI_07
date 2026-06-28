# 上下文纪要：伪谱法专家轨迹生成（GAIL 预训练数据）

> 本文件供新对话直接续接工作。任务：改造 `main_jiaBP3_0421_duo.m`，用「式47 一体化最优制导 + 优化时变权重 θ(t)」生成 1000 条专家规避轨迹，动作=权重，供 GAIL 模仿学习。参考 `Pseudospectral_Expert_Trajectory_Spec.md`（主）与 `新论文_0625.pdf`（公式细节，4.2 节前）。

---

## 0. 核心设定（已与用户确认）

- **决策变量**：权重 **θ(t)=[γ, wT, wE] 时变**（除 w 外随时间变化）；**w 固定 = 1e4**；制导律内嵌的期望脱靶量 **d\* 固定 = 20**（即式47 里的 `r_star`）。
- **权重映射（式23）**：`a1 = w/(1+exp(-γ))`，`a2 = w*exp(-γ)/(1+exp(-γ))`，满足 `a1+a2=w`。a1/a2 是双防 D1/D2 的规避项权重。
- **目标函数 = 固定系数式22**（评分系数为常数，**不是**决策 θ，避免退化）：
  - `J = 0.5*s_w1*(miss1-d*)^2 + 0.5*s_w2*(miss2-d*)^2 + 0.5*s_wE*∫‖u_A‖^2 dt + 0.5*s_wT*zT^2`，其中 `s_w1=s_w2=s_w/2`。
  - **固定评分系数（已重标定，见 §3.1）：`s_w=2e3, s_wE=55, s_wT=22`**（旧值 `1e4/1.44e3/1e3` 致目标 ZEM 项压倒规避项 ~4 个量级，使 miss 贴下限、θ 贴边）。
  - `miss1,miss2` = 规避窗口内对 D1/D2 的最近距离（softmin，≈ z_i(t_fi)）；`zT` = 规避**末端**对目标的预测 ZEM。
- **轨迹范围**：全段 PN打目标 → 式47规避 → PN恢复打目标（三阶段）；引入攻-目标通道（选项B）。
- **场景几何**：坐标系 y 轴指天、x 轴大致指向目标；目标静止于原点。状态 `[x,y,z,theta,psi_v,V]`。

---

## 1. 已完成的代码与功能

### 1.1 数值层（阶段A warm-start）
| 文件 | 功能 | 状态 |
|---|---|---|
| `evasion_helpers.m` | 数值函数集：computeLOS / computeMDDot / computeMTDot / computeBC / computePN / updateState(RK4) / saturate / **computeOptimalCmd_duo_target（式47 数值版，3通道D1/D2/目标耦合）** | ✅ |
| `reconstruct_geometry.m` | 由 `samples_clean.mat` 一行(14列) 重建 M0,D10,D20 初态（`r_AD10=rAD0+beta`、`r_AD20=rAD0-beta`，目标与D1初始在原点等） | ✅ |
| `simulate_warmstart_trajectory.m` | 阶段A 三阶段 RK4 全段仿真：PN→**式47规避**→PN；过载饱和(攻8g/防10g)；早停(到达目标 reach 或持续远离 diverge) | ✅ |
| `solve_warmstart.m` | 对单样本按 wE 降序扫描 `[1440 700 400 250 150 100 60]`，可行性=「不被拦截 且 min双防脱靶≥d\*」（命中目标**不**作硬性要求）；优先选「规避+达标」中最大 wE，否则「规避」中最大 wE，否则评分最高 | ✅ |
| `generate_ps_expert_library.m` | 批处理主入口：载样本→重建→solve_warmstart→统计，含绘图/保存骨架 | ✅（阶段A 部分） |

### 1.2 符号层（CasADi，阶段B）
| 文件 | 功能 | 对拍 |
|---|---|---|
| `casadi_guidance_duo_target.m` | **式47 光滑无分支符号版**：ε正则 Q；可去奇点用级数(sinhc/c2func/d2func/expdiv，`if_else` 小参数切级数防相消)；`calc_Phi12` 用 expdiv 统一公式消 Q1-Q2 奇点；`t≤tf` 用 sigmoid；`Q*tgo` 用硬钳制 `clampval`(L=30,有效区恒等) 防 cosh/sinh/exp 溢出 | ✅ 与数值版 **8.7e-8** |
| `casadi_dynamics.m` | 规避段闭环符号动力学：式47(攻) + PN(双防) + 式5运动学；**含光滑过载饱和 `satf`**(攻8g/防10g)；返回 `[Xdot, aM, aD1, aD2]` | ✅ 与数值版 **1e-11** |
| `casadi_target_zem.m` | 给定攻弹态算对原点目标的 ZEM `zT`（式22 目标项用） | ✅ |
| `validate_guidance_sym.m` / `validate_dynamics_sym.m` | 数值↔符号对拍脚本 | ✅ |

### 1.3 阶段B 求解器
| 文件 | 方法 | 状态 |
|---|---|---|
| `build_ps_nlp.m` + `solve_ps_trajectory.m` | **Radau 伪谱配点**（决策=配点状态+分段θ） | ⚠️ **不收敛，已弃用**（18维刚性闭环态+缺陷约束太难，max-iter/Restoration_Failed） |
| `solve_ss_trajectory.m` | **单打靶（当前主求解器）**：决策仅 θ(t)（K=10 段, nsub=12, 共 3K=30 个），状态由符号动力学 **RK4 滚动**积分得到，无缺陷约束；softmin 求 miss、casadi_target_zem 求 zT；IPOPT + **L-BFGS**。**已重标定 J 系数**（`s_w=2e3/s_wE=55/s_wT=22`, `Jscale=1e5`, `tol=1e-3`）；新增 `cfg.report_terms_only`（warm θ 处只求值不求解，标定用）与 `res.terms` 分项分解；`max_iter/print_level/tol` 改 cfg 可调 | ✅ 典型条收敛，miss≈d*、wT/wE 内点（~25~30s/条） |
| `calibrate_J_weights.m` | **J 系数标定驱动**：跑阶段A 筛规避成功轨迹 → `report_terms_only` 实测三项原始量级中位数 → 按平衡式算 `s_w/s_wE/s_wT` → 对 1~2 条用新系数求解验证 θ 内点/miss≈20/耗时 | ✅ |

---

## 2. 关键架构与设计决策

1. **为何单打靶而非配点**：决策是「权重经式47闭环律」，控制非自由量；配点把18维刚性闭环态当变量+缺陷约束极难收敛（实测不行）。单打靶决策仅 3K 个、状态由积分保证动力学一致，鲁棒得多。**θ(t) 仍时变**（每区间一组，K 段分段常值），符合用户要求。
2. **目标当作「原点静止防御弹」**：论文式3/4 视线系 x 轴均定义为"被规避体→攻击者"，故目标通道用 `computeLOS(T,M)`（T→M）+ `computeMDDot(M, T_static, …)`，与双防通道复用同一套 ZEM/B 公式。
3. **目标项用「规避末端 ZEM」而非窗口内最近距离**：配点/滚动窗口=规避段(stage2)，此时攻弹离目标仍远，目标靠 stage3 PN 命中。故 `missT` 取规避末端对目标的预测 ZEM（越小→规避后越对准目标，利于 PN 收尾），不是窗口内 r_MT 最近值。
4. **stage3 仍用 PN**：试过 stage3 也用式47，但防御弹通过后式47目标项增益(~wT/wE)比规避项弱~10倍，homing 太弱、多发散；PN 才是可靠强目标 homing。
5. **过载饱和做进动力学**（`satf` 光滑保方向 ‖a‖≤amax），**不**作硬约束——式47本身会输出>8g，硬约束会与之冲突致 `Restoration_Failed`。
6. **固定系数 vs 决策权重**：J 的系数 `s_w/s_wE/s_wT` 是常数评分；θ 喂入式47 生成轨迹。二者分离避免"权重既当系数又当变量"的退化。
7. **J 三项必须数量级平衡**：三项（规避/能量/目标）量纲、量级天差地别（目标 ZEM 平方 ~1e6、能量 ~1e4、规避脱靶平方 ~1e2），若沿用论文标称系数会让目标项压倒一切 → miss 被压到下限、θ 贴边、对 GAIL 无区分度。**做法**：以一条典型 warm 轨迹实测各项原始量级，令加权后三项同数量级（规避项基准取"塌到下限"惩罚 `A_floor` 而非接近 0 的 warm 残差，避免系数发散），规避项给温和加成把 miss 钉在 d*。系数仍是**固定常数**（标定一次写死），不破坏式22 语义。

---

## 3. 待办 TODO 与下一步计划

### 3.1 已完成（本轮）：重标定 J 固定系数，消除贴边/退化
- [x] **新系数 `s_w=2e3, s_wE=55, s_wT=22`** 已写入 `solve_ss_trajectory.m` 默认（标定过程见 `calibrate_J_weights.m`；原理见 §2.7）。配套：`Jscale 1e6→1e5`、`tol 1e-4→1e-3`、`max_iter/print_level/tol` 可调、acceptable 放宽、新增 `report_terms_only` 与 `res.terms`。
- **效果（典型条 row1，ok=1, ~28s）**：`miss=[26.8, 16.6]` 接近 d*=20（不再压到下限 6）、wT/wE 落内点；三项加权 avoid 2.9e4 / energy 5.6e4 / target 2.3e5（同数量级，旧时差 ~4 个量级）。
- **遗留（已记录，未深挖）**：① **γ 仍 bang-bang**（分段在 ±3 间切换）——γ 经式47 闭环律对 miss 近单调、最优即 bang-bang 的**结构性**结果，**跨段/跨轨迹变化（非退化恒定贴边），对 GAIL 仍有信号**；如需严格内点可加轻量 γ 正则（会偏离纯式22）。② 个别几何（某防御弹本就贴极近、softmin miss1≈6 近活动约束）miss1 留在下限是**正确物理**。

### 3.2 下一步（阶段B 收尾 → 串通）
- [ ] **接批处理**：把 `generate_ps_expert_library.m` 的 `run_stageB` 分支接上 `solve_ss_trajectory`（阶段A `tau` → 抽 stage==2 段 → 求 θ*）；先跑小批量(5~10条)**检查 θ 跨轨迹区分度**（γ/wT/wE 是否随场景变化，而非全部同值）——这是 GAIL 数据有用性的关键验收。
- [ ] **提速 + 收敛鲁棒性**（1000 条前必须）：当前 L-BFGS **~25~30s/条**（500 iter 上限），1000 条约 7~8h，偏慢；个别硬几何 IPOPT 早停 `ok=0`（restoration/达上限）。思路：减小 `nsub`/`K`、把 RK4 滚动封 `casadi.Function` 复用减小建图、调 IPOPT 选项、对 `ok=0` 条降级/补采。

### 3.3 后续（数据产出）
- [ ] `extract_expert_transitions.m`：把最优轨迹插值到 Δt=0.01~0.05s，抽 (state, action) 对，**动作 a=[w,γ,wT,wE](t)**（w 常值1e4），state 定义见 spec（弹间/弹目相对量等）。
- [ ] 批处理扩到 P2(100)→P3(1000)：质检（规避成功+目标可达）、失败补采（新样本足够，见下），保存 `expert_lib.mat`。
- [ ] 确认最终 GAIL 的 state/action 张量格式（与 spec §"输出"对齐）。

---

## 4. 避坑指南 / 约束条件

### CasADi 语法（MATLAB 接口）
- 绝对值用 `abs`（不是 `fabs`）；向量求和用 `sum1`（不是 `sum`）；向量最小值用 `mmin`；分支用 `if_else`。
- CasADi 路径已在 MATLAB path 上：`E:\matlab\casadi-3.7.2-windows64-matlab2018b`（`import casadi.*` 即可，无需 addpath）。
- 构 `Function` 时多入参可用 `args=arrayfun(@(k)in(k),1:N,'UniformOutput',false); f(args{:})`。

### 数值稳定（已踩过的坑）
- **softmin 必须用 log-sum-exp 稳定化**：`m = mmin(v) - beta*log(sum1(exp(-(v-mmin(v))/beta)))`。直接 `-beta*log(sum(exp(-v/beta)))` 在距离大时下溢→`log(0)=-Inf`→梯度 NaN。softmin 是 min 的**下估计**（≤真实min），用作 `≥6` 约束是保守安全的。beta=3。
- **式47符号版必须钳制 `Q*tgo`**：坏迭代点 `r_dot→0` 使 `tgo=tf-t` 爆炸，`cosh/sinh/exp` 溢出→NaN。用**硬钳制** `clampval(x,L)=if_else(|x|<L, x, L*sign)`（有效区恒等，保对拍精度），L=30。**不要钳制 tgo 本身**（正常 tgo 5~15s 会被扭曲，曾导致对拍误差 1.3）。
- **过载必须在动力学内饱和**（`satf` 光滑），不能用硬约束 `‖aM‖≤8g`（与式47冲突→Restoration_Failed）。
- 单打靶 Hessian 用 **L-BFGS**（`hessian_approximation='limited-memory'`），决策仅 30 个；精确 Hessian 对长滚动图太慢。
- **J 三项别直接套论文标称系数**：量级差 ~4 个量级会让目标项独大、解退化（miss 贴下限、θ 贴边）。必须按一条 warm 轨迹实测后平衡（见 §2.7 / `calibrate_J_weights.m`）。`opti.to_function` 要求**先 `opti.solver(...)` 再调**，否则报 `solver_name_.empty()`。
- **IPOPT 收敛/标度**：J 量级变化后须同步调 `Jscale`（让 `J/Jscale~O(1)`）；`tol` 取 1e-3 足够（权重标签无需 1e-4），并放宽 `acceptable_tol/acceptable_iter` 让"可行且停滞"的解返回 `ok=1` 而非 `ok=0`。`s_w` 越大 miss 越贴 d* 但问题越刚、越易 max-iter/restoration。

### 物理/建模
- **阶段A 现状**：100% 规避成功（脱靶≥d\*、0拦截）；约 **55% 能命中目标**，约 45% 是「规避成功但偏离目标」——**式47无论权重都无法同时充分规避(≥20m)又命中目标**（开局即规避的几何固有矛盾），属正常，stage3 PN 负责收尾命中。用户选**选项A**：warm-start 只要求规避，命中交给 stage B/stage3。
- **样本**：用户已替换 `samples_clean.mat`，有效轨迹 **>1000 条**（与阶段A早期样本不同，但阶段B按单条处理，无结构影响）。14 列结构见 `reconstruct_geometry.m`。若最终需更多可调大 `lhsyangbenshengcheng_beta_0.m` 的 N 重生成。
- 运行方式：`matlab -batch "..."`。控制台中文乱码是编码问题，数值正常。

### 关键调用关系
```
samples_clean.mat → reconstruct_geometry → solve_warmstart(扫wE) → tau(warm,含stage标签)
                                                                      │ 抽 stage==2 段 (x0,Tev)
                                                                      ▼
                                          solve_ss_trajectory → [symbolic RK4 rollout: casadi_dynamics(含式47 casadi_guidance_duo_target + satf)]
                                                              → softmin miss + casadi_target_zem → IPOPT(L-BFGS) → θ*(t)
```

### 接续点（新对话从这里开始）
- **状态**：阶段A + 阶段B 单条求解器（`solve_ss_trajectory`）均可用；J 系数刚重标定完，典型条 miss≈d*、wT/wE 内点、三项同数量级（详见 §3.1）。
- **可直接复现**：`out = calibrate_J_weights(6, true)`（跑阶段A + 标定 + 验证 1~2 条；约 3 分钟，结果可 `save`）。单条求解：`r = solve_ss_trajectory(tau)`（`tau` 取自 `solve_warmstart`，须含 `stage==2` 段）。
- **第一件事**：执行 §3.2「接批处理」——把 `solve_ss_trajectory` 接入 `generate_ps_expert_library` 的 `run_stageB` 分支，小批量(5~10条)跑通并**检查 θ 跨轨迹区分度**；随后处理提速/鲁棒性，再做 `extract_expert_transitions` 抽 (s,a)。
- 注：根目录 `calib_out.mat` 是本轮标定的缓存（含若干 warm `tau`），可用于快速复跑单条验证，非最终产物。
