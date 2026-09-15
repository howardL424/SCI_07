# 参数辨识代码改写会话总结（2026-09-14）

> 2026-09-15 目录整理：专属辨识代码与测试产物已迁至 [`ident/`](../ident/README.md)。
> 下文旧路径仍可通过兼容入口访问；新工作请直接使用 `ident/`。

## 1. 文档用途

本文件用于新对话恢复参数辨识工作的上下文。继续工作前，应优先阅读：

1. 本文件；
2. [`论文文稿/overleaf_SCI_07_0914/sections/02_problem.tex`](../论文文稿/overleaf_SCI_07_0914/sections/02_problem.tex)；
3. [`论文文稿/overleaf_SCI_07_0914/sections/03_identification.tex`](../论文文稿/overleaf_SCI_07_0914/sections/03_identification.tex)；
4. [`论文文稿/overleaf_SCI_07_0914/sections/04_guidance.tex`](../论文文稿/overleaf_SCI_07_0914/sections/04_guidance.tex)；
5. [`rl_td3/matlab/stage2_ident_init.m`](../rl_td3/matlab/stage2_ident_init.m)；
6. [`rl_td3/matlab/stage2_ident_step.m`](../rl_td3/matlab/stage2_ident_step.m)；
7. [`rl_td3/matlab/stage2_step.m`](../rl_td3/matlab/stage2_step.m)。

当前范围仍是“纯 TD3 + 在线参数辨识 + 含噪量测”。暂不处理专家轨迹、BC-TD3 和第6节正式实验方案。

## 2. 已明确的理论与时序口径

三通道分别辨识：

- D1：`[A21, B1, B2, C1]`；
- D2：`[A21, B1, B2, C1]`；
- T：`[A21, B1, B2]`。

共11个参数。参数不是整局不变，也不是在一个制导周期内连续更新，而是逐拍时变、周期内零阶保持：

\[
\vartheta(t)=\vartheta[k],\qquad t\in[t_k,t_{k+1}).
\]

因此，第 \(k\) 拍利用区间 \([t_{k-1},t_k)\) 数据构造的因果回归应为：

\[
y(k)=\phi^\mathrm{T}(k-1)\vartheta[k-1]+e(k).
\]

辨识器输入是带噪量测，不是真值：

- KF 输入为 \(\tilde r,\tilde{\dot r}\)；
- RLS 输入为 KF 后验量和上一周期实际施加的饱和加速度；
- 无噪声参数真值只用于诊断、RMSE 和影子对比，不进入辨识更新。

四体非线性运动学仍由 RK4 连续积分。参数“冻结”是第4节局部线性回归模型的拟冻结假设，不是把物理植物替换成线性状态空间模型。非线性植物与周期拟冻结模型之间的差异记入 `frozen_model_residual`。

需要区分两种误差：

- `theta_err_causal = theta_hat[k] - vartheta[k-1]`：评价第 \(k\) 拍对已发生周期的因果拟合/跟踪；
- `theta_err_command = theta_hat[k] - vartheta[k]`：评价估计值用于下一周期制导时的一拍滞后。

真实闭环参数逐拍变化，因此只能谈“跟踪性能”，不能把其阈值通过率称为经典静态参数“收敛率”。静态收敛性必须由固定参数、带噪、持续激励的独立测试验证。

## 3. 本次对话开始时发现的主要实现偏差

1. 软阈值错误地作用于整个参数向量，包括本应不收缩的 `A21`。这会将小量级 `A21` 反复压到零。
2. 式(52)/(54)要求回归量使用上一拍 KF 后验距离，代码原先使用上一拍原始含噪距离。
3. 论文式(62d)要求 `R_xi = diag(sigma_r^2, sigma_rdot^2)`，代码原先将实际量测噪声设为零，另设 `kf_sigma_r/kf_sigma_rdot` 作为设计协方差。
4. KF 的第三状态初值原先为零，没有使用解析得到的初始相对加速度。
5. 非真值辨识路径的 `tgo/tf` 仍由真值 `r/rdot` 计算，存在真值泄漏。
6. 参数诊断将 `theta_hat[k]` 与当前几何 `vartheta[k]` 直接比较，没有按因果回归对齐 `vartheta[k-1]`。
7. 旧实现没有 RLS 协方差防护，弱激励方向可能因遗忘因子而膨胀。

## 4. 已完成的代码改动：阶段0

阶段0统一了量测噪声与 Kalman 协方差口径。

涉及文件：

- [`rl_td3/ident_cfg.py`](../rl_td3/ident_cfg.py)
- [`rl_td3/matlab/stage2_core_init.m`](../rl_td3/matlab/stage2_core_init.m)
- [`rl_td3/matlab/stage2_ident_init.m`](../rl_td3/matlab/stage2_ident_init.m)
- [`rl_td3/matlab/env_step.m`](../rl_td3/matlab/env_step.m)
- [`plot_table2_matlab/table2_noisy_ident/noisy_ident_cfg.m`](../plot_table2_matlab/table2_noisy_ident/noisy_ident_cfg.m)
- 纯 TD3 的影子诊断、消融和超参扫描入口。

当前默认值：

```text
sigma_r          = 10.0 m
sigma_rdot       = 0.5 m/s
kf_qj            = 1.0e4
kf_sigma_a0      = 50.0 m/s^2
```

`R_xi` 现在直接由实际量测噪声构成：

```matlab
Rxi = diag([ident.sigma_r^2, ident.sigma_rdot^2]);
```

已删除运行配置中的 `kf_sigma_r` 和 `kf_sigma_rdot`。原超参扫描脚本已改成直接扫描 `sigma_r/sigma_rdot`。

当前 `stage2_ident_init` 要求 `sigma_r > 0` 且 `sigma_rdot > 0`。因此旧测试中使用严格零距离噪声的配置需要在阶段2重写。

## 5. 已完成的代码改动：阶段1

### 5.1 软阈值不再收缩 A21

[`rl_td3/matlab/stage2_ident_step.m`](../rl_td3/matlab/stage2_ident_step.m) 中的软阈值现在只处理第2个及后续分量：

```matlab
th_sc = local_soft_threshold(th_sc, eta, 2);
```

`A21` 只进行非负投影，不参与 L1 收缩。

### 5.2 回归量改用上一拍 KF 后验距离

回归向量首项由 `r_meas_prev` 改为 `r_hat_prev`，对应论文式(52)/(54)。

### 5.3 KF 相对加速度初值改为解析初始化

首拍使用 noisy analytic warm-start 参数与初始回归量计算 `rddot0`，不再直接置零。NCV 的 `C1` 初始化继续通过已知 PN 模型重建 NCV 加速度，不读取仿真真值加速度。

### 5.4 增加逐拍参数真值缓存

每拍由当前几何得到 `theta_true_current = vartheta[k]`，在拍末缓存。下一拍 RLS 更新和因果误差使用缓存的 `theta_true_causal = vartheta[k-1]`。

新增诊断：

```text
theta_true_causal
theta_true_current
theta_err_causal
theta_err_command
qdot_err_causal
qdot_err_command
frozen_model_residual
```

MATLAB `info_vec` 由68维扩展为85维，Python 解析和影子诊断脚本已同步。旧字段 `theta_true` 和 `theta_err` 暂时保留兼容，但其语义已明确为因果真值和因果误差。

### 5.5 非真值路径使用估计状态计算 tgo

[`rl_td3/matlab/stage2_step.m`](../rl_td3/matlab/stage2_step.m) 中：

- `analytic_truth` 仍使用真值时间轴；
- `analytic_noisy` 使用 noisy `r/rdot`；
- `sparse_ffrls` 使用 KF 后验 `r_hat/rdot_hat`；
- 影子估计指令也使用辨识路径自己的 `tgo/tf`。

### 5.6 诊断与导出同步

已同步：

- [`rl_td3/evasion_env.py`](../rl_td3/evasion_env.py)
- [`td3_baseline/run_ident_shadow_diag_pure.py`](../td3_baseline/run_ident_shadow_diag_pure.py)
- [`td3_baseline/export_ident_params.py`](../td3_baseline/export_ident_params.py)
- [`plot_table2_matlab/table2_noisy_ident/simulate_table2_noisy_episode.m`](../plot_table2_matlab/table2_noisy_ident/simulate_table2_noisy_episode.m)
- [`rl_td3/matlab/tmp_ident_probe.m`](../rl_td3/matlab/tmp_ident_probe.m)

影子诊断现在分别保存因果真值、当前真值、P 最大特征值/条件数序列和冻结模型残差。

## 6. 阶段3：20 IC 带噪影子诊断

使用现有纯 TD3 checkpoint，不重训：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model
```

无协方差迹限制的结果目录：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/
  ident_shadow_stage3_N20_20260914/
```

关键结果：

- 20局中8局 `theta_norm_rmse_tail > 1e6`；
- 其中5局超过 `1e8`；
- 中位因果归一化参数误差为 `6.49e4`；
- 参数估计绝对值最大达到 `1.32e12`，而20局真值最大约为 `2.46e3`；
- 中位 `Qdot` RMSE 为 `9.40`；
- 中位饱和后指令 RMSE 为 `43.0 m/s^2`；
- 冻结模型残差中位 RMSE 为 `359.5 m/s^2`；
- 部分接近100步的场景冻结残差很小，但一些约75–90步的快速交会场景达到数百至上千 `m/s^2`。

参数发散通常首次出现在第9–28步。发散时：

```text
P_max 约为 1.2e3 到 1.3e4
P_cond 约为 4.8e4 到 3.4e5
```

末段 `P_max` 已经下降到几十量级，因此只看末段 P 会错误地认为协方差没有参与早期发散。

阶段3结论：

1. 修复 `A21` 软阈值和因果下标后，原有数值发散仍存在；
2. 主要爆炸发生在早期高协方差、病态回归阶段；
3. 单周期拟冻结误差在部分场景本身很大，不能只靠调 RLS 超参解释或掩盖；
4. 此结果仍不代表固定参数条件下 FF-RLS 是否收敛，因为阶段2的固定参数测试尚未补做。

## 7. 阶段4：协方差迹投影

已新增配置：

```text
rls_P_trace_max
```

缩放域协方差在初始化和每次式(59)更新后执行：

\[
P\leftarrow
\min\left(1,\frac{P_{\max}}{\operatorname{tr}(P)}\right)P.
\]

该投影保持矩阵对称半正定；`rls_P_trace_max=inf` 可关闭。

对同一20 IC扫描：

```text
inf, 1e4, 1e3, 1e2, 10, 1, 0.1, 0.01
```

综合“消除参数爆炸”和“饱和后指令 RMSE”选择：

```text
rls_P_trace_max = 1.0
```

相对无上限基线：

- 参数估计绝对值最大值：`1.32e12 -> 1.86e3`；
- 参数误差超过 `1e6` 的场景：`8/20 -> 0/20`；
- 中位归一化参数误差：`6.49e4 -> 486.8`；
- 中位 `Qdot` RMSE：`9.40 -> 0.540`；
- 中位饱和后指令 RMSE：`43.0 -> 17.75 m/s^2`。

更小的上限 `0.1/0.01` 虽略微降低参数误差，但饱和后指令 RMSE 回升到约 `39 m/s^2`，说明估计器接近停止更新，不能仅按参数误差选择。

选定结果目录：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/
  ident_shadow_stage4_ptrace1e0_N20_20260914/
```

其他扫描目录以 `ident_shadow_stage4_ptrace*` 命名，均为本次对话产生。

当前默认值已同步到：

- [`rl_td3/ident_cfg.py`](../rl_td3/ident_cfg.py)
- [`rl_td3/matlab/stage2_core_init.m`](../rl_td3/matlab/stage2_core_init.m)
- [`rl_td3/matlab/stage2_ident_init.m`](../rl_td3/matlab/stage2_ident_init.m)
- [`plot_table2_matlab/table2_noisy_ident/noisy_ident_cfg.m`](../plot_table2_matlab/table2_noisy_ident/noisy_ident_cfg.m)

论文第4节已在式(61)后补充协方差迹投影 Remark：

- [`论文文稿/overleaf_SCI_07_0914/sections/03_identification.tex`](../论文文稿/overleaf_SCI_07_0914/sections/03_identification.tex)

LaTeX 已成功编译，未引入新的编译错误。

## 8. 阶段2完成后的结论

### 8.1 阶段2已补做

[`rl_td3/matlab/test_stage2_ident.m`](../rl_td3/matlab/test_stage2_ident.m)
已重写为12组测试并全部通过，覆盖配置与量测协方差、KF灵敏度与解析初值、
软阈值/A21投影、上一拍KF后验距离、逐拍ZOH真值、冻结模型残差、三种
`tgo/tf`路径、协方差迹投影、固定参数带噪持续激励收敛、episode重置、
85维诊断协议及解析系数入口回归。

为避免测试复制生产公式，FF-RLS单步更新已抽取为
[`rl_td3/matlab/stage2_ffrls_update.m`](../rl_td3/matlab/stage2_ffrls_update.m)。
该重构由 `P_trace_max=Inf` 对标准FF-RLS的逐项等价测试约束。

固定参数收敛测试分别检查 `P_trace_max=Inf` 与默认 `P_trace_max=1`；
为单独评价迹限制，收敛测试取 `eta=0`。非零软阈值本身会引入L1稳态偏差，
不应将该偏差误判为迹投影破坏基本收敛能力。

### 8.2 MATLAB/Python协议已建立独立测试

新增 [`rl_td3/ident_protocol.py`](../rl_td3/ident_protocol.py) 集中解析85维诊断向量，
并由 [`rl_td3/tests/test_ident_protocol.py`](../rl_td3/tests/test_ident_protocol.py)
检查默认配置、已删除字段、关键切片、causal兼容别名及首拍NaN。
当前Python环境未安装 `pytest`，本次直接调用3个无框架测试函数，全部通过；
相关文件也已通过 `py_compile`。

### 8.3 P迹限制解决的是数值爆炸，不是辨识理论问题

`P_trace_max=1` 已把估计量控制在与真值相近的数量级，并显著降低制导误差，但：

- 中位归一化参数误差仍约为 `486.8`；
- 冻结模型残差中位仍约为 `359.5 m/s^2`；
- 不能据此声称11参数已经收敛；
- 很小的 P 迹上限可能只是降低更新增益，存在“靠少更新避免发散”的风险；
- 阶段2固定参数测试已确认 `P_trace_max=1` 在 `eta=0` 时未破坏基本收敛能力，
  但这不等价于真实时变闭环中的11参数能够收敛。

### 8.4 单周期拟冻结假设在部分场景偏差很大

当前 `frozen_model_residual` 使用：

```text
区间平均真值加速度：
  (rdot_true[k] - rdot_true[k-1]) / dt

冻结模型预测：
  phi_true[k-1]^T * vartheta[k-1]
```

二者不完全等价：前者是整个周期的平均加速度，后者更接近周期起点的瞬时模型值。当前残差中既包含参数冻结误差，也包含这种离散化口径差异。阶段2已验证残差使用的确为 `k-1` 数据；后续还需决定是否：

1. 保留当前定义，将其解释为“周期平均模型残差”；或
2. 使用更严格的周期内积分/中点真值构造离散回归目标。

在该问题澄清前，不应把全部残差都归因于 FF-RLS。

## 9. 阶段3/4复核与阶段5前停止点

阶段2已经补齐；以下保留覆盖清单，并记录阶段3/4复核结果。

### 9.1 阶段2覆盖范围（已完成）

已完成以下覆盖：

1. **配置和量测协方差**
   - 默认 `sigma_r=10`、`sigma_rdot=0.5`；
   - `R_xi=diag(sigma_r^2,sigma_rdot^2)`；
   - 配置中不再生成 `kf_sigma_r/kf_sigma_rdot`；
   - 非法零值/负值触发预期错误。

2. **KF灵敏度**
   - 非零 `R_xi` 下改变 `kf_qj` 必须改变 `rddot_hat`；
   - 避免重新出现 `kf_qj` 死旋钮。

3. **KF初值**
   - 首拍第三状态等于 noisy analytic 参数计算的 `rddot0`；
   - NCV `C1` 初始化不得读取传入的真值 `a_D`。

4. **软阈值与投影**
   - `A21` 不受软阈值影响；
   - `B1/B2/C1` 按 `eta` 收缩；
   - `A21` 负值仍投影到零。

5. **因果回归距离**
   - 构造非零距离噪声，使 `r_hat_prev != r_meas_prev`；
   - 断言 RLS 使用 `r_hat_prev`。

6. **逐拍 ZOH 下标**
   - 首拍 `theta_true_causal` 为 NaN；
   - 第2拍 `theta_true_causal ==` 第1拍 `theta_true_current`；
   - 第2拍当前真值另存为 `theta_true_current`；
   - 排除 off-by-one。

7. **冻结模型残差**
   - 验证使用 `r_true[k-1]`、`a_sat[k-1]` 和 `vartheta[k-1]`；
   - 不得混入 `vartheta[k]`。

8. **tgo真值泄漏**
   - `analytic_truth` 使用真值 `r/rdot`；
   - `analytic_noisy` 使用 noisy `r/rdot`；
   - `sparse_ffrls` 使用 KF 后验 `r_hat/rdot_hat`；
   - 影子预测指令也使用辨识路径时间轴。

9. **P迹投影**
   - 初始化后 `trace(P) <= P_trace_max`；
   - 每次更新后仍满足上限；
   - 对称性和非负特征值保持；
   - `P_trace_max=Inf` 时与标准 FF-RLS 等价。

10. **固定参数带噪收敛**
    - 在整段测试内固定 `vartheta_star`；
    - 使用持续激励回归量和带噪输出；
    - 先用 `P_trace_max=Inf` 验证基础 FF-RLS 公式收敛；
    - 再用默认 `P_trace_max=1` 检查是否仍能在合理步数内进入误差带；
    - 如果默认上限无法收敛，必须重新评估 `P_trace_max=1`，不能仅因它抑制爆炸就保留。

11. **MATLAB/Python诊断协议**
    - `info_vec` 长度为85；
    - Python 切片与 MATLAB索引逐项一致；
    - `theta_true` 兼容字段确实映射到 causal 真值；
    - 首拍 NaN 不引起画图或汇总警告。

### 9.2 阶段3/4复核（已完成）

由于阶段2将生产FF-RLS更新抽成可测试函数，已按同一checkpoint、同一20个IC、
同一噪声种子显式复跑 `rls_P_trace_max=1`。结果目录：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/
  ident_shadow_stage4_recheck_after_stage2_N20_20260914/
```

复跑的参数、状态、KF、参数误差、冻结残差和P健康度指标与原阶段4结果一致；
指令RMSE最大仅有 `4.2e-9` 的浮点差异。阶段4结论不变，无需重新扫描
`P_trace_max`，默认值继续保持 `1.0`。阶段3无上限发散基线也无需复跑。

### 9.3 阶段5验证集闭环门控（已执行）

为避免测试集泄漏，阶段5已改用 `samples_clean_val.mat` 前30个有效IC，
使用同一 checkpoint、同一IC和同一噪声种子进行配对闭环消融：

```text
analytic_truth
analytic_noisy
sparse_ffrls, eta=0.1, P_trace_max=1
sparse_ffrls, eta=0,   P_trace_max=1
```

结果目录：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/
  ident_ablation_stage5_val_N30_20260914/
```

至少比较：

- 双NCV规避成功率；
- 目标到达成功率；
- 两枚NCV最近距离；
- 目标最近距离/末端误差；
- 控制能量；
- 指令总变差；
- 饱和率；
- 因果参数误差；
- 制导滞后误差；
- 饱和前后指令预测误差。

关键结果：

- `analytic_truth`：联合成功 `29/30`，饱和率 `8.2%`；
- `analytic_noisy`：联合成功 `27/30`，饱和率 `56.6%`；
- `sparse_ffrls, eta=0.1`：联合成功 `30/30`，饱和率 `25.7%`；
- `sparse_ffrls, eta=0`：联合成功 `28/30`，饱和率 `27.2%`。

相对 `analytic_noisy`，`eta=0.1` 联合成功率提高10个百分点，同时平均控制能量、
指令总变差和饱和率分别下降约44%、78%和31个百分点；但相对
`analytic_truth` 仍有明显过度规避，平均控制能量约增加161%，因此结论是
“成功率门控通过、效率门控未完全通过”。

随后在同一验证集30 IC上扫描 `eta=0.15/0.2/0.3`，联合成功分别为
`27/30`、`29/30`、`28/30`，均未超过 `eta=0.1`。故阶段5暂时冻结
`rls_eta=0.1`、`rls_P_trace_max=1.0`，不根据最终测试集继续调参。

阶段5冒烟还发现旧指令误差在闭环应用辨识指令时会退化为“预测减自身”等于零。
现已统一改为“非真值路径指令减 analytic_truth 参考指令”；真值仅进入离线诊断，
不进入辨识更新或控制。

## 10. 已完成的验证

截至本次停止点已经完成：

- MATLAB 阶段2完整测试：12/12通过；
- Python 配置/85维协议测试：3/3通过；
- Python 改动文件 `py_compile`；
- 20 IC、`P_trace_max=1` 阶段4复跑，结论与原结果一致；
- 验证集30 IC阶段5四模式配对闭环消融；
- 验证集30 IC `eta=0.15/0.2/0.3` 闭环扫描；
- 先前已完成论文 XeLaTeX/latexmk 完整编译。

未完成：

- 含噪观测策略重训；
- 超参数冻结后的完整300 IC测试集终验；
- BC-TD3/专家轨迹接入。

## 11. 继续工作时的约束

1. 阶段2与阶段5验证集门控已经完成；下一步是含噪路径重训，随后进行完整300 IC测试集终验。
2. 不要依据旧的8月份结论性 Markdown 直接作技术判断；以当前代码、当前论文和本次新产物为准。
3. 不降维成 C1-only，除非用户明确同意修改论文理论。
4. 暂不改专家轨迹、BC-TD3 和 `rl_td3/train_td3.py`。
5. 已冻结 `rls_eta=0.1`、`rls_P_trace_max=1.0`；最终测试集不得再用于调参。
6. Python 固定使用：

```text
E:\Python\venvs\rl_pytorch\Scripts\python.exe
```

7. 当前工作区原本已有大量未跟踪论文和导出文件，不要删除、覆盖或提交无关用户文件。
8. 本次改动尚未提交 Git。

