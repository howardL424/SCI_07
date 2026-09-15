# 参数辨识 + 纯TD3改造会话总结（二）

日期：2026-09-14  
工作区：`D:\LTY\yan\yanerxia\GAIL`  
前序总结：[`PARAMETER_IDENTIFICATION_SESSION_SUMMARY_20260914.md`](PARAMETER_IDENTIFICATION_SESSION_SUMMARY_20260914.md)

> 2026-09-15 目录整理：专属辨识代码与测试产物已迁至 [`ident/`](../ident/README.md)。
> 下文旧路径仍可通过兼容入口访问；新工作请直接使用 `ident/`。

## 1. 文档用途

本文记录第二段参数辨识改造会话，供后续对话直接恢复上下文。内容包括：

1. 用户与Agent在本次会话中的关键对话；
2. 本会话实际完成的代码改动；
3. 阶段2、阶段3/4复核和阶段5验证集闭环结果；
4. 当前实现边界、尚未解决的问题；
5. 面向最终蒙特卡洛成功率验证的下一步顺序。

当前工作范围仍为：

```text
纯TD3 + 在线参数辨识 + 含噪量测
```

暂不处理专家轨迹、BC-TD3和论文第6节正式实验排版。

## 2. 进入本会话前已有的基础

本会话开始前，前序工作已完成阶段0、阶段1、阶段3和阶段4的大部分改造：

- KF量测协方差改为
  `R_xi = diag(sigma_r^2, sigma_rdot^2)`；
- 删除独立的 `kf_sigma_r/kf_sigma_rdot`；
- 软阈值仅作用于 `B1/B2/C1`，不收缩 `A21`；
- 回归距离改用上一拍KF后验 `r_hat_prev`；
- KF第三状态使用解析相对加速度初始化；
- 参数真值区分 `vartheta[k-1]` 和 `vartheta[k]`；
- 非真值制导路径使用自己的 `r/rdot` 计算 `tgo/tf`；
- MATLAB/Python诊断协议扩展到85维；
- 新增RLS协方差迹投影 `rls_P_trace_max`；
- 阶段4预研暂选 `rls_P_trace_max=1.0`。

前序工作遗漏了阶段2完整测试，因此本会话首先补做阶段2。

## 3. 本次会话关键对话

### 3.1 恢复前序工作并补做阶段2

用户要求：

> 阅读先前的参数辨识代码改造对话总结，继续进行下一步，先补做阶段2，然后看情况再改动阶段3、4，停在阶段5改造前。

Agent执行：

- 阅读前序总结、论文第2至第4节及MATLAB/Python实现；
- 确认旧 `test_stage2_ident.m` 使用零量测噪声和已删除配置，已经过时；
- 重写阶段2测试；
- 将FF-RLS生产更新抽成可直接测试的函数；
- 复跑阶段4的20 IC结果；
- 当时按用户要求停在阶段5之前。

### 3.2 询问距离完整实现还差多少

用户询问：

> 现在距离完成参数辨识+纯TD3实现还差多少？

Agent判断：

- 参数辨识和纯TD3的工程接线已基本完成；
- 仍缺阶段5闭环门控、含噪路径重训和完整蒙特卡洛终验；
- 最大不确定性不是代码能否运行，而是辨识误差、过度机动和最终成功率能否同时达标。

### 3.3 澄清是否已经使用完整300组蒙特卡洛

用户询问：

> 你现在所做的测试和调参是针对现有300组蒙特卡洛测试进行的吗？

核对数据后确认：

```text
samples_clean.mat       2351个有效训练初态
samples_clean_val.mat    300个有效验证初态
samples_clean_test.mat   300个有效测试初态
```

阶段2使用单元测试和合成数据，不使用蒙特卡洛池。

阶段3/4预研只使用了 `samples_clean_test.mat` 的前20个有效初态，并据此选择过
`rls_P_trace_max=1.0`。因此该选择带有测试集参与超参数决策的风险。

### 3.4 解释测试集泄漏

用户询问：

> 为什么说会存在测试集泄漏？

Agent解释：

- 如果先观察测试集前20组在不同超参数下的表现；
- 再选择表现较好的参数；
- 最后又将这20组包含在完整300组测试统计中；
- 则测试集已经参与参数选择，最终成功率会有乐观偏差。

这属于评估协议泄漏，不是辨识器读取真值的算法泄漏。

规范流程应为：

```text
训练集：训练策略
验证集：选择KF/RLS/稀疏参数和策略checkpoint
测试集：所有参数冻结后只做最终评估
```

### 3.5 用户要求继续阶段5

用户要求：

> 那你继续之前的阶段5改造吧，记得最终目标是为了让算法通过蒙特卡洛测试。

Agent据此将阶段5默认数据集从测试集改为验证集，并完成30 IC配对闭环消融和
`eta`小范围扫描。最终300组测试集没有在阶段5中继续用于调参。

### 3.6 本次总结请求

用户要求将本次改造会话、具体改动、当前现状和下一步写入一个新增 `_2` 后缀的
Markdown文件，即本文。

## 4. 本会话完成的阶段2改造

### 4.1 重写MATLAB阶段2测试

文件：

- [`rl_td3/matlab/test_stage2_ident.m`](../rl_td3/matlab/test_stage2_ident.m)

测试重写为12组，覆盖：

1. `analytic_truth` 核心回归；
2. 默认配置与 `R_xi`；
3. 非法零值/负值量测噪声；
4. `kf_qj`灵敏度；
5. KF解析初值和NCV `C1`无真值加速度泄漏；
6. 软阈值只处理第2个及后续分量；
7. `A21`非负投影；
8. 第三拍RLS确实使用上一拍 `r_hat`，而非原始 `r_meas`；
9. `vartheta[k-1]`/`vartheta[k]`两拍ZOH对齐；
10. `frozen_model_residual`使用上一拍距离、饱和指令和参数真值；
11. 三种制导路径与影子预测的 `tgo/tf`来源；
12. 协方差迹限制、固定参数带噪持续激励收敛、episode重置和85维协议。

执行结果：

```text
MATLAB阶段2测试：12/12通过
test_stage2_ident.m checkcode：0 issues
```

### 4.2 抽取生产FF-RLS单步函数

新增：

- [`rl_td3/matlab/stage2_ffrls_update.m`](../rl_td3/matlab/stage2_ffrls_update.m)

修改：

- [`rl_td3/matlab/stage2_ident_step.m`](../rl_td3/matlab/stage2_ident_step.m)

作用：

- 测试直接调用生产FF-RLS公式，避免在测试中复制一份可能漂移的局部算法；
- 在同一函数中完成RLS二次更新、P迹投影、`j>=2`软阈值和 `A21>=0`投影；
- `P_trace_max=Inf` 已与标准FF-RLS逐项比较并通过。

固定参数带噪持续激励测试分别检查：

```text
P_trace_max = Inf
P_trace_max = 1.0
```

为单独判断P迹限制是否破坏基础收敛能力，该测试取 `eta=0`。非零软阈值本身会
带来稳态偏差，不能将该偏差错误归因于P迹投影。

### 4.3 建立独立Python诊断协议

新增：

- [`rl_td3/ident_protocol.py`](../rl_td3/ident_protocol.py)
- [`rl_td3/tests/test_ident_protocol.py`](../rl_td3/tests/test_ident_protocol.py)

修改：

- [`rl_td3/evasion_env.py`](../rl_td3/evasion_env.py)

作用：

- 集中解析MATLAB `env_step` 的85维诊断向量；
- 统一 `theta_true` 与 `theta_true_causal` 的兼容语义；
- 解析 `theta_true_current`、`theta_err_command`、冻结残差和Qdot指令误差；
- 检查首拍causal真值为NaN时不会造成切片错误；
- 检查配置中不再生成 `kf_sigma_r/kf_sigma_rdot`。

当前Python环境未安装 `pytest`，因此直接调用3个测试函数执行，结果为：

```text
protocol tests: 3 passed
py_compile：通过
IDE lints：0
```

## 5. 阶段3/4复核

由于FF-RLS生产函数发生了等价重构，本会话按原checkpoint、原20个IC、
原噪声种子和显式 `P_trace_max=1` 复跑阶段4。

结果目录：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/
  ident_shadow_stage4_recheck_after_stage2_N20_20260914/
```

结论：

- 参数、KF、因果误差、制导滞后误差、冻结残差和P健康度指标与原结果一致；
- 指令RMSE最大差异约 `4.2e-9`，属于浮点误差；
- 阶段2重构没有改变阶段4算法行为；
- 阶段3无P迹限制的发散基线无需重跑。

注意：

- 阶段3/4的前20 IC来自测试集；
- 这些结果应继续视为预研/问题定位结果；
- 不应单独作为最终论文泛化结论。

## 6. 本会话完成的阶段5改造

### 6.1 扩展纯TD3闭环episode统计

修改：

- [`rl_td3/evaluate_td3.py`](../rl_td3/evaluate_td3.py)

`run_episode` 新增逐局统计：

```text
theta_causal_rmse
theta_causal_tail_rmse
theta_command_rmse
theta_command_tail_rmse
frozen_model_rmse
frozen_model_tail_rmse
a_cmd_rmse
a_cmd_tail_rmse
a_cmd_sat_rmse
a_cmd_sat_tail_rmse
```

同时保留并落盘：

```text
规避成功
打击成功
联合成功
是否被拦截
两枚NCV最近距离
目标最近距离
控制能量
指令总变差
饱和率
```

### 6.2 重构阶段5配对消融脚本

修改：

- [`td3_baseline/run_ident_ablation_pure.py`](../td3_baseline/run_ident_ablation_pure.py)

主要变化：

- 默认数据集由 `samples_clean_test.mat` 改为 `samples_clean_val.mat`；
- 默认运行30个有效验证IC；
- 默认包含四种模式：

```text
analytic_truth
analytic_noisy
sparse_ffrls, eta=0.1
sparse_ffrls, eta=0
```

- 同一 `local_idx` 在不同模式间严格配对；
- 每局噪声种子固定为 `base_noise_seed + local_idx`；
- 保存逐局压缩NPZ和完整JSON摘要；
- 为成功率保存Wilson 95%区间；
- 为候选相对基线的配对差值保存bootstrap 95%区间；
- `--episodes 0` 表示遍历完整初态池；
- 新增 `--dataset-role validation/test/custom`；
- 使用测试集时必须显式声明 `--dataset-role test`；
- 禁止在 `dataset_role=test` 时使用 `--eta-grid` 调参；
- 新增 `--eta-grid`，用于只在验证集运行闭环稀疏阈值候选。

### 6.3 修复闭环指令误差诊断退化

阶段5第一次冒烟发现：

- `sparse_ffrls`在闭环实际应用时，旧 `a_cmd_err` 是“辨识指令减本拍采用指令”；
- 二者是同一条指令，因此误差恒为零；
- 该指标在影子模式有效，在闭环应用模式下失效。

修改：

- [`rl_td3/matlab/stage2_step.m`](../rl_td3/matlab/stage2_step.m)
- [`rl_td3/matlab/env_step.m`](../rl_td3/matlab/env_step.m)
- [`rl_td3/matlab/test_stage2_ident.m`](../rl_td3/matlab/test_stage2_ident.m)

新定义：

```text
a_cmd_err =
  非真值路径指令 - analytic_truth参考指令
```

饱和后误差同样比较两条路径各自饱和后的指令。

`analytic_truth`参考只用于诊断，不进入KF、RLS或实际控制，不构成控制真值泄漏。

修复后阶段5冒烟中 `sparse_ffrls` 的饱和后指令误差恢复为有效非零量。

## 7. 阶段5验证集结果

使用checkpoint：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model
```

使用数据：

```text
samples_clean_val.mat 前30个有效IC
noise_seed = local_idx
rls_P_trace_max = 1.0
```

主结果目录：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/
  ident_ablation_stage5_val_N30_20260914/
```

### 7.1 四模式闭环结果

`analytic_truth`：

```text
联合成功       29/30 = 96.7%
平均D1最近距   34.6 m
平均D2最近距   34.3 m
平均目标最近距 0.102 m
平均控制能量   8318.7
平均指令总变差 332.5
平均饱和率     8.2%
```

`analytic_noisy`：

```text
联合成功       27/30 = 90.0%
平均D1最近距   24.0 m
平均D2最近距   104.9 m
平均目标最近距 0.112 m
平均控制能量   38870.8
平均指令总变差 6777.9
平均饱和率     56.6%
```

`sparse_ffrls, eta=0.1`：

```text
联合成功       30/30 = 100.0%
平均D1最近距   98.7 m
平均D2最近距   117.0 m
平均目标最近距 0.138 m
平均控制能量   21756.5
平均指令总变差 1492.6
平均饱和率     25.7%
```

`sparse_ffrls, eta=0`：

```text
联合成功       28/30 = 93.3%
平均D1最近距   131.0 m
平均D2最近距   171.1 m
平均目标最近距 0.112 m
平均控制能量   23264.0
平均指令总变差 1321.5
平均饱和率     27.2%
```

### 7.2 `eta=0.1`相对含噪解析基线

在相同30 IC和相同噪声下：

```text
联合成功率：90.0% -> 100.0%，提高10个百分点
控制能量：下降约44%
指令总变差：下降约78%
饱和率：56.6% -> 25.7%，下降约31个百分点
```

30组中，`sparse_ffrls, eta=0.1` 相对 `analytic_noisy`：

```text
联合成功赢3局
联合成功输0局
```

### 7.3 不能只按成功率判定完全通过

虽然 `eta=0.1` 达到30/30联合成功，但相对 `analytic_truth`：

```text
控制能量约增加161%
平均D1/D2规避距离明显增大
饱和率从8.2%增加到25.7%
```

因此当前判断为：

```text
成功率门控：通过
相对含噪解析路径的鲁棒性与控制平滑性：通过
相对理想真值路径的效率门控：未完全通过
```

这意味着辨识闭环已经具有实用价值，但仍存在过度规避和额外机动。

### 7.4 真实时变参数仍未“收敛”

`eta=0.1` 的验证集结果中：

```text
theta_causal_tail_rmse 平均约 3230.9
theta_causal_tail_rmse 中位约 714.6
```

因此不能声称11个时变参数已经收敛。当前可声称的是：

- P迹投影消除了原有数值爆炸；
- 固定参数、带噪、持续激励测试具备基础收敛能力；
- 真实闭环中参数估计仍是高误差跟踪；
- 即便参数误差较大，闭环成功率和控制品质仍明显优于直接含噪解析代入。

## 8. 验证集 `eta` 扫描

结果目录：

```text
td3_baseline/checkpoints/training_300k_20260730_1917/
  ident_ablation_stage5_eta_scan_val_N30_20260914/
```

同一验证集30 IC、`P_trace_max=1`：

```text
eta=0.15：联合成功27/30，饱和率21.5%
eta=0.20：联合成功29/30，饱和率22.4%
eta=0.30：联合成功28/30，饱和率20.6%
```

这些候选虽部分降低能量或饱和率，但联合成功数均低于 `eta=0.1` 的30/30。

当前暂时冻结：

```text
rls_eta         = 0.1
rls_P_trace_max = 1.0
```

注意：

- `eta=0.1`已通过验证集闭环比较；
- `P_trace_max=1.0`最初来自测试集前20 IC预研扫描；
- 阶段2固定参数测试和阶段5验证集结果支持它可用；
- 但若追求最严格的无泄漏协议，正式重训前仍可在验证集补做一个很窄的
  `P_trace_max`候选复核，或将其明确视为由数值稳定性约束确定的工程参数，
  不再根据测试集表现修改。

## 9. 当前代码和实验状态

### 9.1 已完成

- 三通道11参数在线辨识框架；
- 含噪距离/距离率KF；
- 因果FF-RLS回归；
- `A21`非负投影和选择性软阈值；
- 参数周期ZOH真值对齐；
- 非真值路径 `tgo/tf` 去真值泄漏；
- RLS协方差迹防护；
- 85维MATLAB/Python诊断协议；
- 阶段2完整测试；
- 阶段3无迹限制发散定位；
- 阶段4P迹限制预研与复核；
- 阶段5验证集30 IC四模式闭环门控；
- 验证集 `eta=0.15/0.2/0.3` 扫描；
- 纯TD3训练入口已经能够接收辨识配置。

### 9.2 尚未完成

- 含噪 `sparse_ffrls` 路径下的纯TD3正式重训；
- 重训前的逐episode噪声种子调度检查/改造；
- 明确TD3 Actor观测是否也应使用含噪量测；
- 完整验证集上的最终checkpoint选择；
- 所有配置冻结后的完整300 IC测试终验；
- 多训练种子或多噪声种子的鲁棒性统计；
- 过度规避和控制能量偏大的进一步处理；
- 论文第6节正式实验表格和结论；
- BC-TD3/专家轨迹接入。

### 9.3 当前不能宣称的结论

目前不能宣称：

- 11个真实时变参数已经收敛；
- `eta=0.1, P_trace_max=1` 已在完整300测试IC上最终验证；
- 当前真值路径训练的旧checkpoint就是最终含噪最优策略；
- 参数辨识算法已经达到论文最终蒙特卡洛结论。

目前可以宣称：

- 代码时序、量测口径、诊断协议和协方差防护已有完整测试；
- 当前辨识闭环在验证集前30 IC上达到30/30联合成功；
- 相对直接含噪解析代入，成功率、能量、指令平滑性和饱和率均明显改善；
- 相对理想真值路径仍有明显效率差距。

## 10. 测试集泄漏与最终蒙特卡洛约束

前序阶段3/4已观察过 `samples_clean_test.mat` 的前20 IC，并据此比较过
`P_trace_max`。因此严格意义上，当前测试集不是完全未观察的纯盲测集。

后续可选处理：

1. 最严谨方案：从未使用的数据重新生成独立的300组最终测试集；
2. 次优方案：将已使用的前20组定义为开发样本，只报告剩余280组；
3. 工程方案：在验证集冻结全部参数后仍评估现有300组，但在论文/记录中说明
   前20组曾用于预研，不将结果描述为完全盲测。

无论采用哪种方案，后续都必须遵守：

```text
最终测试数据不得继续用于选择KF、RLS、eta、P迹上限、训练步数或checkpoint。
```

## 11. 下一步建议

### 11.1 第一步：重训前协议检查

检查并明确：

1. 训练期间 `noise_seed` 是否每个episode变化；
2. 训练与验证是否使用独立且可复现的噪声序列；
3. TD3 Actor输入观测是否仍为真值；
4. 若论文要求含噪观测，需将Actor观测同步为可用量测/KF状态；
5. `P_trace_max=1` 是直接冻结，还是在验证集做最后一次窄范围复核。

当前风险：

- 如果每局都使用相同 `noise_seed`，策略可能记住固定的逐拍噪声模式；
- 如果Actor仍读取真值状态，只能称为“制导系数辨识含噪”，不能严格称为
  “全观测含噪纯TD3”。

### 11.2 第二步：纯TD3含噪路径冒烟训练

使用：

```text
coeff_mode       = sparse_ffrls
ident_apply_cmd  = true
rls_eta          = 0.1
rls_P_trace_max  = 1.0
sigma_r          = 10.0
sigma_rdot       = 0.5
kf_qj            = 1.0e4
```

先运行短训练，确认：

- episode能够正常重置；
- 噪声种子不跨局错误复用；
- 评估环境与训练环境隔离；
- checkpoint和辨识配置完整保存；
- 不出现NaN、MATLAB persistent串局或参数爆炸。

### 11.3 第三步：正式纯TD3重训

建议：

- 使用 `samples_clean.mat` 训练；
- 使用 `samples_clean_val.mat` 选checkpoint；
- 不使用最终测试集选训练步数或模型；
- 至少保留训练种子、噪声种子规则和完整 `ident_run_config.json`；
- 资源允许时运行多个训练种子，不只保留单次最好结果。

### 11.4 第四步：验证集完整评估

对候选checkpoint在完整300验证IC上比较：

```text
analytic_truth
analytic_noisy
sparse_ffrls eta=0.1
必要时的冻结消融项
```

门控不只看联合成功率，还应同时检查：

- 双NCV规避成功率；
- 目标打击成功率；
- 联合成功率；
- 两枚NCV最近距离；
- 目标最近距离；
- 控制能量；
- 指令总变差；
- 饱和率；
- 因果参数误差；
- 制导滞后误差；
- 饱和前后指令误差。

### 11.5 第五步：最终蒙特卡洛终验

冻结以下所有内容：

```text
模型checkpoint
KF参数
RLS参数
eta
P_trace_max
量测噪声
动作映射
观测模式
成功判据
```

然后只运行一次最终测试集，报告：

- 成功数/总数；
- 成功率；
- Wilson 95%置信区间；
- 配对成功/失败变化；
- 连续指标均值、中位数、标准差；
- 必要时的bootstrap配对差值区间。

最终目标不是只让 `both_ok` 数值变高，而是：

```text
在含噪条件下保持较高规避+打击联合成功率，
同时避免系统性过度规避、过高控制能量和长期饱和。
```

## 12. 重要文件

核心MATLAB：

- [`rl_td3/matlab/stage2_ident_init.m`](../rl_td3/matlab/stage2_ident_init.m)
- [`rl_td3/matlab/stage2_ident_step.m`](../rl_td3/matlab/stage2_ident_step.m)
- [`rl_td3/matlab/stage2_ffrls_update.m`](../rl_td3/matlab/stage2_ffrls_update.m)
- [`rl_td3/matlab/stage2_step.m`](../rl_td3/matlab/stage2_step.m)
- [`rl_td3/matlab/env_step.m`](../rl_td3/matlab/env_step.m)
- [`rl_td3/matlab/test_stage2_ident.m`](../rl_td3/matlab/test_stage2_ident.m)

Python协议与环境：

- [`rl_td3/ident_cfg.py`](../rl_td3/ident_cfg.py)
- [`rl_td3/ident_protocol.py`](../rl_td3/ident_protocol.py)
- [`rl_td3/evasion_env.py`](../rl_td3/evasion_env.py)
- [`rl_td3/evaluate_td3.py`](../rl_td3/evaluate_td3.py)
- [`rl_td3/tests/test_ident_protocol.py`](../rl_td3/tests/test_ident_protocol.py)

诊断与消融：

- [`td3_baseline/run_ident_shadow_diag_pure.py`](../td3_baseline/run_ident_shadow_diag_pure.py)
- [`td3_baseline/run_ident_hyperparam_scan_pure.py`](../td3_baseline/run_ident_hyperparam_scan_pure.py)
- [`td3_baseline/run_ident_ablation_pure.py`](../td3_baseline/run_ident_ablation_pure.py)

训练入口：

- [`td3_baseline/train_td3_pure.py`](../td3_baseline/train_td3_pure.py)

阶段5正式结果：

- `td3_baseline/checkpoints/training_300k_20260730_1917/ident_ablation_stage5_val_N30_20260914/`
- `td3_baseline/checkpoints/training_300k_20260730_1917/ident_ablation_stage5_eta_scan_val_N30_20260914/`

阶段5冒烟结果：

- `ident_ablation_stage5_smoke_val_N2_20260914/`：修复前，指令误差诊断退化，不应用于结论；
- `ident_ablation_stage5_smoke_v2_val_N2_20260914/`：修复后有效冒烟。

## 13. 验证记录

本会话已完成：

```text
MATLAB test_stage2_ident：12/12通过
stage2_ident_step.m checkcode：0 issues
stage2_ffrls_update.m checkcode：0 issues
stage2_step.m checkcode：0 issues
test_stage2_ident.m checkcode：0 issues
Python protocol tests：3/3通过
Python py_compile：通过
IDE Python lints：0
git diff --check：通过
阶段4复核20 IC：通过，数值与原结果一致
阶段5有效冒烟2 IC：通过
阶段5验证集30 IC四模式消融：完成
阶段5验证集30 IC eta扫描：完成
```

`env_step.m` 的 `checkcode` 仍有3个既有 `MSNU` 提示，来自MATLAB对函数内局部
函数/变量使用的静态分析，不是本次新增错误；完整阶段2运行测试已经通过。

## 14. 工作区约束

1. 本次所有改动尚未提交Git；
2. 工作区原本存在大量未跟踪论文、图片、缓存和实验产物；
3. 不要删除、覆盖或提交与参数辨识无关的用户文件；
4. Python固定使用：

```text
E:\Python\venvs\rl_pytorch\Scripts\python.exe
```

5. 不降维为C1-only，除非用户明确同意修改论文理论；
6. 在纯TD3含噪路径最终通过前，不接入BC-TD3和专家轨迹；
7. 下一次工作应从“重训前噪声/观测协议检查”开始，而不是重新执行阶段2。

