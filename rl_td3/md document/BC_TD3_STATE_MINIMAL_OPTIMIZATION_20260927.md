# BC-TD3 状态量最小必要、短时间优化方案

> 日期：2026-09-27  
> 用途：供新对话直接参考并实施；目标是在不大规模重构、不重新求解参考轨迹的前提下，快速判断并修复当前 13 维观测的信息缺口。
>
> 术语说明：本文将系统统一表述为多实体相对运动与分阶段控制问题。
> `M`、`D1`、`D2`、`T` 仅是代码中的实体标识；`Q` 表示相对方位变化率
> 的合成幅值。本文只讨论观测设计、行为克隆与闭环数值验证。

## 1. 背景与当前判断

当前 Actor 使用 13 维观测：

```text
[r_MD1, rdot_MD1, qy_D1M, qz_D1M,
  r_MD2, rdot_MD2, qy_D2M, qz_D2M,
  r_MT, qy_MT, qz_MT, V_M, tgo_norm]
```

历史纯 BC 诊断结果并不理想：

- 原动作表示下验证集 MSE 约为 `0.360`；
- 比值动作表示下验证集 MSE 约为 `0.321`；
- `gamma` MAE 约为 `0.55`，动作符号和贴边行为学习较差；
- 改动作表示后，闭环表现没有明显改善，诊断结论仍为
  `fail_high_val_error -> state_ablation`。

主要怀疑是：参考控制器实际使用相对方位变化率、实体状态和相对运动
信息，而当前 13 维观测只有相对方位角，没有其变化率。因此相近甚至
相同的 13 维观测可能对应不同参考动作，造成不可约的 BC 误差。

由于可用于算法调整的时间有限，本方案不直接扩展到完整姿态、速度或
18 维绝对状态，而只补充最直接、改动最小的信息。

## 2. 唯一优先方案：13 维增加三个相对方位变化率幅值

将观测由 13 维扩展为 16 维：

```text
obs16 = [obs13, Q1, Q2, QT]
```

其中：

```text
Q1 = sqrt(qd_y1^2  + qd_z1^2  * cos(qy_D1M)^2)
Q2 = sqrt(qd_y2^2  + qd_z2^2  * cos(qy_D2M)^2)
QT = sqrt(qd_yTM^2 + qd_zTM^2 * cos(qy_TM)^2)
```

定义必须与 `rl_td3/matlab/stage2_step.m` 和
`evasion_helpers.m::computeOptimalCmd_duo_target` 完全一致：

- D1、D2 通道分别使用 `M-D1`、`M-D2` 实体对的相对方位变化率；
- T 通道沿用 `stage2_step.m` 中的 `T -> M` 几何口径；
- 使用 `computeMDDot` 的解析结果，不用相邻采样点差分方位角；
- 首轮只加入三个标量幅值 `Q1/Q2/QT`，不加入六个
  `qd_y/qd_z` 分量。

选择三个 `Q` 而不是一次加入姿态和全部速度的原因：

1. 三个 `Q` 被当前参考控制器直接使用；
2. 可以由现有每步 `state_abs [18]` 确定性派生；
3. 不需要重跑 IPOPT、边界值求解器或重新生成参考轨迹；
4. 只增加 3 维，便于定位收益是否来自状态信息补齐；
5. 若无收益，可以迅速停止，避免投入完整 TD3 长训。

## 3. 必须先做的小规模 A/B 测试

不要直接修改正式训练口径并启动 300k TD3。先做纯 BC 小规模对照：

### 3.1 固定条件

- A 组：现有归一化 13 维观测；
- B 组：相同 13 维观测加归一化 `Q1/Q2/QT`；
- 使用完全相同的参考轨迹、`both_ok` 筛选、动作映射；
- 按轨迹而不是按 transition 使用相同训练/验证划分；
- 网络结构、优化器、batch size、训练轮数和 early stopping 相同；
- 至少运行 3 个 seed；
- 保存每个 seed 的配置、数据划分和指标。

### 3.2 数据来源

现有参考 transition 已保存：

```text
state_abs      [N, 18]
next_state_abs [N, 18]
```

直接由这两个字段派生当前步和下一步的 `Q1/Q2/QT`。不得使用阶段结束结果、
未来状态或完整轨迹统计量构造当前观测，避免信息泄漏。

首轮实验最好保持旧 13 维数据不变，在加载阶段按可选
`obs_schema` 派生 16 维数据，不立即覆盖数据集中的权威 `state` 字段。
建议 schema 名称：

```text
rel13_v1
rel13_qmag16_v1
```

### 3.3 归一化

- 原 13 维继续使用现有 `OBS_SCALE`；
- 三个 `Q` 的 scale 只根据训练轨迹统计；
- 推荐分别使用训练集各通道绝对值的 `99.5%` 分位数作为固定 scale；
- scale 落盘并写入新的观测归一化版本；
- 验证集不得参与 scale 选择；
- 仍然只做除法，不裁剪归一化结果；
- 报告验证集各维 `|obs_norm| > 1` 和 `> 5` 的比例。

## 4. 判断指标和继续门槛

不能只看总 MSE。每个 seed 至少报告：

- 总体 validation MSE、MAE；
- `gamma / rhoT / rhoE` 或当前三维动作各自的 MSE、MAE；
- `gamma` 正负号准确率；
- 参考数据中 `|gamma|` 边界样本的符号准确率和 MAE；
- BC 输出边界率与参考动作边界率；
- 13D 相对 16D 的均值、标准差和逐 seed 配对差值。

建议继续进入闭环测试的门槛：

1. 16D 的平均 validation MSE 相比 13D 至少下降 `15%`；并且
2. `gamma` MAE 至少下降 `10%`，或 `gamma` 符号/贴边预测明显改善；并且
3. 改善至少在 3 个 seed 中有 2 个方向一致，没有明显数据泄漏或数值异常。

若未满足上述门槛：

- 不进行完整 BC-TD3 重训；
- 不继续盲目加入大量状态量；
- 优先将问题转向 `gamma` 多模态/动作标签不唯一诊断，例如
  “符号分类 + 幅值回归”或近邻标签方差测试。

## 5. 通过离线门槛后的最小闭环验证

只有纯 BC A/B 通过门槛后才做：

1. 使用相同初态，对 13D-BC 与 16D-BC 做 100～300 条配对闭环；
2. 分别报告 `both_ok`、stage2 提前终止、stage3 终止条件未满足；
3. 同时报告动作饱和率、`gamma` 切换次数和动作总变差；
4. 若 16D 离线改善但闭环不改善，不直接启动 TD3 长训；
5. 只有离线指标和闭环指标均有改善，才进入一次正式 BC-TD3 重训。

在时间很紧时，完整流程应严格按下列顺序停止或继续：

```text
13D/16D 数据一致性测试
    -> 3-seed 纯 BC A/B
        -> 未过门槛：停止状态扩展
        -> 过门槛：100～300 IC 配对闭环
            -> 无闭环改善：停止
            -> 有闭环改善：正式 BC-TD3 重训
```

## 6. 最小实现范围

实施时应优先检查和修改以下位置，但先保留 13D 兼容模式：

- `rl_td3/obs_norm.py`
  - 增加观测 schema 和 16D scale，不应直接破坏旧 13D checkpoint。
- `rl_td3/load_expert_buffer.py`
  - 从 `state_abs/next_state_abs` 派生当前步和下一步三个 `Q`；
  - 取消相关代码中写死的 13 维假设，改为由 schema 决定。
- `rl_td3/train_bc.py`
  - Actor 输入维数由数据或 schema 决定，不再固定 `obs_dim=13`。
- `rl_td3/bc_actor.py`
  - 保持 checkpoint 中显式记录 `obs_dim` 和 `obs_schema`。
- `rl_td3/matlab/stage2_obs_from_abs.m`
  - 在 16D schema 下使用与 `stage2_step.m` 相同的解析口径返回三个 `Q`。
- `rl_td3/evasion_env.py`
  - 仅在进入闭环验证后接入 16D 在线观测；
  - observation space 维度必须随 schema 改变。
- `rl_td3/tests/`
  - 更新或新增维度、归一化、参考数据/在线观测一致性和 checkpoint
    不兼容提示测试。

不要在第一阶段同时改奖励、动作映射、网络结构、数据筛选或 TD3 超参数，
否则无法判断状态扩展是否有效。

## 7. 必须通过的一致性测试

正式训练前至少满足：

1. 同一个 `state_abs` 经数据加载器和在线 MATLAB 环境计算的
   `Q1/Q2/QT` 数值一致；
2. 当前状态和下一状态分别由 `state_abs`、`next_state_abs` 计算，
   没有错一拍；
3. 16D 数据 shape、dtype、NaN/Inf 检查通过；
4. 归一化和反归一化可逆；
5. 13D 模式的旧测试和旧 checkpoint 加载行为保持不变；
6. 16D checkpoint 不能误加载到 13D 环境，反之亦然；
7. 训练集和验证集仍按轨迹隔离。

## 8. 含噪/参数辨识场景的约束

本轮可以先在 `actor_obs_mode=truth` 下判断 16D 是否具有信息价值，但不能
据此声称含噪闭环已经改进。

如果后续用于 `noisy` 或 `filtered` 模式：

- `Q1/Q2/QT` 必须由 Actor 同一时刻实际可获得的量测或滤波状态计算；
- 不能把解析真值 `Q` 拼入含噪 Actor 观测；
- Actor 观测、控制器系数输入和诊断真值三条数据流继续严格分离；
- 初始观测与后续观测必须使用同一数据口径；
- 同一 noise seed 下参考控制器/策略对照必须逐拍可复现。

## 9. 明确不做的内容

本轮不做：

- 不重跑 IPOPT 或边界值求解器；
- 不重建完整参考数据集，除非现有数据确实缺少 `state_abs`；
- 不一次加入全部实体的完整状态、速度或 18 维绝对状态；
- 不同时修改奖励、动作空间和状态空间；
- 不直接开始 300k TD3/BC-TD3 长训；
- 不将验证集用于 scale、超参数或训练轮数选择；
- 不用单 seed 或单一总 MSE 宣布改进成功。

## 10. 相关历史文档

- `rl_td3/md document/0723gpt56sol改进方案提议(含参数辨识).md`
- `rl_td3/md document/SESSION_SUMMARY_20260719_BC_AND_RATIO_ACTION.md`
- `rl_td3/md document/SESSION_SUMMARY_20260718_EXPERT_DATA_AND_TD3_PLAN.md`
- `rl_td3/md document/CONTEXT_TD3_TRAINING.md`
- `rl_td3/md document/BC_TD3_USAGE.md`
- `.cursor/plans/td3稳定改进路线_782e8ebf.plan.md`
- `参数辨识新对话&新增实验清单md/PARAMETER_IDENTIFICATION_REFACTOR_TODO_20260915.md`

## 11. 交给新对话的任务说明

可直接向新 Agent 提交：

> 请按
> `rl_td3/md document/BC_TD3_STATE_MINIMAL_OPTIMIZATION_20260927.md`
> 实施。先完成 13D 与 `13D+Q1/Q2/QT` 的数据派生、一致性测试和
> 3-seed 纯 BC A/B，不要启动完整 TD3 长训，也不要同时修改奖励、
> 动作映射、网络结构或数据筛选。完成后按文档门槛给出继续/停止结论。

