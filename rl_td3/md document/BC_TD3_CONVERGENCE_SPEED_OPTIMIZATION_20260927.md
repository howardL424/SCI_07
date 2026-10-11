# BC-TD3 收敛速度最小改进与分级实验方案

> 日期：2026-09-27  
> 目标：在不改变任务定义、奖励函数和正式评估口径的前提下，优先提高
> BC-TD3 的前期收敛速度，并判断行为克隆是否真正降低了在线交互成本。  
> 原则：一次只增加一个机制；先证明加速，再讨论最终性能和状态扩展。

## 1. 当前问题与改进目标

当前 BC-TD3 使用：

- 专家 transition 预填充 ReplayBuffer；
- 独立专家 `(state, action)` 数据上的衰减 BC 正则；
- 随后进行 TD3 在线训练。

但 `train_td3.py` 当前是先随机创建 `TD3BC` Actor，再执行专家 Replay
预填充和 `set_expert_dataset()`。项目虽然已有
`bc_actor.py::copy_bc_weights_to_td3_actor()`，正式训练入口尚未把纯 BC
checkpoint 权重加载到 `model.actor` 和 `model.actor_target`。

因此，当前实现更准确地说是“专家 Replay + BC 正则”，还不是完整的
“BC Actor 初始化 + TD3 微调”。

现有同协议单 seed 训练曲线中，纯 TD3 前期并不慢于 BC-TD3：

- 25k 步平均评估奖励：纯 TD3 约 `109.24`，BC-TD3 约 `93.62`；
- 75k 步平均评估奖励：纯 TD3 约 `118.15`，BC-TD3 约 `99.18`；
- BC-TD3 到约 225k 步才达到该次训练的较高奖励区。

这些数据不足以形成多 seed 结论，但足以说明：现有流程尚未证明 BC 能够
加速收敛。

本方案的主要目标不是单纯提高 300k 最终成功率，而是：

1. 减少达到目标成功率所需的在线环境交互步数；
2. 提高 0～150k 步内的成功率曲线面积；
3. 在加速前期训练的同时，不明显损害 300k 最终性能。

## 2. 总体优先级

### P0：实验协议与加速指标固化——必须先完成

这是所有对照实验的前置条件，不属于算法改进，但不得跳过。

必须固定：

- 训练 seed：筛选阶段至少 `0/21/42`；
- 专家轨迹 train/validation 划分和采样计划；
- 在线训练初态池及采样规则；
- 固定 validation 初态及顺序；
- 网络结构、奖励、动作映射、训练预算和评估频率；
- 每 25k 步评估同一组 validation 初态；
- test 初态只用于最终冻结模型的一次性评估。

需要新增或统一记录：

- 每个评估点的 `both_ok / evade_ok / strike_ok`；
- 达到目标成功率的首次训练步数；
- 0～150k 成功率曲线面积；
- 在线环境交互步数、离线梯度更新次数和墙钟时间；
- Actor、Actor target 初始化前后的输出一致性；
- `actor_loss / critic_loss / bc_loss / bc_lambda / Q` 尺度；
- 配置、代码版本、数据 hash 和专家采样计划。

若只记录平均奖励而不记录成功率，不能用于本方案的主要验收。

### P1：真正使用纯 BC Actor warm-start——最高优先级

这是改动最小、机制最直接、应首先实施的尝试。

实施内容：

1. 为 `train_td3.py` 增加可选参数，例如 `--bc-init-ckpt`；
2. 使用 `load_bc_checkpoint()` 加载纯 BC Actor；
3. 使用 `copy_bc_weights_to_td3_actor()` 写入 `model.actor`；
4. 将 `model.actor` 完整同步到 `model.actor_target`；
5. 验证同一批观测上，BC Actor、TD3 Actor、TD3 Actor target 的输出
   在浮点误差范围内一致；
6. checkpoint 中记录 BC 初始化文件、hash、epoch、观测维数、
   观测 schema 和动作映射；
7. 不改变 Critic、Replay 采样、BC 正则和其他 TD3 超参数。

注意：

- 只复制 `model.actor` 而不复制 `model.actor_target` 是错误实现；
- 权重必须在模型和优化器创建后原位写入参数，不能替换为未被优化器引用的
  新模块；
- 13D checkpoint 不能加载到 16D Actor；
- 动作映射不一致时必须拒绝加载；
- 当前纯 BC 拟合误差较高，因此 warm-start 必须保留开/关对照，不能假定
  一定有效。

P1 的目的仅是回答：实际使用 BC 权重初始化后，是否比当前
“随机 Actor + BC 正则”更快。

### P2：BC Actor 初始化后的 Critic 预热——高优先级

若 Actor 已由 BC 初始化，而 Critic 仍为随机网络，训练初期的错误 Q 梯度
可能迅速破坏 BC 策略。因此第二优先级是 Critic-only 离线预热。

建议实现独立的 Critic 预热函数，不直接复用当前同时更新 Actor/Critic 的
`TD3BC.train()`：

1. 保持 Actor 和 Actor target 固定；
2. 从专家 Replay transition 采样；
3. 只更新两个 Critic；
4. target action 使用已经同步的 BC Actor target；
5. 预热完成后，将 Critic 硬同步到 Critic target；
6. 再开始正式在线交互和常规 TD3 更新。

筛选阶段仅测试一个中等预算，例如 `10k` 次 Critic 梯度更新。若有正向
趋势，再比较 `5k/10k/20k`，不得一开始进行大网格搜索。

必须单独报告离线 Critic 更新次数和耗时，不能仅按在线环境步数宣称整体
训练时间更短。

### P3：修正 RL 项与 BC 项的相对尺度——中优先级

当前 Actor 损失为：

```text
L_actor = -mean(Q(s, pi(s))) + lambda(t) * MSE(pi(s_E), a_E)
```

固定 `lambda0=1` 并不能保证 Q 项与 BC 项在不同训练阶段具有可比尺度。
BC 过强可能长期锚定有误差的专家标签，BC 过弱又不能保护初始化策略。

P3 按以下顺序尝试，不应同时启用多个新机制：

1. **优先尝试 Q 尺度归一化**  
   使用停止梯度的 `mean(abs(Q))` 对 RL Actor 项归一化，并继续保留原有
   线性 BC 衰减。记录归一化前后的两项损失尺度。

2. **再扫描较短 BC 衰减周期**  
   只比较 `50k/100k/200k`，其余参数不变。若 P1/P2 已提供良好初始化，
   不一定仍需持续 200k 的强 BC 约束。

3. **Q-filtered BC 仅作条件尝试**  
   只有在 Critic 预热后才允许根据
   `Q(s_E,a_E) > Q(s_E,pi(s_E))` 决定是否施加 BC。Critic 未可靠前禁止
   使用该门控。

若 Q-filter 后有效专家 batch 长期过少或剧烈振荡，应立即停止该分支。

### P4：通过状态扩展提高 BC 初始化质量——条件优先级

`13D + Q1/Q2/QT` 的 16D 状态扩展只在下列条件满足后进入收敛速度实验：

1. 3-seed 纯 BC A/B 中，16D validation MSE 平均至少下降 15%；
2. `gamma` MAE 或符号/贴边预测有明确改善；
3. 16D 纯 BC 的配对闭环结果优于 13D；
4. 离线和在线观测的一致性测试通过。

状态扩展的预期作用是提高 BC warm-start 质量，而不是直接保证 TD3 最终
成功率上升。

在 P1～P3 尚未完成前，不应把 16D 与 Critic 预热、自适应 BC 等机制同时
加入正式训练，否则无法归因收敛加速来自哪里。

### P5：可选后续优化——暂不进入最小实验

仅当 P1～P3 已证明方向有效但加速幅度不足时，再依次考虑：

- 独立专家 Replay 与在线 Replay，并显式退火专家 batch 比例；
- BC 初始化后的探索噪声调度；
- 残差策略 `pi = pi_BC + delta_pi` 及逐步放宽残差范围；
- Actor/Critic 分离学习率；
- 每环境步多次梯度更新；
- 多步回报或优先级 Replay。

这些方案会改变优化计算量或训练结构，不得与最小实验同时实施。

## 3. 最小实验序列

纯 TD3 使用相同 seed、初态和评估协议，作为所有曲线中的外部参照。
BC-TD3 按以下顺序实施：

### E0：当前 BC-TD3 基线

- 随机 Actor；
- 专家 Replay 预填充；
- 固定形式的衰减 BC 正则；
- 无 Critic 预热。

必须先用当前代码复现，不能直接拿旧单 seed 曲线代替多 seed 基线。

### E1：仅增加 Actor BC warm-start

相对 E0 唯一变化：

- Actor 和 Actor target 从对应 seed 的纯 BC checkpoint 初始化。

其余完全不变。

### E2：Actor warm-start + Critic 预热

相对 E1 唯一变化：

- 在线训练前增加固定预算的 Critic-only 专家 Replay 预热。

第一轮固定使用一个预算，不同时扫描预热步数。

### E3：E2 + Q 尺度归一化 BC

相对 E2 唯一变化：

- 归一化 Actor RL 项与 BC 项的相对尺度。

第一轮不同时加入 Q-filter、Replay 比例退火、噪声调度或 16D 状态。

若 E3 显示 BC 锚定仍持续过久，再进行 E3a 衰减周期小扫描；Q-filter
属于 E3b 条件分支。

## 4. 公平对照与数据隔离

所有 E0～E3 必须满足：

- 使用同一组训练 seed；
- 每个 seed 使用相同专家轨迹划分和采样索引；
- 使用相同在线训练初态序列；
- 使用相同网络结构、batch size、动作噪声和奖励；
- 使用相同在线环境交互预算；
- validation 用于选择 checkpoint 和实验分支；
- test 不参与超参数、预热步数或 BC 衰减周期选择；
- 报告全部预先约定 seed，不得只选择最优 seed；
- 若增加离线更新，必须同时报告交互样本效率和总计算成本。

纯 BC checkpoint 也必须按 seed 配对生成，不能三个 TD3 seed 共用一个
经过特定划分训练的 seed=0 BC checkpoint。

## 5. 主要指标与验收门槛

### 5.1 主要指标

1. **达到 90% validation `both_ok` 的首次在线训练步数**；
2. **0～150k 步 validation `both_ok` 曲线面积**；
3. 50k、100k、150k、300k 的配对 `both_ok`；
4. 至少 3 个 seed 的均值、标准差和逐 seed 配对差值。

若部分方法始终未达到 90%，其达标步数记为超过训练预算，不得删除。

### 5.2 次要指标

- 平均奖励及标准差；
- `evade_ok / strike_ok / intercepted`；
- Critic loss、Q 值范围和 TD error；
- BC loss、有效 BC 样本比例和动作饱和率；
- 动作总变差、`gamma` 切换次数；
- 环境交互耗时、离线更新时间和总墙钟时间。

### 5.3 筛选阶段通过条件

某个新增机制进入下一优先级，至少应满足：

1. 3 个 seed 中至少 2 个 seed 的前期收敛方向改善；
2. 平均达到 90% 成功率的步数至少提前一个评估间隔，即 25k 步；
3. 或 0～150k 成功率曲线面积相对提高至少 5%；
4. 300k 最终 `both_ok` 不比 E0 平均下降超过 1 个百分点；
5. 没有明显 Q 爆炸、Actor 崩溃或只在单 seed 有效。

筛选通过后，再用至少 5 个 seed 做确认实验和置信区间；3-seed 结果只用于
方案筛选，不用于宣称统计显著。

## 6. 分阶段停止规则

### P1/E1 后

- 若 E1 明显加速且后期稳定：继续 E2；
- 若 E1 初始好但很快退化：仍继续 E2，优先判断随机 Critic 是否破坏策略；
- 若 E1 从初始阶段就差于 E0：检查 BC checkpoint、动作映射、Actor target
  同步及纯 BC 闭环质量，不直接调 TD3 超参数。

### P2/E2 后

- 若 E2 相对 E1 恢复或保持 BC 初始策略，并提高前 100k 表现：进入 P3；
- 若 E1/E2 均无加速：停止 Critic 预热扩展，转向纯 BC 表示能力诊断，
  不进行大规模 λ 和学习率搜索；
- 若只有增加大量离线更新才有效：结论应表述为提高交互样本效率，而不是
  降低总训练成本。

### P3/E3 后

- 若前期加速且最终性能不降：冻结方案，进入 5-seed 确认；
- 若前期改善但后期被 BC 锚定：只做 BC 衰减周期小扫描；
- 若自适应尺度无效：停止该分支，不同时追加 Q-filter 和噪声调度。

### P4/16D 前

只有纯 BC 13D/16D 离线及闭环 A/B 通过既定门槛，才允许将 16D 加入
E1～E3 中表现最好的训练流程。

## 7. 最小实现范围

### `rl_td3/train_td3.py`

- 增加 BC Actor 初始化参数及配置记录；
- 在专家数据加载完成后、在线训练前执行权重拷贝和一致性检查；
- 同步 Actor target；
- 增加 Critic 预热入口及预热统计；
- 不改变默认行为，未提供新参数时仍复现 E0。

### `rl_td3/bc_actor.py`

- 复用现有 checkpoint 加载和 Actor 权重拷贝函数；
- 增加动作映射、观测 schema、归一化版本的兼容检查；
- 必要时增加同时同步 Actor target 的小型辅助函数。

### `rl_td3/td3_bc.py`

- 增加 Critic-only 预热实现；
- 增加可选 Q 尺度归一化模式；
- 保持当前 fixed-decay 模式作为 E0/E1/E2 基线；
- 记录 RL Actor 项和 BC 项各自的未加权尺度。

### `rl_td3/tests/`

至少新增：

1. BC Actor、TD3 Actor、Actor target 权重和输出一致性测试；
2. obs/action schema 不兼容时拒绝加载测试；
3. Critic 预热期间 Actor 权重不变化测试；
4. Critic 预热后 Critic target 同步测试；
5. Q 尺度归一化有限值和梯度测试；
6. 默认参数仍保持旧训练行为的回归测试。

## 8. 暂不同时修改的内容

最小实验期间不修改：

- 奖励函数和成功判据；
- 动作映射；
- 网络宽度和深度；
- 专家轨迹筛选规则；
- 在线任务初值范围；
- 非合作实体性能；
- 状态维数；
- TD3 基础超参数；
- 噪声、辨识和含噪 Actor 观测口径。

状态扩展按 P4 单独进入，任务难度扩展属于后续泛化和鲁棒性实验，不属于
本轮收敛速度机制验证。

## 9. 推荐执行顺序

```text
P0  固化协议、成功率曲线和运行元数据
  -> E0  3-seed 当前 BC-TD3 基线复现
  -> P1/E1  仅增加 BC Actor + Actor target 初始化
  -> P2/E2  再增加固定预算 Critic-only 预热
  -> P3/E3  再增加 Q 尺度归一化
      -> 未通过：按停止规则结束，不堆叠更多机制
      -> 通过：5-seed 确认
          -> P4 仅在纯 BC 16D A/B 通过后测试状态扩展
```

最高优先级结论：先把现有代码中已经具备但尚未接入正式训练入口的纯 BC
Actor 权重真正用于 TD3 Actor 和 Actor target 初始化；随后再判断是否需要
Critic 预热和 BC 损失尺度修正。不要从大规模超参数搜索或提高任务难度开始。

## 10. 2026-09-28 P1 两 seed 结果后的执行修订

> 本节根据 seed `0/42` 的 P1/E1 实测结果补充，覆盖“P1 后立即进入 P2”
> 的默认执行顺序；原 P2 定义和实现要求仍保留。

术语澄清：

- 正常 BC 预训练仅指用专家 `(state, action)` 监督训练 Actor，并将其权重
  加载到 RL Actor；TD3 还需同步 Actor target；
- 专家 Replay 预填充、在线 BC 正则和 Critic-only 预热均为可选增强，不是
  BC 预训练本身；
- E0 没有 Actor BC 预训练；E1 有 Actor BC 预训练，但仍叠加了专家 Replay
  和在线 BC 正则，因此不是最朴素的“BC 初始化 + 标准 TD3”。

P1 两 seed 预筛结果：

- 0～150k 平均 `both_ok` AUC 相对增益为 3.47%，低于 5% 门槛；
- 90% 首达步数平均晚 25k；
- 300k 最终 `both_ok` 均值不变；
- seed 0 加速、seed 42 退化，未证明稳定加速。

在进入 P2 前，先增加不占用 E2 编号的干净基线 **E1V**：

```text
纯 BC Actor checkpoint
  -> 初始化 Actor 与 Actor target
  -> 不预填专家 Replay
  -> 不使用在线 BC 正则
  -> Critic 随机初始化
  -> learning_starts=1000
  -> 标准 TD3 在线训练
```

修订后的顺序：

```text
P1/E1 已完成
  -> E1V 技术冒烟
  -> E1V 50k、seed 0/42 趋势预筛
      -> 无改善：停止 E1V，结合退化时点决定是否进入 P2
      -> 有改善且跨 seed 稳定：E1V 200k 两 seed 预筛
          -> 通过：再决定是否延长 300k
          -> 开始 TD3 更新后仍迅速退化：进入 P2/E2 Critic-only 预热
```

在 E1V 结果出来前，不启动 P2/P3、Q-filter、第三个 P1 seed、16D 状态扩展
或辨识实验。详细结果和新对话交接见
`P1_E1_BC_WARMSTART_20260927/RESULTS_0927.md`。
