# P1/E1 BC Actor warm-start 实验结果与后续交接

> 更新日期：2026-09-28  
> 用途：供新对话直接开展下一阶段实验。  
> 当前状态：所有 P1/E1 训练已结束；没有后续实验在运行。

## 1. 实验范围

- 结果范围：两 seed 预筛（`0/42`），不作为统计显著性结论；
- 训练预算：每个条件 300,000 在线交互步；
- validation：每 25k 遍历同一组 150 个固定初态，并额外记录 `t=0`；
- observation：13D clean，未使用辨识；
- E0/E1 使用相同 TD3 超参数、初态池、评估池和动作映射；
- 第三个 seed 暂缓，未启动 P2/P3。

## 2. 条件定义

### E0：示范增强 TD3 基线

```text
随机 Actor / Actor target
+ 专家 transition 预填充 ReplayBuffer
+ 在线衰减 BC 正则
+ TD3 在线训练
```

E0 **没有纯 BC Actor 预训练**，更准确的名称是“专家 Replay + 在线 BC
正则的 TD3”。

### E1：E0 + BC Actor warm-start

```text
先用专家 (state, action) 监督训练纯 BC Actor
+ 用对应 seed 的 BC checkpoint 初始化 Actor 和 Actor target
+ 专家 transition 预填充 ReplayBuffer
+ 在线衰减 BC 正则
+ TD3 在线训练
```

E1 相对 E0 的唯一增量是 Actor/Actor target 的 BC 初始化。两 seed 的
BC Actor、TD3 Actor 和 Actor target 初始化后最大输出误差均为 `0`。

## 3. P1 结果

P1 未通过既定收敛加速门槛：

- E1 相对 E0 的 0–150k 平均 AUC 相对增益为 **3.47%**，低于 5% 门槛；
- 首次达到 90% 的平均步数差为 **+25k**，即 E1 平均更晚；
- 300k 平均 `both_ok` 差为 **0 个百分点**，最终性能均值未下降；
- 两个 seed 的前期响应不一致，不能认为 BC warm-start 稳定加速。

逐 seed：

- seed 0：90% 首达由 150k 提前到 75k；AUC 相对提高 6.22%；
  300k `both_ok` 由 96% 降至 94%；
- seed 42：90% 首达由 75k 推迟到 200k；AUC 相对提高 0.73%；
  300k `both_ok` 由 93.33% 提高到 95.33%。

详细逐点曲线、墙钟时间和初始化 hash 见 `summary.json`。

## 4. 术语澄清：什么是正常 BC 预训练

最基础的行为克隆预训练只包含：

1. 准备专家 `(state, action)` 数据；
2. 使用与在线 Actor 相同的观测预处理、动作映射和网络结构；
3. 以 `MSE(pi(s_E), a_E)`（确定性策略）监督训练 Actor；
4. 用 validation 选择 checkpoint；
5. 将 checkpoint 加载到 RL Actor；
6. 对 TD3 同步 Actor target，然后开始标准 RL 微调。

以下均不是 BC 预训练本身，而是可选的后续增强：

- 专家 transition 预填充 ReplayBuffer；
- RL 阶段继续使用 BC 正则；
- Critic-only 离线预热；
- Q-filter、Replay 比例退火、Q 尺度归一化；
- 残差策略、探索噪声调度或状态扩展。

BC 预训练阶段的 BC loss 是主要监督目标；只有在 RL 阶段把 BC loss
加到 Actor RL loss 上时，才称为“在线 BC 正则”。因此 BC 正则不是
BC 预训练的必需组成部分。

## 5. 当前实验回答了什么、没有回答什么

当前 E0/E1 回答的是：

> 在“专家 Replay + 在线 BC 正则”的 E0 基础上，再增加 BC Actor
> warm-start 是否能够稳定加快收敛？

两 seed 预筛答案是：**没有证明稳定加速**。

当前实验没有回答：

> 最朴素的“纯 BC Actor 初始化 + 标准 TD3”是否优于 E0/E1？

原因是 E1 仍同时包含专家 Replay 预填充和在线 BC 正则，二者可能保护
BC 策略，也可能因分布失配或标签误差妨碍 TD3。

## 6. 纯 BC 初始化 + 标准 TD3 是否可能更好

目前只能说**有待检验**，不能由 E0/E1 直接推断。

可能更好的原因：

1. 去掉在线 BC 正则后，TD3 不再持续锚定拟合误差较高或多解的专家动作；
2. 去掉专家 Replay 后，Critic 主要学习当前 BC 策略实际访问的在线分布，
   可能减少专家动作与当前动作之间的外推误差；
3. 标准 `learning_starts=1000` 会先积累在线 transition，再开始更新，
   不会像 E1 一样从第 0 步立即用随机 Critic 更新 Actor。

也可能更差的原因：

1. 没有在线 BC 正则保护，随机 Critic 可能更快破坏 BC Actor；
2. 没有专家 Replay，Critic 缺少成功轨迹和稀有高奖励样本；
3. BC Actor 本身存在较明显拟合误差和离线到闭环的分布偏移。

因此该条件必须单独实验，不能假设一定优于 E1。

## 7. 专家数据质量与 BC 可学习性

当前正式 BC checkpoint：

- seed 0：best validation MSE `0.3168`；
- seed 42：best validation MSE `0.2985`。

但 E1 在 `t=0` 的两 seed 平均 `both_ok` 比 E0 随机 Actor 高约
4.33 个百分点，说明 BC 数据中确实含有有用策略信息，只是没有被稳定利用。

“专家轨迹成功”不等于“数据适合 BC”。后续需要区分：

- 专家闭环控制质量；
- 13D observation 是否足以唯一决定动作；
- 相似状态下是否存在多组差异较大的可行动作；
- 专家数据是否覆盖 BC 策略偏离专家轨迹后的状态；
- 各动作维度的误差、方差、贴边率和样本不平衡；
- transition/reward 质量——它影响专家 Replay，但不直接影响纯 `(s,a)` BC。

当前高 validation MSE 可能来自部分可观测、多解或动作平均化，不能简单归因
为专家轨迹本身质量差。

## 8. 修订后的下一步：先做干净基线，不直接进入 P2

新增条件建议命名为 **E1V（vanilla BC warm-start + TD3）**，避免占用
原方案中已定义为 Critic 预热的 E2：

```text
对应 seed 的纯 BC checkpoint
→ 初始化 Actor 和 Actor target
→ 不预填专家 Replay
→ 不设置在线专家数据集
→ bc_lambda0 = 0
→ Critic 随机初始化
→ learning_starts = 1000
→ 标准 TD3 在线训练
```

当前入口原则上可直接表达该条件：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py `
  --no-bc `
  --bc-lambda0 0 `
  --bc-init-ckpt <对应 seed 的 bc_seed*_best.pt> `
  --steps <预算> `
  --seed <seed> `
  --save-dir <新的独立结果目录> `
  --name <独立名称>
```

正式运行前仍需用冒烟确认 `--no-bc` 与 `--bc-init-ckpt` 的组合路径、
初始化元数据、`learning_starts=1000` 和成功率日志均符合预期。

## 9. 建议的分级实验

### 第一级：技术冒烟

- seed 0；
- 1k 左右在线步；
- 只检查代码路径、模型保存、成功率记录、初始化一致性和有限值；
- 不用于判断算法效果。

### 第二级：50k 趋势预筛

- seed 0/42；
- 评估点 `0/25k/50k`，继续使用相同的 150 个 fixed validation IC；
- 直接与现有 E0/E1 的相同评估点配对比较；
- 若两个 seed 均无改善，或出现明显崩溃，则停止 E1V；
- 若平均 0–50k AUC 有改善且没有单 seed 明显退化，再进入 200k。

### 第三级：200k 两 seed 预筛

- 主要指标仍为 90% 首达步数和 0–150k AUC；
- 若 E1V 相对 E0/E1 的平均 AUC 提高、跨 seed 方向更一致且 200k
  性能无明显下降，再决定是否延长到 300k；
- 只有 200k 通过后才运行 300k，不再从极小冒烟直接跳到 300k。

## 10. E1V 结果如何决定后续方向

1. **E1V 优于 E1，且 seed 42 的前期退化消失**  
   说明专家 Replay、在线 BC 正则或其交互可能是主要问题。优先保留干净
   BC→TD3 基线，再做最小单因素消融，不直接进入 P2。

2. **E1V 与 E1 相近或更差，且 BC 初始优势在开始 TD3 更新后消失**  
   随机 Critic 破坏 Actor 的解释更可信，再进入原方案 P2：固定 10k 次
   Critic-only 预热。

3. **E1V 从 `t=0` 就较差**  
   先诊断 BC checkpoint、状态充分性和专家动作可学习性，不做 P2/P3。

4. **P2 仍无效**  
   停止堆叠 Q-filter、P3 或大规模超参数搜索，转向纯 BC 表示能力和
   专家数据诊断。

## 11. 当前明确不做

- 尚未启动 E1V；
- 尚未启动 P2 Critic 预热；
- 不补第三个 P1 seed；
- 不进入 P3、Q-filter、Replay 退火或 16D 状态扩展；
- 不使用辨识。
