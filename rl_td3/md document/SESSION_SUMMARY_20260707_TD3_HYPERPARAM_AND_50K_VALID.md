# TD3 超参数评审与 50k 验证对话总结（2026-07-07）

> **用途**：供后续对话快速恢复上下文。  
> **前置文档**：`rl_td3/SESSION_SUMMARY_20260706_TD3_STABILITY_FIX.md`（稳定性三项修复 + 首次冒烟）  
> **本文档范围**：超参数合理性分析 → 代码小改 → 冒烟复验 → 50k 中等验证（精简总结）

---

## 1. 背景与本次对话目标

在 2026-07-06 已完成三项稳定性修复（`ratio_cap`、梯度裁剪、`bc_ratio`）并通过 5000 步冒烟后，本次对话目标是：

1. 基于旧 500k TensorBoard 日志，评审当前超参数是否在正式训练前需要调整
2. 落实评审结论（`n_eval_episodes` 提升）
3. 先跑冒烟复验，再跑中等规模验证（最终定为 **50k**）
4. 根据验证结果，判断能否进入 500k 正式训练

**Python 解释器**（Agent 默认）：

```text
E:\Python\venvs\rl_pytorch\Scripts\python.exe
```

---

## 2. 旧 500k 训练复盘（修复前基线）

### 2.1 蒙特卡洛评估（`best_model_eval_0705.log`，100 轮）

| 指标 | 结果 |
|------|------|
| 双防规避成功率（d1,d2≥20m） | 86%（86/100） |
| 到达目标率（zT≤25m） | 58%（58/100） |
| 规避+到达全部达标 | 53%（53/100） |
| 被拦截率（r_MD<6m） | 3%（3/100） |

**关键异常**：

- 终端 ZEM `mean±std`：`2.55e15 ± 2.54e16`（离群值污染）
- 累计奖励 `mean±std`：`-9280 ± 54182`（方差极大）
- 动作 20 次采样完全一致：`gamma=-3, wT=1e4, wE=1e4`（策略 collapse 到边界常数动作）
- `best_model` 更像早期偶然优点，而非后期稳定收敛产物

### 2.2 TensorBoard 解析（旧训练日志）

**文件**：`rl_td3/checkpoints/tb_logs/TD3_1/events.out.tfevents.1783165488.LAPTOP-M4UJ842F.63188.0`

| 指标 | 实测（500k 全程） | 解读 |
|------|------------------|------|
| `rollout/ep_rew_mean` | -15090 ~ -2276，均值 **-4686** | 全程负值、无上升趋势 |
| `train/critic_loss` | 1.6e4 ~ **4.27e7**，均值 7.7e5 | 长期高位 + 脉冲，典型 Q 发散 |
| `train/actor_loss` | 206 ~ **9158**，均值 4286 | 持续正且大，未收敛 |
| `eval/mean_reward` | -13410 ~ **+565**，均值 -3978 | 几乎全负，450k 处偶发一次转正 |

**根因归纳**（与 07-06 文档一致）：

- `ratio_cap=30` 时目标惩罚项上限过大，放大 TD 目标方差
- 评估统计直接记录原始 `zT`，极端轨迹污染均值/标准差
- 未设置 `max_grad_norm`，Critic 易大梯度更新
- BC 预填充占比过高时，在线数据不足，策略易固化在边界动作

---

## 3. 已落地稳定性修复（07-06，本次对话沿用）

### 3.1 `rl_td3/matlab/env_step.m`

- `ratio_cap`：`30.0 → 5.0`（单步目标惩罚上限从 -900 压到 -25）
- 新增 `zT_info_cap = 1e5`（`info_vec` 统计裁剪，不影响终止判据）

### 3.2 `rl_td3/load_expert_buffer.py`

- `_compute_reward(..., ratio_cap=5.0)`，与 `env_step.m` 对齐

### 3.3 `rl_td3/train_td3.py`（07-06 修复 + 07-07 增补）

| 参数/功能 | 值/说明 |
|-----------|---------|
| `grad_clip` | `10.0`（`GradClipCallback`，元素级 clamp，SB3 无原生 `max_grad_norm`） |
| `bc_ratio` | `0.33`（BC 最多写入 99,000 / 300,000） |
| `n_eval_episodes` | **`5 → 20`**（07-07 本次对话修改，降低 eval 方差、提升 best_model 可信度） |
| CLI | `--grad-clip`、`--bc-ratio`、**`--n-eval`**（07-07 新增） |

**其余默认超参（未改）**：

```text
total_steps=500_000, lr=3e-4, batch_size=256, gamma=0.99, tau=0.005
buffer_size=300_000, learning_starts=0(有BC), train_freq=(1,"step")
policy_delay=2, net_arch=[256,256], noise_sigma=0.1
eval_freq=10_000
```

---

## 4. 超参数合理性评审结论（50k 验证前）

### 4.1 逐项评估

| 参数 | 当前值 | 结论 |
|------|--------|------|
| `learning_rate` | 3e-4 | **保持**；旧训练发散主因是奖励尺度，非 lr |
| `batch_size` / `gamma` / `tau` | 256 / 0.99 / 0.005 | **保持**，标准配置 |
| `buffer_size` + `bc_ratio` | 300k + 0.33 | **保持**；BC 数据在 ~201k 在线步后被 FIFO 挤出，形成天然退火 |
| `policy_delay` | 2 | **保持**（50k 后见改进建议） |
| `noise_sigma` | 0.1 | **保持** |
| `grad_clip` | 10.0 | **保持**，作为 Critic 安全网 |
| `bc_ratio` | 0.33 | **保持** |
| **`n_eval_episodes`** | **20**（原 5） | **必须改**；旧训练 eval 在 -13410↔+565 乱跳，best_model 靠运气 |

### 4.2 正式训练前必须调整 vs 可保持

- **必须调整**：`n_eval_episodes: 5 → 20`（已在代码中落实）
- **核心 RL 超参无需动**：lr / gamma / tau / batch / net_arch / policy_delay
- **已知近似（非阻塞）**：BC 预填充 reward 中 `a_M_norm=0`，不含能量项，与在线 `env_step.m` 有轻微偏移；BC 数据会被挤出，影响有限

### 4.3 训练耗时参考

旧日志 `fps` 均值约 **23.5**（MATLAB Engine 瓶颈）→ 500k 步约 **6 小时**。建议正式训练前先跑 **30k~50k** 中等验证。

---

## 5. 本次对话执行的验证（50k 前）

### 5.1 冒烟复验（5000 步）

**命令**：

```text
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py --smoke --no-log
```

**结果**：exit code `0`；BC 写入 98,999；GradClip 注册 36 个参数；输出 `rl_td3/checkpoints_smoke/td3_smoke.zip`

| step | ep_rew_mean | actor_loss | critic_loss |
|------|-------------|-----------|------------|
| 804 | -958 | 154 | 1,370 |
| 4853 | -1,050 | 161 | 1,260 |
| 4754 | -867 | 282 | 6,740 |
| 4927 | -867 | 282 | 6,740 |

**与 07-06 首次冒烟对比**：本次奖励未转正（-958→-867），但 `critic_loss` 全程 ~1e3–7e3，**无发散**（旧训练同期 2.8e4+）。随机种子不同导致曲线差异，**稳定性目标已达成**，可进入中等验证。

### 5.2 中等验证计划变更

- 用户最初要求 30k，后改为 **50k**
- 独立目录：`rl_td3/checkpoints_50k/`，运行名：`td3_50k_valid`

---

## 6. 50k 中等验证（精简总结）

### 6.1 运行配置

```text
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py \
  --steps 50000 --save-dir rl_td3/checkpoints_50k --name td3_50k_valid
```

- 总步数：50,000；BC：99,000（33%）；梯度裁剪：±10.0；评估：每 10k、20 episode
- 日志：`rl_td3/logs/td3_50k_valid_20260707_085815.log`
- 产物：`rl_td3/checkpoints_50k/td3_50k_valid.zip`、`best/best_model.zip`、TB `checkpoints_50k/tb_logs/TD3_1/`
- **耗时**：约 13.1 分钟（~72 fps，双 MATLAB Engine 启动后）

### 6.2 五个评估点汇总

| 步数 | eval/mean_reward | std | best_model 更新 |
|------|-----------------|-----|----------------|
| 10k | -998 | ±541 | ✓ |
| 20k | **-946** | ±562 | ✓ |
| 30k | -952 | ±429 | — |
| 40k | **-1110** | ±536 | —（最差） |
| 50k | -932 | ±700 | ✓（最终 best） |

### 6.3 整体结论

**通过项 ✓**

| 检查项 | 结果 |
|--------|------|
| `critic_loss` 稳定性 | 全程 < 2.5e4（旧峰值 4.27e7），三项修复有效 |
| 常数动作 collapse | 未观察到 `gamma=-3, wT=1e4, wE=1e4` 退化 |
| 训练流程 | exit code 0，完整跑通 |

**待解决问题 ✗**

| 现象 | 说明 |
|------|------|
| `actor_loss` 单调上升 | 154 → 875；Actor 持续"变悲观"，20k 后策略震荡退步 |
| eval 奖励震荡 | 20k 最优（-946）→ 40k 最差（-1110）→ 50k 略好（-932）但 std 最大（±700） |
| `critic_loss` 重复脉冲 | step ~28k、~40k 各出现 ~2.5e4 脉冲，可回落但未根除 |
| 奖励未转正 | 全程 `ep_rew_mean` 约 -900 ~ -1100，未见冒烟级别的单调上升 |

**与旧 500k 对比**：稳定性（critic 量级）已根本改善；**策略学习质量**仍不足，尚不能认定可直接上 500k。

---

## 7. 改进建议（正式 500k 前）

| 优先级 | 参数 | 当前 | 建议 | 原因 |
|--------|------|------|------|------|
| ⭐ 高 | Actor/Critic lr | 均为 3e-4 | **Actor 1e-4 / Critic 3e-4** | 抑制 Actor 漂移，TD3 常见做法 |
| ⭐ 高 | `policy_delay` | 2 | **3~4** | 给 Critic 更多收敛时间再更新 Actor |
| 中 | `noise_sigma` | 0.1 固定 | 后期衰减至 0.05（可选） | 减少后期无效探索 |
| 低 | `grad_clip` | 10.0 | 保持 | 已证明有效 |
| 低 | `bc_ratio` | 0.33 | 保持 | 退火机制合理 |
| 低 | `n_eval_episodes` | 20 | 保持或升至 30 | 进一步稳定 best_model 选择 |

**推荐下一步（三选一）**：

1. **改参后重跑 50k**（推荐）：验证 `actor_lr=1e-4`、`policy_delay=3` 是否缓解 actor_loss 上升与 eval 震荡
2. **对当前 best_model 跑 Monte Carlo**（100 ep）：看任务指标（规避率/到达率）是否优于旧模型，再决定是否调参
3. **直接上 500k**：仅当接受当前策略质量风险时

---

## 8. 关键文件索引

| 路径 | 说明 |
|------|------|
| `rl_td3/SESSION_SUMMARY_20260706_TD3_STABILITY_FIX.md` | 稳定性修复原始总结 |
| `rl_td3/train_td3.py` | 训练入口（含 GradClip、bc_ratio、n_eval=20） |
| `rl_td3/checkpoints/tb_logs/TD3_1/events...1783165488...` | **旧** 500k TB 日志 |
| `rl_td3/checkpoints_50k/tb_logs/TD3_1/` | **新** 50k 验证 TB 日志 |
| `rl_td3/checkpoints_50k/best/best_model.zip` | 50k 验证最优模型（eval -932） |
| `rl_td3/logs/td3_50k_valid_20260707_085815.log` | 50k 完整终端日志 |

---

## 9. 新对话续接提示（可直接复制）

```text
请读取以下文档恢复上下文：
- rl_td3/SESSION_SUMMARY_20260706_TD3_STABILITY_FIX.md
- rl_td3/SESSION_SUMMARY_20260707_TD3_HYPERPARAM_AND_50K_VALID.md

当前状态：稳定性修复已验证（critic_loss 可控），50k 中等验证已完成但策略学习仍震荡。

Python: E:\Python\venvs\rl_pytorch\Scripts\python.exe
```
