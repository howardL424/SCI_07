# TD3 训练 NaN 问题排查与修复总结

> **日期**：2026-07-04（总结自对话 Terminal output NAN evaluation issue）  
> **范围**：`evaluate_td3.py` 动作全 NaN → 追溯训练数值爆炸 → reward 标定与学习率调整  
> **关联文件**：`load_expert_buffer.py`、`matlab/env_step.m`、`train_td3.py`

---

## 1. 问题现象

### 1.1 评估阶段

运行：

```bash
python rl_td3/evaluate_td3.py --model rl_td3/checkpoints_smoke/td3_smoke --episodes 20
```

出现以下异常：


| 指标               | 异常值                 |
| ---------------- | ------------------- |
| 策略动作 `γ, wT, wE` | 全部为 `nan`           |
| 终端 ZEM 均值        | **135 亿米**（正常应在百米级） |
| 累计 reward 均值     | **-4.8×10²⁰**       |
| 双防+到达成功率         | 15%（在 NaN 动作下不可信）   |


### 1.2 训练阶段（smoke，修复前）

```
RuntimeWarning: overflow encountered in square
  r_target = -kT * (zT_n / np.maximum(zT0_all, 1.0)) ** 2

| actor_loss  | nan       |
| critic_loss | nan       |
| ep_rew_mean | -4.85e+20 |
```

模型权重检查：Actor **70077/70147** 参数为 NaN，Critic **100%** NaN。

---



## 2. 根因分析



### 2.1 直接原因：网络权重在训练早期变为 NaN

`model.predict()` 输出 NaN，是因为 Actor/Critic 在 BC 预填充后的前几百步更新中就已梯度爆炸，权重全部损坏。评估脚本本身无 bug，只是暴露了已损坏的 checkpoint。

### 2.2 根本原因：reward 目标项数值爆炸

奖励公式（BC 与在线环境共用）：

```
r_evade  = k1 · (log10(r_MD1) + log10(r_MD2))     # 典型量级 ~5–8
r_target = -kT · (zT / zT0)²                       # 可爆炸至 -inf
reward   = r_evade + r_target
```

专家库 **120115** 条 transition 的诊断数据：


| 统计量              | 修复前                     |
| ---------------- | ----------------------- |
| reward 有限比例      | 99.86%（含 0.14% nan/inf） |
| reward 最小值       | **-inf**                |
| reward < -1e6 占比 | **9.87%**               |
| `zT/zT0` 比值 p50  | 2.64                    |
| `zT/zT0` 比值 p95  | **1.55×10¹⁶**           |
| `(zT/zT0)²` p99  | **1.58×10³⁵**           |


**关键结论**：约 10% 的步骤 `zT/zT0` 比值极大，平方后 float32 溢出为 `-inf`。Critic 需拟合含 `-inf` 的 Q 值，配合 `lr=1e-3`，数百步内即 NaN。

### 2.3 为何单靠调小 kT 不够

配平分析显示：中位 `(zT/zT0)² ≈ 7`，与 `r_evade` 中位 ~7.35 相当，**kT≈1 在正常情况下是合理的**。

但 p99 处 `(zT/zT0)² ≈ 1.58×10³⁵`，即使 `kT=1e-6`，`r_target` 仍达 `1.58×10²⁹`。**必须先 clip 比值，再谈 kT 标定。**

### 2.4 BC 预填充 zT0 与 env_reset 不对齐（次要但需修复）

排查初期发现，BC 预填充与在线环境在 **zT0 基准** 上不一致，会放大 `(zT/zT0)²` 并加剧 reward 爆炸：

| 对比项 | BC 预填充（修复前） | 在线环境 `env_reset.m` / `env_step.m` |
|--------|---------------------|--------------------------------------|
| zT0 取值时刻 | 每条轨迹 `state_abs` 的 **首步** `chunk[0]` | warm-start 后 **规避段起点**（`stage==2` 第一个时刻） |
| zT0 计算公式 | `_target_zem_from_state_abs(chunk[0])` | `target_zem(M)`，`M` 为规避段起点攻击弹状态 |
| reward 中 zT0 下限 | `max(zT0, 1.0)` | `max(zT0, 1.0)`（与 BC 相同，但均 **过低**） |

**不对齐的具体表现**：

1. **语义不一致**：`env_reset.m` 明确在规避段起点（`stage==2`）计算 `zT0`；BC 侧若按「轨迹首步」理解，容易与在线 episode 的 zT0 基准脱节（用户亦明确要求「BC 预填充用规避段起点的 zT0，与 env_reset.m 对齐」）。
2. **归一化下限过低**：两侧虽都用 `max(zT0, 1.0)`，但大量轨迹 zT0 远小于 25 m（1103 条轨迹中 **5.1%** zT0 < 1 m，**73.2%** zT0 < 25 m）。以 1 m 为下限时，`(zT/zT0)²` 极易被放大；而到达判据为 `zT ≤ 25 m`，基准应与之对齐。
3. **BC 与在线 reward 尺度分裂**：同一物理态势下，BC buffer 中的 `r_target` 与 `env_step.m` 在线计算的 `r_target` 可能相差数个数量级，Critic 难以同时拟合两类 transition。

**修复方向**（见第 3 节）：

- BC 继续用 `chunk[0]` 计算 zT0（`extract_expert_transitions.m` 已从 `stage==2` 起点提取，`chunk[0]` 即规避段初态，与 `env_reset` 同源）。
- 两侧统一引入 **`zT0_floor = 25.0 m`**（与 `zT_reach` 一致），替代原先的 `max(zT0, 1.0)`。

### 2.5 学习率偏高

修正 reward 后，BC 数据 reward 分布：


| 百分位 | reward            |
| --- | ----------------- |
| p5  | **-892**（clip 边界） |
| p50 | +4.47             |
| p95 | +7.78             |


100 步累计 Q 估算：`min ≈ -90000`，`median ≈ +450`，量级跨越 10 万。`lr=1e-3` 下 smoke 测试 `actor_loss` 从 656 涨到 1760，Critic 不稳定。

---



## 3. 修改内容



### 3.1 `rl_td3/load_expert_buffer.py` — `_compute_reward()` 与 zT0 对齐


| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `kT` | 0.5 | **1.0** |
| `zT0` 下限 | `max(zT0, 1.0)` | **`max(zT0, 25.0)`**（与 `zT_reach` / `env_reset` 语义对齐） |
| 比值处理 | 无 | **`ratio = min(zT/zT0_safe, 30.0)`** |

zT0 仍由每条轨迹 **`chunk[0]`**（规避段起点 `state_abs`）经 `_target_zem_from_state_abs` 计算，与 `env_reset.m` 的 `target_zem(M)` 同源；变更在于 **归一化下限** 与 **比值 clip**，使 BC reward 与 `env_step.m` 一致。

```python
zT0_safe = np.maximum(zT0_all, zT0_floor)   # zT0_floor=25.0，替代原 max(zT0, 1.0)
ratio    = np.minimum(zT_n / zT0_safe, ratio_cap)  # ratio_cap=30.0
r_target = -kT * ratio ** 2
```



### 3.2 `rl_td3/matlab/env_step.m`

默认配置新增：

```matlab
d.kT         = 1.0;    % 原 0.5
d.zT0_floor  = 25.0;   % zT0 归一化下限
d.ratio_cap  = 30.0;   % zT/zT0 比值上限
```

奖励计算同步：

```matlab
zT0_safe = max(zT0, cfg_in.zT0_floor);
ratio_zT = min(zT_n / zT0_safe, cfg_in.ratio_cap);
r_target = -kT * ratio_zT^2;
```



### 3.3 `rl_td3/train_td3.py`


| 参数              | 修改前  | 修改后      |
| --------------- | ---- | -------- |
| `learning_rate` | 1e-3 | **3e-4** |


---



## 4. 修复后效果



### 4.1 BC 预填充 reward 分布


| 指标   | 修复前    | 修复后                  |
| ---- | ------ | -------------------- |
| 有限比例 | 99.86% | **100%**             |
| 最小值  | -inf   | **-895.2**（= -1×30²） |
| 最大值  | 8.5    | 8.5                  |
| 均值   | -inf   | **-94.4**            |




### 4.2 Smoke 训练（5000 步）— 无 NaN

**lr=1e-3（reward 已修）：**


| 步数   | actor_loss | critic_loss | ep_rew_mean |
| ---- | ---------- | ----------- | ----------- |
| 890  | 656        | 3.05e+04    | -1.34e+04   |
| 4350 | 1760       | 3.44e+04    | -1.30e+04   |


**lr=3e-4（最终配置）：**


| 步数   | actor_loss | critic_loss | ep_rew_mean   |
| ---- | ---------- | ----------- | ------------- |
| 539  | 592        | 3.53e+04    | -3.72e+03     |
| 4288 | 1440       | 3.77e+04    | **-3.05e+03** |


- 全程 **无 nan**
- Actor NaN 参数：**0 / 70147**
- `model.predict()` 输出有限值（早期策略倾向 `γ=-3, wT=100, wE=100`，属正常贪心规避）



### 4.3 lr 对比小结


| 指标                  | lr=1e-3         | lr=3e-4                 |
| ------------------- | --------------- | ----------------------- |
| actor_loss 增长（5k 步） | 656→1760（≈2.7×） | 592→1440（≈2.4×，更稳）      |
| ep_rew_mean         | ~-13000         | **~-3000**（Critic 信号更准） |


---



## 5. 设计参数说明


| 参数              | 值      | 含义                                             |
| --------------- | ------ | ---------------------------------------------- |
| `kT`            | 1.0    | 目标 ZEM 项权重；与 `r_evade` 中位量级配平                  |
| `zT0_floor` | 25.0 m | 与到达判据 `zT_reach=25m` 一致；**BC 预填充与 `env_step.m` 统一**，替代原 `max(zT0, 1.0)` |
| `ratio_cap`     | 30.0   | `r_target` 最差约 **-900**，强于拦截惩罚 -100，弱于失控时的 inf |
| `learning_rate` | 3e-4   | 适应 Q 值 -9e4 ~ +8e2 量级，减缓 Critic 发散             |


---



## 6. 后续建议

1. **启动正式训练**（500k 步）：
  ```bash
   E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_td3.py
  ```
2. **删除旧损坏 checkpoint**：修复前保存的 `td3_smoke.zip` / `td3_evasion_bc.zip` 权重已 NaN，不可复用。
3. **观察训练中期动作分布**：早期 Actor 可能饱和在最大规避权重；随在线数据增多，应逐渐学到对准目标。
4. **若 actor_loss 仍持续上升**：可进一步降至 `1e-4`，或对 reward 再做 `clip(-100, 100)` 硬截断。

---



## 7. 因果链总览

```
BC 与 env_reset 的 zT0 基准不一致（下限 1 m vs 设计意图 25 m）
    +
zT/zT0 比值无 clip，部分步骤极大 (p95≈10¹⁶)
    ↓
r_target = -kT·(zT/zT0)² 溢出为 -inf
    ↓
ReplayBuffer 含 -inf reward → Critic 拟合失败
    ↓
actor_loss / critic_loss = nan，权重损坏
    ↓
evaluate 输出动作全 nan，ZEM/reward 天文数字
```

**修复路径**：BC/在线统一 zT0 floor @ 25m + clip 比值 @ 30 + kT=1.0 + lr=3e-4 → reward 有界、BC 与在线尺度一致、训练稳定、无 NaN。