# P1-E2S16D：13D → 16D（+Q1/Q2/QT）状态扩展 · 可撤回实施记录

> 日期：2026-09-28 · 范围：truth 口径 only，**不碰含噪/辨识**。
> 前置结论（STAGE_CONCLUSION_20260928 §6-7）：E1V 未通过稳定加速门，
> 下一阶段先做 13D vs 16D 纯 BC A/B，不堆 TD3 机制。

## 0. 路径说明（文件夹搬家后）

- 本目录：`BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/`
- 交接：`BC-TD3性能改进实验_9月/P1_E1V_VANILLA_20260928/STAGE_CONCLUSION_20260928.md`
- 可学习性：`BC-TD3性能改进实验_9月/P1_E1V_VANILLA_20260928/LEARNABILITY.md`
- 收敛速度总纲：`rl_td3/md document/BC_TD3_CONVERGENCE_SPEED_OPTIMIZATION_20260927.md`
- 状态最小方案：`rl_td3/md document/BC_TD3_STATE_MINIMAL_OPTIMIZATION_20260927.md`
- 用法（部分过时，以本文件为准）：`rl_td3/md document/BC_TD3_USAGE.md`
- md 中旧路径 `P1_E1V_VANILLA_20260928/...` 一律加前缀
  `BC-TD3性能改进实验_9月/`；`P1_E1_BC_WARMSTART_20260927/...` 同理。

## 1. 16D 定义（与 stage2_step.m 同口径）

```text
obs16 = [obs13, Q1, Q2, QT]
Q1 = sqrt(qd_y1^2  + qd_z1^2  * cos(qy_D1M)^2)
Q2 = sqrt(qd_y2^2  + qd_z2^2  * cos(qy_D2M)^2)
QT = sqrt(qd_yTM^2 + qd_zTM^2 * cos(qy_TM)^2)   # T->M 口径
```

- 用 `computeMDDot` 解析结果，不用差分；返回原始 Q（不做 Q*tgo 钳制）；
- 由当拍 `state_abs[18]` 派生，`s` 用 `state_abs`、`s'` 用 `next_state_abs`，
  不错一拍，不用未来/统计量；
- T 静止：`T_pos=[0,0,0]`，`T_state=[0,0,0,0,0,0]`；
- Q scale：各 seed **训练 Actor 子集** 99.5% 分位，验证集不参与，只除不裁剪。

## 2. 可撤回性（已落实）

加法式设计，默认 13D 零行为变化：

| 文件 | 改动性质 |
|---|---|
| `rl_td3/obs_qmag.py`（新） | Q 派生/归一化/反归一化，13D 不 import |
| `rl_td3/matlab/stage2_qmag_from_abs.m`（新） | MATLAB Q 口径，不改旧函数 |
| `rl_td3/matlab/stage2_actor_obs16_from_abs.m`（新） | 16D truth 包装器；noisy/filtered 直接报错 |
| `rl_td3/tests/test_obs_qmag.py`（新） | 8 项验收 |
| `rl_td3/obs_norm.py` | 仅追加 3 个常量 |
| `rl_td3/load_expert_buffer.py` | `load_abs=False` 默认；旧返回字段不变 |
| `rl_td3/train_bc.py` | `--obs-schema` 默认 `rel13_v1`；`obs_dim` 跟随 schema |
| `rl_td3/bc_actor.py` | `validate` 追加 16D 分支；13D 分支原样 |
| `rl_td3/evasion_env.py` | `obs_schema` 默认 `rel13_v1`；16D 要求 truth+q_scale |

回退（按需逐级使用）：

```powershell
# L0（日常）：什么都不做——默认即 13D，16D 必须显式 opt-in 才生效。
# L1（停用 16D 代码路径，但保留文件备查）：无需操作；所有 16D 入口默认为关。
# L2（撤销本次 5 个跟踪文件的改动）：注意这 5 个文件在改动前已各有未提交修改，
#     执行前务必先备份当前工作区！
git diff -- rl_td3/obs_norm.py rl_td3/load_expert_buffer.py rl_td3/train_bc.py rl_td3/bc_actor.py rl_td3/evasion_env.py > E:\16D_revert_backup.patch
git checkout -- rl_td3/obs_norm.py rl_td3/load_expert_buffer.py rl_td3/train_bc.py rl_td3/bc_actor.py rl_td3/evasion_env.py
# L3（彻底清理，含新文件；同样先做 L2 的备份）
Remove-Item rl_td3/obs_qmag.py, rl_td3/tests/test_obs_qmag.py, rl_td3/matlab/stage2_qmag_from_abs.m, rl_td3/matlab/stage2_actor_obs16_from_abs.m
```

另：用户另有 git 备份；本目录 `PLAN_16D.md` + `derive_q_stats.py` 为实验记录，
删留随意，不影响 13D 主干。

## 3. 已通过验收（2026-09-28）

`rl_td3/tests/test_obs_qmag.py` 8/8：shape/有限/非负、确定性、
16D 前 13 维 ≡ 13D 归一化、归一化可逆、scale 为正、
13D/16D checkpoint 互斥、**Python vs MATLAB Q 一致（1e-9）**。

`test_bc_actor.py` 有 1 项失败（`val_actor_step_indices` 为空），经
stash 对照确认为**既有问题**（`expert_sampling.py` 未动过），与本次改动无关。

## 4. 进行中：只读 Q 预检

`derive_q_stats.py`（后台）：连续性抽查、Q 分布/先验 scale、溢出率。
只读专家库，不训练。结果回填 §5。

## 5. Q 预检结果（2026-09-28，`derive_q_stats.py`）

- 映射：1287 轨 / 136896 步，**首步差 max = 0.0**（`source_idx` 映射对齐，无错配）；
- Q 分布：Q1 p50=0.0076/p99.5=0.1695/max=902；Q2 p50=0.0087/p99.5=0.3665/max=358；
  QT p50=0.0056/p99.5=0.3097/max=130；
- 全局 99.5% 先验 scale = `[0.1695, 0.3665, 0.3097]`，
  已存 `q_scale_prior.npz`（正式实验各 seed 在训练集重估）；
- 先验 scale 下溢出：`|x|>1` obs13=0.08%/Q=0.50%；`|x|>5` obs13=0.00%/Q=0.16%；
- ⚠️ 全轨漂移 max=27057（tid=1026）：库记录态 vs 重放态在个别轨严重发散，
  `analyze_drift.py` 追查中（漂移分布/拦截相关/Q 极值来源），结论回填 §5.1。

### 5.1 漂移与极值追查（`analyze_drift.py` + `analyze_q_actor.py`，2026-09-28）

- clip 总量：12019/136896 步（8.8%）；拦截轨 307/1287，both_ok 轨 903/1287；
- 原始漂移：逐轨 max 中位 0.0066，但 198 轨 >1m、46 轨 >100m，
  几乎全是非拦截长轨（>100m 中拦截仅 1 条）；
- 归一化漂移（13 维，`/OBS_SCALE`）：步占比 >0.01=6.6%、>0.05=3.4%、
  >0.2=1.9%、>1.0=0.95%；轨级 p50=0.0058/p90=0.0119/p99≈2.0/max=2.0——
  **max 恰为 2.0 疑似角度跳变（±π）伪影，未做环绕处理，待 wrap-aware 复核**；
- actor（both_ok，79896 步/58.4%）Q：Q1 p99.5=0.120/max=76；
  Q2 p99.5=0.305/max=358；QT p99.5=0.0215/max=0.03；
  actor scale=`[0.1201, 0.3053, 0.0215]`；归一化后 `|x|>1`=0.50%、
  `|x|>5`=0.11%、max=1174（Q2 极值步，训练稳定性风险）；
- Q 已落盘复用：`q_all_aligned_order.npy`；报告：`q_actor_report.json`。

### 5.2 wrap-aware 复核（`wrap_recheck.py`，2026-09-28）——因果链锤死

- 复算 clip=11846 步 vs aligned 记录 12019（差 1.4%，口径一致）；
- wrap 后漂移>0.05 步占比 3.25%，>1.0 仅 0.004%——证实此前>1.0 几乎全是
  角度 `±π` 跳变伪影；
- 漂移>0.05 的步中 94.7% 有 clip 暴露；clip 暴露步中 14.4% 漂移超标；
  无 clip 暴露步中仅 0.22% 超标——**clip→重放分叉**因果成立；
- 可信步（wrap 漂移≤0.05）占比 96.75%，已存 `trusted_step_mask.npy`；
- **both_ok 步中可信占比 99.93%**——Actor BC 子集几乎不受漂移污染，
  16D 纯 BC A/B 无需清洗数据；漂移只影响 Replay 区（16D 进 TD3 时再议）；
- 遗留风险：actor 子集 Q2 极值归一化后 max=1174（0.11% 步>5），
  BC 冒烟时监控梯度/输入分布。报告：`wrap_report.json`。

## 6. 下一步（需用户确认，不擅自启动 >50k 训练）

1. `train_bc.py --smoke --obs-schema rel13_qmag16_v1` 烟雾（小预算）；
2. 13D vs 16D 纯 BC A/B：seed `0/21/42`，同 seed 同划分同动作标签，
   同网络/epoch/early-stopping；报告总 MSE、gamma MAE、符号准确率、
   贴边率、近邻歧义（13D vs 16D）；
3. 门槛（沿用 STAGE_CONCLUSION §7）：16D val MSE 平均 −15% 且
   gamma/符号/贴边明确改善且纯 BC 配对闭环更优，才谈 TD3；
4. 任何在线 TD3（即使 ≤50k）与 >50k 训练都先询问。

## 8. A/B 结果（2026-09-28，`AB_SUMMARY.md` / `ab_summary.json`）

| seed | 13D MSE | 16D MSE | ΔMSE | ΔγMAE | Δ符号acc | 近邻γ歧义 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.3168 | 0.2728 | −13.9% | −0.054 | +0.040 | 95.0%→95.0% |
| 42 | 0.2985 | 0.2762 | −7.5% | −0.041 | +0.011 | 94.0%→92.9% |
| 21 | 0.3039 | 0.2793 | −8.1% | −0.017 | +0.003 | 95.5%→93.2% |
| 均值 | 0.3064 | 0.2761 | **−9.9%** | 三 seed 同向 | 三 seed 同向 | 基本不变 |

对照门槛（§7/STAGE_CONCLUSION §7）：①均值 −15% ✗（−9.9%）；
②γ/符号/贴边改善 ✓（方向三 seed 一致，但幅度小；BC 贴边 0.1%→~1.2%，
距专家 ~55% 仍远）；③配对闭环 ✗（见 §10）；④一致性 ✓。
另：16D 最优 epoch 更早（14/10/11 vs 21/10/19）；`ab13_s0repeat_init42`
为误跑归档（seed-0 划分×init42，可作重复性参考，不计入 A/B）。

## 9. C项深挖结论（2026-09-28，`DIG_Q_AMBIGUITY.md` / `dig_q_report.json`，`dig_q_ambiguity.py`）

- C1：同query 13D-NN vs 16D-NN Jaccard均值0.36/0.40/0.32，Q距离占比中位~19%——邻居换了；
  但近邻γ极差>1几乎不动（95.0→95.0 / 94.0→92.9 / 95.5→93.2）——**Q不解歧义**；
- C2：pear(Q1/Q2,γ)|r|<0.02，QT仅+0.12~+0.14；Q分箱γ方差收缩0.7~2.3%；线性探针R²仅+0.001~+0.007；
- C3：ΔSE均值0.022~0.044，58~61%改善但22~29%大幅恶化，corr(ΔSE,|Q|)弱负——弥散小增益；
- C4：首层Q列范数比达1.2~2.6，Q置均值消融MSE 0.27→0.50~0.52（回退526~1104%）——**重度依赖+重尾（Q归一化std 4~8，max 1174），外推风险高**；QT在Actor子集近乎退化（std~0.16）；
- C5：BC贴边0.1%→~1.2%（专家~55%），precision 0.49~0.69但recall仅~0.01——贴边提高≠变好；
- **结论**：当前Q幅值口径非多解主因；建议按STAGE_CONCLUSION §7转向多模态/残差/动作重参化，不进TD3。

## 10. 纯 BC 配对闭环（2026-09-28，`PAIRED_CLOSED_LOOP.md`）

协议：`samples_clean_val.mat`、`order_seed=0`，与 P1/E0/E1/E1V 相同的
150 个固定 IC；truth、deterministic；同 IC 下比较各 seed 的 13D/16D best
checkpoint，无训练。13D seed 0/42 的 84.00%/83.33% 与 E1/E1V 的 t=0
完全一致，验证评估口径正确。

| seed | 13D both_ok | 16D both_ok | Δpp | 13成功→16失败 | 13失败→16成功 |
|---:|---:|---:|---:|---:|---:|
| 0 | 84.00% | 75.33% | −8.67 | 22 | 9 |
| 42 | 83.33% | 76.67% | −6.67 | 23 | 13 |
| 21 | 77.33% | 83.33% | +6.00 | 11 | 20 |
| 宏平均 | 81.56% | 78.44% | **−3.11** | 合计 56 | 合计 42 |

- 无 warm-start 失败；差异来自策略本身；
- seed 0 的配对 McNemar exact p=0.029，退化方向有较强证据；seed 42/21
  分别 p=0.132/0.150，单 seed 不作显著性宣称；
- 闭环改善明显 seed 依赖（2 退化 / 1 改善），且宏平均下降；
- **裁决：16D 未通过闭环门，不能因离线 MSE −9.9% 而进入 E0-16D 或
  16D TD3。** E0 本身最终成功率高不改变该裁决：当前问题是 Q 扩维对初始
  Actor 的净效应不稳定且平均为负，交给 E0 的 Replay/BC 正则去“修复”会混淆
  状态增益与训练机制的贡献，也不支持“加速收敛”的原目标。

## 11. 明确不做

- 不碰含噪（noisy/filtered）与辨识（ident/）任何逻辑；
- 不改奖励、动作映射、网络结构、数据筛选、TD3 超参数；
- 不重跑 IPOPT/重建专家库；验证集不参与 scale 选择；
- 未经用户确认不越门；当前 16D 已明确未过闭环门，不开 16D TD3。
