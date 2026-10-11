# 16D（13D+Q1/Q2/QT）代码结构目录与使用说明（BC-TD3 / TD3）

> 目的：给新对话一份“16D 相关实验”的接手文档：改了哪些文件、BC 怎么跑、
> TD3/BC-TD3 侧还缺什么、产物在哪、禁区是什么。
> 日期：2026-09-28。解释器一律用 `E:\Python\venvs\rl_pytorch\Scripts\python.exe`，
> 不要用裸 `python`（见工作区规则）。

## 0. 状态快照（先读结论，避免重复踩坑）

- 16D 定义：`obs16 = [obs13, Q1, Q2, QT]`，见 §1。
- **纯 BC 16D 已可用**（`train_bc.py --obs-schema rel13_qmag16_v1`）。
- **TD3/BC-TD3 的 16D 尚未接线**：`rl_td3/train_td3.py` 与
  `rl_td3/evaluate_td3.py` 当前只建 13D 环境、无 `--obs-schema/--q-scale`
  参数（细节见 §6）。**不要直接拿 16D checkpoint 去跑现有 TD3 训练/评估脚
  本，会因 `obs_dim` 不一致被拒绝。**
- 实验结论（`PLAN_16D.md` §8–§10）：13D vs 16D 纯 BC A/B 为均值 −9.9%
 （未达 −15% 门槛）；深挖（`DIG_Q_AMBIGUITY.md`）证实 Q 幅值信息量极低且
  网络对其重度依赖；固定 150 IC 配对闭环宏平均 −3.11pp（2 退化/1 改善）。
  **16D 未过闭环门，不开 16D TD3 / E0-16D。**

## 1. 16D 定义（与 `stage2_step.m` 同口径，truth only）

```text
obs16 = [obs13, Q1, Q2, QT]
Q1 = sqrt(qd_y1^2  + qd_z1^2  * cos(qy_D1M)^2)
Q2 = sqrt(qd_y2^2  + qd_z2^2  * cos(qy_D2M)^2)
QT = sqrt(qd_yTM^2 + qd_zTM^2 * cos(qy_TM)^2)   # T->M 口径
```

- 用 `computeMDDot` 解析结果，不用差分；**返回原始 Q，不做 `Q*tgo` 钳制**；
- 由当拍 `state_abs[18]` 派生：`s` 用 `state_abs`、`s'` 用 `next_state_abs`，
  不错一拍，不用未来/统计量；T 静止（`T_pos=[0,0,0]`，
  `T_state=[0,0,0,0,0,0]`）；
- Q scale：各 seed **训练 Actor 子集 99.5% 分位**，验证集不参与，
  只除不裁剪（`Q_SCALE_QUANTILE = 0.995`）；
- **仅 truth**：16D 要求 `actor_obs_mode='truth'`；noisy/filtered 直接报错
  （防把真值 Q 拼入含噪观测，Q 口径待定义）。

## 2. 代码结构（16D 相关文件一览）

```text
rl_td3/
  obs_qmag.py                  # 新增：Q 派生/归一化/反归一化（13D 默认不 import）
  obs_norm.py                  # 追加 3 个 16D 常量（13D 逻辑不动）
  load_expert_buffer.py        # 加 load_abs 开关（默认 False，旧返回字段不变）
  train_bc.py                  # 加 --obs-schema（默认 rel13_v1），obs_dim 跟随 schema
  bc_actor.py                  # validate 加 16D 分支；13D 分支原样；13D/16D checkpoint 互斥
  evasion_env.py               # 加 obs_schema/q_scale（默认 13D）；16D 要求 truth+q_scale
  td3_bc.py                    # 未动：TD3BC 模型（obs 维数随 env 自适应；见 §6 的 16D 缺口）
  train_td3.py                 # 未动：仍 13D 硬编码（见 §6）
  evaluate_td3.py              # 未动：EvasionEnv() 无 schema 参数（见 §6）
  fixed_ic_eval.py             # 未动：与 obs 维数无关，16D 可复用（150 IC 固定评估）
  expert_sampling.py           # 未动：采样计划被 13D/16D 共用（配对关键）
  action_mapping.py            # 未动：ratio_log_box_v1
  matlab/
    stage2_qmag_from_abs.m         # 新增：MATLAB Q 口径，不改旧函数
    stage2_actor_obs16_from_abs.m  # 新增：16D truth 包装器；noisy/filtered 直接报错
  tests/
    test_obs_qmag.py           # 新增：8 项验收（§4）
  data/
    expert_transitions_aligned.mat  # 对齐表（1287 轨/136896 步，与 P1 同源）
    expert_terminal_cache.npz       # both_ok 划分
    expert_set_99k.npz              # seed0 默认 99k 采样计划（ab13_s0 配对复用它）
```

### 各文件 13D/16D 行为对照

| 文件 | 默认（13D） | 16D opt-in |
|---|---|---|
| `obs_norm.py` | `OBS_DIM=13`，`OBS_SCHEMA_VERSION`，`OBS_NORM_VERSION` 不变 | 追加 `OBS_SCHEMA_REL13_QMAG16="rel13_qmag16_v1"`、`OBS_DIM_16=16`、`OBS16_NORM_VERSION="obs_scale_0714_v1+qmag_q995_v1"` |
| `obs_qmag.py` | 13D 路径不 import | `qmag_from_state_abs` / `qmag_batch`（单步/批量派生）；`qmag_for_aligned_data`（按 `source_idx` 前缀映射对齐，首步错配/越界直接 raise）；`estimate_q_scales`（训练集 99.5% 分位）；`normalize_obs16`（前 13 维 ≡ 13D 归一化）；`denormalize_obs16`；`overflow_ratio_q`；`Q_BOX_HIGH=5.0`（仅空间定义，不裁剪） |
| `load_expert_buffer.py` | `load_abs=False`，返回字段与历史一致 | `load_abs=True` 才额外返回 `state_abs/next_state_abs`（`[N,18]`，16D Q 派生用） |
| `train_bc.py` | `--obs-schema rel13_v1`，`obs_dim=13` | `--obs-schema rel13_qmag16_v1` → `obs_dim=16`；Q scale 只用训练 Actor 子集估计并落盘 `q_scale_seed{S}.npz`（`--q-scale-path` 可复用）；checkpoint config 写 `obs_schema_version/obs_norm_version/q_scale` |
| `bc_actor.py` | `BcActor(obs_dim=13)`，13D 校验分支原样 | `BcActor(obs_dim=16)`；`validate_bc_checkpoint` 16D 只认 `rel13_qmag16_v1` + `obs_scale_0714_v1+qmag_q995_v1`；13D/16D checkpoint 互斥（误加载 raise） |
| `evasion_env.py` | `obs_schema="rel13_v1"`，行为不变 | `EvasionEnv(..., obs_schema="rel13_qmag16_v1", q_scale=[..3..], actor_obs_mode="truth", obs_normalize=True)`；`_format_obs` 由当拍 `_state_abs` 现算 Q 并拼归一化值；`obs_dim=16`，`observation_space` 后 3 维界为 `Q_BOX_HIGH` |
| `matlab/stage2_qmag_from_abs.m` | 不被 13D 调用 | MATLAB 端 Q 口径（与 `stage2_step.m` 一致） |
| `matlab/stage2_actor_obs16_from_abs.m` | 不被 13D 调用 | `obs16=[obs13;Q1;Q2;QT]`，非 truth 直接 error |

## 3. 数据源与实验产物

数据源（与 P1/E0/E1/E1V 同源，保证可比）：

- 专家库：`E:\matlab\result_0714\expert_lib_1000.mat`
- 对齐表：`rl_td3/data/expert_transitions_aligned.mat`
- 终端 cache：`rl_td3/data/expert_terminal_cache.npz`
- 初态池：`samples_clean.mat`（训练）、`samples_clean_val.mat`（150 IC 固定评估，
  `order_seed=0`，`eval_ic_pool.json` 落盘备查）

本实验目录：`BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/`

```text
P1_E2S16D_QMAG_20260928/
  PLAN_16D.md / AB_SUMMARY.md / DIG_Q_AMBIGUITY.md / PAIRED_CLOSED_LOOP.md
  16D_CODEMAP_USAGE.md              # 本文件
  derive_q_stats.py / analyze_drift.py / analyze_q_actor.py / wrap_recheck.py
  summarize_ab.py / dig_q_ambiguity.py
  eval_paired_closed_loop.py / summarize_paired_closed_loop.py
  probe_data_layout.py
  q_scale_prior.npz                 # 全局 99.5% 先验 scale（正式实验各 seed 在训练集重估）
  q_all_aligned_order.npy           # 与 states 同序的全量 Q（复用，避免重算）
  trusted_step_mask.npy             # wrap-aware 可信步（both_ok 步可信 99.93%）
  q_actor_report.json / wrap_report.json / ab_summary.json
  dig_q_report.json / paired_closed_loop_summary.json
  paired_closed_loop_13d_150ic.json / paired_closed_loop_16d_150ic.json
  runs/
    ab13_s0/  ab13_s42/  ab13_s21/   # 13D 纯 BC（best/final pt + report + 采样计划）
    ab16_s0/  ab16_s42/  ab16_s21/   # 16D 纯 BC（+ q_scale_seed{S}.npz）
    ab13_s0repeat_init42/            # 误跑归档（seed0 划分×init42，不计入 A/B）
    smoke16_s0/ smoke16_s42/         # 16D BC 冒烟
```

命名约定：`runs/ab{13|16}_s{seed}/bc_ab{13|16}_s{seed}_{best,final}.pt`、
`bc_ab{13|16}_s{seed}_report.json`、`bc_sample_plan_seed{seed}.npz`、
`runs/ab16_s{seed}/q_scale_seed{seed}.npz`。
注意 seed0 配对复用了 `rl_td3/data/expert_set_99k.npz`
（与 P1 BC 同划分；`ab13_s0` 的 0.3168/epoch21 与 P1 纯 BC 完全一致即为证据）。

## 4. BC 用法（16D 已可用）

冒烟（小预算，先跑通再谈 A/B）：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_bc.py --smoke --obs-schema rel13_qmag16_v1 --seed 0 --sample-seed 0 --save-dir "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/runs/smoke16_s0" --name bc16_smoke
```

完整 13D vs 16D 纯 BC A/B（seed 0/21/42；**同 seed 同划分同标签**是铁律）：

```powershell
# 13D 建划分（seed21 示例；seed0 复用 expert_set_99k.npz 即可）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_bc.py --budget 99000 --epochs 80 --seed 21 --sample-seed 21 --sample-plan "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/runs/ab13_s21/bc_sample_plan_seed21.npz" --save-dir "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/runs/ab13_s21" --name bc_ab13_s21
# 16D 加载同一划分（关键：--sample-plan 指向上面的 13D 计划）
E:\Python\venvs\rl_pytorch\Scripts\python.exe rl_td3/train_bc.py --budget 99000 --epochs 80 --obs-schema rel13_qmag16_v1 --seed 21 --sample-seed 21 --sample-plan "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/runs/ab13_s21/bc_sample_plan_seed21.npz" --save-dir "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/runs/ab16_s21" --name bc_ab16_s21
```

- 网络/epoch/early-stopping 与 P1 BC 一致（`net_arch=(256,256)`，80 epoch，
  `patience=15`，`budget=99000`，`batch=256`）；报告看总 MSE、γ MAE、符号
  准确率、贴边率、近邻歧义；
- 汇总：`summarize_ab.py` → `AB_SUMMARY.md` / `ab_summary.json`；
- 深挖：`dig_q_ambiguity.py` → `dig_q_report.json` / `DIG_Q_AMBIGUITY.md`；
- 验收：`rl_td3/tests/test_obs_qmag.py`（8/8：shape/有限/非负、确定性、
  16D 前 13 维 ≡ 13D、归一化可逆、scale 为正、13D/16D checkpoint 互斥、
  Python vs MATLAB Q 一致 1e-9；直接 `python.exe <文件>` 即可跑，
  MATLAB 一致性项需 Engine，缺失则跳过）。

配对闭环（无训练，同一 150 IC 比较纯 BC 策略本身质量）：

```powershell
E:\Python\venvs\rl_pytorch\Scripts\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/eval_paired_closed_loop.py" --arm 13D --n-ic 150
E:\Python\venvs\rl_pytorch\Scripts\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/eval_paired_closed_loop.py" --arm 16D --n-ic 150
E:\Python\venvs\rl_pytorch\Scripts\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/summarize_paired_closed_loop.py"
```

输出 `paired_closed_loop_13d/16d_150ic.json` → `PAIRED_CLOSED_LOOP.md`
（含逐 IC 成功→失败/失败→成功转移与 McNemar p）。

## 5. TD3 / BC-TD3 条件映射（13D 现状，供对照）

`train_td3.py` 的条件由 flags 组合（均为 13D）：

| 条件 | flags |
|---|---|
| E0（专家 Replay 预填＋在线衰减 BC 正则，随机 Actor） | 默认（`--no-bc` 不加，`--bc-lambda0>0`，无 `--bc-init-ckpt`） |
| E1（E0 ＋ BC Actor warm-start） | E0 flags ＋ `--bc-init-ckpt <纯BC best.pt>` |
| E1V（纯 BC 初始化＋标准 TD3：无预填、无在线专家集、`bc_lambda0=0`、Critic 随机、`learning_starts=1000`） | `--no-bc --bc-lambda0 0 --bc-init-ckpt <对应seed纯BC best.pt>` |

评估：`train_td3.py` 内 `FixedICEvalCallback` 每 25k 步遍历同一组 150 val 初态；
`run_config.json` 记录 `obs_schema_version/obs_norm_version/bc_init`。

## 6. 16D 上 TD3/BC-TD3 的缺口（动工前必读）

现状：`train_td3.py` 无 `--obs-schema/--q-scale` 参数，`make_env()` 只按
`(action_mapping, samples_file)` 建 13D 环境；`run_config` 与
`_initialize_from_bc_checkpoint` 的元数据硬编码
`OBS_SCHEMA_VERSION/OBS_NORM_VERSION`；`evaluate_td3.py` 同样无 schema 参数。
因此 **E0-16D / E1-16D / E1V-16D 当前都跑不起来**，不是“换个 checkpoint 就行”。

若将来要接 16D-TD3，按序改（保持默认 13D 零行为变化，全部 opt-in）：

1. `train_td3.py` 加 `--obs-schema`（`rel13_v1` 默认）＋ `--q-scale-path/--q-scale-seed`，
   `make_env()` 透传 `obs_schema/q_scale` 给 `EvasionEnv`（训练 env 与评估 env
   都要，且评估 env 的 `q_scale` 必须与训练一致）；
2. `load_expert_buffer` 的 Replay 预填 obs 必须走 16D（`normalize_obs16`＋同一种
   `q_scale`），`model.set_expert_dataset` 的 states 维数须与 16D 策略一致；
   `q_scale` 只允许从**训练集**估计，落盘进 `run_config`；
3. `--bc-init-ckpt` 的 `validate_bc_checkpoint` 按 `model.observation_space`
   维数校验（16D 要求 16D checkpoint，沿用 `bc_actor.py` 现有分支）；
4. `run_config.json` 的 `obs_schema_version/obs_norm_version/q_scale` 如实记录
   16D 值，不再硬编码 13D；
5. `evaluate_td3.py` 加同样的 `--obs-schema/--q-scale` 并建对应 env；
6. 先跑 `test_obs_qmag.py` ＋ 16D 环境冒烟（2 IC，如
   `eval_paired_closed_loop.py --n-ic 2` 的量级），再谈 50k；
7. MATLAB 侧若用 `stage2_actor_obs16_from_abs`，保持 truth 口径；含噪/辨识不动。

## 7. 可撤回性与回退

加法式设计，默认即 13D，16D 必须显式 opt-in：

```powershell
# L0（日常）：什么都不做——默认即 13D。
# L1（停用 16D 代码路径）：无需操作；所有 16D 入口默认为关。
# L2（撤销 5 个跟踪文件的改动；执行前务必先备份，改动前它们各有未提交修改）
git diff -- rl_td3/obs_norm.py rl_td3/load_expert_buffer.py rl_td3/train_bc.py rl_td3/bc_actor.py rl_td3/evasion_env.py > E:\16D_revert_backup.patch
git checkout -- rl_td3/obs_norm.py rl_td3/load_expert_buffer.py rl_td3/train_bc.py rl_td3/bc_actor.py rl_td3/evasion_env.py
# L3（彻底清理新文件；同样先做 L2 的备份）
Remove-Item rl_td3/obs_qmag.py, rl_td3/tests/test_obs_qmag.py, rl_td3/matlab/stage2_qmag_from_abs.m, rl_td3/matlab/stage2_actor_obs16_from_abs.m
```

## 8. 禁区与审批（沿用既有约定）

- 不碰含噪（noisy/filtered）与辨识（`ident/`）任何逻辑；
- 不改奖励、动作映射、网络结构、数据筛选、TD3 超参数；
- 不重跑 IPOPT/重建专家库；验证集不参与 scale 选择；
- **任何在线 TD3（即使 ≤50k）与 >50k 训练都先询问用户**；
- 未过离线门槛不开闭环，未过闭环不开 TD3 长训（16D 当前两门皆未过）。
