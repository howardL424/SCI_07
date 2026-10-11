# ps_gail_td3 对照组代码编写方案 (供 Agent 执行)

> 目标:在 `ps_gail_td3/` 新独立文件夹内实现"伪谱直优加速度 + GAIL-TD3"对照组,
> 与 latex 论文 (BC-TD3 主路线) 做同条件算法比较。
> **冻结口径**:动力学 Eq.1–5、13 维状态 Eq.72、仅过载过程约束、规避 6m、打击 10m、
> 奖励 Eq.74–75,动作/控制量改为 `(ay,az)`。式 48 在本支线**完全不用**。
> GAIL 生成器 = 简化路线 (判别器奖励 + TD3 微调);主干先 TD3,留 DDPG 切换口。
> **红线**:只在 `ps_gail_td3/` 内新建文件;**不得修改** `rl_td3/`、`td3_baseline/`、
> `expert_trajectory/`、`ident/` 及根目录任何现有文件;复用一律 import/addpath,不许拷贝后改原文件。
> **BC 隔离**:本支线任何代码不得 import `td3_bc`(TD3BC)、`bc_actor`、`load_expert_buffer`、
> `expert_sampling`、`train_bc`,不得做 BC 预填/BC 正则/BC warm-start。判别器是唯一的模仿机制。

Python 解释器统一用 `E:\Python\venvs\rl_pytorch\Scripts\python.exe`,工作目录 `D:\LTY\yan\yanerxia\GAIL`。

---

## 1. 目录结构 (全部新建)

```text
ps_gail_td3/
├── AGENT_PLAN.md            # 本文件
├── matlab/                  # 直驱 (ay,az) 动力学核心 (新增,不碰 rl_td3/matlab)
│   ├── ps_core_init.m       # 初始化 helper/参数/版本元数据 (对标 stage2_core_init)
│   ├── ps_step.m            # 单步: (ay,az)饱和 → RK4 推进18维状态 (对标 stage2_step)
│   ├── ps_env_reset.m       # 从 samples_clean 行重建初态 + 定 Tev 窗口
│   ├── ps_env_step.m        # 在线 env 单步包装 (过程奖励,拦截/结束判定)
│   └── ps_transition_meta.m # 版本戳: ps_schema_v1 / ps_direct_accel_v1 / obs_scale_0714_v1
├── expert/                  # Radau 伪谱专家库 (全新,不复用 build_ps_nlp 权重版)
│   ├── radau_dynamics_accel.m      # CasADi 符号动力学: 式5 + 非合作飞行器方（D1,D2）PN式9嵌入,控制=(ay,az)
│   ├── build_radau_nlp_accel.m     # Radau NLP 构造器 (配点/缺陷/约束/目标)
│   ├── solve_radau_accel.m         # 单条求解 (初值/IPOPT/debug 提取)
│   ├── generate_radau_library.m    # 批处理主入口 (LHS池循环/checkpoint/入库判定)
│   └── extract_accel_transitions.m # 重放抽取 (s,a,s',done) + terminal_abs
├── direct_env.py            # Gymnasium 环境 (MATLAB engine 调 ps_env_*)
├── accel_mapping.py         # 动作映射: [-1,1]^2 ↔ (ay,az)=a_norm*8g, 往返测试
├── expert_data.py           # 读专家库 → 归一化 (s,a) 供判别器 + 诊断
├── gail_discriminator.py    # 判别器 D(s,a) MLP + 训练 step + 奖励函数
├── train_gail_td3.py        # 两阶段训练入口 (SB3 原生 TD3/DDPG, 无BC)
├── evaluate_direct.py       # MC 评估 (同 test 池/同判据/同日志格式)
├── tests/
│   ├── test_accel_mapping.py
│   ├── test_direct_env_smoke.py
│   └── test_expert_consistency.py
├── checkpoints*/  data/  logs/   # 本支线自有产物目录, 不写 rl_td3/ 下
└── md document/PS_GAIL_TD3_USAGE.md  # 完成后补用法说明 (最后一步)
```

---

## 2. 复用清单 (只读引用, 禁止修改源文件)

### 2.1 允许 import/addpath 的白名单

| 复用源 | 用途 | 方式 |
|---|---|---|
| `rl_td3/obs_norm.py` (`OBS_SCALE`, `normalize_obs`, 13维定义) | 在线/专家观测归一化, 与主路线同一尺度 | `sys.path` import |
| `rl_td3/reward_projection.py` (`project_terminal`, `DEFAULT_REWARD_CFG`) | 终端奖励结算, 保证奖励公式同一实现 | import |
| `rl_td3/fixed_ic_eval.py` (`FixedICEvalCallback`, `FixedICEvalEnv`, `save_ic_pool`) | 训练内固定初态评估协议 | import (env 接口需与其兼容: `reset(local_idx)` 约定见 M4) |
| `rl_td3/matlab/stage2_obs_from_abs.m` | 18维绝对态 → 13维相对观测 (与动作无关,可直接用) | `addpath` 调用 |
| `rl_td3/matlab/stage3_rollout.m` | stage3 PN 真实打击结算 (与动作无关) | `addpath` 调用 |
| `rl_td3/matlab/stage2_process_reward.m` | 过程奖励公式 (若其签名与权重无关则调, 否则在 `ps_env_step.m` 内按 Eq.74 重写并做数值对拍) | 二选一, 对拍通过为准 |
| `expert_trajectory/reconstruct_geometry.m` | `samples_clean` 14列 → 18维绝对初态 | `addpath` 调用 |
| 根目录 `samples_clean*.mat` | 训练/val/test 初态池 (与主路线同一分布) | 只读 |
| `td3_baseline/train_td3_pure.py` | 训练入口结构模板 (seed/tee日志/callback/DEFAULTS) | **照抄结构, 不许 import 其 `EvasionEnv`** |
| SB3 原生 `TD3` / `DDPG` | 微调主干 (`--algo td3|ddpg` 切换, 默认 td3) | 直接用 |

### 2.2 禁止复用/写入的黑名单

- 禁止 import: `td3_bc`, `bc_actor`, `load_expert_buffer`, `expert_sampling`, `train_bc`,
  `action_mapping`(权重版, 本支线用自有 `accel_mapping.py`)。
- 禁止调用: `computeOptimalCmd_duo_target`(式48数值版) 及任何权重→加速度链路。
- 禁止写入: `rl_td3/checkpoints*`, `rl_td3/data/`, `rl_td3/logs/`, `td3_baseline/` 下任何路径。
- 禁止复用遗留 `expert_trajectory/build_ps_nlp.m` (它是权重版配点, 控制量不对, 只可参考配点写法)。

---

## 3. 模块规格

### M1 MATLAB 直驱动力学核心 (`matlab/`)

- `ps_core_init.m`:入参 `dt/dstar/KD1/KD2/g0/amax_M/amax_D`;返回 core (含 meta 版本戳)。
  防御方 PN 增益/速度与 latex 一致 (`KD=4`, `VD=600`, `VA=500` 常值)。
- `ps_step.m`:输入 `(sabs, t, Tev, a_norm[2])`:
  1. `a=(ay,az)=clip(a_norm,-1,1)*amax_M`, 硬饱和 `‖a‖≤8g` (保方向);
  2. 非合作飞行器方（D1, D2）由当前态按式9算 PN 指令并饱和到 10g;
  3. 三飞行器按式5 RK4 推进 `dt=0.1`;
  4. 返回 `(obs_next(13, 调 stage2_obs_from_abs), sabs_next, t_new, fact)`。
  `fact` 字段对齐主路线:`r_MD1/r_MD2/r_MT/intercepted(6m)/stage2_done/done/terminated`。
- `ps_env_reset.m`:输入 samples 行 → `reconstruct_geometry` 得 18维初态;
  Tev 窗口确定规则:**复用主路线 warm-start 产出的 stage2 时长** (只取时间窗口与初态,
  不取其权重动作), 若某行无有效规避段则 `ok=false` (与主路线 `env_reset` 语义一致)。
  > 实现提示: 最简单的是 addpath 后调 `solve_warmstart` 取 `tau_warm` 的 stage2 段
  > `i0/i1/Tev`; 不得将其权重用于任何监督标签。
- `ps_env_step.m`:调 `ps_step` + 过程奖励 (Eq.74: `0.05*log10(min(r,d*)-5)`, 拦截当步按主路线
  语义处理——以 `stage2_process_reward.m` 实现为准, 先读再对齐)。
- `ps_transition_meta.m`:声明 `ps_schema_v1`, `ps_direct_accel_v1`, `obs_scale_0714_v1`。

### M2 Radau 伪谱专家 NLP (`expert/`)

- `radau_dynamics_accel.m`:CasADi 符号版 M1 动力学。控制 `U=[ay;az]` 为自由变量;
  非合作飞行器（D1, D2） PN 嵌入为状态函数 (不是决策量); 运动学式5 (V 常值, `dV=0`); 光滑饱和保证可微
  (参考 `casadi_dynamics.m` 的 `satf`, 但**删掉其内部式48调用**, 换成直通 `U`)。
- `build_radau_nlp_accel.m`:Radau 配点, 默认 `K=20` 区间 × `deg=3`。
  - 决策:各区间起点/配点状态 `X(18)` + **每区间常值控制 `U(2)`** (相对 Hui"配点控制"
    的降维, 必须在注释与论文中声明);
  - 约束:初值、Radau 缺陷、区间衔接、配点处 `‖a‖≤8g`、softmin 窗口
    `miss1/miss2≥6` (`beta=3`, 参考 `solve_ss_trajectory.m` 的 softmin);
  - 目标 (式23的直接加速度版, 固定评分系数):
    `J=0.25*s_w*[(miss1-d*)^2+(miss2-d*)^2]+0.5*s_wE*∫‖a‖²+0.5*s_wT*missT²`,
    其中 `missT` 为 stage2 末端目标 ZEM (调现有 `casadi_target_zem.m`。
    注意:能量项量级与权重版不同, `s_w/s_wE/s_wT` **必须重标定**, 不得沿用 2e3/55/22:
    先实现 `report_terms_only` 标定模式 (抄 `solve_ss_trajectory.m` 同名逻辑),
    在 5–10 条 warm 初值上测三项原始量级, 再定系数使三项平衡, 系数写入注释与落盘 meta。
  - `Jscale`:按标定结果设 (初值 1e5, 随系数调)。
- `solve_radau_accel.m`:IPOPT (`max_iter=300`, `tol=1e-3`, `limited-memory`);
  初值:状态用 warm stage2 插值, 控制初值 0; catch 后取 `opti.debug` 中间解。
  返回与主路线同形:`ok/theta→U_K/miss/J/terms` (字段名 `U_K [2×K]`, 不得叫 theta)。
- 入库判定 (抄主路线语义):`ok` 直接入库; `debug` 需 `miss1≥6 & miss2≥6 & missT≤30`;
  其余 fail。记录求解器状态与 `J` 终值。

### M3 专家库批处理与抽取

- `generate_radau_library.m`:循环 LHS 有效行 → `ps_env_reset`(取窗口) → `solve_radau_accel`
  → `extract_accel_transitions` → 汇总落盘 `data/radau_expert_lib.mat`。
  支持 `valid_pos` 子集、`checkpoint_every`、`resume` (结构抄
  `generate_ps_expert_library.m`, 变量名 `results_ckpt/results` 同规范)。
  库级 meta 必须含版本戳 + 评分系数 + IPOPT 配置 + `samples_clean` 指纹 (行数/sha)。
- `extract_accel_transitions.m`:**先映射裁剪、再用同一动作积分** (铁律, 抄
  `extract_expert_transitions.m` 流程):
  `U_K → a_norm=U/8g → clip → ps_step(dt=0.1)重放` → 存 `state[13]/a_norm[2]/next_state/`
  `next_state_abs/done/intercepted/stage2_done/t/r_min1/r_min2/terminal_abs/t_end/n_clipped`。
  拦截立刻截断; 末步 `terminal_abs` 为执行最后一拍后的真实末态。
- stage3 sidecar:从 `terminal_abs` 调 `stage3_rollout` (`dt=1e-3`) 得 `r_minT`,
  落盘 `data/radau_terminal_cache.npz` (格式对齐主路线 sidecar 字段名, 便于同脚本评估)。
  允许 Python 脚本 (`tools/precompute_radau_terminals.py`) 或 MATLAB 二选一实现。

### M4 Python 在线环境 (`direct_env.py` + `accel_mapping.py`)

- `accel_mapping.py`:`a_phys=a_norm*AMAX (AMAX=8*9.81)`; `a_norm=a_phys/AMAX`;
  提供 `clip` 计数返回值; 单测往返误差 `<1e-9`。
- `direct_env.py` (结构抄 `EvasionEnv`, 不得继承它):
  - `action_space=Box(-1,1,(2,))`, `observation_space` 用 `OBS_BOX_HIGH` (import `obs_norm`);
  - `reset(seed, options{local_idx?})`:MATLAB `ps_env_reset`; 定点/随机双协议与主路线一致
    (支持 `FixedICEvalEnv` 的 `local_idx` 约定);
  - `step(a_norm)`:调 `ps_env_step`; 正常结束时调 `stage3_rollout` 并用
    `reward_projection.project_terminal` 结算终端 (与主路线**同一函数同一阈值**);
    info 字段对齐主路线 (`r_minT/R1/bonus/both_ok/evade_ok/strike_ok/reached/stage3_energy_int`);
  - `close()` 退出 MATLAB engine。
- 一致性验收 (必须做, 见 §5)。

### M5 GAIL 判别器 (`gail_discriminator.py`)

- 网络:MLP `[256,256]` → sigmoid, 输入 `concat(s_norm(13), a_norm(2))`, 输出 `D∈(0,1)`。
  `s_norm` 用 `normalize_obs`, `a_norm` 用 `accel_mapping` 归一化, 专家/在线同管线。
- 训练 step:二元交叉熵, 专家 label=1 (论文式53的 `log(1-D(real))` 写法等价实现,
  以标准 `BCE(D(real),1)+BCE(D(fake),0)` 为准, 注释注明对应关系);
  每个外层迭代更新 `d_steps=1`, batch=256, lr=3e-4, Adam。
- GAIL 奖励 (忠实 Hui 式49 `u_t=lnD`, 加数值护栏):
  `r_gail=clip(log(D+1e-8), -10, 0)`; 若训练崩溃, 备选 `r=-log(1-D+1e-8)` clip 到 `[0,10]`,
  切换需记录在 run_config (只允许二选一, 不许自创第三种)。
- 专家数据:从 `data/radau_expert_lib.mat` 经 `expert_data.py` 读 `(s,a_norm)`,
  **只取 both_ok 轨迹** (重算 `¬intercepted ∧ r_min≥6 ∧ r_minT≤10`, 不得信任缓存列——
  与主路线 07-23 约定一致); 训练/val 按轨迹 ID 划分用于诊断过拟合。
- **策略侧样本 (新鲜同策略, 与 Hui Step2 同口径)**:判别器的负样本**只许来自
  当前策略的新鲜 rollout**, 不许从 replay 里抽陈旧样本。外层每轮:
  1. 用当前 actor (+ 与训练相同的探索噪声 `--noise`) 现采 `--fresh-episodes`
     (默认 10) 个 episode, 得新鲜 batch `B_fresh`;
  2. 用当前判别器给 `B_fresh` 打分得 `r_gail` 后, 判别器在
     `(expert_batch, B_fresh)` 上更新 `d_steps` 次;
  3. `B_fresh` (带 `r_gail`) 写入 Phase I 专用小 rollout buffer
     (默认容量 50k, FIFO 覆盖, 与 Phase II 的大 replay 物理隔离),
     TD3 在该 buffer 上做固定 `--td3-steps-per-iter` (默认 1000) 次梯度更新。
  顺序与 Hui Step2→5 对应 (先 rollout → 打分 → 交替更新)。备选降级:
  若新鲜 buffer 导致 TD3 不稳, 允许回退到"判别器吃新鲜、TD3 吃大 replay 混合样本",
  但必须在 run_config 中记录 (只允许这两种, 不许第三种)。

### M6 两阶段训练 (`train_gail_td3.py`, 结构抄 `train_td3_pure.py`)

- `--algo {td3(默认),ddpg}`:td3 用 SB3 `TD3`, ddpg 用 SB3 `DDPG` (原生类即有, 无需自写)。
  主超参/DEFAULTS/smoke/log/seed/callback 全部对齐 `train_td3_pure.py` (去掉 ident 相关亦可,
  但 `FixedICEvalCallback` + 固定 val 池必须保留)。
- Phase I (GAIL, `--gail-steps` 默认 200k 对应论文~200ep的概念, 以步数为准):
  环境奖励被 `r_gail` 替换 (wrapper 实现, 不得改 `direct_env.py` 内公式);
  数据流按 M5 新鲜样本协议 (现采 → 打分 → 判别器更新 → 小 buffer TD3 更新);
  不预填专家 transition 进任何 buffer; Phase I 结束时**丢弃小 rollout buffer**,
  Phase II 用全新大 replay 从零开始 (避免新鲜/陈旧样本串扰, 与 Hui 两阶段语义一致)。
- Phase II (微调, 剩余步数至 `--steps` 默认 300k):切回环境奖励 (Eq.74–75),
  保留 Phase I 权重热启动 (对应 Hui Step6 的"生成器→Actor"传参, 本路线 Actor 即 TD3 actor,
  无需跨网络拷贝, 注释说明), 探索噪声与主路线一致 (`--noise 0.1`)。
- 落盘:最终/`best`/周期 checkpoint、`eval_logs/eval_ic_pool.json`、`run_config.json`
  (必须记录 `algo/phase边界/判别器配置/奖励版本/专家库指纹/是否用备选奖励/
  新鲜样本协议[fresh-episodes/小buffer容量/td3-steps-per-iter/是否回退混合]/rollout噪声`)。
- 产物目录默认 `ps_gail_td3/checkpoints/`, smoke 用 `ps_gail_td3/checkpoints_smoke/`。

### M7 评估 (`evaluate_direct.py`)

- 协议与主路线逐项一致:默认测完全部 `samples_clean_test` 有效行、
  `deterministic=True`、同成功判据 (双防≥6m、打击≤10m、联合)、同日志命名
  (`monte_carlo_eval_*`)。允许 import 主路线评估函数做转发 (参考
  `td3_baseline/evaluate_td3_pure.py` 的做法), 模型加载用 SB3 `TD3/DDPG.load`
  (不得用 `TD3BC.load`)。
- 输出对照表:PN / 正弦 / TD3直驱(Phase II 起点? 不, 应为纯TD3直驱基线, 见§4 P2)
  / GAIL-TD3直驱 / 主路线BC-TD3(引用已发表数, 不重跑)。

---

## 4. 实施顺序与每步验收

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| P0 环境骨架 | M1 + M4 + `accel_mapping` 单测 | `test_accel_mapping` 通过; `direct_env` smoke 5局不崩; 零动作弹道与解析直线偏差 `<1e-6` (V恒定直飞); 全零行过滤计数正确 |
| P1 伪谱专家 | M2 标定模式 → 定系数 → M3 跑 5 条 → 100 条 | 标定报告三项量级平衡; 5条全链路 `(s,a,s',done)` 非空、`done` 唯一、`terminal_abs` 为真末态; 100条可用率与求解时长记录; 与主路线同 5 个初态的 `r_min1/2` 可比对 (不要求更优) |
| P2 纯TD3直驱基线 | M6 Phase II 单独跑 (跳过GAIL, `--gail-steps 0`), 单种子短训 | 训练曲线/评估成功率产出, 作为"环境可学习"证据; 若完全学不动, 先修 M1/M4, 不进入 P3 |
| P3 GAIL两阶段 | M5 + M6 全流程 smoke → 正式单种子 | smoke 内判别器 loss 下降、Phase切换不断链; 断言判别器负样本全部来自新鲜 rollout (按 batch 时间戳检查, 不得含超 `fresh-episodes` 轮次之前的样本); 正式跑产出 best 模型 + MC 表; 与 P2 基线同表比较 |
| P4 文档 | `md document/PS_GAIL_TD3_USAGE.md` | 复现命令齐全; 与本方案的偏离清单 (如有) 如实记录 |

---

## 5. 必做的一致性检查 (Agent 不得跳过)

1. 观测对拍:同一起始 `sabs`, `ps_step` 与主路线 `stage2_step` 在**等效加速度**下
   (主路线用某权重算出 `a_M_sat` 后喂给 `ps_step`) 单步 `sabs'` 误差 `<1e-9`。
2. 奖励对拍:同一段 `(r_MD1,r_MD2,r_minT)` 序列, 本支线结算与主路线
   `reward_projection` 输出逐项相等。
3. 专家-在线对拍:专家库任一条轨的 `(s,a_norm)` 序列喂给 `direct_env.reset_from_abs`
   (需实现, 对标 `EvasionEnv.reset_from_abs`) 逐步重放, `s'` 与库内 `next_state`
   误差 `<1e-9`, `done` 位置一致。
4. 动作裁剪审计:专家抽取与在线训练分别统计 `n_clipped`, 写入 meta/log;
   若专家裁剪率 `>5%`, 先收紧 NLP 控制盒, 不许 mute。
5. 版本戳:每次落盘/训练必须写清 `ps_schema_v1 / ps_direct_accel_v1 / obs_scale_0714_v1 /
   reward_version / 专家库指纹`; 半库合并脚本如需新增, 必须复制全部库级 meta
   (主路线 P3 合并丢 meta 的教训)。
6. 新鲜性断言:判别器每个训练 batch 必须携带 rollout 轮次戳; 单测/冒烟中断言
   负样本轮次戳与当前轮次差 ≤1, 从根上杜绝 replay 陈旧样本混入 (P3 验收同)。

---

## 6. 风险与已知坑

- J 系数不可沿用权重版, 必须重标定 (M2 已规定流程); 否则能量项或目标项主导导致退化解。
- Hui 的 `R3=log10(R_AD-149)` 等公式**不得引入**, 本支线奖励就是 Eq.74–75。
- GAIL 在 2 维连续动作下可能震荡:先调判别器 lr/batch/d_steps, 再考虑备选奖励,
  最后才动网络结构; 每次只动一个变量并记录。
- 与标准 GAIL 的剩余差距 (论文须如实声明):判别器样本已同策略新鲜,
  但生成器仍是 TD3 (非 TRPO) 且无熵正则, 故称 "GAIL-guided TD3 (on-policy
  discriminator)" 而非标准 GAIL。本次新鲜样本改造已顺带实现 PPO/TRPO 扩展预留的
  第2条 (on-policy 数据路径), 日后切 PPO 只剩换主干, 切 TRPO 只剩写优化器。
- MATLAB engine/CasADi 路径:CasADi 优先 toolbox, 回退机器路径 (抄主入口写法);
  MATLAB 函数只加 `ps_gail_td3/matlab` + 只读 addpath, 不得改动被复用目录。
- 专家规模:与主路线同一 LHS 池; 可用率若远低于主路线 55%, 优先放宽 `missT≤30`→诊断,
  不得放宽 6m/10m 判据本身。
- DDPG 切换:仅 `--algo ddpg` 一处切换, 不得为 DDPG 另写训练循环; 若 SB3 DDPG 缺某回调兼容,
  以 TD3 结果为准并记录。

## 7. 明确非目标

- 不做辨识 (ident) 与含噪评估; 本支线只做无噪对照 (§6.2 口径)。
- 不做 16D 观测扩展; 只用 13 维。
- 不回写 latex 论文; 论文改稿另行讨论。

---

## 8. TRPO 标准扩展口 (未来改造指南, 本次不实施)

> 目标:把 Phase I 生成器从 TD3 换成**标准 TRPO**, 使 GAIL 阶段忠实 Ho & Ermon
> (及 Hui Step4 式50–51), 判别器/奖励/新鲜样本路径保持不动。
> 前置条件已满足:新鲜 rollout 路径 (§M5/M6) 即 TRPO 必需的 on-policy 数据流,
> 切 TRPO 时数据管线不用重排, 只换优化器后端。

### 8.1 新增文件 (全部落在 `ps_gail_td3/` 内)

| 文件 | 作用 |
|---|---|
| `train_generator.py` | 生成器优化器抽象层:统一 `update(policy, batch) -> info` 接口;现有 TD3 更新逻辑搬迁为 `TD3Backend`, TRPO 实现为 `TRPOBackend`, 由 `--generator {td3,trpo}` 切换 |
| `trpo_backend.py` | TRPO 后端:见 §8.2 |
| `value_baseline.py` | GAIL 回报的价值基线 (MLP `[256,256]`, 仅 TRPO 模式用, 供 GAE advantage) |

### 8.2 TRPO 后端规格

- 策略网:随机高斯策略, 均值 MLP `[256,256]+tanh` (与 TD3 actor **同结构**,
  保证 Phase II 热启动可直接拷贝均值权重, 见 §8.4), 独立 `logstd` 向量
  (初值 `log(0.5)`, 与策略同优化, 或固定二选一, run_config 记录)。
- 优化量 (Hui 式50–51):`max E[π(a|s)/π_old(a|s) · A(s,a)]`,
  s.t. `KL(π_old‖π) ≤ max_kl (默认 0.01)`; `A` 用 GAE (`γ=0.99`, `gae_lambda=0.95`)
  在 GAIL 奖励 (`r_gail`, M5 定义, Phase I 内不变) 上计算, 回报标准化后使用。
- 求解:KL 的 Fisher 向量积 + 共轭梯度 (默认 10 步, damping 0.1) 求自然梯度方向,
  backtracking line search (默认 10 步, 缩减系数 0.5) 保证替代目标提升且满足 KL。
  SB3 无 TRPO, 必须自写 (~200 行) 或引入经评审的第三方实现 (二选一, 禁止手写近似
  如"大 batch 普通梯度冒充 TRPO"; 是否满足 KL 约束必须在 log 中实测记录)。
- 价值基线:每轮在新鲜 batch 上用 MSE 拟合 GAIL 回报 (默认 5 epoch, lr 3e-4),
  只用于 GAE, 不进 Phase II。

### 8.3 Phase I 循环改造 (TRPO 模式)

```text
for iter in 1..N:
  1. 新鲜 rollout (复用现有路径, 采样改用随机高斯策略, 不加额外探索噪声)
  2. 判别器打分得 r_gail (M5 不动)
  3. 判别器更新 (M5 不动)
  4. TRPOBackend.update: GAE advantage → 共轭梯度自然梯度 → 线搜索更新策略;
     价值基线拟合
  (删除: 小 rollout buffer 上的 TD3 更新; 小 buffer 整体退役, replay 留给 Phase II)
```

判别器、奖励 wrapper、`--fresh-episodes`、新鲜性断言 (§5-6) **全部不动**。

### 8.4 Phase II 衔接

- 热启动 = Hui Step6 的"生成器→Actor"传参:拷贝 TRPO 策略均值 MLP 权重到
  TD3/DDPG actor (同结构故直接 `load_state_dict`, 均值对齐误差 `<1e-6` 断言),
  `logstd` 丢弃, 探索改回 `--noise` 高斯。
- Phase II 及之后 (大 replay、环境奖励、评估、MC 表) 与现有 M6/M7 完全一致。

### 8.5 验收与记录

- 新增:KL 实测值 (均值/超限比例) 每轮记录; 线搜索失败率; 与 TD3-生成器版本
  **同专家库、同种子、同评估表**三方比较 (PN/正弦/纯TD3直驱/GAIL-TD3/Hui式GAIL-TRPO)。
- run_config 新增:`generator=trpo`, `max_kl/cg_damping/gae_lambda/价值基线配置/
  TRPO实现来源(自写版本或第三方名+commit)`。
- 论文口径升级:此后 GAIL 阶段可宣称标准 GAIL (Ho & Ermon 生成器 + Hui 流程),
  Phase II 仍为 TD3/DDPG 微调 (与 Hui 的混合结构一致, 如实写明)。

### 8.6 明确不做的事

- 不为 TRPO 另写 rollout/评估/MC 链路, 一律复用现有;
- 不把价值基线与 TD3 critic 混用 (两者目标不同:GAIL 回报 vs 环境回报);
- 不动 M1–M4、M7 及任何原有文件夹文件。
