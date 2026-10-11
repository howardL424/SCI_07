# 伪谱直驱专家 + GAIL-guided TD3 使用说明

本模块在 `ps_gail_td3/` 内独立实现直接加速度对照组。观测为共享尺度的
13 维相对状态，动作为二维 `(ay, az)`，攻击方合成过载上限 8g，防御方
PN 增益为 4、合成过载上限 10g。判别器是唯一模仿机制，没有 BC 预训练、
专家 replay 预填或 BC 正则。没有实现方案 §8 的未来 TRPO 扩展。

算法表述应使用 **GAIL-guided TD3 (on-policy discriminator)**：判别器使用
当前 actor 的新鲜 rollout，生成器由 TD3/DDPG 更新，没有标准 TRPO 或熵正则。

## 环境与工作目录

以下命令均在 PowerShell、项目根目录执行：

```powershell
Set-Location 'D:\LTY\yan\yanerxia\GAIL'
$psPython = 'E:\Python\venvs\rl_pytorch\Scripts\python.exe'
```

实测环境：MATLAB R2024a、机器现有 CasADi、PyTorch 2.6.0+cu124、SB3 2.9.0。
Python 环境还需 Gymnasium、NumPy、SciPy、matplotlib 和 MATLAB Engine。
CasADi 优先查找 MATLAB 路径和 `toolbox/casadi`，再查机器已有
`E:\matlab\casadi-3.7.2-windows64-matlab2018b`。

默认初态池与 `td3_baseline/train_td3_pure.py` 一致：训练
`samples_clean.mat`、验证 `samples_clean_val.mat`、测试
`samples_clean_test.mat`。若使用扩大的 `samples_clean_kuoda_*` 池，必须在
专家生成、训练、验证和测试命令中同时指定对应文件，并重新标定评分系数。
本模块不会修改这些初态文件或共享代码。

## 先验收环境

```powershell
& $psPython -m unittest discover -s ps_gail_td3/tests -v
$env:PS_RUN_MATLAB = '1'
& $psPython -m unittest discover -s ps_gail_td3/tests -v
& $psPython -m ps_gail_td3.tools.verify_matlab
```

未设置 `PS_RUN_MATLAB=1` 时，真实 MATLAB 的环境和专家重放测试会明确跳过；
它们不属于离线 Python 测试的通过结果。专家重放测试需要先生成下面的默认
5 条专家库。`verify_matlab` 保存零动作直飞、等效加速度单步、奖励对拍、
5 个初态的主路线固定权重对照以及 MATLAB 静态检查结果到
`ps_gail_td3/logs/matlab_verification.json`。

## 标定并生成 Radau 专家

```powershell
& $psPython -m ps_gail_td3.tools.prepare_experts --calibrate --count 5
& $psPython -m ps_gail_td3.tools.audit_library

& $psPython -m ps_gail_td3.tools.prepare_experts --count 100 `
  --out ps_gail_td3/data/radau_expert_lib_100.mat `
  --terminal-cache ps_gail_td3/data/radau_terminal_cache_100.npz
& $psPython -m ps_gail_td3.tools.audit_library `
  --expert ps_gail_td3/data/radau_expert_lib_100.mat `
  --terminal-cache ps_gail_td3/data/radau_terminal_cache_100.npz `
  --out ps_gail_td3/logs/expert_audit_100.json
```

第一次必须加 `--calibrate`，后续复用
`ps_gail_td3/data/radau_calibration.mat`。不接受没有标定 SHA 的正式求解。
5–10 条初态的标定也可在 MATLAB 直接执行：

```matlab
addpath('D:/LTY/yan/yanerxia/GAIL/ps_gail_td3/expert');
calibrate_radau_weights( ...
    'D:/LTY/yan/yanerxia/GAIL/samples_clean.mat', 1:10, ...
    'D:/LTY/yan/yanerxia/GAIL/ps_gail_td3/data/radau_calibration.mat');
```

标定将 Eq.23 中不含评分系数的三项中位数各缩放到 `1e5`，`Jscale=3e5`。
2026-10-10 在默认训练池前 5 个有效初态上的实测结果为：

| 项 | 原始项中位数 | 评分系数 | 加权项中位数 |
|---|---:|---:|---:|
| `0.25*((miss1-d*)²+(miss2-d*)²)` | 213.1696 | 469.110038 | 100000 |
| `0.5*积分(||a||²)` | 826.3951 | 121.007493 | 100000 |
| `0.5*missT²` | 28212.9299 | 3.544474 | 100000 |

正式 NLP 控制初值为零；标定能量项则从 warm 状态的姿态导数恢复物理加速度，
仅作为量级探针，不作为专家标签、控制初值或 replay 数据。

Radau 默认 `K=20`、`deg=3`。每区间只有一个常值二维控制，区别于每配点
一个控制的方案，论文比较时需要说明这一降维。防御方 PN 在符号动力学中
使用光滑饱和；专家最终从 `U_K` 投影后用在线的硬饱和动力学重新积分，
保存执行后的真实末态。求解器成功轨迹直接入库；debug 解仅在
`miss1>=6 & miss2>=6 & missT<=30` 时入库。入库并不等于联合成功。

判别器加载器只接收重放未拦截、真实双防最小距离均不小于 6m、真实
stage3 最小目标距离不大于 10m 的轨迹。它不信任缓存的 `both_ok` 和规避
最小距离列；训练/验证按轨迹原始行 ID 分割。

断点恢复使用相同命令加 `--resume`，checkpoint 每 5 条保存一次。
初态池、标定文件和求解配置必须一致。库/缓存指纹必须匹配；重新保存 MAT
可能改变文件 SHA，此时须重新预计算终端缓存，`prepare_experts` 会自动完成。
所有 MATLAB `valid_pos` 均为过滤零行后的 **1-based** 编号，Python 的
`local_idx` 和保存的原始 `row_idx` 则为 **0-based**。

可在 MATLAB 指定有效行子集和自定义配点数：

```matlab
opts = struct('valid_pos', [1 3 5 7 9], ...
    'save_file', 'D:/LTY/yan/yanerxia/GAIL/ps_gail_td3/data/subset.mat', ...
    'cfg', struct('K',20,'deg',3), 'checkpoint_every',2);
generate_radau_library(5, opts);
```

单独生成 sidecar 的命令为：

```powershell
& $psPython -m ps_gail_td3.tools.precompute_radau_terminals `
  --expert ps_gail_td3/data/subset.mat `
  --out ps_gail_td3/data/subset_terminal_cache.npz
```

## 纯 TD3 直驱基线和 GAIL 两阶段训练

环境短训与 GAIL 冒烟：

```powershell
& $psPython -m ps_gail_td3.train_gail_td3 --smoke --gail-steps 0
& $psPython -m ps_gail_td3.train_gail_td3 --smoke
```

纯 TD3 正式入口：

```powershell
& $psPython -m ps_gail_td3.train_gail_td3 --gail-steps 0 `
  --steps 300000 --seed 0 --save-dir ps_gail_td3/checkpoints_pure
```

基于已生成 100 条库的正式两阶段入口：

```powershell
& $psPython -m ps_gail_td3.train_gail_td3 `
  --steps 300000 --gail-steps 200000 --seed 0 `
  --expert ps_gail_td3/data/radau_expert_lib_100.mat `
  --terminal-cache ps_gail_td3/data/radau_terminal_cache_100.npz `
  --save-dir ps_gail_td3/checkpoints_gail
```

`--algo ddpg` 切换到 SB3 原生 DDPG，两阶段循环不变。默认 TD3 参数与纯基线
对齐：学习率 `1e-4`、batch 256、gamma 0.99、tau 0.005、policy delay 3、
网络 `[256,256]`、探索噪声 0.1、Phase II replay 300k、warmup 1000 步。
Phase II 的新 replay 自己重新经历 1000 步 warmup。

Phase I 每轮冻结当前 actor，采集默认 10 个新鲜 episode；先用更新前的
判别器打分，再用专家和这批新鲜负样本更新判别器，最后在专用 50k FIFO
buffer 上进行默认 1000 次 TD3/DDPG 梯度更新。判别器负样本携带轮次戳，
允许与当前轮次相差至多 1，当前实现全部来自当前轮次。步数预算最后可能
截取一个 episode 的前缀，以保持阶段边界准确。

判别器使用执行后的饱和动作；TD3 replay 保存 actor 在 Box 中请求的动作，
由环境实施同一个过载投影。Phase I 使用 `GAILRewardWrapper` 替换奖励，
物理环境仍返回原 Eq.74–75 奖励与终端事实。Phase I 结束时释放小 buffer，
新建 Phase II 大 replay，保留 actor/critic 及优化器。没有专家预填。

默认 GAIL 奖励为 `clip(log(D+1e-8),-10,0)`。显式指定
`--reward-mode neg_log_one_minus_d` 才使用备选奖励，选择记录在
`run_config.json`；当前没有自动奖励切换或混合 replay 回退。

每次训练创建一个 `run_时间戳` 子目录，含：

- `run_config.json`：版本、输入指纹、专家轨迹划分、阶段边界和判别器配置。
- `phase1_model.zip`、`discriminator.pt`、`gail_metrics.jsonl`：GAIL 阶段产物。
- `final_model.zip`、`best/best_model.zip` 和每 50k 步模型。
- `eval_logs/eval_ic_pool.json`、评估奖励/成功率记录、训练 CSV。

所有输出路径强制落在本支线内。正式训练默认为 150 个固定 val 初态，
`--n-eval 0` 使用整个验证池。`--help` 列出全部入口参数。

## MC 评估与对照表

默认遍历全部有效 test 行，固定初态逐行执行、`deterministic=True`，失败
warm-start 单独记录，不用其他初态替代。短流程验证才使用 `--limit`。

```powershell
& $psPython -m ps_gail_td3.evaluate_direct `
  --model 'ps_gail_td3/checkpoints_gail/实际run目录/best/best_model.zip' `
  --out ps_gail_td3/logs/monte_carlo_eval_gail
```

输出 `.json/.csv/.log/.npz`，含主路线兼容字段 `r_MD1_min/r_MD2_min/r_minT/`
`evade_ok/strike_ok/both_ok/intercepted/row_idx/local_idx/energy_kJ`。
双防规避采用 6m、打击采用 10m，终端奖励直接调用共享 `project_terminal`。

可创建 JSON，将 `纯TD3直驱` 和 `GAIL-TD3直驱` 映射到对应评估报告的路径，
并在评估时指定 `--comparison-inputs ps_gail_td3/data/comparison_inputs.json`。
对照表保存到 `ps_gail_td3/logs/comparison.md`。PN、正弦与主路线 BC-TD3
默认只引用本地论文 `tab:perf-nonoise`，不重跑或改写论文；它们的独立
打击率未单列，原表的“目标到达率”包含规避成功条件，按联合率列出。
只有相同池、相同样本规模的完整结果才适合正式算法比较。

绘制完成训练的固定验证池曲线：

```powershell
& $psPython -m ps_gail_td3.tools.plot_training --runs `
  ps_gail_td3/checkpoints_smoke_pure/run_20261010_184454 `
  ps_gail_td3/checkpoints_smoke_gail/run_20261010_184846
```

## 本次实际完成的验证与产物

验证日期：2026-10-10。完整 Python 验收集 **11 项通过、无跳过**，其中包含
真实 MATLAB 的 5 局环境测试和 5 条专家逐拍重放；MATLAB 静态检查无警告。
零动作直飞误差与主路线等效加速度单步绝对态误差均为 0。奖励对拍、真实
末态、唯一 done、零行过滤、新鲜性拒绝与 TD3/DDPG replay 切换均已验证。

5 条库：5 条求解入库、5 条联合成功、440 拍、裁剪 0 次。
100 条库：49 条求解入库；重放后其中 8 条被拦截，**41 条联合成功**可用于
判别器，轨迹划分为 train 33 / val 8。入库率 49%，联合可用率占全部初态
41%；平均求解 4.64 秒、中位数 5.14 秒，solver 总计约 464.41 秒（不含
warm-start 和 stage3）。入库轨迹 4301 拍，动作裁剪率为 0。
失败解保留诊断，没有放宽 6m/10m 或 debug 的 30m 门槛。100 条 checkpoint
已验证可以恢复且不重复求解。

真实短训：纯 TD3 1500 步；GAIL-TD3 1500 步，其中 Phase I 400 步。两者
都用 seed 0、batch 64、replay 10k 和同一 5 条固定 val 池。GAIL 冒烟使用
fresh episodes 1、每轮 TD3 更新 10 次、判别器 batch 64，最后 replay 切换
时大小为 0。判别器 loss 首末为 1.38464 / 1.37523，只证明这次更新链路
正常，不能据此宣称正式收敛。

两种 best 模型各在相同 5 条 test 初态上完成流程评估，联合率分别为
80% / 60%。少量初态与短训不能支持优劣结论。模型、曲线和原始记录都在
`checkpoints_smoke_pure/`、`checkpoints_smoke_gail/`、`logs/` 内。
**正式 300k 单种子训练、全量 test MC 和正式性能对照尚未运行。**

## 对方案的具体实现选择与边界

1. warm 窗口使用主路线 `env_reset` 实际调用的
   `simulate_warmstart_trajectory`，没有改用扫描权重的 `solve_warmstart`；
   这保证同一输入行得到相同初态和 Tev。历史 warm helper 内部仍计算其
   固有制导，但本支线只使用状态/窗口，不使用权重监督。直驱核心和 NLP
   没有式48控制链路。数值一致性测试和固定权重审计单独调用旧主路线。
2. 数值步进复用 `evasion_helpers.updateState`，防御方 PN 每个 0.1s 步开始
   时计算一次，RK4 子步保持指令不变，确保与主路线单步一致。
3. NLP 采用连续符号 PN 和光滑防御方饱和；0.1s 硬饱和重放才是训练专家
   的权威物理事实，配点成功不能替代重放及真实 stage3 成功过滤。
4. 标定能量使用 warm 状态导数，解决零控制初值的能量项为零问题；评分
   系数由文件注入，未沿用原权重版系数。
5. 默认协议仅实现专用小 buffer 的 GAIL 路径，没有启用可选混合 replay
   降级。DDPG 已通过原生后端与阶段切换测试，未开展正式 DDPG 实验。
6. 新增了本支线内的标定、指纹/输出校验、奖励 wrapper、审计与绘图辅助
   文件；没有更改原有目录的源代码或所附 `AGENT_PLAN.md`。

库、sidecar、训练配置及审计使用 `ps_schema_v1 / ps_direct_accel_v1 /`
`obs_scale_0714_v1 / r1_bonus_v1`。动作语义、观测尺度、动力学或初态库变化
时必须重新生成/标定相应产物，不能沿用旧指纹缓存。
