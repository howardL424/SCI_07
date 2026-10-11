# 参数辨识独立目录

本目录只放**专属于参数辨识**的代码、测试和实验产物。
训练环境、制导核心、评估入口和论文 Table 2 出图流水线仍留在原共用位置。

## 目录

```text
ident/
  ident_cfg.py              CLI / matlab_cfg / 运行配置落盘
  ident_protocol.py         96 维诊断向量解析
  paths.py                  本目录路径常量
  matlab/                   KF + FF-RLS 与阶段2测试
  scripts/                  影子诊断、消融、超参扫描、参数导出
  tests/                    Python 协议测试
  results/                  辨识实验产物（按 checkpoint 运行名分子目录）
```

## 仍留在共用位置的文件

这些文件被训练、评估或制导闭环共用，不能整文件搬走：

- `rl_td3/evasion_env.py`、`rl_td3/evaluate_td3.py`
- `rl_td3/matlab/env_reset.m`、`env_step.m`、`stage2_core_init.m`、`stage2_step.m`
- `td3_baseline/train_td3_pure.py`
- `plot_table2_matlab/table2_noisy_ident/`（论文 Table 2 出图，不是辨识算法本体）
- `参数辨识新对话&新增实验清单md/`（会话与待办文档）

旧路径保留薄兼容入口，因此现有 `from ident_cfg import ...` 和
`td3_baseline/run_ident_*.py` 命令仍然可用。新工作请直接使用本目录。

## 常用命令

```text
E:\Python\venvs\rl_pytorch\Scripts\python.exe ident/tests/test_ident_protocol.py

matlab -batch "addpath('ident/matlab'); addpath('rl_td3/matlab'); addpath('.'); addpath('expert_trajectory'); ok=test_stage2_ident(); exit(~ok);"

E:\Python\venvs\rl_pytorch\Scripts\python.exe ident/scripts/run_ident_ablation_pure.py --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model --episodes 30

E:\Python\venvs\rl_pytorch\Scripts\python.exe ident/scripts/run_p0_4_evaluation.py --episodes 2 --dataset-role validation

E:\Python\venvs\rl_pytorch\Scripts\python.exe ident/scripts/run_h1_shared_replay_diag.py --episodes 40
E:\Python\venvs\rl_pytorch\Scripts\python.exe ident/scripts/analyze_h1_shared_replay.py --run-dir ident/results/h1_val_N40
```

## H1 共享回放诊断

`run_h1_shared_replay_diag.py` 用影子模式（指令恒为 `analytic_truth`）在**同一条
轨迹、同一组量测**上同时导出三路因果估计，避免 `analytic_noisy` 与 FF-RLS 的
直接误差比较被两条闭环轨迹分叉混淆：

- `analytic_noisy` —— 含噪量测代入解析式，P0-4 的“无辨识”基线；
- `sparse_ffrls` —— 因果稀疏 FF-RLS 估计；
- `zoh_drift_floor` —— 神谕 `vartheta[k]` 对 `vartheta[k-1]`，度量“周期内参数
  保持常值”这一约定被违背的代价。

脚本内置测试集访问保护，默认数据为 `samples_clean_val.mat`。
`analyze_h1_shared_replay.py` 给出滞后—误差曲线、逐分量归因、
交汇末段剔除敏感性，以及**恒零/warm-start 两个无技能参照**——统一 NRMSE 的
无技能水平是 1.0，逐分量小于 1.0 才算该参数上真正有技能。

新实验结果默认写到 `ident/results/<checkpoint运行名>/`，不再写进策略 checkpoint 目录。
正式 P0-4 的四组配对统一使用 `run_p0_4_evaluation.py`；该脚本默认采用
`actor_obs_mode=filtered`，并保存实际逐局噪声种子及全程/去初始化阶段 NRMSE。
