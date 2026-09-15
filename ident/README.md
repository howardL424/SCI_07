# 参数辨识独立目录

本目录只放**专属于参数辨识**的代码、测试和实验产物。
训练环境、制导核心、评估入口和论文 Table 2 出图流水线仍留在原共用位置。

## 目录

```text
ident/
  ident_cfg.py              CLI / matlab_cfg / 运行配置落盘
  ident_protocol.py         85 维诊断向量解析
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
```

新实验结果默认写到 `ident/results/<checkpoint运行名>/`，不再写进策略 checkpoint 目录。
