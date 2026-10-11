# E1V（vanilla BC warm-start + 标准 TD3）阶段结果

- 预算：50,000 在线步；AUC 口径：0~50k；完成 seed：[0, 42]
- 共用评估点：[0, 25000, 50000]
- E1V−E0 平均 AUC 差：{'mean': 0.04583333333333328, 'std': 0.08603132504436323}
- E1V−E1 平均 AUC 差：{'mean': 0.00500000000000006, 'std': 0.08249579113843034}
- E1V−E0 平均 90% 首达差：{'mean': -25000.0, 'std': 35355.33905932738}
- E1V−E1 平均 90% 首达差：{'mean': -25000.0, 'std': 35355.33905932738}

- 预筛提示：{"mean_auc_improved_vs_E0": true, "mean_auc_improved_vs_E1": true, "no_single_seed_severe_regression": false, "suggestion": "若相对 E0/E1 的平均 0~50k AUC 均有改善且无单 seed 明显退化(>-2pp)，可进入 200k；若两 seed 均无改善或出现明显崩溃，则停止 E1V，按 RESULTS_0927 §10 结合退化时点决定是否进入 P2。"}

逐 seed 配对差值与初始化 hash 见 `summary_e1v.json`。
