"""derive_q_stats.py — 16D 只读预检（不训练、不写专家库）。

映射依据：aligned 轨 tid 经 source_idx → 专家库 1-based 轨号，
取 state_abs 前缀逐拍派生 Q（见 obs_qmag.qmag_for_aligned_data）。
顺带校验首步一致、全轨漂移、Q 分布/先验 scale、归一化溢出。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/derive_q_stats.py"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

GAIL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAIL_ROOT / "rl_td3"))

from load_expert_buffer import load_transition_data  # noqa: E402
from obs_qmag import (  # noqa: E402
    estimate_q_scales,
    normalize_obs16,
    overflow_ratio_q,
    qmag_for_aligned_data,
)

EXPERT = r"E:\matlab\result_0714\expert_lib_1000.mat"
ALIGNED = str(GAIL_ROOT / "rl_td3" / "data" / "expert_transitions_aligned.mat")
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    t0 = time.time()
    print(f"[qstats] 加载 aligned：{ALIGNED}")
    data = load_transition_data(EXPERT, aligned_path=ALIGNED)
    print(f"[qstats] aligned：{len(data['states'])} 步 / {len(data['n_steps'])} 轨")
    print("[qstats] 映射派生 Q（需加载专家库，约2分钟）…")
    q, report = qmag_for_aligned_data(data, EXPERT)
    print(f"[qstats] 映射校验：{report}")
    for j, name in enumerate(["Q1", "Q2", "QT"]):
        col = q[:, j]
        qs = np.quantile(col, [0.5, 0.9, 0.99, 0.995, 0.999, 1.0])
        print(f"[qstats] {name}: p50={qs[0]:.5f} p90={qs[1]:.5f} p99={qs[2]:.5f} "
              f"p99.5={qs[3]:.5f} p99.9={qs[4]:.5f} max={qs[5]:.5f}")

    scale = estimate_q_scales(q)
    print(f"[qstats] 全局 99.5% 分位 scale（先验）：{scale}")
    out = OUT_DIR / "q_scale_prior.npz"
    np.savez(out, q_scale=scale, quantile=np.array(0.995), n_steps=np.array(len(q)))
    print(f"[qstats] 已保存 {out}（正式实验仍按各 seed 训练集重估）")

    o13 = np.asarray(data["states"], dtype=np.float64)
    x16 = np.asarray(normalize_obs16(o13, q, scale), dtype=np.float64)
    for lim in (1.0, 5.0):
        ov = overflow_ratio_q(x16, limit=lim)
        print(f"[qstats] |x|>{lim}：obs13={ov['obs13']*100:.2f}% "
              f"Q={ov['q']*100:.2f}% all={ov['all']*100:.2f}%")
    print(f"[qstats] 全程耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
