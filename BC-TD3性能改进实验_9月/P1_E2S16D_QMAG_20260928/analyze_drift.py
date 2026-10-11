"""analyze_drift.py — 只读追查：逐轨漂移分布、漂移 vs 拦截、Q 极值来源。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/analyze_drift.py"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

GAIL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAIL_ROOT / "rl_td3"))

from load_expert_buffer import (  # noqa: E402
    _as_NxD,
    _get_transition_field,
    _iter_trajectories,
    load_transition_data,
)
from obs_qmag import qmag_batch  # noqa: E402

EXPERT = r"E:\matlab\result_0714\expert_lib_1000.mat"
ALIGNED = str(GAIL_ROOT / "rl_td3" / "data" / "expert_transitions_aligned.mat")


def main() -> None:
    t0 = time.time()
    data = load_transition_data(EXPERT, aligned_path=ALIGNED)
    n_steps = np.asarray(data["n_steps"], dtype=np.int64)
    source_idx = np.asarray(data["source_idx"], dtype=np.int64)
    intercepted = np.asarray(data["intercepted"], dtype=bool)
    states = np.asarray(data["states"], dtype=np.float64)
    offs = np.concatenate([[0], np.cumsum(n_steps)])
    print(f"[drift] aligned clip 总数: {data.get('n_actions_clipped')} / "
          f"{len(states)} 步; 拦截轨: {intercepted.sum()}/{len(n_steps)}")

    import mat73

    lib = mat73.loadmat(EXPERT)["results"]
    trajs = _iter_trajectories(lib)
    abs_by_src, st_by_src = {}, {}
    for i, tr in enumerate(trajs):
        sabs = _get_transition_field(tr, "state_abs")
        st = _get_transition_field(tr, "state")
        if sabs is None or st is None:
            continue
        abs_by_src[i + 1] = _as_NxD(sabs, 18)
        st_by_src[i + 1] = _as_NxD(st, 13)

    drifts = np.empty(len(n_steps))
    qmax = np.empty((len(n_steps), 3))
    for tid in range(len(n_steps)):
        n, k = int(n_steps[tid]), int(source_idx[tid])
        s, e = int(offs[tid]), int(offs[tid + 1])
        drifts[tid] = float(np.abs(st_by_src[k][:n] - states[s:e]).max())
        qmax[tid] = qmag_batch(abs_by_src[k][:n]).max(axis=0)

    for q in [0.5, 0.9, 0.99, 0.995]:
        print(f"[drift] 逐轨漂移 p{int(q*100)}={np.quantile(drifts, q):.4f} "
              f"| >1m 的轨数={(drifts > 1.0).sum()} "
              f"| >100m 的轨数={(drifts > 100).sum()}")
    print(f"[drift] 漂移最大的5轨: {np.argsort(drifts)[-5:][::-1]} "
          f"值={np.sort(drifts)[-5:][::-1].round(1)}")
    print(f"[drift] 拦截轨漂移均值={drifts[intercepted].mean():.2f} "
          f"非拦截轨漂移均值={drifts[~intercepted].mean():.4f}")
    big = drifts > 100
    if big.sum():
        print(f"[drift] 漂移>100m 的轨数={int(big.sum())}，其中拦截轨="
              f"{int(intercepted[big].sum())}")
    else:
        print("[drift] 无>100m漂移轨")

    # Q 极值：是否来自拦截轨？
    for j, name in enumerate(["Q1", "Q2", "QT"]):
        ti = int(np.argmax(qmax[:, j]))
        print(f"[drift] {name} 轨内最大值最大@${ti}（拦截={bool(intercepted[ti])}）"
              f"={qmax[ti, j]:.2f}；非拦截轨的{name}全局max="
              f"{qmax[~intercepted][:, j].max():.4f}")
    keep = ~intercepted
    print(f"[drift] 非拦截轨占比={(keep).mean()*100:.1f}%")
    print(f"[drift] 耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
