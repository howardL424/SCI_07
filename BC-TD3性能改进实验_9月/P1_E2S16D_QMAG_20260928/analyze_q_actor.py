"""analyze_q_actor.py — 只读：归一化漂移 + both_ok 子集 Q 统计 + 落盘复用 Q。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/analyze_q_actor.py"
"""

from __future__ import annotations

import json
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
from obs_norm import OBS_SCALE  # noqa: E402
from obs_qmag import estimate_q_scales, qmag_batch  # noqa: E402

EXPERT = r"E:\matlab\result_0714\expert_lib_1000.mat"
ALIGNED = str(GAIL_ROOT / "rl_td3" / "data" / "expert_transitions_aligned.mat")
CACHE = str(GAIL_ROOT / "rl_td3" / "data" / "expert_terminal_cache.npz")
OUT_DIR = Path(__file__).resolve().parent

KILL_R = 6.0
STRIKE_OK = 10.0


def main() -> None:
    t0 = time.time()
    data = load_transition_data(EXPERT, aligned_path=ALIGNED)
    n_steps = np.asarray(data["n_steps"], dtype=np.int64)
    source_idx = np.asarray(data["source_idx"], dtype=np.int64)
    intercepted = np.asarray(data["intercepted"], dtype=bool)
    states = np.asarray(data["states"], dtype=np.float64)
    offs = np.concatenate([[0], np.cumsum(n_steps)])

    cache = np.load(CACHE, allow_pickle=True)
    r_minT = np.asarray(cache["r_minT"], dtype=np.float64).ravel()
    n_traj = len(n_steps)
    if len(r_minT) != n_traj:
        print(f"[qactor] cache 轨数 {len(r_minT)} 与 aligned {n_traj} 不一致，both_ok 无法计算")
        both_ok_traj = np.zeros(n_traj, dtype=bool)
    else:
        r1 = np.asarray(data["r_min1"], dtype=np.float64)
        r2 = np.asarray(data["r_min2"], dtype=np.float64)
        both_ok_traj = ((~intercepted) & (r1 >= KILL_R) & (r2 >= KILL_R)
                        & np.isfinite(r_minT) & (r_minT <= STRIKE_OK))
    print(f"[qactor] both_ok 轨: {both_ok_traj.sum()}/{n_traj}")

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

    q_parts, dn_parts, actor_mask_parts = [], [], []
    for tid in range(n_traj):
        n, k = int(n_steps[tid]), int(source_idx[tid])
        s, e = int(offs[tid]), int(offs[tid + 1])
        lib_st = st_by_src[k][:n]
        dn = np.abs(lib_st - states[s:e]) / OBS_SCALE  # [n,13] 归一化漂移
        dn_parts.append(dn.max(axis=1))
        q_parts.append(qmag_batch(abs_by_src[k][:n]))
        actor_mask_parts.append(np.full(n, bool(both_ok_traj[tid])))
    dn_max = np.concatenate(dn_parts)
    q_all = np.concatenate(q_parts, axis=0)
    actor_mask = np.concatenate(actor_mask_parts)

    np.save(OUT_DIR / "q_all_aligned_order.npy", q_all)
    print(f"[qactor] Q 已落盘复用：{OUT_DIR / 'q_all_aligned_order.npy'}")

    for thr in (0.01, 0.05, 0.2, 1.0):
        print(f"[qactor] 归一化漂移>{thr} 的步占比={(dn_max > thr).mean()*100:.3f}%")
    per_traj_dn = np.array([dn_parts[i].max() for i in range(n_traj)])
    print(f"[qactor] 轨级归一化漂移 p50={np.quantile(per_traj_dn, .5):.4f} "
          f"p90={np.quantile(per_traj_dn, .9):.4f} p99={np.quantile(per_traj_dn, .99):.4f} "
          f"max={per_traj_dn.max():.3f}")

    qa = q_all[actor_mask]
    print(f"[qactor] actor(both_ok)步数={len(qa)} ({len(qa)/len(q_all)*100:.1f}%)")
    for j, name in enumerate(["Q1", "Q2", "QT"]):
        col = qa[:, j]
        qs = np.quantile(col, [0.5, 0.99, 0.995, 0.999, 1.0])
        print(f"[qactor] actor {name}: p50={qs[0]:.5f} p99={qs[1]:.5f} "
              f"p99.5={qs[2]:.5f} p99.9={qs[3]:.5f} max={qs[4]:.2f}")
    scale_a = estimate_q_scales(qa)
    print(f"[qactor] actor 子集 99.5% scale={scale_a}")
    xa = qa / scale_a
    print(f"[qactor] actor Q归一化后 |x|>1={(np.abs(xa) > 1).mean()*100:.3f}% "
          f"|x|>5={(np.abs(xa) > 5).mean()*100:.3f}% max={np.abs(xa).max():.1f}")

    rep = {
        "step_frac_dn_gt": {str(t): float((dn_max > t).mean())
                            for t in (0.01, 0.05, 0.2, 1.0)},
        "traj_dn_p50": float(np.quantile(per_traj_dn, .5)),
        "traj_dn_p90": float(np.quantile(per_traj_dn, .9)),
        "traj_dn_p99": float(np.quantile(per_traj_dn, .99)),
        "traj_dn_max": float(per_traj_dn.max()),
        "actor_scale_995": [float(v) for v in scale_a],
        "actor_frac": float(actor_mask.mean()),
    }
    (OUT_DIR / "q_actor_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qactor] 报告已存 q_actor_report.json，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
