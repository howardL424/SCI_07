"""wrap_recheck.py — 只读：环绕感知漂移复核 + 裁剪重合度 + 可信步掩码。

- 角度维（qy/qz×3通道）先环绕到 [-π,π] 再归一化比较；
- 用比值映射复算库记录动作的逐拍 clip 标记（与 rebuild 同口径），
  核对总数≈12019，并统计 clip 与漂移的重合度；
- 落盘 trusted_step_mask.npy（wrap 漂移<=0.05 的步），供 A/B 清洗规则选用；
  本脚本只写本实验目录，不碰主干与专家库。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "BC-TD3性能改进实验_9月/P1_E2S16D_QMAG_20260928/wrap_recheck.py"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

GAIL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAIL_ROOT / "rl_td3"))

from action_mapping import ACTION_MAPPING_RATIO, phys_to_action_norm  # noqa: E402
from load_expert_buffer import (  # noqa: E402
    _as_NxD,
    _get_transition_field,
    _iter_trajectories,
    load_transition_data,
)
from obs_norm import OBS_SCALE  # noqa: E402

EXPERT = r"E:\matlab\result_0714\expert_lib_1000.mat"
ALIGNED = str(GAIL_ROOT / "rl_td3" / "data" / "expert_transitions_aligned.mat")
OUT_DIR = Path(__file__).resolve().parent

ANGLE_IDX = (2, 3, 6, 7, 9, 10)
TRUST_THR = 0.05


def wrap_angles(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    t0 = time.time()
    data = load_transition_data(EXPERT, aligned_path=ALIGNED)
    n_steps = np.asarray(data["n_steps"], dtype=np.int64)
    source_idx = np.asarray(data["source_idx"], dtype=np.int64)
    intercepted = np.asarray(data["intercepted"], dtype=bool)
    states = np.asarray(data["states"], dtype=np.float64)
    offs = np.concatenate([[0], np.cumsum(n_steps)])

    import mat73

    lib = mat73.loadmat(EXPERT)["results"]
    trajs = _iter_trajectories(lib)
    rec = {}
    for i, tr in enumerate(trajs):
        st = _get_transition_field(tr, "state")
        ac = _get_transition_field(tr, "action")
        if st is None or ac is None:
            continue
        rec[i + 1] = (_as_NxD(st, 13), _as_NxD(ac, 4))

    dn_list, clip_list, cumclip_list = [], [], []
    total_clip = 0
    for tid in range(len(n_steps)):
        n, k = int(n_steps[tid]), int(source_idx[tid])
        lib_st, lib_ac = rec[k]
        s, e = int(offs[tid]), int(offs[tid + 1])
        d = lib_st[:n] - states[s:e]
        d[:, ANGLE_IDX] = wrap_angles(d[:, ANGLE_IDX])
        dn_list.append((np.abs(d) / OBS_SCALE).max(axis=1))
        _, clip = phys_to_action_norm(
            lib_ac[:n, 1], lib_ac[:n, 2], lib_ac[:n, 3],
            version=ACTION_MAPPING_RATIO,
        )
        clip = np.asarray(clip, dtype=bool)
        clip_list.append(clip)
        cumclip_list.append(np.maximum.accumulate(clip))
        total_clip += int(clip.sum())
    dn = np.concatenate(dn_list)
    clip = np.concatenate(clip_list)
    cumclip = np.concatenate(cumclip_list)
    print(f"[wrap] 复算 clip 步数={total_clip}（aligned 记录 12019，应接近）")
    for thr in (0.01, 0.05, 0.2, 1.0):
        print(f"[wrap] wrap漂移>{thr} 步占比={(dn > thr).mean()*100:.3f}%")
    per_traj = np.array([dn_list[i].max() for i in range(len(n_steps))])
    print(f"[wrap] 轨级 p50={np.quantile(per_traj,.5):.4f} "
          f"p90={np.quantile(per_traj,.9):.4f} p99={np.quantile(per_traj,.99):.4f} "
          f"max={per_traj.max():.4f}")
    print(f"[wrap] clip步占比={clip.mean()*100:.2f}%；"
          f"clip暴露（本拍及之前被clip）步占比={cumclip.mean()*100:.2f}%")
    print(f"[wrap] 漂移>0.05步中 clip暴露占比="
          f"{cumclip[dn > 0.05].mean()*100:.1f}%；"
          f"clip暴露步中 漂移>0.05占比={(dn[cumclip] > 0.05).mean()*100:.1f}%；"
          f"无clip暴露步中 漂移>0.05占比={(dn[~cumclip] > 0.05).mean()*100:.2f}%")

    trusted = dn <= TRUST_THR
    np.save(OUT_DIR / "trusted_step_mask.npy", trusted)
    print(f"[wrap] 可信步(>{TRUST_THR}剔除)占比={trusted.mean()*100:.2f}%，"
          f"已存 trusted_step_mask.npy")
    # both_ok 轨上的可信度（需 cache）
    try:
        cache = np.load(str(GAIL_ROOT / "rl_td3" / "data" / "expert_terminal_cache.npz"),
                        allow_pickle=True)
        rT = np.asarray(cache["r_minT"], dtype=np.float64).ravel()
        r1 = np.asarray(data["r_min1"], dtype=np.float64)
        r2 = np.asarray(data["r_min2"], dtype=np.float64)
        bok = ((~intercepted) & (r1 >= 6.0) & (r2 >= 6.0)
               & np.isfinite(rT) & (rT <= 10.0))
        bok_step = np.concatenate([np.full(int(n_steps[i]), bool(bok[i]))
                                   for i in range(len(n_steps))])
        print(f"[wrap] both_ok步中可信占比={trusted[bok_step].mean()*100:.2f}% "
              f"（both_ok步占比={bok_step.mean()*100:.1f}%）")
    except Exception as e:
        print(f"[wrap] both_ok 可信度未计算：{e}")
    rep = {
        "total_clip_recomputed": int(total_clip),
        "step_frac_wrap_dn_gt": {str(t): float((dn > t).mean())
                                 for t in (0.01, 0.05, 0.2, 1.0)},
        "traj_wrap_dn_max": float(per_traj.max()),
        "trusted_thr": TRUST_THR,
        "trusted_frac": float(trusted.mean()),
        "clip_exposed_frac": float(cumclip.mean()),
        "drift_given_clip_exposed": float((dn[cumclip] > 0.05).mean()),
        "drift_given_clean": float((dn[~cumclip] > 0.05).mean()),
    }
    (OUT_DIR / "wrap_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrap] 报告已存 wrap_report.json，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
