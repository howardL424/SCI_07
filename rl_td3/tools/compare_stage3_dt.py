"""
compare_stage3_dt.py — 在代表性轨迹子集上对比 stage3 dt=1e-2 vs 1e-3。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/compare_stage3_dt.py
  ... --n-sample 40 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matlab.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evasion_env import (
    BONUS_BOTH,
    EXPERT_DIR,
    GAIL_ROOT,
    KILL_R,
    MATLAB_DIR,
    STRIKE_OK,
    terminal_R1,
)
from load_expert_buffer import DEFAULT_ALIGNED_PATH, load_aligned_transitions


def _stratified_indices(
    n_traj: int,
    intercepted: np.ndarray,
    n_steps: np.ndarray,
    n_sample: int,
    seed: int,
) -> np.ndarray:
    """按拦截/非拦截与轨迹长度四分位分层抽样。"""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n_traj)
    safe = idx_all[~intercepted.astype(bool)]
    if len(safe) == 0:
        return rng.choice(idx_all, size=min(n_sample, n_traj), replace=False)

    lengths = n_steps[safe]
    try:
        qs = np.quantile(lengths, [0.0, 0.25, 0.5, 0.75, 1.0])
    except Exception:
        qs = np.array([lengths.min(), lengths.max()])

    buckets = []
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        if i < len(qs) - 2:
            mask = (lengths >= lo) & (lengths < hi)
        else:
            mask = (lengths >= lo) & (lengths <= hi)
        buckets.append(safe[mask])

    per = max(1, n_sample // max(1, len([b for b in buckets if len(b) > 0])))
    chosen = []
    for b in buckets:
        if len(b) == 0:
            continue
        take = min(len(b), per)
        chosen.extend(rng.choice(b, size=take, replace=False).tolist())

    chosen = np.unique(np.asarray(chosen, dtype=np.int32))
    if len(chosen) < n_sample:
        remain = np.setdiff1d(safe, chosen)
        extra = min(len(remain), n_sample - len(chosen))
        if extra > 0:
            chosen = np.concatenate(
                [chosen, rng.choice(remain, size=extra, replace=False)]
            )
    if len(chosen) > n_sample:
        chosen = rng.choice(chosen, size=n_sample, replace=False)
    return np.sort(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description="对比 stage3 dt=1e-2 vs 1e-3")
    p.add_argument("--aligned", type=str, default=DEFAULT_ALIGNED_PATH)
    p.add_argument("--n-sample", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "stage3_dt_compare.npz"),
    )
    args = p.parse_args()

    data = load_aligned_transitions(args.aligned)
    n_traj = len(data["n_steps"])
    intercepted = data["intercepted"]
    n_steps = data["n_steps"]
    idxs = _stratified_indices(n_traj, intercepted, n_steps, args.n_sample, args.seed)
    print(f"[compare] 对齐轨迹 {n_traj}，抽样 {len(idxs)} 条（仅未拦截）")

    eng = matlab.engine.start_matlab()
    eng.addpath(MATLAB_DIR, nargout=0)
    eng.addpath(GAIL_ROOT, nargout=0)
    eng.addpath(EXPERT_DIR, nargout=0)

    rows = []
    for j, tid in enumerate(idxs):
        if bool(intercepted[tid]):
            continue
        last_abs = np.asarray(data["terminal_abs"][tid], dtype=np.float64).ravel()
        t0 = float(data["t_end"][tid])
        r_min1 = float(data["r_min1"][tid])
        r_min2 = float(data["r_min2"][tid])
        sabs_ml = matlab.double(last_abs.tolist())

        r_fast, _, _, _ = eng.stage3_rollout(sabs_ml, t0, {"dt": 1e-2}, nargout=4)
        r_ref, _, _, _ = eng.stage3_rollout(sabs_ml, t0, {"dt": 1e-3}, nargout=4)
        r_fast = float(r_fast)
        r_ref = float(r_ref)

        strike_fast = r_fast <= STRIKE_OK
        strike_ref = r_ref <= STRIKE_OK
        evade_ok = (r_min1 >= KILL_R) and (r_min2 >= KILL_R)
        both_fast = evade_ok and strike_fast
        both_ref = evade_ok and strike_ref
        R1_fast = terminal_R1(r_fast)
        R1_ref = terminal_R1(r_ref)
        bonus_fast = BONUS_BOTH if both_fast else 0.0
        bonus_ref = BONUS_BOTH if both_ref else 0.0

        rows.append(
            {
                "tid": tid,
                "r_fast": r_fast,
                "r_ref": r_ref,
                "strike_fast": strike_fast,
                "strike_ref": strike_ref,
                "both_fast": both_fast,
                "both_ref": both_ref,
                "R1_fast": R1_fast,
                "R1_ref": R1_ref,
                "bonus_fast": bonus_fast,
                "bonus_ref": bonus_ref,
            }
        )
        if (j + 1) % 10 == 0 or j == 0:
            print(
                f"  [{j+1}/{len(idxs)}] tid={tid}  "
                f"r_minT 1e-2={r_fast:.3f}  1e-3={r_ref:.3f}  "
                f"Δ={r_fast-r_ref:.3f}"
            )

    eng.quit()
    if not rows:
        print("[compare] 无可用样本（全为拦截？）")
        return

    r_fast = np.array([r["r_fast"] for r in rows])
    r_ref = np.array([r["r_ref"] for r in rows])
    abs_err = np.abs(r_fast - r_ref)
    strike_agree = np.mean(
        [r["strike_fast"] == r["strike_ref"] for r in rows]
    )
    both_agree = np.mean([r["both_fast"] == r["both_ref"] for r in rows])
    corr = np.corrcoef(r_fast, r_ref)[0, 1] if len(rows) > 1 else np.nan

    print("\n========== stage3 dt 对比摘要 ==========")
    print(f"  n={len(rows)}")
    print(f"  |Δr_minT|  med={np.median(abs_err):.4f}  p95={np.percentile(abs_err,95):.4f}  max={abs_err.max():.4f}")
    print(f"  corr(r_fast,r_ref)={corr:.4f}")
    print(f"  打击标签一致率={strike_agree*100:.1f}%")
    print(f"  联合成功标签一致率={both_agree*100:.1f}%")
    # 标签一致但 |Δr_minT| 大时，R1=80/(5+r) 仍会偏离在线 1e-3
    substantial = (np.median(abs_err) > 0.2) or (np.percentile(abs_err, 95) > 1.0)
    if substantial:
        print("  建议: |Δr_minT| 实质偏大 → 正式 cache 用 --dt 1e-3（与在线一致）")
    elif strike_agree >= 0.98 and both_agree >= 0.98:
        print("  建议: 标签高度一致且 |Δ| 较小 → 正式 cache 可用 --dt 1e-2")
    else:
        print("  建议: 存在标签分歧 → 正式 cache 用 --dt 1e-3")

    out = Path(args.out)
    np.savez_compressed(
        out,
        tid=np.array([r["tid"] for r in rows], dtype=np.int32),
        r_minT_1e2=r_fast,
        r_minT_1e3=r_ref,
        strike_1e2=np.array([r["strike_fast"] for r in rows], dtype=np.int8),
        strike_1e3=np.array([r["strike_ref"] for r in rows], dtype=np.int8),
        both_1e2=np.array([r["both_fast"] for r in rows], dtype=np.int8),
        both_1e3=np.array([r["both_ref"] for r in rows], dtype=np.int8),
    )
    print(f"[compare] 明细已保存 → {out}")


if __name__ == "__main__":
    main()
