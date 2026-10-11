"""Per-IC analysis: did TD3 updates destroy t=0 BC successes?"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
P1 = ROOT.parent / "P1_E1_BC_WARMSTART_20260927"


def load_successes(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    return np.asarray(z["successes"]).astype(bool)


def report(seed: int) -> None:
    e1v = load_successes(ROOT / "runs" / f"seed_{seed}" / "E1V" / "eval_logs" / "evaluations.npz")
    print(f"--- seed {seed} E1V successes shape {e1v.shape} ---")
    t0, t25, t50 = e1v[0], e1v[1], e1v[2]
    print(f"t0 good={t0.sum()} t25 good={t25.sum()} t50 good={t50.sum()}")
    print(f"t0->t25: newly_failed={(t0 & ~t25).sum()} recovered={(~t0 & t25).sum()} kept_good={(t0 & t25).sum()}")
    print(f"t0->t50: newly_failed={(t0 & ~t50).sum()} recovered={(~t0 & t50).sum()} kept_good={(t0 & t50).sum()}")
    for cond in ("E0", "E1"):
        try:
            ref = load_successes(P1 / "runs" / f"seed_{seed}" / cond / "eval_logs" / "evaluations.npz")
        except FileNotFoundError:
            print(f"{cond}: missing")
            continue
        # compare at shared timesteps 0/25k/50k
        for i, t in enumerate((0, 25000, 50000)):
            a = e1v[i]
            b = ref[i] if i < len(ref) else None
            if b is None or len(b) != len(a):
                print(f"{cond} t={t}: shape mismatch ref={None if b is None else len(b)}")
                continue
            print(f"{cond} t={t}: E1V_good={a.sum()} {cond}_good={b.sum()} "
                  f"both_good={(a & b).sum()} E1V_only={(a & ~b).sum()} {cond}_only={(~a & b).sum()}")


if __name__ == "__main__":
    report(0)
    report(42)
