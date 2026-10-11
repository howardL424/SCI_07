"""纯 TD3 基线与 E0/E1/E1V 在同一 150 固定 val 初态上的配对比较。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

MONTH_DIR = Path(__file__).resolve().parents[1]
P1_ROOT = MONTH_DIR / "P1_E1_BC_WARMSTART_20260927"
E1V_ROOT = MONTH_DIR / "P1_E1V_VANILLA_20260928"
SEEDS = (0, 21, 42)
HORIZON = 150_000
STEPS = (0, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000)


def load_curve(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        "timesteps": np.asarray([r["timestep"] for r in rows], dtype=np.int64),
        "both_ok": np.asarray([r["both_ok_rate"] for r in rows], dtype=float),
    }


def value_at(curve: dict[str, np.ndarray], step: int) -> float | None:
    idx = np.flatnonzero(curve["timesteps"] == step)
    return float(curve["both_ok"][idx[0]]) if len(idx) else None


def first_hit(curve: dict[str, np.ndarray], horizon: int, threshold: float = 0.9) -> int | None:
    mask = (curve["both_ok"] >= threshold) & (curve["timesteps"] <= horizon)
    idx = np.flatnonzero(mask)
    return int(curve["timesteps"][idx[0]]) if len(idx) else None


def normalized_auc(curve: dict[str, np.ndarray], horizon: int) -> float | None:
    mask = curve["timesteps"] <= horizon
    x, y = curve["timesteps"][mask], curve["both_ok"][mask]
    if len(x) < 2 or x[0] != 0 or x[-1] < horizon:
        return None
    return float(np.trapezoid(y, x) / horizon)


def summarize(curve: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "both_ok_at": {str(s): value_at(curve, s) for s in STEPS},
        "auc_0_50k": normalized_auc(curve, 50_000),
        "auc_0_150k": normalized_auc(curve, HORIZON),
        "first_90_step_le_150k": first_hit(curve, HORIZON),
        "first_90_step_le_50k": first_hit(curve, 50_000),
    }


def metrics_path(root: Path, seed: int, cond: str) -> Path:
    return root / "runs" / f"seed_{seed}" / cond / "eval_logs" / "success_metrics.json"


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None]
    return float(np.mean(finite)) if finite else None


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%" if pct else f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "TD3": (args.root, "TD3"),
        "E0": (P1_ROOT, "E0"),
        "E1": (P1_ROOT, "E1"),
        "E1V": (E1V_ROOT, "E1V"),
    }
    records: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for name, (root, cond) in sources.items():
            curve = load_curve(metrics_path(root, seed, cond))
            if curve is not None:
                records.setdefault(str(seed), {})[name] = summarize(curve)

    censor = HORIZON + 25_000
    paired: list[dict[str, Any]] = []
    for seed, rec in records.items():
        if "TD3" not in rec:
            continue
        td3 = rec["TD3"]
        for name in ("E0", "E1", "E1V"):
            if name not in rec:
                continue
            other = rec[name]
            row: dict[str, Any] = {"seed": int(seed), "condition": name}
            for key in ("auc_0_50k", "auc_0_150k"):
                a, b = other[key], td3[key]
                row[f"{key}_minus_TD3"] = None if a is None or b is None else a - b
            for s in STEPS:
                a, b = other["both_ok_at"][str(s)], td3["both_ok_at"][str(s)]
                row[f"both_ok_{s}_minus_TD3"] = None if a is None or b is None else a - b
            if other["auc_0_150k"] is not None and td3["auc_0_150k"] is not None:
                a = other["first_90_step_le_150k"]
                b = td3["first_90_step_le_150k"]
                row["first_90_minus_TD3_censored"] = (a if a is not None else censor) - (
                    b if b is not None else censor
                )
            paired.append(row)

    summary = {
        "horizon": HORIZON,
        "seeds_with_td3": [int(s) for s, r in records.items() if "TD3" in r],
        "records": records,
        "paired_vs_TD3": paired,
        "aggregate": {
            name: {
                key: mean_or_none(
                    [r[name][key] for r in records.values() if name in r and "TD3" in r]
                )
                for key in ("auc_0_50k", "auc_0_150k")
            }
            for name in sources
        },
    }
    (args.root / "summary_td3_pure.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 纯 TD3 基线 vs E0/E1/E1V（同 150 固定 val 初态）",
        "",
        "E0/E1 截取 0–150k；E1V 只有 0–50k。首达 90% 只统计 150k 以内。",
        "",
        "| seed | 条件 | t=0 | 25k | 50k | 75k | 100k | 150k | AUC 0–50k | AUC 0–150k | 90% 首达 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted(records, key=int):
        for name in sources:
            if name not in records[seed]:
                continue
            r = records[seed][name]
            at = r["both_ok_at"]
            hit = r["first_90_step_le_150k"]
            lines.append(
                f"| {seed} | {name} | "
                + " | ".join(fmt(at[str(s)]) for s in (0, 25_000, 50_000, 75_000, 100_000, 150_000))
                + f" | {fmt(r['auc_0_50k'], False)} | {fmt(r['auc_0_150k'], False)} | "
                + (f"{hit // 1000}k" if hit is not None else "未达")
                + " |"
            )
    lines += ["", "## 配对差值（条件 - TD3）", ""]
    for p in paired:
        d50 = p.get("auc_0_50k_minus_TD3")
        d150 = p.get("auc_0_150k_minus_TD3")
        dh = p.get("first_90_minus_TD3_censored")
        lines.append(
            f"- seed {p['seed']} {p['condition']}：ΔAUC 0–50k="
            + (f"{d50:+.4f}" if d50 is not None else "—")
            + "，ΔAUC 0–150k="
            + (f"{d150:+.4f}" if d150 is not None else "—")
            + "，Δ90% 首达="
            + (f"{dh:+d} 步" if dh is not None else "—")
        )
    (args.root / "RESULTS_TD3_PURE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
