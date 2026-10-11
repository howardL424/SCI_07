"""汇总 P1/E1 配对 validation 成功率曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (0, 21, 42)
CONDITIONS = ("E0", "E1")
SELECTED_STEPS = (50_000, 100_000, 150_000, 300_000)


def load_curve(path: Path) -> dict[str, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        "timesteps": np.asarray([r["timestep"] for r in rows], dtype=np.int64),
        "both_ok": np.asarray([r["both_ok_rate"] for r in rows], dtype=float),
        "evade_ok": np.asarray([r["evade_ok_rate"] for r in rows], dtype=float),
        "strike_ok": np.asarray([r["strike_ok_rate"] for r in rows], dtype=float),
        "intercepted": np.asarray(
            [r["intercepted_rate"] for r in rows], dtype=float
        ),
    }


def value_at(curve: dict[str, np.ndarray], step: int) -> float | None:
    idx = np.flatnonzero(curve["timesteps"] == step)
    return float(curve["both_ok"][idx[0]]) if len(idx) else None


def first_hit(curve: dict[str, np.ndarray], threshold: float = 0.9) -> int | None:
    idx = np.flatnonzero(curve["both_ok"] >= threshold)
    return int(curve["timesteps"][idx[0]]) if len(idx) else None


def normalized_auc(curve: dict[str, np.ndarray], horizon: int = 150_000) -> float:
    mask = curve["timesteps"] <= horizon
    x = curve["timesteps"][mask]
    y = curve["both_ok"][mask]
    if len(x) < 2 or x[0] != 0 or x[-1] < horizon:
        return float("nan")
    return float(np.trapezoid(y, x) / horizon)


def mean_std(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return {"mean": None, "std": None}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=300_000)
    args = parser.parse_args()
    auc_horizon = min(150_000, args.budget)
    manifest_path = args.root / "experiment_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest_path.exists()
        else {}
    )
    seeds = tuple(int(seed) for seed in manifest.get("seeds", SEEDS))

    records: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for seed in seeds:
        seed_record: dict[str, Any] = {}
        for condition in CONDITIONS:
            metrics = (
                args.root
                / "runs"
                / f"seed_{seed}"
                / condition
                / "eval_logs"
                / "success_metrics.json"
            )
            if not metrics.exists():
                missing.append(str(metrics))
                continue
            curve = load_curve(metrics)
            config_path = metrics.parents[1] / "run_config.json"
            config = (
                json.loads(config_path.read_text(encoding="utf-8"))
                if config_path.exists()
                else {}
            )
            seed_record[condition] = {
                "curve": {
                    key: value.tolist() for key, value in curve.items()
                },
                "first_90_step": first_hit(curve),
                "auc_0_150k": normalized_auc(curve, horizon=auc_horizon),
                "both_ok_at": {
                    str(step): value_at(curve, step) for step in SELECTED_STEPS
                },
                "wall_time_seconds": config.get("wall_time_seconds"),
                "bc_init": config.get("bc_init"),
                "identification_enabled": config.get(
                    "identification_enabled", False
                ),
            }
        if seed_record:
            records[str(seed)] = seed_record

    complete_seeds = [
        seed for seed in seeds if all(c in records.get(str(seed), {}) for c in CONDITIONS)
    ]
    paired: list[dict[str, Any]] = []
    censor = args.budget + 25_000
    for seed in complete_seeds:
        e0 = records[str(seed)]["E0"]
        e1 = records[str(seed)]["E1"]
        e0_hit = e0["first_90_step"]
        e1_hit = e1["first_90_step"]
        paired.append(
            {
                "seed": seed,
                "first_90_step_delta": (
                    (e1_hit if e1_hit is not None else censor)
                    - (e0_hit if e0_hit is not None else censor)
                ),
                "auc_0_150k_delta": e1["auc_0_150k"] - e0["auc_0_150k"],
                "final_both_ok_delta": (
                    (e1["both_ok_at"].get(str(args.budget)) or 0.0)
                    - (e0["both_ok_at"].get(str(args.budget)) or 0.0)
                ),
            }
        )

    aggregate: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [records[str(seed)][condition] for seed in complete_seeds]
        aggregate[condition] = {
            "first_90_step_censored": mean_std(
                [
                    float(r["first_90_step"])
                    if r["first_90_step"] is not None
                    else float(censor)
                    for r in rows
                ]
            ),
            "auc_0_150k": mean_std([float(r["auc_0_150k"]) for r in rows]),
            "final_both_ok": mean_std(
                [
                    float(r["both_ok_at"][str(args.budget)])
                    for r in rows
                    if r["both_ok_at"].get(str(args.budget)) is not None
                ]
            ),
        }

    n_direction_better = sum(
        p["first_90_step_delta"] < 0 or p["auc_0_150k_delta"] > 0 for p in paired
    )
    mean_hit_delta: float | None = (
        float(np.mean([p["first_90_step_delta"] for p in paired]))
        if paired
        else None
    )
    e0_auc = aggregate.get("E0", {}).get("auc_0_150k", {}).get("mean")
    e1_auc = aggregate.get("E1", {}).get("auc_0_150k", {}).get("mean")
    auc_relative_gain = (
        (e1_auc - e0_auc) / e0_auc
        if e0_auc not in (None, 0.0) and e1_auc is not None
        else None
    )
    e0_final = aggregate.get("E0", {}).get("final_both_ok", {}).get("mean")
    e1_final = aggregate.get("E1", {}).get("final_both_ok", {}).get("mean")
    final_delta = (
        e1_final - e0_final
        if e0_final is not None and e1_final is not None
        else None
    )
    screening_complete = (
        len(seeds) >= 3
        and len(complete_seeds) == len(seeds)
        and args.budget >= 300_000
    )
    gate = {
        "screening_complete": screening_complete,
        "direction_better_seed_count": n_direction_better,
        "mean_first_90_advance_at_least_25k": bool(
            mean_hit_delta is not None and mean_hit_delta <= -25_000
        ),
        "auc_relative_gain_at_least_5pct": bool(
            auc_relative_gain is not None and auc_relative_gain >= 0.05
        ),
        "final_drop_within_1pp": bool(
            final_delta is not None and final_delta >= -0.01
        ),
    }
    gate["pass"] = bool(
        screening_complete
        and n_direction_better >= 2
        and (
            gate["mean_first_90_advance_at_least_25k"]
            or gate["auc_relative_gain_at_least_5pct"]
        )
        and gate["final_drop_within_1pp"]
    )

    summary = {
        "budget": args.budget,
        "planned_seeds": list(seeds),
        "result_scope": (
            "three_seed_screening" if len(seeds) >= 3 else "two_seed_prescreen"
        ),
        "complete_seeds": complete_seeds,
        "missing": missing,
        "records": records,
        "paired": paired,
        "aggregate": aggregate,
        "auc_relative_gain": auc_relative_gain,
        "mean_first_90_step_delta": mean_hit_delta,
        "final_both_ok_delta": final_delta,
        "screening_gate": gate,
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# P1/E1 BC Actor warm-start 实验结果",
        "",
        f"- 完整配对 seed：{complete_seeds}",
        f"- 训练预算：{args.budget:,} 在线步",
        f"- E1 相对 E0 的 0–150k AUC 增益：{auc_relative_gain}",
        f"- 首次达到 90% 的平均步数差（E1−E0，未达标按 {censor:,} 计）：{mean_hit_delta}",
        f"- 最终 both_ok 差（E1−E0）：{final_delta}",
        f"- 筛选门结论：{'通过' if gate['pass'] else '未通过/尚未完成'}",
        "",
        "详细逐 seed 曲线及初始化 hash 见 `summary.json`。",
    ]
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
