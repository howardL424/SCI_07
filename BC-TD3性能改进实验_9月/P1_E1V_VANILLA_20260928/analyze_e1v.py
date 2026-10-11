"""汇总 E1V 并与 P1/E1 的 E0/E1 在相同评估点配对比较。

E1V 定义（见 RESULTS_0927.md §8）：
    配对纯 BC checkpoint -> 初始化 Actor/Actor-target，
    --no-bc（不预填专家 Replay），--bc-lambda0 0（无在线 BC 正则），
    Critic 随机初始化，learning_starts=1000，标准 TD3 在线训练。

比较原则：
    - 只比较相同 timestep 的评估点（P1 每 25k + t=0，E1V 必须同样 --eval-at-start）；
    - 50k 预筛主要看 0~50k AUC 与 0/25k/50k 配对 both_ok；
    - 200k 预筛主要看 90% 首达与 0~150k AUC（与 P1 门槛一致）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_curve(path: Path) -> dict[str, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        "timesteps": np.asarray([r["timestep"] for r in rows], dtype=np.int64),
        "both_ok": np.asarray([r["both_ok_rate"] for r in rows], dtype=float),
        "evade_ok": np.asarray([r["evade_ok_rate"] for r in rows], dtype=float),
        "strike_ok": np.asarray([r["strike_ok_rate"] for r in rows], dtype=float),
        "intercepted": np.asarray([r["intercepted_rate"] for r in rows], dtype=float),
    }


def value_at(curve: dict[str, np.ndarray], step: int) -> float | None:
    idx = np.flatnonzero(curve["timesteps"] == step)
    return float(curve["both_ok"][idx[0]]) if len(idx) else None


def first_hit_truncated(curve: dict[str, np.ndarray], budget: int, threshold: float = 0.9) -> int | None:
    mask = curve["timesteps"] <= budget
    ts = curve["timesteps"][mask]
    y = curve["both_ok"][mask]
    idx = np.flatnonzero(y >= threshold)
    return int(ts[idx[0]]) if len(idx) else None


def normalized_auc(curve: dict[str, np.ndarray], horizon: int) -> float:
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


def load_condition_curve(root: Path, seed: int, condition: str) -> dict[str, np.ndarray] | None:
    p = root / "runs" / f"seed_{seed}" / condition / "eval_logs" / "success_metrics.json"
    if not p.exists():
        return None
    return load_curve(p)


def load_run_config(root: Path, seed: int, condition: str) -> dict[str, Any]:
    p = root / "runs" / f"seed_{seed}" / condition / "run_config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="E1V 实验根目录")
    parser.add_argument("--p1-root", type=Path, required=True, help="P1/E1 参考根目录")
    parser.add_argument("--budget", type=int, default=50_000)
    args = parser.parse_args()

    manifest_path = args.root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
    seeds = tuple(int(s) for s in manifest.get("seeds", (0, 42)))
    budget = int(args.budget)
    auc_horizon = min(150_000, budget)
    # 配对比较用的共有评估点：0/25k/.../budget（要求 E1V 与 P1 均为 25k 间隔 + t=0）
    common_steps = list(range(0, budget + 1, 25_000))
    if budget == 1000 or budget < 25_000:
        # 冒烟预算下不做 AUC 决策，只记录原始曲线
        common_steps = []

    records: dict[str, Any] = {}
    missing: list[str] = []
    for seed in seeds:
        e1v = load_condition_curve(args.root, seed, "E1V")
        if e1v is None:
            missing.append(f"seed_{seed}/E1V")
            continue
        cfg = load_run_config(args.root, seed, "E1V")
        rec: dict[str, Any] = {
            "curve": {k: v.tolist() for k, v in e1v.items()},
            "first_90_step": first_hit_truncated(e1v, budget),
            "auc_horizon": auc_horizon,
            f"auc_0_{auc_horizon // 1000}k": normalized_auc(e1v, horizon=auc_horizon),
            "both_ok_at": {str(s): value_at(e1v, s) for s in common_steps},
            "wall_time_seconds": cfg.get("wall_time_seconds"),
            "bc_init": cfg.get("bc_init"),
            "arguments": cfg.get("arguments"),
        }
        # 与 P1 E0/E1 在相同评估点的配对差值
        for ref in ("E0", "E1"):
            ref_curve = load_condition_curve(args.p1_root, seed, ref)
            if ref_curve is None:
                rec[f"vs_{ref}"] = {"available": False}
                continue
            deltas = {}
            for s in common_steps:
                a = value_at(e1v, s)
                b = value_at(ref_curve, s)
                deltas[str(s)] = (a - b) if (a is not None and b is not None) else None
            auc_e1v = normalized_auc(e1v, horizon=auc_horizon)
            auc_ref = normalized_auc(ref_curve, horizon=auc_horizon)
            deltas_auc = auc_e1v - auc_ref if np.isfinite([auc_e1v, auc_ref]).all() else float("nan")
            e1v_hit = first_hit_truncated(e1v, budget)
            ref_hit = first_hit_truncated(ref_curve, budget)
            censor = budget + 25_000
            deltas_hit = (e1v_hit if e1v_hit is not None else censor) - (ref_hit if ref_hit is not None else censor)
            rec[f"vs_{ref}"] = {
                "available": True,
                "both_ok_delta": deltas,
                "auc_delta": float(deltas_auc) if np.isfinite(deltas_auc) else None,
                "first_90_step_delta": int(deltas_hit),
                "first_90": {"E1V": e1v_hit, ref: ref_hit},
            }
        records[str(seed)] = rec

    # 聚合：E1V 自身 + 相对 E0/E1 的平均 AUC 差
    aucs = [records[s][f"auc_0_{auc_horizon // 1000}k"] for s in records]
    summary = {
        "budget": budget,
        "auc_horizon": auc_horizon,
        "common_steps": common_steps,
        "planned_seeds": list(seeds),
        "complete_seeds": sorted(int(s) for s in records),
        "missing": missing,
        "records": records,
        "aggregate": {
            f"E1V_auc_0_{auc_horizon // 1000}k": mean_std([float(v) for v in aucs if v is not None and np.isfinite(v)]),
        },
    }
    for ref in ("E0", "E1"):
        for key in ("auc_delta", "first_90_step_delta"):
            vals = [records[s][f"vs_{ref}"][key] for s in records
                    if records[s].get(f"vs_{ref}", {}).get("available") and records[s][f"vs_{ref}"][key] is not None
                    and (np.isfinite(records[s][f"vs_{ref}"][key]) if key == "auc_delta" else True)]
            summary["aggregate"][f"E1V_minus_{ref}_{key}"] = mean_std([float(v) for v in vals]) if vals else {"mean": None, "std": None}

    # 50k 预筛决策提示（不自动判 pass/fail，只给出是否满足进入 200k 的两个必要条件）
    hints: dict[str, Any] = {}
    if budget == 50_000 and len(records) >= 2:
        mean_auc_vs_e0 = summary["aggregate"]["E1V_minus_E0_auc_delta"]["mean"]
        mean_auc_vs_e1 = summary["aggregate"]["E1V_minus_E1_auc_delta"]["mean"]
        per_seed_ok = True
        for s in records:
            for ref in ("E0", "E1"):
                d = records[s][f"vs_{ref}"]["auc_delta"]
                # 单 seed 明显退化：相对任一参考的 0~50k AUC 下降超过 2pp
                if d is not None and d < -0.02:
                    per_seed_ok = False
        hints = {
            "mean_auc_improved_vs_E0": bool(mean_auc_vs_e0 is not None and mean_auc_vs_e0 > 0),
            "mean_auc_improved_vs_E1": bool(mean_auc_vs_e1 is not None and mean_auc_vs_e1 > 0),
            "no_single_seed_severe_regression": bool(per_seed_ok),
            "suggestion": (
                "若相对 E0/E1 的平均 0~50k AUC 均有改善且无单 seed 明显退化(>-2pp)，可进入 200k；"
                "若两 seed 均无改善或出现明显崩溃，则停止 E1V，按 RESULTS_0927 §10 结合退化时点决定是否进入 P2。"
            ),
        }
        summary["prescreen_hints"] = hints

    (args.root / "summary_e1v.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# E1V（vanilla BC warm-start + 标准 TD3）阶段结果",
        "",
        f"- 预算：{budget:,} 在线步；AUC 口径：0~{auc_horizon // 1000}k；完成 seed：{summary['complete_seeds']}",
        f"- 共用评估点：{common_steps}",
        f"- E1V−E0 平均 AUC 差：{summary['aggregate'].get('E1V_minus_E0_auc_delta')}",
        f"- E1V−E1 平均 AUC 差：{summary['aggregate'].get('E1V_minus_E1_auc_delta')}",
        f"- E1V−E0 平均 90% 首达差：{summary['aggregate'].get('E1V_minus_E0_first_90_step_delta')}",
        f"- E1V−E1 平均 90% 首达差：{summary['aggregate'].get('E1V_minus_E1_first_90_step_delta')}",
    ]
    if hints:
        lines += ["", f"- 预筛提示：{json.dumps(hints, ensure_ascii=False)}"]
    lines += ["", "逐 seed 配对差值与初始化 hash 见 `summary_e1v.json`。"]
    (args.root / "RESULTS_E1V.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
