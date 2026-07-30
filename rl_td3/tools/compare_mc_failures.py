"""compare_mc_failures.py — 对比多个模型在同一初态池上的失败集合。

输入为 `evaluate_td3.py` 落盘的逐轮指标 npz（含 row_idx / both_ok /
evade_ok / strike_ok / intercepted / r_MD1_min / r_MD2_min / r_minT）。
按 row_idx 对齐后给出：各模型失败率与 Wilson 区间、两两与全体交集、
"失败在 k 个模型上"的观测分布与独立假设下的期望分布、以及共同失败初态明细。

若观测到的高重叠显著超过独立期望，说明残余失败集中在固定的难初态上
（几何/可行性问题）；若接近独立期望，则更像训练随机性导致的散点失败。

Examples
--------
  python rl_td3/tools/compare_mc_failures.py \\
    --npz "pure_best=td3_baseline/checkpoints/training_260k_20260726_1337/mc_fulltest_pure_best_20260730.npz" \\
    --npz "bc_250k=rl_td3/checkpoints/training_500k_20260728_1455/mc_fulltest_bc250k_20260730.npz" \\
    --npz "bc_best=rl_td3/checkpoints/training_500k_20260728_1455/mc_fulltest_bc_best_20260730.npz" \\
    --out-json rl_td3/out_pictures/mc_failure_overlap_20260730.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

Z95 = 1.959963985


def wilson(k: int, n: int) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间（返回百分数）。"""
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1.0 + Z95**2 / n
    center = (ph + Z95**2 / (2 * n)) / den
    half = Z95 * np.sqrt(ph * (1 - ph) / n + Z95**2 / (4 * n * n)) / den
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def poisson_binomial_pmf(probs: list[float]) -> np.ndarray:
    """独立但概率不同的 Bernoulli 之和的分布（Poisson-binomial）。"""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对比多模型 MC 失败集合的重叠")
    p.add_argument(
        "--npz",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="逐轮指标 npz，形如 label=path，可重复指定（≥2 个）",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="将失败集合与重叠统计写入 json",
    )
    p.add_argument(
        "--permutations",
        type=int,
        default=20000,
        help="全体交集显著性的置换检验次数（0=跳过）",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_runs(specs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    runs: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--npz 需为 label=path 形式，收到：{spec}")
        label, path_str = spec.split("=", 1)
        path = Path(path_str)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            raise SystemExit(f"找不到 npz：{path}")
        d = np.load(path)
        need = ["row_idx", "both_ok", "evade_ok", "strike_ok"]
        missing = [k for k in need if k not in d]
        if missing:
            raise SystemExit(f"{path} 缺少字段：{missing}")
        runs[label.strip()] = {k: d[k] for k in d.files}
        print(f"[load] {label:10s} n={len(d['row_idx']):4d}  {path}")
    if len(runs) < 2:
        raise SystemExit("至少需要 2 个 npz 才能比较")
    return runs


def align(runs: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """取所有 run 共有的 row_idx（升序）；不一致时报告差异。"""
    sets = {lab: set(int(x) for x in r["row_idx"]) for lab, r in runs.items()}
    common = set.intersection(*sets.values())
    union = set.union(*sets.values())
    if len(common) != len(union):
        print(
            f"[align] 警告：row_idx 不完全一致，交集 {len(common)} / 并集 {len(union)}"
        )
        for lab, s in sets.items():
            extra = sorted(s - common)
            if extra:
                print(f"  {lab}: 独有 {len(extra)} 个 row_idx，例如 {extra[:10]}")
    else:
        print(f"[align] 三份结果 row_idx 完全一致，共 {len(common)} 个初态")
    return np.array(sorted(common), dtype=np.int64)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.npz)
    rows = align(runs)
    n = len(rows)
    labels = list(runs.keys())

    # 按 row_idx 重排每个 run 的字段
    field = {}
    for lab, r in runs.items():
        order = {int(v): i for i, v in enumerate(r["row_idx"])}
        sel = np.array([order[int(x)] for x in rows], dtype=np.int64)
        field[lab] = {k: v[sel] for k, v in r.items() if len(v) == len(r["row_idx"])}

    fail = {lab: ~field[lab]["both_ok"].astype(bool) for lab in labels}
    fail_evade = {lab: ~field[lab]["evade_ok"].astype(bool) for lab in labels}
    fail_strike = {
        lab: field[lab]["evade_ok"].astype(bool) & ~field[lab]["strike_ok"].astype(bool)
        for lab in labels
    }

    print(f"\n{'='*74}")
    print(f"  各模型失败率（同一 {n} 个 test 初态，both_ok 为成功判据）")
    print(f"{'='*74}")
    print(f"  {'模型':<12}{'失败':>8}{'失败率':>9}{'  95%CI':>16}{'  规避失败':>10}{'  仅打击失败':>12}")
    for lab in labels:
        k = int(fail[lab].sum())
        lo, hi = wilson(k, n)
        print(
            f"  {lab:<12}{k:>6d}/{n:<3d}{k/n*100:>7.2f}%   [{lo:5.2f},{hi:5.2f}]"
            f"{int(fail_evade[lab].sum()):>10d}{int(fail_strike[lab].sum()):>12d}"
        )

    # ---- 两两重叠 ----
    print(f"\n{'='*74}")
    print("  两两交集（Jaccard = |A∩B| / |A∪B|）")
    print(f"{'='*74}")
    pair_stats = {}
    for a, b in combinations(labels, 2):
        inter = int((fail[a] & fail[b]).sum())
        union = int((fail[a] | fail[b]).sum())
        exp_ind = fail[a].mean() * fail[b].mean() * n
        jac = inter / union if union else 0.0
        pair_stats[f"{a}&{b}"] = {
            "inter": inter,
            "union": union,
            "jaccard": jac,
            "expected_if_independent": exp_ind,
        }
        ratio = inter / exp_ind if exp_ind > 0 else float("nan")
        print(
            f"  {a:<12}∩ {b:<12} = {inter:3d}  (并集 {union:3d}, Jaccard {jac:.3f})"
            f"   独立期望 {exp_ind:5.2f}  实测/期望 = {ratio:.1f}×"
        )

    # ---- 失败重数分布 ----
    stack = np.stack([fail[lab] for lab in labels], axis=0)
    kfail = stack.sum(axis=0)
    pmf = poisson_binomial_pmf([float(fail[lab].mean()) for lab in labels])
    print(f"\n{'='*74}")
    print("  「失败于 k 个模型」的初态数：实测 vs 独立假设期望")
    print(f"{'='*74}")
    for k in range(len(labels) + 1):
        obs = int((kfail == k).sum())
        exp = pmf[k] * n
        print(f"  k={k}: 实测 {obs:3d}   独立期望 {exp:7.2f}")

    all_fail = np.where(kfail == len(labels))[0]
    any_fail = np.where(kfail >= 1)[0]
    print(
        f"\n  并集 {len(any_fail)} 个初态至少被一个模型打败；"
        f"其中 {len(all_fail)} 个（{len(all_fail)/max(1,len(any_fail))*100:.1f}%）被全部 {len(labels)} 个模型打败"
    )

    # ---- 全体交集的置换检验 ----
    perm_p = None
    if args.permutations > 0:
        rng = np.random.default_rng(args.seed)
        counts = [int(fail[lab].sum()) for lab in labels]
        obs_all = len(all_fail)
        hits = 0
        for _ in range(args.permutations):
            acc = np.ones(n, dtype=bool)
            for c in counts:
                m = np.zeros(n, dtype=bool)
                m[rng.choice(n, size=c, replace=False)] = True
                acc &= m
            if acc.sum() >= obs_all:
                hits += 1
        perm_p = (hits + 1) / (args.permutations + 1)
        print(
            f"  置换检验（各模型失败数固定、位置随机，{args.permutations} 次）："
            f"P(交集 ≥ {obs_all}) = {perm_p:.4g}"
        )

    # ---- 共同失败初态明细 ----
    if len(all_fail) > 0:
        print(f"\n{'='*74}")
        print(f"  被全部模型打败的 {len(all_fail)} 个初态明细")
        print(f"{'='*74}")
        hdr = f"  {'row_idx':>8}{'local':>7}"
        for lab in labels:
            hdr += f" | {lab[:11]:>11} r_MD_min / r_minT"
        print(hdr)
        for i in all_fail:
            line = f"  {int(rows[i]):>8d}{int(field[labels[0]]['local_idx'][i]):>7d}"
            for lab in labels:
                f_ = field[lab]
                dmin = min(float(f_["r_MD1_min"][i]), float(f_["r_MD2_min"][i]))
                rT = float(f_["r_minT"][i])
                rT_s = f"{rT:8.2f}" if np.isfinite(rT) else "     nan"
                line += f" | {dmin:11.2f} /{rT_s}"
            print(line)

    # ---- 各模型独有失败 ----
    print(f"\n{'='*74}")
    print("  仅被单一模型打败的初态（row_idx）")
    print(f"{'='*74}")
    for lab in labels:
        others = np.zeros(n, dtype=bool)
        for o in labels:
            if o != lab:
                others |= fail[o]
        only = np.where(fail[lab] & ~others)[0]
        print(f"  {lab:<12} {len(only):3d} 个: {[int(rows[i]) for i in only]}")

    if args.out_json:
        out = Path(args.out_json)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_ic": int(n),
            "labels": labels,
            "fail_row_idx": {
                lab: [int(rows[i]) for i in np.where(fail[lab])[0]] for lab in labels
            },
            "fail_counts": {lab: int(fail[lab].sum()) for lab in labels},
            "fail_evade_counts": {lab: int(fail_evade[lab].sum()) for lab in labels},
            "fail_strike_counts": {lab: int(fail_strike[lab].sum()) for lab in labels},
            "pairwise": pair_stats,
            "k_fail_observed": {
                str(k): int((kfail == k).sum()) for k in range(len(labels) + 1)
            },
            "k_fail_expected_independent": {
                str(k): float(pmf[k] * n) for k in range(len(labels) + 1)
            },
            "all_fail_row_idx": [int(rows[i]) for i in all_fail],
            "permutation_p_all_fail": perm_p,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[save] 重叠统计已写入 {out}")


if __name__ == "__main__":
    main()
