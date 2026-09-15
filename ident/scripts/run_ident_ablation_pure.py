"""纯 TD3 参数辨识配对闭环消融与蒙特卡洛门控。

默认在验证集前30个有效IC上运行四种模式；各模式使用同一IC和噪声种子。
超参数冻结后，可显式指定测试集并令 ``--episodes 0`` 跑完整300组终验。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
    ident/scripts/run_ident_ablation_pure.py \\
    --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model \\
    --episodes 30 --noise-seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_IDENT = _SCRIPT.parent
_RL = _IDENT.parent / "rl_td3"
sys.path.insert(0, str(_IDENT))
sys.path.insert(0, str(_RL))

from paths import default_results_dir  # noqa: E402

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from evasion_env import EvasionEnv, SAMPLES_VAL_FILE  # noqa: E402
from evaluate_td3 import run_episode  # noqa: E402
from ident_cfg import IDENT_CLI_DEFAULTS, matlab_cfg_from_args  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="纯 TD3 参数辨识配对闭环消融")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--samples", type=str, default=SAMPLES_VAL_FILE)
    p.add_argument(
        "--dataset-role",
        choices=["validation", "test", "custom"],
        default="validation",
        help="记录数据集角色；调参必须使用validation，最终冻结评估才使用test",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="取前N个有效IC；0表示遍历完整初态池",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--action-mapping", type=str, default=DEFAULT_ACTION_MAPPING)
    p.add_argument(
        "--also-eta0",
        "--also-mu0",
        dest="also_eta0",
        action="store_true",
        help="包含eta=0对照（阶段5默认已包含；--also-mu0为旧别名）",
    )
    p.add_argument(
        "--no-eta0",
        dest="also_eta0",
        action="store_false",
        help="仅调试时跳过eta=0对照",
    )
    p.set_defaults(also_eta0=True)
    p.add_argument(
        "--eta-grid",
        type=float,
        nargs="+",
        default=None,
        help="验证集闭环调参：仅运行给定eta候选，不重复解析基线",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="结果目录（默认 ident/results/<checkpoint运行名>/ident_ablation_{ts}）",
    )
    # 噪声 / KF / RLS（与 evaluate 同名）
    p.add_argument("--sigma-scale", type=float, default=IDENT_CLI_DEFAULTS["sigma_scale"])
    p.add_argument("--sigma-q", type=float, default=IDENT_CLI_DEFAULTS["sigma_q"])
    p.add_argument("--sigma-qdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_qdot"])
    p.add_argument(
        "--sigma-theta", type=float, default=IDENT_CLI_DEFAULTS["sigma_theta"]
    )
    p.add_argument("--sigma-psi", type=float, default=IDENT_CLI_DEFAULTS["sigma_psi"])
    p.add_argument("--sigma-r", type=float, default=IDENT_CLI_DEFAULTS["sigma_r"])
    p.add_argument("--sigma-rdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_rdot"])
    p.add_argument("--kf-qj", type=float, default=IDENT_CLI_DEFAULTS["kf_qj"])
    p.add_argument(
        "--kf-sigma-a0", type=float, default=IDENT_CLI_DEFAULTS["kf_sigma_a0"]
    )
    p.add_argument("--rls-lambda", type=float, default=IDENT_CLI_DEFAULTS["rls_lambda"])
    p.add_argument("--rls-mu", type=float, default=IDENT_CLI_DEFAULTS["rls_mu"])
    p.add_argument("--rls-eta", type=float, default=IDENT_CLI_DEFAULTS["rls_eta"])
    p.add_argument("--rls-P0", type=float, default=IDENT_CLI_DEFAULTS["rls_P0"])
    p.add_argument(
        "--rls-P-trace-max",
        type=float,
        default=IDENT_CLI_DEFAULTS["rls_P_trace_max"],
    )
    p.add_argument("--rls-r0", type=float, default=IDENT_CLI_DEFAULTS["rls_r0"])
    p.add_argument("--rls-a0", type=float, default=IDENT_CLI_DEFAULTS["rls_a0"])
    return p.parse_args()


def _modes(args: argparse.Namespace) -> list[tuple[str, dict]]:
    base = argparse.Namespace(**vars(args))
    base.ident_shadow = False
    if args.eta_grid:
        modes = [
            (
                f"sparse_ffrls_eta_{eta:g}".replace(".", "p"),
                {"coeff_mode": "sparse_ffrls", "rls_eta": float(eta)},
            )
            for eta in args.eta_grid
        ]
    else:
        modes = [
            ("analytic_truth", {"coeff_mode": "analytic_truth"}),
            ("analytic_noisy", {"coeff_mode": "analytic_noisy"}),
            ("sparse_ffrls", {"coeff_mode": "sparse_ffrls", "rls_eta": args.rls_eta}),
        ]
        if args.also_eta0:
            modes.append(
                ("sparse_ffrls_eta0", {"coeff_mode": "sparse_ffrls", "rls_eta": 0.0})
            )
    out = []
    for name, over in modes:
        ns = argparse.Namespace(**vars(base))
        for k, v in over.items():
            setattr(ns, k, v)
        cfg = matlab_cfg_from_args(ns)
        out.append((name, cfg))
    return out


_CONTINUOUS_METRICS = (
    "r_MD1_min",
    "r_MD2_min",
    "r_minT",
    "a_energy",
    "a_total_variation",
    "saturation_rate",
    "theta_causal_rmse",
    "theta_causal_tail_rmse",
    "theta_command_rmse",
    "theta_command_tail_rmse",
    "frozen_model_rmse",
    "frozen_model_tail_rmse",
    "a_cmd_rmse",
    "a_cmd_tail_rmse",
    "a_cmd_sat_rmse",
    "a_cmd_sat_tail_rmse",
)


def _wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    radius = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _finite_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "n_finite": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
        }
    return {
        "n_finite": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
    }


def _summarize_rows(rows: list[dict], cfg: dict) -> dict:
    n = len(rows)
    out = {"n": n, "cfg": cfg}
    for key in ("evade_ok", "strike_ok", "both_ok", "intercepted"):
        count = int(sum(bool(r[key]) for r in rows))
        out[key] = {
            "count": count,
            "rate": float(count / n) if n else float("nan"),
            "ci95_wilson": _wilson_ci(count, n),
        }
    for key in _CONTINUOUS_METRICS:
        out[key] = _finite_stats(np.array([r.get(key, np.nan) for r in rows]))
    out["per_episode"] = [
        {
            "local_idx": int(r["local_idx"]),
            "row_idx": int(r["row_idx"]),
            "steps": int(r["steps"]),
            **{
                key: bool(r[key])
                for key in ("evade_ok", "strike_ok", "both_ok", "intercepted")
            },
            **{key: float(r.get(key, np.nan)) for key in _CONTINUOUS_METRICS},
        }
        for r in rows
    ]
    return out


def _save_mode_metrics(out_dir: Path, mode_name: str, rows: list[dict]) -> None:
    payload = {
        "local_idx": np.asarray([r["local_idx"] for r in rows], dtype=np.int64),
        "row_idx": np.asarray([r["row_idx"] for r in rows], dtype=np.int64),
        "steps": np.asarray([r["steps"] for r in rows], dtype=np.int64),
    }
    for key in ("evade_ok", "strike_ok", "both_ok", "intercepted"):
        payload[key] = np.asarray([r[key] for r in rows], dtype=bool)
    for key in _CONTINUOUS_METRICS:
        payload[key] = np.asarray(
            [r.get(key, np.nan) for r in rows], dtype=np.float64
        )
    np.savez_compressed(out_dir / f"metrics_{mode_name}.npz", **payload)


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, n_boot: int = 5000
) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [float(lo), float(hi)]


def _paired_comparison(
    baseline: list[dict], candidate: list[dict], *, seed: int
) -> dict:
    base_by_ic = {int(r["local_idx"]): r for r in baseline}
    cand_by_ic = {int(r["local_idx"]): r for r in candidate}
    common = sorted(set(base_by_ic) & set(cand_by_ic))
    out = {"n_paired": len(common), "continuous_delta_candidate_minus_baseline": {}}
    for key in ("evade_ok", "strike_ok", "both_ok"):
        base = np.asarray([base_by_ic[i][key] for i in common], dtype=int)
        cand = np.asarray([cand_by_ic[i][key] for i in common], dtype=int)
        delta = cand - base
        out[f"{key}_rate_delta"] = float(np.mean(delta)) if common else float("nan")
        out[f"{key}_delta_ci95_bootstrap"] = _bootstrap_mean_ci(
            delta, seed=seed + len(key)
        )
        out[f"{key}_wins"] = int(np.sum(delta > 0))
        out[f"{key}_losses"] = int(np.sum(delta < 0))
    for j, key in enumerate(_CONTINUOUS_METRICS):
        delta = np.asarray(
            [
                float(cand_by_ic[i].get(key, np.nan))
                - float(base_by_ic[i].get(key, np.nan))
                for i in common
            ],
            dtype=float,
        )
        stats = _finite_stats(delta)
        stats["ci95_bootstrap_mean"] = _bootstrap_mean_ci(
            delta, seed=seed + 100 + j
        )
        out["continuous_delta_candidate_minus_baseline"][key] = stats
    return out


def main() -> None:
    args = parse_args()
    sample_name = Path(args.samples).name.lower()
    if sample_name == "samples_clean_test.mat" and args.dataset_role != "test":
        raise SystemExit(
            "使用 samples_clean_test.mat 时必须显式指定 --dataset-role test"
        )
    if args.dataset_role == "test" and args.eta_grid:
        raise SystemExit("测试集禁止 --eta-grid 调参；请先在验证集冻结超参数")
    model_path = args.model
    if not model_path.endswith(".zip"):
        zip_path = model_path + ".zip"
    else:
        zip_path = model_path
        model_path = model_path[:-4]
    if not Path(zip_path).exists():
        print(f"[ablation] 找不到模型 {zip_path}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = default_results_dir(model_path, "ident_ablation", ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = _modes(args)
    summary = {
        "model": model_path,
        "protocol": {
            "dataset": str(Path(args.samples)),
            "dataset_role": args.dataset_role,
            "requested_episodes": int(args.episodes),
            "paired_ic": True,
            "noise_seed_rule": "base_noise_seed + local_idx",
            "base_noise_seed": int(args.noise_seed),
            "hyperparameters_frozen_for_test": args.dataset_role == "test",
        },
        "modes": {},
        "comparisons": {},
    }
    rows_by_mode: dict[str, list[dict]] = {}

    # 先建一个 env 只为列 IC；各模式各自起 engine 以免 persistent 串扰
    probe = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        matlab_cfg={"coeff_mode": "analytic_truth"},
    )
    n_ic = probe.n_samples if args.episodes == 0 else min(args.episodes, probe.n_samples)
    probe.close()
    summary["protocol"]["episodes"] = int(n_ic)
    print(
        f"[ablation] dataset_role={args.dataset_role} samples={args.samples} "
        f"episodes={n_ic} paired=True"
    )

    for mode_name, cfg in modes:
        print(f"\n===== mode={mode_name} =====")
        env = EvasionEnv(
            samples_file=args.samples,
            action_mapping=args.action_mapping,
            matlab_cfg=cfg,
        )
        model = TD3.load(model_path, env=env)
        rows = []
        for i in range(n_ic):
            # 每局独立噪声：noise_seed + local_idx，保证跨模式同 IC 对齐
            cfg_i = dict(cfg)
            cfg_i["noise_seed"] = int(args.noise_seed) + i
            env.matlab_cfg = cfg_i
            # 迫使下一拍 env_step 重建 cache（通过改 cfg 字段）
            # MATLAB persistent 靠 cfg_cache_key；Python 每次 step 传 cfg
            ep = run_episode(env, model, seed=args.seed if i == 0 else None, local_idx=i)
            if ep is None:
                print(f"  [skip] local_idx={i}")
                continue
            rows.append(ep)
            rT = ep["r_minT"]
            rT_s = f"{rT:.2f}" if np.isfinite(rT) else "nan"
            print(
                f"  IC{i}: both_ok={ep['both_ok']}  "
                f"r1={ep['r_MD1_min']:.1f} r2={ep['r_MD2_min']:.1f} "
                f"rT={rT_s}"
            )
        env.close()
        rows_by_mode[mode_name] = rows

        if not rows:
            summary["modes"][mode_name] = {"n": 0}
            continue
        summary["modes"][mode_name] = _summarize_rows(rows, cfg)
        _save_mode_metrics(out_dir, mode_name, rows)

    for baseline_name in ("analytic_truth", "analytic_noisy"):
        baseline = rows_by_mode.get(baseline_name, [])
        for candidate_name in ("sparse_ffrls", "sparse_ffrls_eta0"):
            candidate = rows_by_mode.get(candidate_name, [])
            if baseline and candidate:
                key = f"{candidate_name}_vs_{baseline_name}"
                summary["comparisons"][key] = _paired_comparison(
                    baseline, candidate, seed=args.seed
                )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[ablation] 结果写入 {out_dir}")
    for name, st in summary["modes"].items():
        if st.get("n", 0) == 0:
            continue
        both = st["both_ok"]
        print(
            f"  {name}: both_ok={both['count']}/{st['n']}="
            f"{100*both['rate']:.1f}%  "
            f"evade={100*st['evade_ok']['rate']:.1f}%  "
            f"strike={100*st['strike_ok']['rate']:.1f}%  "
            f"r1={st['r_MD1_min']['mean']:.1f} "
            f"r2={st['r_MD2_min']['mean']:.1f} "
            f"rT={st['r_minT']['mean']:.2f}  "
            f"sat={100*st['saturation_rate']['mean']:.1f}%"
        )


if __name__ == "__main__":
    main()
