"""纯 TD3 辨识超参小网格：kf_qj × rls_lambda × rls_mu（影子诊断）。

复用 run_ident_shadow_diag_pure.run_shadow_episode，对同一 checkpoint / IC
比较末段 θ / r̈ / 指令 RMSE，并汇总 μ>0 vs μ=0。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
    td3_baseline/run_ident_hyperparam_scan_pure.py \\
    --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model \\
    --episodes 5 --noise-seed 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_RL = _ROOT.parent / "rl_td3"
sys.path.insert(0, str(_RL))
sys.path.insert(0, str(_ROOT))

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from evasion_env import EvasionEnv, SAMPLES_TEST_FILE  # noqa: E402
from ident_cfg import IDENT_CLI_DEFAULTS, matlab_cfg_from_args  # noqa: E402
from run_ident_shadow_diag_pure import (  # noqa: E402
    _THETA_HOLD,
    _THETA_RMSE_THRESH,
    run_shadow_episode,
)
from stable_baselines3 import TD3  # noqa: E402

# 默认小网格（可 CLI 覆盖）
_DEFAULT_KF_QJ = [1e2, 1e3, 1e4, 1e5]
_DEFAULT_LAMBDA = [0.95, 0.98, 0.995]
_DEFAULT_MU = [0.0, 1e-4, 1e-3, 1e-2]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="纯 TD3 辨识超参影子网格扫描")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--samples", type=str, default=SAMPLES_TEST_FILE)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--action-mapping", type=str, default=DEFAULT_ACTION_MAPPING)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--sigma-q", type=float, default=IDENT_CLI_DEFAULTS["sigma_q"])
    p.add_argument("--sigma-qdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_qdot"])
    p.add_argument("--sigma-theta", type=float, default=IDENT_CLI_DEFAULTS["sigma_theta"])
    p.add_argument("--sigma-psi", type=float, default=IDENT_CLI_DEFAULTS["sigma_psi"])
    p.add_argument("--sigma-r", type=float, default=IDENT_CLI_DEFAULTS["sigma_r"])
    p.add_argument("--sigma-rdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_rdot"])
    p.add_argument("--kf-sigma-a0", type=float, default=IDENT_CLI_DEFAULTS["kf_sigma_a0"])
    p.add_argument("--rls-P0", type=float, default=IDENT_CLI_DEFAULTS["rls_P0"])
    p.add_argument("--rls-r0", type=float, default=IDENT_CLI_DEFAULTS["rls_r0"])
    p.add_argument("--rls-a0", type=float, default=IDENT_CLI_DEFAULTS["rls_a0"])
    p.add_argument(
        "--kf-qj-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_KF_QJ),
    )
    p.add_argument(
        "--rls-lambda-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_LAMBDA),
    )
    p.add_argument(
        "--rls-mu-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_MU),
    )
    p.add_argument("--theta-rmse-thresh", type=float, default=_THETA_RMSE_THRESH)
    p.add_argument("--theta-hold", type=int, default=_THETA_HOLD)
    return p.parse_args()


def _agg_eps(ep_stats: list[dict], key: str) -> dict:
    arr = np.asarray([e[key] for e in ep_stats], dtype=float)
    return {
        f"mean_{key}": float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"),
        f"median_{key}": float(np.nanmedian(arr))
        if np.isfinite(arr).any()
        else float("nan"),
        f"max_{key}": float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan"),
    }


def run_one_cfg(
    env: EvasionEnv,
    model: TD3,
    cfg: dict,
    *,
    n_ic: int,
    noise_seed: int,
    seed: int,
    theta_thresh: float,
    theta_hold: int,
) -> dict:
    ep_stats = []
    for i in range(n_ic):
        cfg_i = dict(cfg)
        cfg_i["noise_seed"] = int(noise_seed) + i
        env.matlab_cfg = cfg_i
        ep = run_shadow_episode(
            env,
            model,
            local_idx=i,
            seed=seed if i == 0 else None,
            theta_thresh=theta_thresh,
            theta_hold=theta_hold,
        )
        if ep is not None:
            ep_stats.append(ep)
    if not ep_stats:
        return {"n": 0}

    out: dict = {
        "n": len(ep_stats),
        "converge_rate": float(
            np.mean([np.isfinite(e["converge_step"]) for e in ep_stats])
        ),
        "kf_qj": float(cfg["kf_qj"]),
        "rls_lambda": float(cfg["rls_lambda"]),
        "rls_mu": float(cfg["rls_mu"]),
    }
    for key in (
        "theta_rmse_tail",
        "rddot_rmse_tail",
        "a_cmd_rmse_tail",
        "r_rmse_tail",
        "rdot_rmse_tail",
    ):
        out.update(_agg_eps(ep_stats, key))
    out["per_episode"] = [
        {
            "local_idx": e["local_idx"],
            "theta_rmse_tail": e["theta_rmse_tail"],
            "rddot_rmse_tail": e["rddot_rmse_tail"],
            "a_cmd_rmse_tail": e["a_cmd_rmse_tail"],
            "converge_step": e["converge_step"],
        }
        for e in ep_stats
    ]
    return out


def _mu_paired_delta(rows: list[dict]) -> dict:
    """对相同 (kf_qj, λ)，比较 μ>0 相对 μ=0 的 median a_cmd / θ。"""
    by_base: dict[tuple[float, float], dict[float, dict]] = {}
    for r in rows:
        if r.get("n", 0) == 0:
            continue
        key = (r["kf_qj"], r["rls_lambda"])
        by_base.setdefault(key, {})[r["rls_mu"]] = r

    pairs = []
    for (qj, lam), mus in by_base.items():
        if 0.0 not in mus:
            continue
        base = mus[0.0]
        for mu, row in mus.items():
            if mu == 0.0:
                continue
            pairs.append(
                {
                    "kf_qj": qj,
                    "rls_lambda": lam,
                    "rls_mu": mu,
                    "delta_median_a_cmd": row["median_a_cmd_rmse_tail"]
                    - base["median_a_cmd_rmse_tail"],
                    "delta_median_theta": row["median_theta_rmse_tail"]
                    - base["median_theta_rmse_tail"],
                    "delta_median_rddot": row["median_rddot_rmse_tail"]
                    - base["median_rddot_rmse_tail"],
                    "a_cmd_mu0": base["median_a_cmd_rmse_tail"],
                    "a_cmd_mu": row["median_a_cmd_rmse_tail"],
                    "theta_mu0": base["median_theta_rmse_tail"],
                    "theta_mu": row["median_theta_rmse_tail"],
                }
            )
    if not pairs:
        return {"n_pairs": 0}

    d_a = np.array([p["delta_median_a_cmd"] for p in pairs], dtype=float)
    d_th = np.array([p["delta_median_theta"] for p in pairs], dtype=float)
    # 负值 = μ>0 更好
    better_a = int(np.sum(d_a < -1e-9))
    worse_a = int(np.sum(d_a > 1e-9))
    return {
        "n_pairs": len(pairs),
        "mean_delta_median_a_cmd": float(np.nanmean(d_a)),
        "mean_delta_median_theta": float(np.nanmean(d_th)),
        "frac_mu_better_a_cmd": better_a / len(pairs),
        "frac_mu_worse_a_cmd": worse_a / len(pairs),
        "recommend_disable_l1": bool(np.nanmean(d_a) > 0 and worse_a >= better_a),
        "pairs": pairs,
    }


def main() -> None:
    args = parse_args()
    model_path = args.model
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    zip_path = model_path + ".zip"
    if not Path(zip_path).exists():
        print(f"[scan] 找不到模型 {zip_path}")
        sys.exit(1)

    qj_grid = _parse_float_list(args.kf_qj_grid)
    lam_grid = _parse_float_list(args.rls_lambda_grid)
    mu_grid = _parse_float_list(args.rls_mu_grid)
    grid = list(itertools.product(qj_grid, lam_grid, mu_grid))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(model_path).parent.parent / f"ident_hyperparam_scan_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 基准 cfg（网格字段稍后覆盖）
    args.coeff_mode = "sparse_ffrls"
    args.ident_shadow = True
    args.kf_qj = qj_grid[0]
    args.rls_lambda = lam_grid[0]
    args.rls_mu = mu_grid[0]
    base_cfg = matlab_cfg_from_args(args)

    env = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        matlab_cfg=base_cfg,
    )
    model = TD3.load(model_path, env=env)
    n_ic = min(args.episodes, env.n_samples)

    print(
        f"[scan] model={model_path}  ICs={n_ic}  "
        f"grid={len(grid)}  "
        f"qj={qj_grid}  lambda={lam_grid}  mu={mu_grid}  "
        f"sigma_r/rdot={base_cfg['sigma_r']}/{base_cfg['sigma_rdot']}"
    )

    rows: list[dict] = []
    for k, (qj, lam, mu) in enumerate(grid, start=1):
        cfg = dict(base_cfg)
        cfg["kf_qj"] = float(qj)
        cfg["rls_lambda"] = float(lam)
        cfg["rls_mu"] = float(mu)
        row = run_one_cfg(
            env,
            model,
            cfg,
            n_ic=n_ic,
            noise_seed=args.noise_seed,
            seed=args.seed,
            theta_thresh=args.theta_rmse_thresh,
            theta_hold=args.theta_hold,
        )
        rows.append(row)
        print(
            f"  [{k:02d}/{len(grid)}] qj={qj:.3g} lam={lam:.4g} mu={mu:.3g}  "
            f"med_a={row.get('median_a_cmd_rmse_tail', float('nan')):.3g}  "
            f"med_theta={row.get('median_theta_rmse_tail', float('nan')):.3g}  "
            f"med_rdd={row.get('median_rddot_rmse_tail', float('nan')):.3g}"
        )

    env.close()

    valid = [r for r in rows if r.get("n", 0) > 0]
    # 主排序键：median a_cmd（对 IC 爆点更稳）；次键 median θ
    ranked = sorted(
        valid,
        key=lambda r: (
            r["median_a_cmd_rmse_tail"],
            r["median_theta_rmse_tail"],
            r["median_rddot_rmse_tail"],
        ),
    )
    mu_cmp = _mu_paired_delta(valid)
    best = ranked[0] if ranked else None

    summary = {
        "model": model_path,
        "episodes": n_ic,
        "sigma_r": base_cfg["sigma_r"],
        "sigma_rdot": base_cfg["sigma_rdot"],
        "grid": {
            "kf_qj": qj_grid,
            "rls_lambda": lam_grid,
            "rls_mu": mu_grid,
            "n": len(grid),
        },
        "best": best,
        "mu_vs_mu0": mu_cmp,
        "ranked_top10": ranked[:10],
        "all": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )

    # 扁平 CSV 便于表格查看
    csv_path = out_dir / "scan_table.csv"
    headers = [
        "kf_qj",
        "rls_lambda",
        "rls_mu",
        "median_a_cmd_rmse_tail",
        "mean_a_cmd_rmse_tail",
        "median_theta_rmse_tail",
        "mean_theta_rmse_tail",
        "median_rddot_rmse_tail",
        "mean_rddot_rmse_tail",
        "converge_rate",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in ranked:
            f.write(
                ",".join(
                    str(r.get(h, ""))
                    for h in headers
                )
                + "\n"
            )

    print(f"\n[scan] 结果写入 {out_dir}")
    if best:
        print(
            f"  BEST: qj={best['kf_qj']:.3g} lam={best['rls_lambda']:.4g} "
            f"mu={best['rls_mu']:.3g}  "
            f"med_a={best['median_a_cmd_rmse_tail']:.3g}  "
            f"med_theta={best['median_theta_rmse_tail']:.3g}"
        )
    if mu_cmp.get("n_pairs", 0):
        print(
            f"  mu vs mu0: mean d_med_a={mu_cmp['mean_delta_median_a_cmd']:.3g}  "
            f"better={100*mu_cmp['frac_mu_better_a_cmd']:.0f}%  "
            f"worse={100*mu_cmp['frac_mu_worse_a_cmd']:.0f}%  "
            f"recommend_disable_l1={mu_cmp['recommend_disable_l1']}"
        )


if __name__ == "__main__":
    main()
