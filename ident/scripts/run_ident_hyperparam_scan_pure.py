"""纯 TD3 辨识超参网格：KF(Q/R) × lambda × eta（影子诊断）。

复用 run_ident_shadow_diag_pure.run_shadow_episode，对同一 checkpoint / IC
比较末段归一化 θ / r̈ / 饱和指令 RMSE，并汇总 eta>0 vs eta=0。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
    ident/scripts/run_ident_hyperparam_scan_pure.py \\
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

_SCRIPT = Path(__file__).resolve().parent
_IDENT = _SCRIPT.parent
_RL = _IDENT.parent / "rl_td3"
sys.path.insert(0, str(_SCRIPT))
sys.path.insert(0, str(_IDENT))
sys.path.insert(0, str(_RL))

from paths import default_results_dir  # noqa: E402

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
_DEFAULT_KF_QJ = [1e3, 1e4]
_DEFAULT_SIGMA_R = [5.0, 10.0]
_DEFAULT_SIGMA_RDOT = [0.5, 1.0]
_DEFAULT_LAMBDA = [0.95, 0.98, 0.995]
_DEFAULT_ETA = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


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
    p.add_argument("--sigma-scale", type=float, default=IDENT_CLI_DEFAULTS["sigma_scale"])
    p.add_argument("--sigma-q", type=float, default=IDENT_CLI_DEFAULTS["sigma_q"])
    p.add_argument("--sigma-qdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_qdot"])
    p.add_argument("--sigma-theta", type=float, default=IDENT_CLI_DEFAULTS["sigma_theta"])
    p.add_argument("--sigma-psi", type=float, default=IDENT_CLI_DEFAULTS["sigma_psi"])
    p.add_argument("--kf-sigma-a0", type=float, default=IDENT_CLI_DEFAULTS["kf_sigma_a0"])
    p.add_argument("--rls-mu", type=float, default=IDENT_CLI_DEFAULTS["rls_mu"])
    p.add_argument("--rls-P0", type=float, default=IDENT_CLI_DEFAULTS["rls_P0"])
    p.add_argument(
        "--rls-P-trace-max",
        type=float,
        default=IDENT_CLI_DEFAULTS["rls_P_trace_max"],
    )
    p.add_argument("--rls-r0", type=float, default=IDENT_CLI_DEFAULTS["rls_r0"])
    p.add_argument("--rls-a0", type=float, default=IDENT_CLI_DEFAULTS["rls_a0"])
    p.add_argument(
        "--kf-qj-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_KF_QJ),
    )
    p.add_argument(
        "--sigma-r-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_SIGMA_R),
        help="距离量测标准差网格；同时作为式(62d)的 KF R",
    )
    p.add_argument(
        "--sigma-rdot-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_SIGMA_RDOT),
        help="距离率量测标准差网格；同时作为式(62d)的 KF R",
    )
    p.add_argument(
        "--rls-lambda-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_LAMBDA),
    )
    p.add_argument(
        "--rls-eta-grid",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_ETA),
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
        "sigma_r": float(cfg["sigma_r"]),
        "sigma_rdot": float(cfg["sigma_rdot"]),
        "rls_lambda": float(cfg["rls_lambda"]),
        "rls_eta": float(cfg["rls_eta"]),
    }
    for key in (
        "theta_rmse_tail",
        "theta_norm_rmse_tail",
        "a21_norm_rmse_tail",
        "b_rmse_tail",
        "c1_norm_rmse_tail",
        "qdot_rmse_tail",
        "rddot_rmse_tail",
        "a_cmd_rmse_tail",
        "a_cmd_sat_rmse_tail",
        "P_max_tail",
        "P_cond_max_tail",
        "r_rmse_tail",
        "rdot_rmse_tail",
    ):
        out.update(_agg_eps(ep_stats, key))
    out["per_episode"] = [
        {
            "local_idx": e["local_idx"],
            "theta_rmse_tail": e["theta_rmse_tail"],
            "theta_norm_rmse_tail": e["theta_norm_rmse_tail"],
            "qdot_rmse_tail": e["qdot_rmse_tail"],
            "rddot_rmse_tail": e["rddot_rmse_tail"],
            "a_cmd_sat_rmse_tail": e["a_cmd_sat_rmse_tail"],
            "converge_step": e["converge_step"],
        }
        for e in ep_stats
    ]
    return out


def _eta_paired_delta(rows: list[dict]) -> dict:
    """相同 KF/λ 下比较 eta>0 相对 eta=0 的末段中位指标。"""
    by_base: dict[tuple[float, float, float, float], dict[float, dict]] = {}
    for r in rows:
        if r.get("n", 0) == 0:
            continue
        key = (
            r["kf_qj"],
            r["sigma_r"],
            r["sigma_rdot"],
            r["rls_lambda"],
        )
        by_base.setdefault(key, {})[r["rls_eta"]] = r

    pairs = []
    for (qj, sig_r, sig_rd, lam), etas in by_base.items():
        if 0.0 not in etas:
            continue
        base = etas[0.0]
        for eta, row in etas.items():
            if eta == 0.0:
                continue
            pairs.append(
                {
                    "kf_qj": qj,
                    "sigma_r": sig_r,
                    "sigma_rdot": sig_rd,
                    "rls_lambda": lam,
                    "rls_eta": eta,
                    "delta_median_a_cmd_sat": row["median_a_cmd_sat_rmse_tail"]
                    - base["median_a_cmd_sat_rmse_tail"],
                    "delta_median_theta_norm": row["median_theta_norm_rmse_tail"]
                    - base["median_theta_norm_rmse_tail"],
                    "delta_median_rddot": row["median_rddot_rmse_tail"]
                    - base["median_rddot_rmse_tail"],
                    "a_cmd_eta0": base["median_a_cmd_sat_rmse_tail"],
                    "a_cmd_eta": row["median_a_cmd_sat_rmse_tail"],
                    "theta_eta0": base["median_theta_norm_rmse_tail"],
                    "theta_eta": row["median_theta_norm_rmse_tail"],
                }
            )
    if not pairs:
        return {"n_pairs": 0}

    d_a = np.array([p["delta_median_a_cmd_sat"] for p in pairs], dtype=float)
    d_th = np.array([p["delta_median_theta_norm"] for p in pairs], dtype=float)
    # 负值 = eta>0 更好
    better_a = int(np.sum(d_a < -1e-9))
    worse_a = int(np.sum(d_a > 1e-9))
    return {
        "n_pairs": len(pairs),
        "mean_delta_median_a_cmd_sat": float(np.nanmean(d_a)),
        "mean_delta_median_theta_norm": float(np.nanmean(d_th)),
        "frac_eta_better_a_cmd": better_a / len(pairs),
        "frac_eta_worse_a_cmd": worse_a / len(pairs),
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
    sig_r_grid = _parse_float_list(args.sigma_r_grid)
    sig_rd_grid = _parse_float_list(args.sigma_rdot_grid)
    lam_grid = _parse_float_list(args.rls_lambda_grid)
    eta_grid = _parse_float_list(args.rls_eta_grid)
    grid = list(itertools.product(qj_grid, sig_r_grid, sig_rd_grid, lam_grid, eta_grid))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = default_results_dir(model_path, "ident_hyperparam_scan", ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 基准 cfg（网格字段稍后覆盖）
    args.coeff_mode = "sparse_ffrls"
    args.ident_shadow = True
    args.kf_qj = qj_grid[0]
    args.sigma_r = sig_r_grid[0]
    args.sigma_rdot = sig_rd_grid[0]
    args.rls_lambda = lam_grid[0]
    args.rls_eta = eta_grid[0]
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
        f"qj={qj_grid}  sigma_r/rdot={sig_r_grid}/{sig_rd_grid}  "
        f"lambda={lam_grid}  eta={eta_grid}"
    )

    rows: list[dict] = []
    for k, (qj, sig_r, sig_rd, lam, eta) in enumerate(grid, start=1):
        cfg = dict(base_cfg)
        cfg["kf_qj"] = float(qj)
        cfg["sigma_r"] = float(sig_r)
        cfg["sigma_rdot"] = float(sig_rd)
        cfg["rls_lambda"] = float(lam)
        cfg["rls_eta"] = float(eta)
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
            f"  [{k:02d}/{len(grid)}] qj={qj:.3g} "
            f"R={sig_r:.3g}/{sig_rd:.3g} lam={lam:.4g} eta={eta:.3g}  "
            f"med_a_sat={row.get('median_a_cmd_sat_rmse_tail', float('nan')):.3g}  "
            f"med_theta_norm={row.get('median_theta_norm_rmse_tail', float('nan')):.3g}  "
            f"med_rdd={row.get('median_rddot_rmse_tail', float('nan')):.3g}"
        )

    env.close()

    valid = [r for r in rows if r.get("n", 0) > 0]
    # 主排序键：饱和后指令误差；次键归一化参数误差。
    ranked = sorted(
        valid,
        key=lambda r: (
            r["median_a_cmd_sat_rmse_tail"],
            r["median_theta_norm_rmse_tail"],
            r["median_rddot_rmse_tail"],
        ),
    )
    eta_cmp = _eta_paired_delta(valid)
    best = ranked[0] if ranked else None

    summary = {
        "model": model_path,
        "episodes": n_ic,
        "sigma_r": base_cfg["sigma_r"],
        "sigma_rdot": base_cfg["sigma_rdot"],
        "grid": {
            "kf_qj": qj_grid,
            "sigma_r": sig_r_grid,
            "sigma_rdot": sig_rd_grid,
            "rls_lambda": lam_grid,
            "rls_eta": eta_grid,
            "n": len(grid),
        },
        "best": best,
        "eta_vs_eta0": eta_cmp,
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
        "sigma_r",
        "sigma_rdot",
        "rls_lambda",
        "rls_eta",
        "median_a_cmd_sat_rmse_tail",
        "mean_a_cmd_sat_rmse_tail",
        "median_theta_norm_rmse_tail",
        "mean_theta_norm_rmse_tail",
        "median_qdot_rmse_tail",
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
            f"  BEST: qj={best['kf_qj']:.3g} "
            f"R={best['sigma_r']:.3g}/{best['sigma_rdot']:.3g} "
            f"lam={best['rls_lambda']:.4g} eta={best['rls_eta']:.3g}  "
            f"med_a_sat={best['median_a_cmd_sat_rmse_tail']:.3g}  "
            f"med_theta_norm={best['median_theta_norm_rmse_tail']:.3g}"
        )
    if eta_cmp.get("n_pairs", 0):
        print(
            f"  eta vs eta0: mean d_med_a_sat="
            f"{eta_cmp['mean_delta_median_a_cmd_sat']:.3g}  "
            f"better={100*eta_cmp['frac_eta_better_a_cmd']:.0f}%  "
            f"worse={100*eta_cmp['frac_eta_worse_a_cmd']:.0f}%"
        )


if __name__ == "__main__":
    main()
