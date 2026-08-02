"""纯 TD3 影子辨识诊断：sparse_ffrls + ident_shadow，落盘逐步 RMSE。

记录相对真值：
  - KF r / rdot / rddot 误差
  - 11 维 θ RMSE（D1:4 + D2:4 + T:3）
  - 收敛步数（θ RMSE 首次连续 K 步低于阈值）
  - 指令预测误差 ||a_hat - a_truth||

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
    td3_baseline/run_ident_shadow_diag_pure.py \\
    --model td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model \\
    --episodes 5 --noise-seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_RL = _ROOT.parent / "rl_td3"
sys.path.insert(0, str(_RL))

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from evasion_env import EvasionEnv, SAMPLES_TEST_FILE  # noqa: E402
from ident_cfg import IDENT_CLI_DEFAULTS, matlab_cfg_from_args, save_ident_run_config  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402

# θ RMSE 收敛：连续 hold 步低于阈值
_THETA_RMSE_THRESH = 0.5
_THETA_HOLD = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="纯 TD3 影子辨识逐步诊断")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--samples", type=str, default=SAMPLES_TEST_FILE)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--action-mapping", type=str, default=DEFAULT_ACTION_MAPPING)
    p.add_argument("--rls-mu", type=float, default=IDENT_CLI_DEFAULTS["rls_mu"])
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--sigma-q", type=float, default=IDENT_CLI_DEFAULTS["sigma_q"])
    p.add_argument("--sigma-qdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_qdot"])
    p.add_argument("--sigma-theta", type=float, default=IDENT_CLI_DEFAULTS["sigma_theta"])
    p.add_argument("--sigma-psi", type=float, default=IDENT_CLI_DEFAULTS["sigma_psi"])
    p.add_argument("--sigma-r", type=float, default=IDENT_CLI_DEFAULTS["sigma_r"])
    p.add_argument("--sigma-rdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_rdot"])
    p.add_argument("--kf-qj", type=float, default=IDENT_CLI_DEFAULTS["kf_qj"])
    p.add_argument("--kf-sigma-a0", type=float, default=IDENT_CLI_DEFAULTS["kf_sigma_a0"])
    p.add_argument("--rls-lambda", type=float, default=IDENT_CLI_DEFAULTS["rls_lambda"])
    p.add_argument("--rls-P0", type=float, default=IDENT_CLI_DEFAULTS["rls_P0"])
    p.add_argument("--rls-r0", type=float, default=IDENT_CLI_DEFAULTS["rls_r0"])
    p.add_argument("--rls-a0", type=float, default=IDENT_CLI_DEFAULTS["rls_a0"])
    p.add_argument(
        "--theta-rmse-thresh",
        type=float,
        default=_THETA_RMSE_THRESH,
        help="收敛判定：θ RMSE 阈值",
    )
    p.add_argument(
        "--theta-hold",
        type=int,
        default=_THETA_HOLD,
        help="收敛判定：连续低于阈值的步数",
    )
    return p.parse_args()


def _first_converge_step(
    theta_rmse: np.ndarray, thresh: float, hold: int, *, min_start: int = 10
) -> float:
    """首次从 min_start 起连续 hold 步 RMSE<thresh 的起始步；未收敛则 NaN。

    min_start 避开首拍 warm-start 偶然偏低、随后发散的误报。
    """
    ok = np.isfinite(theta_rmse) & (theta_rmse < thresh)
    start = max(0, int(min_start))
    if hold <= 1:
        idx = np.where(ok[start:])[0]
        return float(idx[0] + start) if idx.size else float("nan")
    run = 0
    for i in range(start, len(ok)):
        run = run + 1 if ok[i] else 0
        if run >= hold:
            return float(i - hold + 1)
    return float("nan")


def run_shadow_episode(
    env: EvasionEnv,
    model: TD3,
    *,
    local_idx: int,
    seed: int | None,
    theta_thresh: float,
    theta_hold: int,
) -> dict | None:
    try:
        obs, info0 = env.reset(seed=seed, options={"local_idx": int(local_idx)})
    except RuntimeError as e:
        if "warm-start" in str(e):
            return None
        raise

    rows: list[dict] = []
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, _, info = env.step(action)
        d = info.get("ident_diag")
        if d is None:
            continue
        th_err = np.asarray(d["theta_err"], dtype=float)
        r_err = np.asarray(d["r_err"], dtype=float)
        rd_err = np.asarray(d["rdot_err"], dtype=float)
        rdd_err = np.asarray(d["rddot_err"], dtype=float)
        a_err = np.asarray(d["a_cmd_err"], dtype=float)
        rows.append(
            {
                "r_rmse": float(np.sqrt(np.mean(r_err**2))),
                "rdot_rmse": float(np.sqrt(np.mean(rd_err**2))),
                "rddot_rmse": float(np.sqrt(np.nanmean(rdd_err**2)))
                if np.isfinite(rdd_err).any()
                else float("nan"),
                "theta_rmse": float(np.sqrt(np.mean(th_err**2))),
                "a_cmd_rmse": float(np.sqrt(np.nanmean(a_err**2)))
                if np.isfinite(a_err).any()
                else float("nan"),
                "did_rls": bool(d["did_rls"]),
                "theta_err": th_err,
                "r_err": r_err,
                "rdot_err": rd_err,
                "rddot_err": rdd_err,
                "a_cmd_err": a_err,
            }
        )

    if not rows:
        return None

    theta_rmse = np.array([r["theta_rmse"] for r in rows], dtype=float)
    conv = _first_converge_step(theta_rmse, theta_thresh, theta_hold)
    # 末段（后 20% 或至少末 5 步）均值
    n_tail = max(5, int(0.2 * len(rows)))
    tail = rows[-n_tail:]

    def _mean_key(key: str) -> float:
        v = np.array([r[key] for r in tail], dtype=float)
        return float(np.nanmean(v))

    theta_tail = _mean_key("theta_rmse")
    # 仅当末段也低于阈值才算真正收敛（避免中途短暂达标后发散）
    if not (np.isfinite(conv) and theta_tail < theta_thresh):
        conv = float("nan")

    return {
        "local_idx": int(info0.get("local_idx", local_idx)),
        "row_idx": int(info0.get("row_idx", -1)),
        "steps": len(rows),
        "converge_step": conv,
        "theta_rmse_final": float(theta_rmse[-1]),
        "theta_rmse_tail": theta_tail,
        "r_rmse_tail": _mean_key("r_rmse"),
        "rdot_rmse_tail": _mean_key("rdot_rmse"),
        "rddot_rmse_tail": _mean_key("rddot_rmse"),
        "a_cmd_rmse_tail": _mean_key("a_cmd_rmse"),
        "theta_rmse_series": theta_rmse,
        "r_rmse_series": np.array([r["r_rmse"] for r in rows], dtype=float),
        "rdot_rmse_series": np.array([r["rdot_rmse"] for r in rows], dtype=float),
        "rddot_rmse_series": np.array([r["rddot_rmse"] for r in rows], dtype=float),
        "a_cmd_rmse_series": np.array([r["a_cmd_rmse"] for r in rows], dtype=float),
        "theta_err_last": rows[-1]["theta_err"],
    }


def main() -> None:
    args = parse_args()
    model_path = args.model
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    zip_path = model_path + ".zip"
    if not Path(zip_path).exists():
        print(f"[shadow] 找不到模型 {zip_path}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(model_path).parent.parent / f"ident_shadow_diag_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args.coeff_mode = "sparse_ffrls"
    args.ident_shadow = True
    cfg = matlab_cfg_from_args(args)
    save_ident_run_config(out_dir / "ident_run_config.json", cfg, extra={"task": "shadow_diag"})

    env = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        matlab_cfg=cfg,
    )
    model = TD3.load(model_path, env=env)
    n_ic = min(args.episodes, env.n_samples)

    ep_stats = []
    print(
        f"[shadow] model={model_path}  episodes={n_ic}  "
        f"sigma_r={cfg['sigma_r']} sigma_rdot={cfg['sigma_rdot']}  "
        f"rls_mu={cfg['rls_mu']}  shadow=True"
    )
    for i in range(n_ic):
        cfg_i = dict(cfg)
        cfg_i["noise_seed"] = int(args.noise_seed) + i
        env.matlab_cfg = cfg_i
        ep = run_shadow_episode(
            env,
            model,
            local_idx=i,
            seed=args.seed if i == 0 else None,
            theta_thresh=args.theta_rmse_thresh,
            theta_hold=args.theta_hold,
        )
        if ep is None:
            print(f"  [skip] local_idx={i}")
            continue
        ep_stats.append(ep)
        conv_s = (
            f"{ep['converge_step']:.0f}"
            if np.isfinite(ep["converge_step"])
            else "nan"
        )
        print(
            f"  IC{i}: steps={ep['steps']}  "
            f"θ_rmse_tail={ep['theta_rmse_tail']:.3g}  "
            f"r_rmse_tail={ep['r_rmse_tail']:.3g}  "
            f"rdot_rmse_tail={ep['rdot_rmse_tail']:.3g}  "
            f"rddot_rmse_tail={ep['rddot_rmse_tail']:.3g}  "
            f"a_cmd_rmse_tail={ep['a_cmd_rmse_tail']:.3g}  "
            f"conv_step={conv_s}"
        )

    env.close()
    if not ep_stats:
        print("[shadow] 无有效 episode")
        sys.exit(1)

    def _agg(key: str) -> float:
        vals = [e[key] for e in ep_stats]
        arr = np.asarray(vals, dtype=float)
        if not np.isfinite(arr).any():
            return float("nan")
        return float(np.nanmean(arr))

    summary = {
        "model": model_path,
        "episodes": len(ep_stats),
        "cfg": cfg,
        "theta_rmse_thresh": args.theta_rmse_thresh,
        "theta_hold": args.theta_hold,
        "mean_theta_rmse_tail": _agg("theta_rmse_tail"),
        "mean_r_rmse_tail": _agg("r_rmse_tail"),
        "mean_rdot_rmse_tail": _agg("rdot_rmse_tail"),
        "mean_rddot_rmse_tail": _agg("rddot_rmse_tail"),
        "mean_a_cmd_rmse_tail": _agg("a_cmd_rmse_tail"),
        "mean_converge_step": _agg("converge_step"),
        "converge_rate": float(
            np.mean([np.isfinite(e["converge_step"]) for e in ep_stats])
        ),
        "per_episode": [
            {
                "local_idx": e["local_idx"],
                "steps": e["steps"],
                "converge_step": e["converge_step"],
                "theta_rmse_tail": e["theta_rmse_tail"],
                "r_rmse_tail": e["r_rmse_tail"],
                "rdot_rmse_tail": e["rdot_rmse_tail"],
                "rddot_rmse_tail": e["rddot_rmse_tail"],
                "a_cmd_rmse_tail": e["a_cmd_rmse_tail"],
            }
            for e in ep_stats
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    np.savez_compressed(
        out_dir / "shadow_diag_series.npz",
        local_idx=np.array([e["local_idx"] for e in ep_stats], dtype=np.int64),
        converge_step=np.array([e["converge_step"] for e in ep_stats], dtype=float),
        theta_rmse_tail=np.array([e["theta_rmse_tail"] for e in ep_stats], dtype=float),
        r_rmse_tail=np.array([e["r_rmse_tail"] for e in ep_stats], dtype=float),
        rdot_rmse_tail=np.array([e["rdot_rmse_tail"] for e in ep_stats], dtype=float),
        rddot_rmse_tail=np.array([e["rddot_rmse_tail"] for e in ep_stats], dtype=float),
        a_cmd_rmse_tail=np.array([e["a_cmd_rmse_tail"] for e in ep_stats], dtype=float),
        # 变长序列：用 object 数组存；读取时自行处理
        theta_rmse_series=np.array(
            [e["theta_rmse_series"] for e in ep_stats], dtype=object
        ),
        r_rmse_series=np.array([e["r_rmse_series"] for e in ep_stats], dtype=object),
        rdot_rmse_series=np.array(
            [e["rdot_rmse_series"] for e in ep_stats], dtype=object
        ),
        rddot_rmse_series=np.array(
            [e["rddot_rmse_series"] for e in ep_stats], dtype=object
        ),
        a_cmd_rmse_series=np.array(
            [e["a_cmd_rmse_series"] for e in ep_stats], dtype=object
        ),
    )

    print(f"\n[shadow] 结果写入 {out_dir}")
    print(
        f"  mean θ_rmse_tail={summary['mean_theta_rmse_tail']:.3g}  "
        f"r={summary['mean_r_rmse_tail']:.3g}  "
        f"rdot={summary['mean_rdot_rmse_tail']:.3g}  "
        f"rddot={summary['mean_rddot_rmse_tail']:.3g}  "
        f"a_cmd={summary['mean_a_cmd_rmse_tail']:.3g}"
    )
    print(
        f"  converge_rate={100*summary['converge_rate']:.0f}%  "
        f"mean_converge_step={summary['mean_converge_step']:.1f}  "
        f"(thresh={args.theta_rmse_thresh}, hold={args.theta_hold})"
    )


if __name__ == "__main__":
    main()
