"""四方法带噪 MC 中的 TD3 / BC-TD3 部分（不带辨识，coeff_mode=analytic_noisy）。

同一 checkpoint、同一 test-IC 顺序、每局
noise_seed = noise_seed0 + local_idx。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
    plot_table2_matlab/mc_noisy_300/run_mc_rl_noisy.py \\
    --algo td3 --episodes 300
  # 默认 σ×30；显式指定可用 --sigma-scale 1

  --algo bctd3  测 BC-TD3
  --algo both   依次测两者
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_GAIL = Path(__file__).resolve().parents[2]
_RL = _GAIL / "rl_td3"
sys.path.insert(0, str(_RL))

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from evasion_env import EvasionEnv, SAMPLES_TEST_FILE  # noqa: E402
from evaluate_td3 import print_stats, run_episode, save_mc_metrics  # noqa: E402
from ident_cfg import IDENT_CLI_DEFAULTS, matlab_cfg_from_args  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402
from td3_bc import TD3BC  # noqa: E402

DEFAULT_MODELS = {
    "td3": _GAIL
    / "td3_baseline"
    / "checkpoints"
    / "training_300k_20260730_1917"
    / "best"
    / "best_model",
    "bctd3": _GAIL
    / "rl_td3"
    / "checkpoints"
    / "training_300k_20260730_1917"
    / "best"
    / "best_model",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TD3/BC-TD3 带测量噪声 MC（analytic_noisy，无辨识）"
    )
    p.add_argument(
        "--algo",
        choices=["td3", "bctd3", "both"],
        default="both",
        help="测哪种策略（默认 both）",
    )
    p.add_argument("--model-td3", type=str, default=str(DEFAULT_MODELS["td3"]))
    p.add_argument("--model-bctd3", type=str, default=str(DEFAULT_MODELS["bctd3"]))
    p.add_argument("--samples", type=str, default=SAMPLES_TEST_FILE)
    p.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="取前 N 个有效 IC（0=全池）",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=0, help="每局实际种子 = 此值 + local_idx")
    p.add_argument(
        "--sigma-scale",
        type=float,
        # default=1.0,  # 原默认：1× 辨识基准 σ
        default=30.0,  # 2026-08-06：改为 30×
        help="相对 ident 默认 σ 的统一倍率（默认 30）",
    )
    p.add_argument("--action-mapping", type=str, default=DEFAULT_ACTION_MAPPING)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="结果目录（默认本目录 results/sigma_x<scale>）",
    )
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args()


def _scaled_sigmas(scale: float) -> dict:
    s = float(scale)
    return {
        "sigma_q": IDENT_CLI_DEFAULTS["sigma_q"] * s,
        "sigma_qdot": IDENT_CLI_DEFAULTS["sigma_qdot"] * s,
        "sigma_theta": IDENT_CLI_DEFAULTS["sigma_theta"] * s,
        "sigma_psi": IDENT_CLI_DEFAULTS["sigma_psi"] * s,
        "sigma_r": IDENT_CLI_DEFAULTS["sigma_r"] * s,
        "sigma_rdot": IDENT_CLI_DEFAULTS["sigma_rdot"] * s,
    }


def _build_cfg(args: argparse.Namespace) -> dict:
    ns = argparse.Namespace(**IDENT_CLI_DEFAULTS)
    ns.coeff_mode = "analytic_noisy"  # 不带辨识：带噪测量 → 解析系数
    ns.ident_shadow = False
    for k, v in _scaled_sigmas(args.sigma_scale).items():
        setattr(ns, k, v)
    ns.noise_seed = int(args.noise_seed)
    ns.action_mapping = args.action_mapping
    return matlab_cfg_from_args(ns, action_mapping=args.action_mapping)


def _load_model(model_path: str, env: EvasionEnv, prefer_td3bc: bool):
    path = model_path
    if path.endswith(".zip"):
        path = path[:-4]
    zip_path = path + ".zip"
    if not Path(zip_path).exists():
        raise FileNotFoundError(f"找不到模型: {zip_path}")
    if prefer_td3bc:
        try:
            return TD3BC.load(path, env=env)
        except Exception:
            pass
    return TD3.load(path, env=env)


def run_one_algo(
    algo: str,
    model_path: str,
    prefer_td3bc: bool,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    cfg0 = _build_cfg(args)
    print(f"\n===== {algo.upper()} / analytic_noisy / σ×{args.sigma_scale} =====")
    print(f"model: {model_path}")
    print(
        f"σ_q={cfg0['sigma_q']:.4g} rad  σ_qdot={cfg0['sigma_qdot']:.4g}  "
        f"σ_r={cfg0['sigma_r']:.4g}  noise_seed0={args.noise_seed}"
    )

    env = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        matlab_cfg=dict(cfg0),
    )
    n_ic = env.n_samples if args.episodes <= 0 else min(args.episodes, env.n_samples)
    model = _load_model(model_path, env, prefer_td3bc)

    results = []
    n_skip = 0
    t0 = datetime.now()
    for i in range(n_ic):
        cfg_i = dict(cfg0)
        cfg_i["noise_seed"] = int(args.noise_seed) + i
        env.matlab_cfg = cfg_i

        ep = run_episode(
            env,
            model,
            seed=args.seed if i == 0 else None,
            local_idx=i,
        )
        if ep is None:
            n_skip += 1
            print(f"  [skip local_idx={i}] warm-start 失败")
            continue
        results.append(ep)
        done = len(results)
        if done % args.progress_every == 0 or i + 1 == n_ic:
            both = sum(1 for r in results if r["both_ok"])
            rT = ep["r_minT"]
            rT_s = f"{rT:.2f}" if np.isfinite(rT) else "nan"
            print(
                f"  [{done:3d}/{n_ic}] both={100 * both / done:.1f}%  "
                f"rMD=[{ep['r_MD1_min']:.1f},{ep['r_MD2_min']:.1f}] rT={rT_s}"
            )

    elapsed = (datetime.now() - t0).total_seconds()
    env.close()

    label = f"{algo.upper()} analytic_noisy σ×{args.sigma_scale}"
    print_stats(label, results)
    if n_skip:
        print(f"[mc] warm-start 跳过 {n_skip}/{n_ic}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = out_dir / f"mc_{algo}_N{len(results)}_noisy_x{args.sigma_scale}_{ts}.npz"
    if results:
        save_mc_metrics(results, npz_path)

    summary = {
        "algo": algo,
        "coeff_mode": "analytic_noisy",
        "sigma_scale": args.sigma_scale,
        "noise_seed0": args.noise_seed,
        "model": model_path,
        "n": len(results),
        "n_skip": n_skip,
        "elapsed_s": elapsed,
        "rate_evade": float(np.mean([r["evade_ok"] for r in results])) if results else 0.0,
        "rate_strike": float(np.mean([r["strike_ok"] for r in results])) if results else 0.0,
        "rate_both": float(np.mean([r["both_ok"] for r in results])) if results else 0.0,
        "rate_intercepted": float(np.mean([r["intercepted"] for r in results]))
        if results
        else 0.0,
        "metrics_file": str(npz_path) if results else "",
        "cfg": {k: cfg0[k] for k in (
            "coeff_mode", "sigma_q", "sigma_qdot", "sigma_theta", "sigma_psi",
            "sigma_r", "sigma_rdot",
        )},
    }
    (out_dir / f"summary_{algo}_x{args.sigma_scale}_{ts}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[mc] 已保存: {npz_path}")
    return summary


def main() -> None:
    args = parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = f"sigma_x{args.sigma_scale:g}"
        out_dir = Path(__file__).resolve().parent / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[mc] samples={args.samples}  episodes={args.episodes}  "
        f"σ×{args.sigma_scale}  out={out_dir}"
    )

    jobs = []
    if args.algo in ("td3", "both"):
        jobs.append(("td3", args.model_td3, False))
    if args.algo in ("bctd3", "both"):
        jobs.append(("bctd3", args.model_bctd3, True))

    summaries = []
    for algo, model, prefer_bc in jobs:
        summaries.append(run_one_algo(algo, model, prefer_bc, args, out_dir))

    print("\n==================== RL 汇总 ====================")
    print(f"{'方法':<8} {'N':>6} {'双规避':>12} {'打击':>12} {'联合':>12}")
    for s in summaries:
        n = s["n"]
        print(
            f"{s['algo']:<8} {n:6d} "
            f"{100 * s['rate_evade']:5.1f}%{'':>6} "
            f"{100 * s['rate_strike']:5.1f}%{'':>6} "
            f"{100 * s['rate_both']:5.1f}%"
        )
    print("=================================================")


if __name__ == "__main__":
    main()
