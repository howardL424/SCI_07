"""纯 TD3 三模式配对验证：analytic_truth / analytic_noisy / sparse_ffrls。

同一 checkpoint、同一 test IC、同一 noise_seed，比较闭环指标。
另可选 --also-eta0 追加 sparse_ffrls(eta=0)。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
    td3_baseline/run_ident_ablation_pure.py \\
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
from evaluate_td3 import run_episode  # noqa: E402
from ident_cfg import IDENT_CLI_DEFAULTS, matlab_cfg_from_args  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="纯 TD3 参数辨识配对消融")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--samples", type=str, default=SAMPLES_TEST_FILE)
    p.add_argument("--episodes", type=int, default=5, help="取前 N 个有效 IC")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--action-mapping", type=str, default=DEFAULT_ACTION_MAPPING)
    p.add_argument(
        "--also-eta0",
        "--also-mu0",
        dest="also_eta0",
        action="store_true",
        help="追加 eta=0 对照（--also-mu0 为旧兼容别名）",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="结果目录（默认模型旁 ident_ablation_{ts}）",
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
    p.add_argument("--kf-sigma-r", type=float, default=IDENT_CLI_DEFAULTS["kf_sigma_r"])
    p.add_argument(
        "--kf-sigma-rdot", type=float, default=IDENT_CLI_DEFAULTS["kf_sigma_rdot"]
    )
    p.add_argument("--rls-lambda", type=float, default=IDENT_CLI_DEFAULTS["rls_lambda"])
    p.add_argument("--rls-mu", type=float, default=IDENT_CLI_DEFAULTS["rls_mu"])
    p.add_argument("--rls-eta", type=float, default=IDENT_CLI_DEFAULTS["rls_eta"])
    p.add_argument("--rls-P0", type=float, default=IDENT_CLI_DEFAULTS["rls_P0"])
    p.add_argument("--rls-r0", type=float, default=IDENT_CLI_DEFAULTS["rls_r0"])
    p.add_argument("--rls-a0", type=float, default=IDENT_CLI_DEFAULTS["rls_a0"])
    return p.parse_args()


def _modes(args: argparse.Namespace) -> list[tuple[str, dict]]:
    base = argparse.Namespace(**vars(args))
    base.ident_shadow = False
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


def main() -> None:
    args = parse_args()
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
        out_dir = Path(model_path).parent.parent / f"ident_ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = _modes(args)
    summary = {"model": model_path, "episodes": args.episodes, "modes": {}}

    # 先建一个 env 只为列 IC；各模式各自起 engine 以免 persistent 串扰
    probe = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        matlab_cfg={"coeff_mode": "analytic_truth"},
    )
    n_ic = min(args.episodes, probe.n_samples)
    probe.close()

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

        if not rows:
            summary["modes"][mode_name] = {"n": 0}
            continue
        d1 = np.array([r["r_MD1_min"] for r in rows], dtype=float)
        d2 = np.array([r["r_MD2_min"] for r in rows], dtype=float)
        rT = np.array([r["r_minT"] for r in rows], dtype=float)
        both = np.array([r["both_ok"] for r in rows], dtype=bool)
        energy = np.array([r["a_energy"] for r in rows], dtype=float)
        total_variation = np.array(
            [r["a_total_variation"] for r in rows], dtype=float
        )
        saturation_rate = np.array(
            [r["saturation_rate"] for r in rows], dtype=float
        )
        summary["modes"][mode_name] = {
            "n": len(rows),
            "both_ok_rate": float(both.mean()),
            "r_MD1_min_mean": float(d1.mean()),
            "r_MD2_min_mean": float(d2.mean()),
            "r_minT_mean": float(np.nanmean(rT)),
            "a_energy_mean": float(energy.mean()),
            "a_total_variation_mean": float(total_variation.mean()),
            "saturation_rate_mean": float(saturation_rate.mean()),
            "cfg": cfg,
        }
        np.savez(
            out_dir / f"metrics_{mode_name}.npz",
            r_MD1_min=d1,
            r_MD2_min=d2,
            r_minT=rT,
            both_ok=both,
            a_energy=energy,
            a_total_variation=total_variation,
            saturation_rate=saturation_rate,
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[ablation] 结果写入 {out_dir}")
    for name, st in summary["modes"].items():
        if st.get("n", 0) == 0:
            continue
        print(
            f"  {name}: both_ok={100*st['both_ok_rate']:.1f}%  "
            f"r1={st['r_MD1_min_mean']:.1f} r2={st['r_MD2_min_mean']:.1f} "
            f"rT={st['r_minT_mean']:.2f}  "
            f"sat={100*st['saturation_rate_mean']:.1f}%"
        )


if __name__ == "__main__":
    main()
