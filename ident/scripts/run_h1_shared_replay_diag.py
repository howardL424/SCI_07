"""H1 共享回放逐拍辨识诊断（仅训练/验证数据）。

影子模式下制导指令恒由 ``analytic_truth`` 产生，因此一次运行得到的单条轨迹
同时承载三路因果估计，直接误差比较不再被两条闭环轨迹分叉混淆：

  - ``analytic_noisy``      含噪量测代入解析式，vartheta_noisy[k-1]
  - ``sparse_ffrls``        本拍因果 FF-RLS 估计 theta_hat[k]
  - ``zoh_drift_floor``     神谕估计器 vartheta_true[k]，仍按协议与
                            vartheta_true[k-1] 比较

第三路是与估计器无关的误差下界：它度量"同一制导周期内真值参数保持常值"
这一约定被违背的代价。若 ``analytic_noisy`` 的 NRMSE 已经接近该下界，
则 H2 的"FF-RLS 误差低于含噪解析基线"验收条件在当前问题设定下不可达，
必须先改口径而不是继续调 lambda/eta。

NRMSE 定义直接复用 ``evaluate_td3._theta_scaled_nrmse``，与 P0-4 正式结果同口径。

用法::

    E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
      ident/scripts/run_h1_shared_replay_diag.py --episodes 5
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

from paths import IDENT_RESULTS  # noqa: E402

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from evaluate_td3 import (  # noqa: E402
    DEFAULT_NRMSE_BURN_IN_STEPS,
    DEFAULT_NRMSE_DENOM_FLOOR,
    _metric_vector,
    _theta_scaled_nrmse,
    tee_terminal_log,
)
from evasion_env import EvasionEnv, SAMPLES_TEST_FILE, SAMPLES_VAL_FILE  # noqa: E402
from ident_cfg import IDENT_CLI_DEFAULTS, matlab_cfg_from_args  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402

PARAM_NAMES = (
    "A21_D1",
    "B1_D1",
    "B2_D1",
    "C1_D1",
    "A21_D2",
    "B1_D2",
    "B2_D2",
    "C1_D2",
    "A21_T",
    "B1_T",
    "B2_T",
)

ESTIMATORS = ("analytic_noisy", "sparse_ffrls", "zoh_drift_floor")

DEFAULT_TD3_MODEL = (
    "td3_baseline/checkpoints/training_300k_20260730_1917/best/best_model"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H1 共享回放逐拍辨识诊断")
    p.add_argument("--model", default=DEFAULT_TD3_MODEL)
    p.add_argument("--samples", default=SAMPLES_VAL_FILE)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=0, help="局级种子基址")
    p.add_argument("--action-mapping", default=DEFAULT_ACTION_MAPPING)
    p.add_argument("--actor-obs-mode", choices=["truth", "noisy", "filtered"],
                   default="filtered")
    p.add_argument("--nrmse-burn-in-steps", type=int,
                   default=DEFAULT_NRMSE_BURN_IN_STEPS)
    p.add_argument("--nrmse-denominator-floor", type=float,
                   default=DEFAULT_NRMSE_DENOM_FLOOR)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--allow-test-set",
        action="store_true",
        help="解除测试集保护；H1/H2 诊断阶段不得使用",
    )
    for name in (
        "sigma_scale", "sigma_q", "sigma_qdot", "sigma_theta", "sigma_psi",
        "sigma_r", "sigma_rdot", "kf_qj", "kf_sigma_a0", "rls_lambda",
        "rls_mu", "rls_eta", "rls_P0", "rls_P_trace_max", "rls_r0", "rls_a0",
        "rls_gate_r_min", "rls_gate_tgo_min",
    ):
        p.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=IDENT_CLI_DEFAULTS[name],
        )
    args = p.parse_args()
    args.coeff_mode = "sparse_ffrls"
    args.ident_shadow = True
    return args


def _guard_dataset(samples: str, allow_test: bool) -> None:
    """阻止 H1 诊断阶段读取 P0-4 最终测试集。"""
    if allow_test:
        return
    if Path(samples).resolve() == Path(SAMPLES_TEST_FILE).resolve():
        raise SystemExit(
            "拒绝运行：samples 指向 P0-4 最终测试集。"
            "H1 诊断只允许训练/验证数据；确需覆盖请显式加 --allow-test-set。"
        )


def run_shared_replay_episode(
    env: EvasionEnv,
    model: TD3,
    *,
    local_idx: int,
    noise_seed: int,
    seed: int | None,
    burn_in: int,
    denom_floor: float,
) -> dict | None:
    """跑一局影子回放，返回三路因果估计的逐拍序列与 NRMSE。"""
    obs, info0 = env.reset(
        seed=seed, options={"local_idx": int(local_idx), "noise_seed": int(noise_seed)}
    )

    series: dict[str, list[np.ndarray]] = {
        "theta_hat": [],
        "theta_analytic_noisy_causal": [],
        "theta_true_causal": [],
        "theta_true_current": [],
    }
    frozen_rows: list[np.ndarray] = []
    p_trace_rows: list[np.ndarray] = []
    rddot_err_rows: list[np.ndarray] = []
    kf_err_rows: list[np.ndarray] = []
    a_cmd_rows: list[np.ndarray] = []
    a_cmd_sat_rows: list[np.ndarray] = []
    did_rls: list[bool] = []

    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, _truncated, info = env.step(action)
        d = info.get("ident_diag")
        if d is None:
            continue
        for key in series:
            series[key].append(_metric_vector(d.get(key), 11))
        frozen_rows.append(_metric_vector(d.get("frozen_model_residual"), 3))
        p_trace_rows.append(_metric_vector(d.get("P_trace"), 3))
        rddot_err_rows.append(_metric_vector(d.get("rddot_err"), 3))
        kf_err_rows.append(
            np.concatenate([
                _metric_vector(d.get("r_err"), 3),
                _metric_vector(d.get("rdot_err"), 3),
                _metric_vector(d.get("rddot_err"), 3),
            ])
        )
        a_cmd_rows.append(_metric_vector(d.get("a_cmd_err"), 2))
        a_cmd_sat_rows.append(_metric_vector(d.get("a_cmd_err_sat"), 2))
        did_rls.append(bool(d.get("did_rls", False)))

    if not series["theta_true_causal"]:
        return None

    r0 = float(env.matlab_cfg["rls_r0"])
    a0 = float(env.matlab_cfg["rls_a0"])
    truth = series["theta_true_causal"]
    estimator_rows = {
        "analytic_noisy": series["theta_analytic_noisy_causal"],
        "sparse_ffrls": series["theta_hat"],
        "zoh_drift_floor": series["theta_true_current"],
    }

    out: dict = {
        "local_idx": int(info0.get("local_idx", local_idx)),
        "row_idx": int(info0.get("row_idx", -1)),
        "noise_seed": int(info0.get("noise_seed", noise_seed)),
        "steps": len(truth),
        "did_rls_rate": float(np.mean(did_rls)) if did_rls else float("nan"),
    }
    for name, rows in estimator_rows.items():
        agg, comp, n_valid = _theta_scaled_nrmse(
            rows, truth, r0=r0, a0=a0, denominator_floor=denom_floor
        )
        agg_b, comp_b, n_valid_b = _theta_scaled_nrmse(
            rows, truth, r0=r0, a0=a0,
            start_step=burn_in, denominator_floor=denom_floor,
        )
        out[f"nrmse__{name}"] = agg
        out[f"nrmse_components__{name}"] = comp
        out[f"nrmse_valid_steps__{name}"] = int(n_valid)
        out[f"nrmse_post_burnin__{name}"] = agg_b
        out[f"nrmse_post_burnin_components__{name}"] = comp_b
        out[f"nrmse_post_burnin_valid_steps__{name}"] = int(n_valid_b)

    frozen = np.stack(frozen_rows)
    rddot_err = np.stack(rddot_err_rows)
    p_trace = np.stack(p_trace_rows)
    out["frozen_model_rmse"] = float(np.sqrt(np.nanmean(frozen**2)))
    out["frozen_model_rmse_per_channel"] = np.sqrt(np.nanmean(frozen**2, axis=0))
    out["rddot_err_rmse_per_channel"] = np.sqrt(np.nanmean(rddot_err**2, axis=0))
    out["p_trace_max_per_channel"] = np.nanmax(p_trace, axis=0)
    out["p_trace_tail_per_channel"] = np.nanmean(
        p_trace[-max(5, len(p_trace) // 5):], axis=0
    )
    for key, rows in series.items():
        out[f"series__{key}"] = np.stack(rows)
    out["series__frozen_model_residual"] = frozen
    out["series__P_trace"] = p_trace
    out["series__kf_err_r_rdot_rddot"] = np.stack(kf_err_rows)
    out["series__a_cmd_err"] = np.stack(a_cmd_rows)
    out["series__a_cmd_err_sat"] = np.stack(a_cmd_sat_rows)
    out["series__did_rls"] = np.asarray(did_rls, dtype=bool)
    out["a_cmd_sat_rmse"] = float(
        np.sqrt(np.nanmean(np.stack(a_cmd_sat_rows) ** 2))
    )
    return out


def _percentiles(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "iqr_low": float(np.percentile(v, 25)),
        "iqr_high": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def main() -> None:
    args = parse_args()
    _guard_dataset(args.samples, args.allow_test_set)

    model_path = args.model[:-4] if args.model.endswith(".zip") else args.model
    if not Path(model_path + ".zip").exists():
        raise SystemExit(f"找不到模型 {model_path}.zip")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else IDENT_RESULTS / f"h1_shared_replay_N{args.episodes}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with tee_terminal_log(out_dir / "run.log"):
        cfg = matlab_cfg_from_args(args, action_mapping=args.action_mapping)
        print(
            f"[H1] model={model_path}\n"
            f"[H1] samples={args.samples}  episodes={args.episodes}\n"
            f"[H1] shadow=True (指令恒为 analytic_truth，三路估计共享同一轨迹)\n"
            f"[H1] actor_obs_mode={cfg['actor_obs_mode']}  "
            f"lambda={cfg['rls_lambda']}  eta={cfg['rls_eta']}  "
            f"P_trace_max={cfg['rls_P_trace_max']}"
        )

        env = EvasionEnv(
            samples_file=args.samples,
            action_mapping=args.action_mapping,
            matlab_cfg=cfg,
        )
        try:
            model = TD3.load(model_path, env=env)
            n_ic = min(args.episodes, env.n_samples)
            episodes: list[dict] = []
            for local_idx in range(n_ic):
                ep = run_shared_replay_episode(
                    env,
                    model,
                    local_idx=local_idx,
                    noise_seed=int(args.noise_seed) + local_idx,
                    seed=args.seed if local_idx == 0 else None,
                    burn_in=args.nrmse_burn_in_steps,
                    denom_floor=args.nrmse_denominator_floor,
                )
                if ep is None:
                    print(f"  [skip] local_idx={local_idx} 无辨识诊断")
                    continue
                episodes.append(ep)
                print(
                    f"  IC{local_idx}: steps={ep['steps']} seed={ep['noise_seed']} "
                    f"noisy={ep['nrmse__analytic_noisy']:.4g} "
                    f"ffrls={ep['nrmse__sparse_ffrls']:.4g} "
                    f"zoh_floor={ep['nrmse__zoh_drift_floor']:.4g} "
                    f"frozen_res={ep['frozen_model_rmse']:.4g}"
                )
        finally:
            env.close()

        if not episodes:
            raise SystemExit("[H1] 无有效 episode")

        summary: dict = {
            "task": "h1_shared_replay_diag",
            "timestamp": timestamp,
            "model": model_path,
            "samples": str(Path(args.samples).resolve()),
            "dataset_role": "validation",
            "episodes": len(episodes),
            "shadow_mode": True,
            "command_source": "analytic_truth",
            "matlab_python_config": cfg,
            "nrmse": {
                "definition": "componentwise_scaled_parameter_nrmse",
                "shared_with": "evaluate_td3._theta_scaled_nrmse",
                "burn_in_steps": int(args.nrmse_burn_in_steps),
                "denominator_floor": float(args.nrmse_denominator_floor),
                "causal_alignment": "estimate[k] versus vartheta[k-1]",
                "parameter_order": list(PARAM_NAMES),
            },
            "estimators": {
                "analytic_noisy": "含噪量测代入解析式（P0-4 无辨识基线）",
                "sparse_ffrls": "因果稀疏 FF-RLS 估计",
                "zoh_drift_floor": (
                    "神谕 vartheta_true[k] 对 vartheta_true[k-1]；"
                    "与估计器无关的 ZOH 违背下界"
                ),
            },
        }

        for name in ESTIMATORS:
            summary[f"nrmse__{name}"] = _percentiles(
                np.array([e[f"nrmse__{name}"] for e in episodes])
            )
            summary[f"nrmse_post_burnin__{name}"] = _percentiles(
                np.array([e[f"nrmse_post_burnin__{name}"] for e in episodes])
            )
            comp = np.stack([e[f"nrmse_components__{name}"] for e in episodes])
            summary[f"nrmse_components_median__{name}"] = {
                PARAM_NAMES[j]: float(np.nanmedian(comp[:, j])) for j in range(11)
            }
            summary[f"nrmse_components_max__{name}"] = {
                PARAM_NAMES[j]: float(np.nanmax(comp[:, j])) for j in range(11)
            }

        summary["frozen_model_rmse"] = _percentiles(
            np.array([e["frozen_model_rmse"] for e in episodes])
        )
        summary["rddot_err_rmse_per_channel_median"] = [
            float(x)
            for x in np.nanmedian(
                np.stack([e["rddot_err_rmse_per_channel"] for e in episodes]), axis=0
            )
        ]
        summary["p_trace_tail_per_channel_median"] = [
            float(x)
            for x in np.nanmedian(
                np.stack([e["p_trace_tail_per_channel"] for e in episodes]), axis=0
            )
        ]
        summary["per_episode"] = [
            {
                "local_idx": e["local_idx"],
                "row_idx": e["row_idx"],
                "noise_seed": e["noise_seed"],
                "steps": e["steps"],
                "did_rls_rate": e["did_rls_rate"],
                **{f"nrmse__{n}": e[f"nrmse__{n}"] for n in ESTIMATORS},
                **{
                    f"nrmse_post_burnin__{n}": e[f"nrmse_post_burnin__{n}"]
                    for n in ESTIMATORS
                },
                "frozen_model_rmse": e["frozen_model_rmse"],
                "a_cmd_sat_rmse": e["a_cmd_sat_rmse"],
            }
            for e in episodes
        ]

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )

        payload: dict = {
            "local_idx": np.array([e["local_idx"] for e in episodes], dtype=np.int64),
            "row_idx": np.array([e["row_idx"] for e in episodes], dtype=np.int64),
            "noise_seed": np.array([e["noise_seed"] for e in episodes], dtype=np.int64),
            "steps": np.array([e["steps"] for e in episodes], dtype=np.int64),
        }
        for name in ESTIMATORS:
            payload[f"nrmse__{name}"] = np.array(
                [e[f"nrmse__{name}"] for e in episodes], dtype=float
            )
            payload[f"nrmse_post_burnin__{name}"] = np.array(
                [e[f"nrmse_post_burnin__{name}"] for e in episodes], dtype=float
            )
            payload[f"nrmse_components__{name}"] = np.stack(
                [e[f"nrmse_components__{name}"] for e in episodes]
            )
            payload[f"nrmse_post_burnin_components__{name}"] = np.stack(
                [e[f"nrmse_post_burnin_components__{name}"] for e in episodes]
            )
        for key in (
            "theta_hat",
            "theta_analytic_noisy_causal",
            "theta_true_causal",
            "theta_true_current",
            "frozen_model_residual",
            "P_trace",
            "kf_err_r_rdot_rddot",
            "a_cmd_err",
            "a_cmd_err_sat",
            "did_rls",
        ):
            payload[f"series__{key}"] = np.array(
                [e[f"series__{key}"] for e in episodes], dtype=object
            )
        np.savez_compressed(out_dir / "h1_shared_replay_series.npz", **payload)

        print("\n[H1] 统一 NRMSE（全程，跨 IC 中位数 / P95）")
        for name in ESTIMATORS:
            s = summary[f"nrmse__{name}"]
            print(f"  {name:<18s} median={s['median']:.4g}  p95={s['p95']:.4g}  "
                  f"max={s['max']:.4g}")
        print("\n[H1] 逐参数 NRMSE 中位数")
        header = f"  {'param':<8s}" + "".join(f"{n:>20s}" for n in ESTIMATORS)
        print(header)
        for j, pname in enumerate(PARAM_NAMES):
            cells = "".join(
                f"{summary[f'nrmse_components_median__{n}'][pname]:>20.4g}"
                for n in ESTIMATORS
            )
            print(f"  {pname:<8s}{cells}")
        print(f"\n[H1] 结果写入 {out_dir}")


if __name__ == "__main__":
    main()
