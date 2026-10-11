"""统一运行 P0-4 的 TD3/BC-TD3 × analytic_noisy/sparse_ffrls 配对评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_IDENT = _SCRIPT.parent
_ROOT = _IDENT.parent
_RL = _ROOT / "rl_td3"
sys.path.insert(0, str(_IDENT))
sys.path.insert(0, str(_RL))

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from evaluate_td3 import run_episode, tee_terminal_log  # noqa: E402
from evasion_env import EvasionEnv, SAMPLES_VAL_FILE  # noqa: E402
from experiment_protocol import DEFAULT_PROTOCOL_PATH, load_protocol  # noqa: E402
from ident_cfg import IDENT_CLI_DEFAULTS, matlab_cfg_from_args  # noqa: E402
from stable_baselines3 import TD3  # noqa: E402
from td3_bc import TD3BC  # noqa: E402


POLICY_NAMES = ("td3", "bc_td3")
COEFF_MODES = ("analytic_noisy", "sparse_ffrls")
BOOL_FIELDS = ("evade_ok", "strike_ok", "both_ok", "intercepted")
FLOAT_FIELDS = (
    "r_MD1_min",
    "r_MD2_min",
    "r_MD_min",
    "r_AT_end",
    "a_energy",
    "energy_int",
    "energy_kJ",
    "energy_stage2_kJ",
    "energy_stage3_kJ",
    "a_total_variation",
    "saturation_rate",
    "theta_scaled_nrmse",
    "theta_scaled_nrmse_post_burnin",
)


def _add_ident_numeric_args(parser: argparse.ArgumentParser) -> None:
    defaults = IDENT_CLI_DEFAULTS
    parser.add_argument("--sigma-scale", type=float, default=defaults["sigma_scale"])
    parser.add_argument("--sigma-q", type=float, default=defaults["sigma_q"])
    parser.add_argument("--sigma-qdot", type=float, default=defaults["sigma_qdot"])
    parser.add_argument("--sigma-theta", type=float, default=defaults["sigma_theta"])
    parser.add_argument("--sigma-psi", type=float, default=defaults["sigma_psi"])
    parser.add_argument("--sigma-r", type=float, default=defaults["sigma_r"])
    parser.add_argument("--sigma-rdot", type=float, default=defaults["sigma_rdot"])
    parser.add_argument("--kf-qj", type=float, default=defaults["kf_qj"])
    parser.add_argument("--kf-sigma-a0", type=float, default=defaults["kf_sigma_a0"])
    parser.add_argument("--rls-lambda", type=float, default=defaults["rls_lambda"])
    parser.add_argument("--rls-mu", type=float, default=defaults["rls_mu"])
    parser.add_argument("--rls-eta", type=float, default=defaults["rls_eta"])
    parser.add_argument("--rls-P0", type=float, default=defaults["rls_P0"])
    parser.add_argument(
        "--rls-P-trace-max", type=float, default=defaults["rls_P_trace_max"]
    )
    parser.add_argument("--rls-r0", type=float, default=defaults["rls_r0"])
    parser.add_argument("--rls-a0", type=float, default=defaults["rls_a0"])


def parse_args() -> argparse.Namespace:
    protocol = load_protocol()
    policies = protocol["policies"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["all", *POLICY_NAMES], default="all")
    parser.add_argument("--td3-model", default=policies["td3"]["checkpoint"])
    parser.add_argument("--bc-td3-model", default=policies["bc_td3"]["checkpoint"])
    parser.add_argument("--samples", default=SAMPLES_VAL_FILE)
    parser.add_argument(
        "--dataset-role",
        choices=["validation", "test", "custom"],
        default="validation",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="前 N 个有效 IC；0 表示完整初态池"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-seed",
        "--base-noise-seed",
        dest="noise_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--actor-obs-mode",
        choices=["noisy", "filtered"],
        default="filtered",
    )
    parser.add_argument("--action-mapping", default=DEFAULT_ACTION_MAPPING)
    parser.add_argument("--nrmse-burn-in-steps", type=int, default=5)
    parser.add_argument("--nrmse-denominator-floor", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
    _add_ident_numeric_args(parser)
    args = parser.parse_args()
    args.coeff_mode = "analytic_noisy"
    args.ident_shadow = False
    return args


def _normalize_model_path(path: str) -> tuple[str, Path]:
    model = Path(path)
    if not model.is_absolute():
        model = _ROOT / model
    model_zip = model if model.suffix == ".zip" else Path(f"{model}.zip")
    model_base = model.with_suffix("") if model.suffix == ".zip" else model
    if not model_zip.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {model_zip}")
    return str(model_base), model_zip


def _resolve_root_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    return resolved.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
    ).strip()
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=_ROOT, text=True
    ).splitlines()
    return {
        "commit": commit,
        "dirty": bool(porcelain),
        "porcelain_entry_count": len(porcelain),
        "porcelain": porcelain,
    }


def _load_policy(policy_name: str, model_path: str, env: EvasionEnv):
    if policy_name == "td3":
        return TD3.load(model_path, env=env)
    if policy_name == "bc_td3":
        return TD3BC.load(model_path, env=env)
    raise ValueError(f"未知策略 {policy_name!r}")


def _wilson_ci(successes: int, n: int) -> list[float | None]:
    if n <= 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    radius = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _finite_stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "n_finite": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
        }
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {
        "n_finite": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
    }


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, n_boot: int = 5000
) -> list[float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [None, None]
    if finite.size == 1:
        value = float(finite[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(n_boot, finite.size), replace=True)
    low, high = np.quantile(np.mean(draws, axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def _mcnemar_exact_p(wins: int, losses: int) -> float:
    """配对二分类的双侧精确 McNemar/binomial p 值。"""
    discordant = int(wins) + int(losses)
    if discordant == 0:
        return 1.0
    lower = min(int(wins), int(losses))
    tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return float(min(1.0, 2.0 * tail / (2.0**discordant)))


def _p95_cvar95(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": 0, "p95": None, "cvar95": None}
    p95 = float(np.quantile(finite, 0.95))
    tail = finite[finite >= p95]
    return {
        "n": int(finite.size),
        "p95": p95,
        "cvar95": float(np.mean(tail)),
    }


def _pair_keys(rows: list[dict]) -> list[tuple[int, int, int]]:
    return [
        (int(row["local_idx"]), int(row["row_idx"]), int(row["noise_seed"]))
        for row in rows
    ]


def validate_strict_pairing(rows_by_group: dict[str, list[dict]]) -> dict[str, Any]:
    """校验所有已运行组的 IC、原始行号和实际噪声种子逐项一致。"""
    nonempty = {key: rows for key, rows in rows_by_group.items() if rows}
    if not nonempty:
        raise RuntimeError("没有可用于配对校验的 episode")
    reference_name = next(iter(nonempty))
    reference = _pair_keys(nonempty[reference_name])
    for name, rows in nonempty.items():
        keys = _pair_keys(rows)
        if keys != reference:
            raise RuntimeError(
                f"严格配对失败：{name} 的 (local_idx,row_idx,noise_seed) "
                f"与 {reference_name} 不一致"
            )
    return {
        "passed": True,
        "reference_group": reference_name,
        "n_paired": len(reference),
        "pair_keys": [list(key) for key in reference],
    }


def validate_group_rows(
    rows: list[dict],
    *,
    policy_name: str,
    coeff_mode: str,
    actor_obs_mode: str,
) -> None:
    """正式逐样本字段必须完整，且连续主指标不得含 NaN/Inf。"""
    for index, row in enumerate(rows):
        if row.get("policy_name") != policy_name:
            raise RuntimeError(f"第{index}局 policy_name 字段不一致")
        if row.get("coeff_mode") != coeff_mode:
            raise RuntimeError(f"第{index}局 coeff_mode 字段不一致")
        if row.get("actor_obs_mode") != actor_obs_mode:
            raise RuntimeError(f"第{index}局 actor_obs_mode 字段不一致")
        missing = [
            field
            for field in (
                "local_idx",
                "row_idx",
                "noise_seed",
                "steps",
                *BOOL_FIELDS,
                *FLOAT_FIELDS,
            )
            if field not in row
        ]
        if missing:
            raise RuntimeError(f"第{index}局缺少字段: {', '.join(missing)}")
        nonfinite = [
            field for field in FLOAT_FIELDS if not np.isfinite(float(row[field]))
        ]
        if nonfinite:
            raise RuntimeError(
                f"第{index}局主指标包含 NaN/Inf: {', '.join(nonfinite)}"
            )


def _summarize_group(rows: list[dict]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for field in BOOL_FIELDS:
        count = int(sum(bool(row[field]) for row in rows))
        summary[field] = {
            "count": count,
            "rate": float(count / len(rows)) if rows else None,
            "ci95_wilson": _wilson_ci(count, len(rows)),
        }
    for field in FLOAT_FIELDS:
        summary[field] = _finite_stats([float(row[field]) for row in rows])
    return summary


def _paired_delta(
    baseline: list[dict], candidate: list[dict], *, bootstrap_seed: int
) -> dict[str, Any]:
    if _pair_keys(baseline) != _pair_keys(candidate):
        raise RuntimeError("配对比较的样本键不一致")
    output: dict[str, Any] = {"n_paired": len(baseline), "candidate_minus_baseline": {}}
    for field in ("evade_ok", "strike_ok", "both_ok"):
        delta = np.asarray(
            [int(c[field]) - int(b[field]) for b, c in zip(baseline, candidate)],
            dtype=np.float64,
        )
        output["candidate_minus_baseline"][field] = {
            "mean_rate_delta": float(np.mean(delta)) if delta.size else None,
            "ci95_paired_bootstrap": _bootstrap_mean_ci(
                delta, seed=bootstrap_seed + len(field)
            ),
            "wins": int(np.sum(delta > 0)),
            "losses": int(np.sum(delta < 0)),
            "mcnemar_exact_p": _mcnemar_exact_p(
                int(np.sum(delta > 0)), int(np.sum(delta < 0))
            ),
        }
    for field_index, field in enumerate(FLOAT_FIELDS):
        delta = np.asarray(
            [
                float(candidate[i][field]) - float(baseline[i][field])
                for i in range(len(baseline))
            ],
            dtype=np.float64,
        )
        stats = _finite_stats(delta.tolist())
        stats["ci95_mean_paired_bootstrap"] = _bootstrap_mean_ci(
            delta, seed=bootstrap_seed + 100 + field_index
        )
        output["candidate_minus_baseline"][field] = stats
    return output


def _common_evasion_tail(
    td3_rows: list[dict], bc_rows: list[dict]
) -> dict[str, Any]:
    if _pair_keys(td3_rows) != _pair_keys(bc_rows):
        raise RuntimeError("长尾统计的 TD3/BC-TD3 样本键不一致")
    common = [
        (td3_row, bc_row)
        for td3_row, bc_row in zip(td3_rows, bc_rows)
        if bool(td3_row["evade_ok"]) and bool(bc_row["evade_ok"])
    ]
    td3_values = [float(pair[0]["r_AT_end"]) for pair in common]
    bc_values = [float(pair[1]["r_AT_end"]) for pair in common]
    return {
        "subset_definition": "TD3 and BC-TD3 both evade D1/D2 successfully",
        "n_common": len(common),
        "td3_r_AT_end": _p95_cvar95(td3_values),
        "bc_td3_r_AT_end": _p95_cvar95(bc_values),
        "paired_delta_bc_minus_td3": _finite_stats(
            [
                bc_value - td3_value
                for td3_value, bc_value in zip(td3_values, bc_values)
            ]
        ),
    }


def _save_group(path: Path, rows: list[dict]) -> None:
    payload: dict[str, np.ndarray] = {
        "policy_name": np.asarray([row["policy_name"] for row in rows]),
        "coeff_mode": np.asarray([row["coeff_mode"] for row in rows]),
        "actor_obs_mode": np.asarray([row["actor_obs_mode"] for row in rows]),
        "local_idx": np.asarray([row["local_idx"] for row in rows], dtype=np.int64),
        "row_idx": np.asarray([row["row_idx"] for row in rows], dtype=np.int64),
        "noise_seed": np.asarray([row["noise_seed"] for row in rows], dtype=np.int64),
        "steps": np.asarray([row["steps"] for row in rows], dtype=np.int64),
    }
    for field in BOOL_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows], dtype=bool)
    for field in FLOAT_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows], dtype=np.float64)
    for field in (
        "theta_scaled_nrmse_components",
        "theta_scaled_nrmse_post_burnin_components",
    ):
        payload[field] = np.stack(
            [np.asarray(row[field], dtype=np.float64) for row in rows]
        )
    np.savez_compressed(path, **payload)


def _selected_policies(args: argparse.Namespace) -> list[str]:
    return list(POLICY_NAMES) if args.policy == "all" else [args.policy]


def _model_arg(args: argparse.Namespace, policy_name: str) -> str:
    return args.td3_model if policy_name == "td3" else args.bc_td3_model


def _run(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    policies = _selected_policies(args)
    protocol_path = _resolve_root_path(args.protocol)
    protocol = load_protocol(protocol_path)
    samples_path = _resolve_root_path(args.samples)
    implementation_checks = {}
    for relative_path, expected_sha256 in protocol.get(
        "implementation_files_sha256", {}
    ).items():
        actual_sha256 = _sha256(_resolve_root_path(relative_path))
        implementation_checks[relative_path] = {
            "expected_sha256": str(expected_sha256).lower(),
            "actual_sha256": actual_sha256,
            "matches": actual_sha256 == str(expected_sha256).lower(),
        }
    model_records = {}
    normalized_models = {}
    for policy_name in policies:
        model_base, model_zip = _normalize_model_path(_model_arg(args, policy_name))
        normalized_models[policy_name] = model_base
        model_records[policy_name] = {
            "checkpoint": str(model_zip),
            "sha256": _sha256(model_zip),
            "matches_protocol_sha256": (
                _sha256(model_zip)
                == str(protocol["policies"][policy_name]["sha256"]).lower()
            ),
        }

    base_cfg = matlab_cfg_from_args(args, action_mapping=args.action_mapping)
    probe = EvasionEnv(
        samples_file=str(samples_path),
        action_mapping=args.action_mapping,
        matlab_cfg={**base_cfg, "coeff_mode": "analytic_noisy"},
    )
    n_ic = probe.n_samples if args.episodes == 0 else min(args.episodes, probe.n_samples)
    probe.close()
    if n_ic <= 0:
        raise RuntimeError("没有可运行的有效初态")

    run_metadata = {
        "schema_version": "p0_4_evaluation_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "status": protocol["status"],
            "phase": protocol["phase"],
        },
        "repository": _git_snapshot(),
        "implementation_file_checks": implementation_checks,
        "dataset": str(samples_path),
        "dataset_sha256": _sha256(samples_path),
        "dataset_matches_protocol_sha256": (
            _sha256(samples_path)
            == str(
                protocol["datasets"]["final_test"].get("sha256", "")
            ).lower()
        ),
        "dataset_role": args.dataset_role,
        "requested_episodes": int(args.episodes),
        "episodes": int(n_ic),
        "policies": model_records,
        "coeff_modes": list(COEFF_MODES),
        "actor_obs_mode": args.actor_obs_mode,
        "action_mapping": args.action_mapping,
        "base_noise_seed": int(args.noise_seed),
        "noise_seed_rule": "base_noise_seed + local_idx",
        "matlab_python_config": base_cfg,
        "nrmse": {
            "name": "componentwise_scaled_parameter_nrmse",
            "parameter_order": [
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
            ],
            "scale": [
                args.rls_r0,
                args.rls_a0,
                args.rls_a0,
                1.0,
                args.rls_r0,
                args.rls_a0,
                args.rls_a0,
                1.0,
                args.rls_r0,
                args.rls_a0,
                args.rls_a0,
            ],
            "denominator": "max(RMS(scaled_truth_j), denominator_floor)",
            "denominator_floor": float(args.nrmse_denominator_floor),
            "aggregation": "RMS of the 11 component NRMSE values",
            "causal_alignment": "estimate[k] versus vartheta[k-1]",
            "burn_in_steps": int(args.nrmse_burn_in_steps),
        },
        "selection_audit": {
            "eta_grid_performed": False,
            "P_trace_scan_performed": False,
            "checkpoint_selection_performed": False,
            "dataset_roles_are_metadata_only": True,
        },
        "statistical_plan": {
            "success_rate_interval": "Wilson 95%",
            "paired_success_difference": "5000-sample paired bootstrap 95% CI",
            "paired_binary_test": "two-sided exact McNemar/binomial test",
            "continuous_summary": "median, IQR, mean, standard deviation",
            "long_tail": "r_AT_end P95 and CVaR95 on common dual-evasion subset",
        },
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows_by_group: dict[str, list[dict]] = {}
    for policy_name in policies:
        for coeff_mode in COEFF_MODES:
            group = f"{policy_name}__{coeff_mode}"
            cfg = {**base_cfg, "coeff_mode": coeff_mode}
            env = EvasionEnv(
                samples_file=str(samples_path),
                action_mapping=args.action_mapping,
                matlab_cfg=cfg,
            )
            try:
                model = _load_policy(
                    policy_name, normalized_models[policy_name], env
                )
                rows = []
                print(f"\n===== {group} =====")
                for local_idx in range(n_ic):
                    noise_seed = int(args.noise_seed) + local_idx
                    episode = run_episode(
                        env,
                        model,
                        seed=args.seed if local_idx == 0 else None,
                        local_idx=local_idx,
                        noise_seed=noise_seed,
                        nrmse_burn_in_steps=args.nrmse_burn_in_steps,
                        nrmse_denominator_floor=args.nrmse_denominator_floor,
                    )
                    if episode is None:
                        raise RuntimeError(
                            f"{group} local_idx={local_idx} warm-start 失败"
                        )
                    episode["policy_name"] = policy_name
                    rows.append(episode)
                    print(
                        f"  IC{local_idx}: seed={episode['noise_seed']} "
                        f"both={int(episode['both_ok'])} "
                        f"r_min={episode['r_MD_min']:.2f} "
                        f"r_AT={episode['r_AT_end']:.2f} "
                        f"NRMSE={episode['theta_scaled_nrmse']:.4g}"
                    )
            finally:
                env.close()
            rows_by_group[group] = rows
            validate_group_rows(
                rows,
                policy_name=policy_name,
                coeff_mode=coeff_mode,
                actor_obs_mode=args.actor_obs_mode,
            )
            _save_group(out_dir / f"metrics_{group}.npz", rows)

    pairing = validate_strict_pairing(rows_by_group)
    comparisons: dict[str, Any] = {}
    for policy_name in policies:
        baseline = rows_by_group[f"{policy_name}__analytic_noisy"]
        candidate = rows_by_group[f"{policy_name}__sparse_ffrls"]
        comparisons[f"{policy_name}__sparse_vs_analytic"] = _paired_delta(
            baseline,
            candidate,
            bootstrap_seed=args.seed + 1000 * (POLICY_NAMES.index(policy_name) + 1),
        )
    long_tail = {}
    if set(policies) == set(POLICY_NAMES):
        for mode_index, coeff_mode in enumerate(COEFF_MODES):
            comparisons[f"{coeff_mode}__bc_td3_vs_td3"] = _paired_delta(
                rows_by_group[f"td3__{coeff_mode}"],
                rows_by_group[f"bc_td3__{coeff_mode}"],
                bootstrap_seed=args.seed + 3000 + 1000 * mode_index,
            )
            long_tail[coeff_mode] = _common_evasion_tail(
                rows_by_group[f"td3__{coeff_mode}"],
                rows_by_group[f"bc_td3__{coeff_mode}"],
            )

    summary = {
        **run_metadata,
        "strict_pairing": pairing,
        "groups": {
            name: _summarize_group(rows) for name, rows in rows_by_group.items()
        },
        "comparisons": comparisons,
        "long_tail_common_evasion_subset": long_tail,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.episodes < 0:
        raise SystemExit("--episodes 须为非负整数")
    if args.nrmse_burn_in_steps < 0:
        raise SystemExit("--nrmse-burn-in-steps 须为非负整数")
    if args.nrmse_denominator_floor <= 0:
        raise SystemExit("--nrmse-denominator-floor 须为正数")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _IDENT / "results" / f"p0_4_{args.dataset_role}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with tee_terminal_log(out_dir / "run.log"):
        print(f"[p0-4] 结果目录：{out_dir}")
        summary = _run(args, out_dir)
        print(
            f"[p0-4] 严格配对通过："
            f"{summary['strict_pairing']['n_paired']} 个 IC"
        )


if __name__ == "__main__":
    main()
