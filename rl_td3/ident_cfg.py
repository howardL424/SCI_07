"""辨识 / 测量噪声 matlab_cfg 构建（纯 TD3 Plan 1）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


IDENT_CLI_DEFAULTS = dict(
    coeff_mode="analytic_truth",
    ident_apply_cmd=True,
    sigma_q=0.1 * 3.141592653589793 / 180.0,  # 0.1 deg
    sigma_qdot=0.001,
    sigma_theta=0.1 * 3.141592653589793 / 180.0,
    sigma_psi=0.1 * 3.141592653589793 / 180.0,
    sigma_r=0.0,  # 2026-08-01：由占位 10.0 改为 0（无距离测量噪声）
    sigma_rdot=0.0,  # 2026-08-01：由占位 1.0 改为 0
    kf_qj=1.0e4,
    kf_sigma_a0=50.0,
    rls_lambda=0.98,
    rls_mu=1.0e-3,  # 2026-08-01：曾误改为 0（依据已作废的 5.2 结论），已回退
    rls_P0=1.0e4,
    rls_r0=1.0e4,
    rls_a0=50.0,
    noise_seed=0,
)


def add_ident_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 argparse 增加辨识相关只读/训练共享参数（默认 analytic_truth）。"""
    g = parser.add_argument_group("参数辨识 / 测量噪声（Plan 1，默认关闭）")
    g.add_argument(
        "--coeff-mode",
        type=str,
        default=IDENT_CLI_DEFAULTS["coeff_mode"],
        choices=["analytic_truth", "analytic_noisy", "sparse_ffrls"],
        help="制导系数来源（默认 analytic_truth，与旧行为一致）",
    )
    g.add_argument(
        "--ident-shadow",
        action="store_true",
        help="影子模式：跑 Kalman/RLS 诊断但不改指令（仅 sparse_ffrls 有意义）",
    )
    g.add_argument("--sigma-q", type=float, default=IDENT_CLI_DEFAULTS["sigma_q"])
    g.add_argument("--sigma-qdot", type=float, default=IDENT_CLI_DEFAULTS["sigma_qdot"])
    g.add_argument(
        "--sigma-theta", type=float, default=IDENT_CLI_DEFAULTS["sigma_theta"]
    )
    g.add_argument("--sigma-psi", type=float, default=IDENT_CLI_DEFAULTS["sigma_psi"])
    g.add_argument(
        "--sigma-r",
        type=float,
        default=IDENT_CLI_DEFAULTS["sigma_r"],
        help="距离测量标准差 (m)；默认 0（曾占位 10）",
    )
    g.add_argument(
        "--sigma-rdot",
        type=float,
        default=IDENT_CLI_DEFAULTS["sigma_rdot"],
        help="距离率测量标准差 (m/s)；默认 0（曾占位 1）",
    )
    g.add_argument("--kf-qj", type=float, default=IDENT_CLI_DEFAULTS["kf_qj"])
    g.add_argument(
        "--kf-sigma-a0", type=float, default=IDENT_CLI_DEFAULTS["kf_sigma_a0"]
    )
    g.add_argument(
        "--rls-lambda", type=float, default=IDENT_CLI_DEFAULTS["rls_lambda"]
    )
    g.add_argument(
        "--rls-mu",
        type=float,
        default=IDENT_CLI_DEFAULTS["rls_mu"],
        help="L1 soft-threshold 强度；0=关闭正则（默认 1e-3）",
    )
    g.add_argument("--rls-P0", type=float, default=IDENT_CLI_DEFAULTS["rls_P0"])
    g.add_argument("--rls-r0", type=float, default=IDENT_CLI_DEFAULTS["rls_r0"])
    g.add_argument("--rls-a0", type=float, default=IDENT_CLI_DEFAULTS["rls_a0"])
    g.add_argument(
        "--noise-seed",
        type=int,
        default=IDENT_CLI_DEFAULTS["noise_seed"],
        help="测量噪声可复现种子（hash RNG）",
    )
    return parser


def matlab_cfg_from_args(
    args: argparse.Namespace,
    *,
    action_mapping: Optional[str] = None,
) -> Dict[str, Any]:
    """从 CLI namespace 构建传给 EvasionEnv 的 matlab_cfg。"""
    apply_cmd = not bool(getattr(args, "ident_shadow", False))
    cfg: Dict[str, Any] = {
        "coeff_mode": str(args.coeff_mode),
        "ident_apply_cmd": apply_cmd,
        "sigma_q": float(args.sigma_q),
        "sigma_qdot": float(args.sigma_qdot),
        "sigma_theta": float(args.sigma_theta),
        "sigma_psi": float(args.sigma_psi),
        "sigma_r": float(args.sigma_r),
        "sigma_rdot": float(args.sigma_rdot),
        "kf_qj": float(args.kf_qj),
        "kf_sigma_a0": float(args.kf_sigma_a0),
        "rls_lambda": float(args.rls_lambda),
        "rls_mu": float(args.rls_mu),
        "rls_P0": float(args.rls_P0),
        "rls_r0": float(args.rls_r0),
        "rls_a0": float(args.rls_a0),
        "noise_seed": int(args.noise_seed),
    }
    if action_mapping is not None:
        cfg["action_mapping_version"] = str(action_mapping)
    elif hasattr(args, "action_mapping"):
        cfg["action_mapping_version"] = str(args.action_mapping)
    return cfg


def save_ident_run_config(path: Path, cfg: Dict[str, Any], extra: Optional[dict] = None) -> None:
    """将完整辨识配置写入运行目录 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(cfg)
    if extra:
        payload["_extra"] = extra
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
