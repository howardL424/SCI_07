"""辨识 / 测量噪声 matlab_cfg 构建（纯 TD3 Plan 1）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


IDENT_CLI_DEFAULTS = dict(
    coeff_mode="analytic_truth",
    ident_apply_cmd=True,
    sigma_scale=30.0,
    sigma_q=3.0 * 3.141592653589793 / 180.0,  # 0.1 deg × 30
    sigma_qdot=0.03,  # 0.001 rad/s × 30
    sigma_theta=3.0 * 3.141592653589793 / 180.0,
    sigma_psi=3.0 * 3.141592653589793 / 180.0,
    sigma_r=10.0,  # 距离量测小噪声；同时用于式(62d)的 R_xi
    sigma_rdot=0.5,  # 距离率量测小噪声；同时用于式(62d)的 R_xi
    kf_qj=1.0e4,
    kf_sigma_a0=50.0,
    rls_lambda=0.98,
    rls_mu=1.0e-3,  # 式(31)目标权重，保留用于记录
    rls_eta=0.1,  # 2026-09-08：20 IC 离线标定候选
    rls_P0=1.0e4,
    rls_P_trace_max=1.0,  # 20 IC 影子扫描选定；inf=关闭
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
    g.add_argument(
        "--sigma-scale",
        type=float,
        default=IDENT_CLI_DEFAULTS["sigma_scale"],
        help="噪声标定倍率元数据；默认30，sigma-* 参数保存的是已缩放最终值",
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
        help="距离测量标准差 (m)，并直接构成式(62d)的 R_xi",
    )
    g.add_argument(
        "--sigma-rdot",
        type=float,
        default=IDENT_CLI_DEFAULTS["sigma_rdot"],
        help="距离率测量标准差 (m/s)，并直接构成式(62d)的 R_xi",
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
        help="式(31)稀疏度权重（运行记录；递推阈值由 --rls-eta 指定）",
    )
    g.add_argument(
        "--rls-eta",
        type=float,
        default=IDENT_CLI_DEFAULTS["rls_eta"],
        help="缩放域逐元素软阈值；0=关闭收缩",
    )
    g.add_argument("--rls-P0", type=float, default=IDENT_CLI_DEFAULTS["rls_P0"])
    g.add_argument(
        "--rls-P-trace-max",
        type=float,
        default=IDENT_CLI_DEFAULTS["rls_P_trace_max"],
        help="缩放域 RLS 协方差迹上限；inf=关闭",
    )
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
        "sigma_scale": float(getattr(args, "sigma_scale", IDENT_CLI_DEFAULTS["sigma_scale"])),
        "sigma_q": float(getattr(args, "sigma_q", IDENT_CLI_DEFAULTS["sigma_q"])),
        "sigma_qdot": float(getattr(args, "sigma_qdot", IDENT_CLI_DEFAULTS["sigma_qdot"])),
        "sigma_theta": float(getattr(args, "sigma_theta", IDENT_CLI_DEFAULTS["sigma_theta"])),
        "sigma_psi": float(getattr(args, "sigma_psi", IDENT_CLI_DEFAULTS["sigma_psi"])),
        "sigma_r": float(getattr(args, "sigma_r", IDENT_CLI_DEFAULTS["sigma_r"])),
        "sigma_rdot": float(getattr(args, "sigma_rdot", IDENT_CLI_DEFAULTS["sigma_rdot"])),
        "kf_qj": float(getattr(args, "kf_qj", IDENT_CLI_DEFAULTS["kf_qj"])),
        "kf_sigma_a0": float(getattr(args, "kf_sigma_a0", IDENT_CLI_DEFAULTS["kf_sigma_a0"])),
        "rls_lambda": float(getattr(args, "rls_lambda", IDENT_CLI_DEFAULTS["rls_lambda"])),
        "rls_mu": float(getattr(args, "rls_mu", IDENT_CLI_DEFAULTS["rls_mu"])),
        "rls_eta": float(getattr(args, "rls_eta", IDENT_CLI_DEFAULTS["rls_eta"])),
        "rls_P0": float(getattr(args, "rls_P0", IDENT_CLI_DEFAULTS["rls_P0"])),
        "rls_P_trace_max": float(
            getattr(args, "rls_P_trace_max", IDENT_CLI_DEFAULTS["rls_P_trace_max"])
        ),
        "rls_r0": float(getattr(args, "rls_r0", IDENT_CLI_DEFAULTS["rls_r0"])),
        "rls_a0": float(getattr(args, "rls_a0", IDENT_CLI_DEFAULTS["rls_a0"])),
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
