"""MATLAB ``env_step`` 参数辨识诊断向量协议。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


IDENT_INFO_LEN = 96


def parse_ident_diag(info_vec: np.ndarray) -> Optional[Dict[str, Any]]:
    """解析 ``env_step`` 的辨识扩展；基础9维向量返回 ``None``。"""
    vec = np.asarray(info_vec, dtype=np.float64).reshape(-1)
    if vec.size < 32:
        return None

    theta_err_causal = vec[18:29].copy()
    diag: Dict[str, Any] = {
        "r_err": vec[9:12].copy(),
        "rdot_err": vec[12:15].copy(),
        "rddot_err": vec[15:18].copy(),
        "theta_err": theta_err_causal,
        "theta_err_causal": theta_err_causal.copy(),
        "did_rls": bool(vec[29] > 0.5),
        "a_cmd_err": vec[30:32].copy(),
    }
    if vec.size >= 46:
        diag.update(
            {
                "a_cmd_err_sat": vec[32:34].copy(),
                "qdot_err": vec[34:37].copy(),
                "qdot_err_causal": vec[34:37].copy(),
                "P_trace": vec[37:40].copy(),
                "P_min_eig": vec[40:43].copy(),
                "P_max_eig": vec[43:46].copy(),
            }
        )
    if vec.size >= 68:
        theta_true_causal = vec[57:68].copy()
        diag.update(
            {
                "theta_hat": vec[46:57].copy(),
                "theta_true": theta_true_causal,
                "theta_true_causal": theta_true_causal.copy(),
            }
        )
    if vec.size >= 85:
        theta_true_current = vec[68:79].copy()
        theta_hat = vec[46:57]
        diag.update(
            {
                "theta_true_current": theta_true_current,
                "theta_err_command": theta_hat - theta_true_current,
                "frozen_model_residual": vec[79:82].copy(),
                "qdot_err_command": vec[82:85].copy(),
            }
        )
    if vec.size >= IDENT_INFO_LEN:
        diag["theta_analytic_noisy_causal"] = vec[85:96].copy()
    return diag
