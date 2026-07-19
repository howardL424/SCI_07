"""
action_mapping.py — 动作归一化映射（Python 权威；与 MATLAB 成对）。

版本:
  td3_box_log_v1  — 旧: [γ_n, wT_n, wE_n]
  ratio_log_box_v1 — 新: [ρE_n, ρT_n, γ_n]，ρE=w/wE，ρT=wT/wE

比值盒由固定 w=1e4 与旧权重盒推出:
  wE∈[1e2,1e4] → ρE=w/wE ∈[1,100]
  wT∈[1e2,1e5], wE∈[1e2,1e4] → ρT=wT/wE ∈[1e-2,1e3]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

W_FIXED = 1e4

ACTION_MAPPING_TD3_BOX = "td3_box_log_v1"
ACTION_MAPPING_RATIO = "ratio_log_box_v1"
DEFAULT_ACTION_MAPPING = ACTION_MAPPING_RATIO

ACTION_NAMES_TD3_BOX = ("gamma_n", "wT_n", "wE_n")
ACTION_NAMES_RATIO = ("rhoE_n", "rhoT_n", "gamma_n")

# 比值物理盒
RHO_E_LO, RHO_E_HI = 1.0, 100.0
RHO_T_LO, RHO_T_HI = 1e-2, 1e3
# 旧权重盒
WT_LO, WT_HI = 1e2, 1e5
WE_LO, WE_HI = 1e2, 1e4
GAMMA_LO, GAMMA_HI = -3.0, 3.0


def action_names(version: str = DEFAULT_ACTION_MAPPING) -> tuple[str, str, str]:
    if version == ACTION_MAPPING_RATIO:
        return ACTION_NAMES_RATIO
    if version == ACTION_MAPPING_TD3_BOX:
        return ACTION_NAMES_TD3_BOX
    raise ValueError(f"未知 action_mapping_version={version!r}")


def phys_to_action_norm(
    gamma: np.ndarray | float,
    wT: np.ndarray | float,
    wE: np.ndarray | float,
    *,
    version: str = DEFAULT_ACTION_MAPPING,
    w: float = W_FIXED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    物理 (γ,wT,wE) → 归一化动作 ∈[-1,1]^3。
    返回 (action_norm, clipped_mask)；clipped_mask 与 action 同广播形状的 bool。
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    wT = np.asarray(wT, dtype=np.float64)
    wE = np.asarray(wE, dtype=np.float64)
    if version == ACTION_MAPPING_TD3_BOX:
        return _phys_to_td3_box(gamma, wT, wE)
    if version == ACTION_MAPPING_RATIO:
        return _phys_to_ratio(gamma, wT, wE, w=w)
    raise ValueError(f"未知 action_mapping_version={version!r}")


def action_norm_to_phys(
    action_norm: np.ndarray,
    *,
    version: str = DEFAULT_ACTION_MAPPING,
    w: float = W_FIXED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """归一化动作 → 物理 (γ, wT, wE)。假定 action 已在 [-1,1]。"""
    a = np.asarray(action_norm, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[-1] != 3:
        raise ValueError(f"action_norm 末维须为 3，收到 {a.shape}")
    if version == ACTION_MAPPING_TD3_BOX:
        gamma = a[..., 0] * 3.0
        wT = 10.0 ** (a[..., 1] * 1.5 + 3.5)
        wE = 10.0 ** (a[..., 2] + 3.0)
        return gamma, wT, wE
    if version == ACTION_MAPPING_RATIO:
        rho_e = 10.0 ** (a[..., 0] + 1.0)
        rho_t = 10.0 ** (a[..., 1] * 2.5 + 0.5)
        gamma = a[..., 2] * 3.0
        wE = float(w) / np.maximum(rho_e, 1e-12)
        wT = rho_t * wE
        return gamma, wT, wE
    raise ValueError(f"未知 action_mapping_version={version!r}")


def action_norm_to_ratios(
    action_norm: np.ndarray,
    *,
    version: str = DEFAULT_ACTION_MAPPING,
    w: float = W_FIXED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    归一化动作 → 式(47) 直接使用的 (w1/wE, w2/wE, wT/wE)。
    比值版本不经过旧权重盒二次裁剪。
    """
    a = np.asarray(action_norm, dtype=np.float64)
    squeeze = a.ndim == 1
    if squeeze:
        a = a.reshape(1, -1)
    if version == ACTION_MAPPING_RATIO:
        rho_e = 10.0 ** (a[..., 0] + 1.0)
        rho_t = 10.0 ** (a[..., 1] * 2.5 + 0.5)
        gamma = a[..., 2] * 3.0
    elif version == ACTION_MAPPING_TD3_BOX:
        gamma, wT, wE = action_norm_to_phys(a, version=version, w=w)
        rho_e = float(w) / np.maximum(wE, 1e-12)
        rho_t = wT / np.maximum(wE, 1e-12)
    else:
        raise ValueError(f"未知 action_mapping_version={version!r}")
    e_g = np.exp(-gamma)
    r1 = rho_e / (1.0 + e_g)
    r2 = rho_e * e_g / (1.0 + e_g)
    if squeeze:
        return r1.ravel()[0], r2.ravel()[0], rho_t.ravel()[0]
    return r1, r2, rho_t


def remap_action_norm(
    action_norm: np.ndarray,
    *,
    from_version: str,
    to_version: str,
    w: float = W_FIXED,
) -> np.ndarray:
    """确定性标签变换：旧权威 action_norm → 新坐标（经裁剪后物理量）。"""
    if from_version == to_version:
        return np.asarray(action_norm, dtype=np.float32)
    gamma, wT, wE = action_norm_to_phys(action_norm, version=from_version, w=w)
    an, _ = phys_to_action_norm(gamma, wT, wE, version=to_version, w=w)
    return an.astype(np.float32)


def old_box_feasible_from_ratio_action(
    action_norm: np.ndarray,
    *,
    w: float = W_FIXED,
    rtol: float = 1e-9,
) -> np.ndarray:
    """
    比值动作反解出的 (wT,wE) 是否落在旧权重盒内（诊断用）。
    在 w 固定且动作∈[-1,1] 时理论应恒为 True。
    """
    gamma, wT, wE = action_norm_to_phys(
        action_norm, version=ACTION_MAPPING_RATIO, w=w
    )
    ok = (
        (wT >= WT_LO * (1 - rtol))
        & (wT <= WT_HI * (1 + rtol))
        & (wE >= WE_LO * (1 - rtol))
        & (wE <= WE_HI * (1 + rtol))
        & (gamma >= GAMMA_LO - 1e-12)
        & (gamma <= GAMMA_HI + 1e-12)
    )
    return np.asarray(ok, dtype=bool)


def _phys_to_td3_box(
    gamma: np.ndarray, wT: np.ndarray, wE: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gn0 = gamma / 3.0
    wTc = np.clip(wT, WT_LO, WT_HI)
    wEc = np.clip(wE, WE_LO, WE_HI)
    wTn0 = (np.log10(wTc) - 3.5) / 1.5
    wEn0 = np.log10(wEc) - 3.0
    gn = np.clip(gn0, -1.0, 1.0)
    wTn = np.clip(wTn0, -1.0, 1.0)
    wEn = np.clip(wEn0, -1.0, 1.0)
    an = np.stack([gn, wTn, wEn], axis=-1)
    clipped = (
        (np.abs(gn - gn0) > 1e-12)
        | (np.abs(wTn - wTn0) > 1e-12)
        | (np.abs(wEn - wEn0) > 1e-12)
        | (wTc != wT)
        | (wEc != wE)
    )
    return an.astype(np.float32), np.asarray(clipped, dtype=bool)


def _phys_to_ratio(
    gamma: np.ndarray,
    wT: np.ndarray,
    wE: np.ndarray,
    *,
    w: float,
) -> tuple[np.ndarray, np.ndarray]:
    wEc = np.maximum(wE, 1e-12)
    rho_e0 = float(w) / wEc
    rho_t0 = wT / wEc
    rho_e = np.clip(rho_e0, RHO_E_LO, RHO_E_HI)
    rho_t = np.clip(rho_t0, RHO_T_LO, RHO_T_HI)
    a_e0 = np.log10(rho_e) - 1.0
    a_t0 = (np.log10(rho_t) - 0.5) / 2.5
    g0 = gamma / 3.0
    a_e = np.clip(a_e0, -1.0, 1.0)
    a_t = np.clip(a_t0, -1.0, 1.0)
    g = np.clip(g0, -1.0, 1.0)
    an = np.stack([a_e, a_t, g], axis=-1)
    clipped = (
        (np.abs(a_e - a_e0) > 1e-12)
        | (np.abs(a_t - a_t0) > 1e-12)
        | (np.abs(g - g0) > 1e-12)
        | (np.abs(rho_e - rho_e0) > 1e-12)
        | (np.abs(rho_t - rho_t0) > 1e-12)
    )
    return an.astype(np.float32), np.asarray(clipped, dtype=bool)
