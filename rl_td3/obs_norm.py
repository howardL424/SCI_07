"""
obs_norm.py — 专家与在线环境共用的固定逐维物理观测归一化。

首版 scale 依据 0714 全库分位数（非 Box obs_high）：
  只除 scale、不裁剪；调用方可统计越过 [-1,1] 的比例。
"""

from __future__ import annotations

import numpy as np

# [r_MD1, rdot_MD1, qy_D1, qz_D1, r_MD2, rdot_MD2, qy_D2, qz_D2,
#  r_MT, qy_MT, qz_MT, V_M, tgo_norm]
OBS_DIM = 13

# 与 MATLAB transition_schema_meta.obs_norm_version 保持同步
OBS_NORM_VERSION = "obs_scale_0714_v1"
OBS_SCHEMA_VERSION = "stage2_relative_obs_13d_v1"

# ── 16D 扩展（加法式新增；13D 默认路径不受影响） ──
# rel13_qmag16_v1 = [obs13, Q1, Q2, QT]，详见 rl_td3/obs_qmag.py 与
# rl_td3/matlab/stage2_qmag_from_abs.m。默认训练仍使用 13D。
OBS_SCHEMA_REL13_QMAG16 = "rel13_qmag16_v1"
OBS_DIM_16 = 16
OBS16_NORM_VERSION = "obs_scale_0714_v1+qmag_q995_v1"

# [r_MD1, rdot_MD1, qy_D1, qz_D1, r_MD2, rdot_MD2, qy_D2, qz_D2,
#  r_MT, qy_MT, qz_MT, V_M, tgo_norm]
OBS_SCALE = np.array(
    [
        3.0e4,
        1.2e3,
        np.pi / 2,
        np.pi,
        3.0e4,
        1.2e3,
        np.pi / 2,
        np.pi,
        1.5e4,
        np.pi / 2,
        np.pi,
        1.0e3,
        1.0,
    ],
    dtype=np.float64,
)

# 归一化后 observation_space 实用上界（不裁剪数值，仅定义空间）
OBS_BOX_HIGH = 5.0


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """obs / OBS_SCALE，不裁剪。保持输入 shape。"""
    x = np.asarray(obs, dtype=np.float64)
    return (x / OBS_SCALE).astype(np.float32)


def denormalize_obs(obs_n: np.ndarray) -> np.ndarray:
    """还原物理量。"""
    x = np.asarray(obs_n, dtype=np.float64)
    return (x * OBS_SCALE).astype(np.float64)


def overflow_ratio(obs_n: np.ndarray, limit: float = 1.0) -> float:
    """归一化后 |x|>limit 的元素比例。"""
    x = np.asarray(obs_n, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) > limit))
