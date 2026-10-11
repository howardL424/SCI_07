"""
obs_qmag.py — 13D → 16D 状态扩展（Q1/Q2/QT 方位变化率幅值）的离线派生与归一化。

可撤回性设计：
  - 本文件为纯新增模块；默认 13D 训练/评估路径完全不 import、不调用它。
  - 16D schema 必须显式 opt-in（--obs-schema rel13_qmag16_v1 或 obs_schema 参数）。
  - Q 由每步 state_abs[18] 确定性派生（解析 computeMDDot 口径，与
    rl_td3/matlab/stage2_step.m 及 evasion_helpers.m::computeMDDot 一致），
    不使用未来状态、轨迹统计量或相邻步差分。
  - T 通道沿用 stage2_step.m 的 T->M 几何口径：
    [qy_TM,qz_TM,r_TM] = computeLOS(T_pos, M)，再 computeMDDot(M, T_state, ...)。

物理布局：state_abs[18] = [M(6); D1(6); D2(6)]，每 6 维 = [x,y,z,theta,psi,V]。
T 为静止目标：T_pos=[0,0,0]，T_state=[0,0,0,0,0,0]。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from obs_norm import OBS_SCALE  # noqa: F401
except ImportError:  # pragma: no cover - 允许直接脚本调试
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from obs_norm import OBS_SCALE  # type: ignore[no-redef]

# ── schema 常量（新增，不改动 obs_norm.OBS_SCHEMA_VERSION） ──
SCHEMA_REL13 = "rel13_v1"  # 语义等价于 stage2_relative_obs_13d_v1 的别名
SCHEMA_REL13_QMAG16 = "rel13_qmag16_v1"
OBS_DIM_13 = 13
OBS_DIM_16 = 16
Q_NAMES: Tuple[str, str, str] = ("Q1", "Q2", "QT")
Q_SCALE_QUANTILE = 0.995

# 16D 归一化版本（13D 的 OBS_NORM_VERSION 保持不变）
OBS16_NORM_VERSION = "obs_scale_0714_v1+qmag_q995_v1"

# observation_space 实用上界（Q 归一化后；不裁剪数值，仅定义空间）
Q_BOX_HIGH = 5.0


# ── MATLAB evasion_helpers.m 的 numpy 移植（必须逐行对齐） ──

def compute_los(pos1: np.ndarray, pos2: np.ndarray) -> Tuple[float, float, float]:
    """移植 computeLOS：从 pos1 看 pos2。返回 (q_y, q_z, r)。"""
    p1 = np.asarray(pos1, dtype=np.float64).ravel()[:3]
    p2 = np.asarray(pos2, dtype=np.float64).ravel()[:3]
    d = p2 - p1
    r = max(float(np.linalg.norm(d)), 1e-3)
    sin_qy = max(min(float(d[1]) / r, 1.0), -1.0)
    q_y = float(np.arcsin(sin_qy))
    q_z = float(-np.arctan2(float(d[2]), float(d[0])))
    return q_y, q_z, r


def compute_md_dot(
    pos1: np.ndarray,
    pos2: np.ndarray,
    r: float,
    q_y: float,
    q_z: float,
) -> Tuple[float, float, float]:
    """移植 computeMDDot。pos1/pos2 为 6 维 [x,y,z,theta,psi,V]。返回 (r_dot, qd_y, qd_z)。"""
    p1 = np.asarray(pos1, dtype=np.float64).ravel()
    p2 = np.asarray(pos2, dtype=np.float64).ravel()
    V_M, theta_M, psi_M = float(p1[5]), float(p1[3]), float(p1[4])
    V_D, theta_D, psi_D = float(p2[5]), float(p2[3]), float(p2[4])

    term_D = V_D * (np.cos(theta_D) * np.cos(q_y) * np.cos(q_z - psi_D)
                    + np.sin(theta_D) * np.sin(q_y))
    term_M = V_M * (np.cos(theta_M) * np.cos(q_y) * np.cos(q_z - psi_M)
                    + np.sin(theta_M) * np.sin(q_y))
    r_dot = float(-term_D + term_M)

    if abs(r) < 1e-6:
        qd_y = 0.0
    else:
        term_D_qy = V_D * (np.cos(theta_D) * np.sin(q_y) * np.cos(q_z - psi_D)
                           - np.sin(theta_D) * np.cos(q_y))
        term_M_qy = V_M * (np.cos(theta_M) * np.sin(q_y) * np.cos(q_z - psi_M)
                           - np.sin(theta_M) * np.cos(q_y))
        qd_y = float((term_D_qy - term_M_qy) / r)

    cos_qy_safe = max(abs(float(np.cos(q_y))), 1e-5)
    r_safe = max(abs(float(r)), 1e-3)
    term_D_qz = V_D * np.cos(theta_D) * np.sin(q_z - psi_D)
    term_M_qz = V_M * np.cos(theta_M) * np.sin(q_z - psi_M)
    qd_z = float((term_D_qz - term_M_qz) / (r_safe * cos_qy_safe))
    return r_dot, qd_y, qd_z


def qmag_from_state_abs(
    state_abs: np.ndarray,
    t_pos: Optional[np.ndarray] = None,
    t_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """由单步 state_abs[18] 计算 (Q1, Q2, QT) 原始物理量（未归一化）。

    Q1 = sqrt(qd_y1^2 + qd_z1^2 * cos(qy_D1M)^2)，Q2 同理；
    QT = sqrt(qd_yTM^2 + qd_zTM^2 * cos(qy_TM)^2)，T 通道为 T->M 口径。
    注意：此处为原始 Q（不做控制器内的 Q*tgo>100 钳制），观测应反映真实几何。
    """
    s = np.asarray(state_abs, dtype=np.float64).ravel()
    if s.size != 18:
        raise ValueError(f"state_abs 须为 18 维，收到 {s.size}")
    M, D1, D2 = s[0:6], s[6:12], s[12:18]
    Tp = np.zeros(3) if t_pos is None else np.asarray(t_pos, dtype=np.float64).ravel()[:3]
    Ts = np.zeros(6) if t_state is None else np.asarray(t_state, dtype=np.float64).ravel()[:6]

    qy1, qz1, r1 = compute_los(D1[:3], M[:3])
    _, qd_y1, qd_z1 = compute_md_dot(M, D1, r1, qy1, qz1)
    Q1 = float(np.sqrt(qd_y1 ** 2 + qd_z1 ** 2 * np.cos(qy1) ** 2))

    qy2, qz2, r2 = compute_los(D2[:3], M[:3])
    _, qd_y2, qd_z2 = compute_md_dot(M, D2, r2, qy2, qz2)
    Q2 = float(np.sqrt(qd_y2 ** 2 + qd_z2 ** 2 * np.cos(qy2) ** 2))

    qyT, qzT, rT = compute_los(Tp, M[:3])
    _, qd_yT, qd_zT = compute_md_dot(M, Ts, rT, qyT, qzT)
    QT = float(np.sqrt(qd_yT ** 2 + qd_zT ** 2 * np.cos(qyT) ** 2))
    return np.array([Q1, Q2, QT], dtype=np.float64)


def qmag_batch(state_abs_batch: np.ndarray) -> np.ndarray:
    """批量计算 [N,18] -> [N,3]。"""
    S = np.asarray(state_abs_batch, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] != 18:
        raise ValueError(f"state_abs_batch 须为 [N,18]，收到 {S.shape}")
    out = np.empty((S.shape[0], 3), dtype=np.float64)
    for i in range(S.shape[0]):
        out[i] = qmag_from_state_abs(S[i])
    return out


def qmag_for_aligned_data(
    aligned_data: dict,
    expert_path: str,
    *,
    state_tol: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """旧库 + aligned 组合下，按 flat 顺序给出与 data['states'] 对齐的 Q [N,3]。

    映射依据（见 rl_td3/matlab/rebuild_expert_transitions.m）：
      - aligned 轨 tid 的 source_idx[tid] = 专家库 1-based 轨号 k；
      - aligned 步 j ↔ 库轨 k 的记录步 j（前缀对应；拦截截断只取前缀）。
    Q 由库轨 state_abs 前缀逐拍解析派生。逐轨校验：
      - 库记录步数 >= aligned n_steps；
      - 库记录 state[0] 与 aligned 首步一致（防排序/版本错配）；
      - 全轨记录 state 与 aligned states 的最大漂移（clip 重放所致，如有则报告）。
    返回 (q[N,3], report)。任何硬不一致直接 raise，不静默拼。
    """
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from load_expert_buffer import _as_NxD, _get_transition_field, _iter_trajectories

    try:
        import mat73
    except ImportError as e:
        raise ImportError(f"读取 v7.3 专家库需要 mat73：{e}")

    n_steps = np.asarray(aligned_data["n_steps"], dtype=np.int64)
    source_idx = np.asarray(aligned_data["source_idx"], dtype=np.int64)
    states = np.asarray(aligned_data["states"], dtype=np.float64)
    if len(states) != int(n_steps.sum()):
        raise ValueError("aligned states 步数与 n_steps 之和不一致")
    offs = np.concatenate([[0], np.cumsum(n_steps)])

    results = mat73.loadmat(expert_path)["results"]
    lib_trajs = _iter_trajectories(results)
    abs_by_src: dict[int, np.ndarray] = {}
    state_by_src: dict[int, np.ndarray] = {}
    for i, traj in enumerate(lib_trajs):
        sabs = _get_transition_field(traj, "state_abs")
        st = _get_transition_field(traj, "state")
        if sabs is None or st is None:
            continue
        abs_by_src[i + 1] = _as_NxD(sabs, 18)
        state_by_src[i + 1] = _as_NxD(st, 13)

    q_parts: list[np.ndarray] = []
    max_first_diff = 0.0
    max_full_drift = 0.0
    worst_tid = -1
    for tid in range(len(n_steps)):
        n = int(n_steps[tid])
        k = int(source_idx[tid])
        if k not in abs_by_src:
            raise ValueError(f"aligned 轨 tid={tid} 的 source_idx={k} 在库中缺 state_abs")
        lib_abs = abs_by_src[k]
        lib_st = state_by_src[k]
        if len(lib_abs) < n or len(lib_st) < n:
            raise ValueError(
                f"aligned 轨 tid={tid}(src={k}) n_steps={n} 超出库记录 "
                f"{len(lib_abs)}/{len(lib_st)}"
            )
        s, e = int(offs[tid]), int(offs[tid + 1])
        d0 = float(np.abs(lib_st[0] - states[s]).max())
        max_first_diff = max(max_first_diff, d0)
        if d0 > state_tol:
            raise ValueError(
                f"aligned 轨 tid={tid}(src={k}) 首步与库记录不一致 "
                f"max|diff|={d0:.3e}（排序/版本错配，拒绝拼接）"
            )
        drift = float(np.abs(lib_st[:n] - states[s:e]).max())
        if drift > max_full_drift:
            max_full_drift = drift
            worst_tid = tid
        q_parts.append(qmag_batch(lib_abs[:n]))
    q = np.concatenate(q_parts, axis=0)
    if len(q) != len(states):
        raise ValueError(f"Q 步数 {len(q)} 与 states {len(states)} 不一致")
    if not np.isfinite(q).all():
        raise ValueError("派生 Q 含 NaN/Inf")
    report = {
        "n_traj": int(len(n_steps)),
        "n_steps": int(len(q)),
        "max_first_step_diff": max_first_diff,
        "max_full_traj_drift": max_full_drift,
        "worst_drift_tid": int(worst_tid),
    }
    return q, report


# ── 归一化 ──

def estimate_q_scales(
    q_train: np.ndarray,
    quantile: float = Q_SCALE_QUANTILE,
) -> np.ndarray:
    """仅用训练集估计每通道 Q scale（绝对值分位数）。验证集不得参与。"""
    q = np.asarray(q_train, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"q_train 须为 [N,3]，收到 {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("q_train 含 NaN/Inf")
    scales = np.quantile(np.abs(q), quantile, axis=0).astype(np.float64)
    scales[scales <= 0] = 1.0
    return scales


def normalize_obs16(
    obs13_phys: np.ndarray,
    q_phys: np.ndarray,
    q_scale: Sequence[float],
) -> np.ndarray:
    """[N,13] 物理量 + [N,3] Q 物理量 -> [N,16] 归一化（只除、不裁剪）。"""
    from obs_norm import normalize_obs as _norm13

    o = np.asarray(obs13_phys, dtype=np.float64)
    q = np.asarray(q_phys, dtype=np.float64)
    qs = np.asarray(q_scale, dtype=np.float64).ravel()
    if o.ndim == 1:
        o = o.reshape(1, -1)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if o.shape[1] != 13:
        raise ValueError(f"obs13 须为 13 维，收到 {o.shape}")
    if q.shape[1] != 3 or qs.size != 3:
        raise ValueError(f"Q 须为 3 维，收到 {q.shape} / scale {qs.shape}")
    if len(o) != len(q):
        raise ValueError(f"obs 与 Q 步数不一致：{len(o)} vs {len(q)}")
    on = np.asarray(_norm13(o), dtype=np.float32)
    qn = (q / qs).astype(np.float32)
    return np.concatenate([on, qn], axis=1)


def denormalize_obs16(obs16_n: np.ndarray, q_scale: Sequence[float]) -> np.ndarray:
    """归一化 16D -> 物理量 16D（前 13 维用 OBS_SCALE，后 3 维用 q_scale）。"""
    from obs_norm import denormalize_obs as _denorm13

    x = np.asarray(obs16_n, dtype=np.float64)
    qs = np.asarray(q_scale, dtype=np.float64).ravel()
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)
    if x.shape[1] != 16 or qs.size != 3:
        raise ValueError(f"obs16 须为 16 维，收到 {x.shape}")
    o_phys = np.asarray(_denorm13(x[:, :13]), dtype=np.float64)
    q_phys = x[:, 13:16] * qs
    out = np.concatenate([o_phys, q_phys], axis=1)
    return out[0] if single else out


def overflow_ratio_q(obs16_n: np.ndarray, limit: float = 1.0) -> dict:
    """报告 13D 部分与 Q 部分各自 |x|>limit 的比例。"""
    x = np.asarray(obs16_n, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 16:
        raise ValueError(f"obs16 须为 16 维，收到 {x.shape}")
    if x.size == 0:
        return {"obs13": 0.0, "q": 0.0, "all": 0.0}
    return {
        "obs13": float(np.mean(np.abs(x[:, :13]) > limit)),
        "q": float(np.mean(np.abs(x[:, 13:16]) > limit)),
        "all": float(np.mean(np.abs(x) > limit)),
    }
