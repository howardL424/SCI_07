"""
load_expert_buffer.py — 从伪谱专家库预填充 SB3 TD3 ReplayBuffer（BC 热启动）。

expert_lib_1000.mat 格式（由 generate_ps_expert_library.m 生成）：
    results.expert_sa  [N_total_steps × 17]
        第 1-13 列  = state  (13维态势)
        第 14 列    = w      (固定 1e4，丢弃)
        第 15 列    = gamma  (权重分配系数)
        第 16 列    = wT     (目标打击权重)
        第 17 列    = wE     (控制能量权重)

    results.trajectories(k).transitions.t   [N_k×1]   时间
    结构体中 episode 边界由轨迹索引决定（非逐步 done 标志）。

归一化规则（与 EvasionEnv.norm_action 一致）：
    gamma_n = clip(gamma / 3, -1, 1)
    wT_n    = clip(log10(wT) - 3, -1, 1)
    wE_n    = clip(log10(wE) - 3, -1, 1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
from stable_baselines3.common.buffers import ReplayBuffer


# 专家库默认路径（E盘归档）
DEFAULT_EXPERT_PATH = r"E:\matlab\result_0702\expert_lib_1000.mat"


def _load_results(mat_path: str):
    """读取 expert_lib .mat（支持 v7 与 v7.3/HDF5，不修改原文件）。"""
    try:
        raw = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return raw["results"]
    except NotImplementedError:
        import mat73
        data = mat73.loadmat(mat_path)
        return data["results"]


def _get_expert_sa(results) -> np.ndarray:
    if isinstance(results, dict):
        return np.asarray(results["expert_sa"], dtype=np.float32)
    return np.array(results.expert_sa, dtype=np.float32)


def _iter_trajectories(results):
    if isinstance(results, dict):
        trajs = results.get("trajectories", [])
    else:
        trajs = results.trajectories
    if trajs is None:
        return []
    if isinstance(trajs, dict):
        return [trajs[k] for k in sorted(trajs.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
    if not hasattr(trajs, "__len__") or isinstance(trajs, (str, bytes)):
        return [trajs]
    try:
        return list(trajs)
    except TypeError:
        return [trajs]


def _traj_n_steps(traj) -> int:
    if traj is None:
        return 0
    if isinstance(traj, dict):
        trans = traj.get("transitions")
    else:
        trans = getattr(traj, "transitions", None)
    if trans is None:
        return 0
    if isinstance(trans, dict):
        t_arr = trans.get("t")
    else:
        t_arr = trans.t
    if t_arr is None:
        return 0
    return int(np.asarray(t_arr).size)


def _get_transition_field(traj, field: str):
    if isinstance(traj, dict):
        trans = traj.get("transitions")
    else:
        trans = getattr(traj, "transitions", None)
    if trans is None:
        return None
    if isinstance(trans, dict):
        return trans.get(field)
    return getattr(trans, field, None)


def _norm_actions(gamma: np.ndarray, wT: np.ndarray, wE: np.ndarray) -> np.ndarray:
    """将物理权重参数归一化到 [-1,1]³。"""
    gn  = np.clip(gamma / 3.0, -1.0, 1.0)
    wTn = np.clip(np.log10(np.clip(wT, 1e2, 1e4)) - 3.0, -1.0, 1.0)
    wEn = np.clip(np.log10(np.clip(wE, 1e2, 1e4)) - 3.0, -1.0, 1.0)
    return np.stack([gn, wTn, wEn], axis=-1).astype(np.float32)


def _target_zem_from_state_abs(state_abs: np.ndarray) -> np.ndarray:
    """按 target_zem.m / casadi_target_zem.m 的公式批量计算目标 ZEM。"""
    sabs = np.asarray(state_abs, dtype=np.float64)
    if sabs.ndim == 1:
        sabs = sabs.reshape(1, -1)
    M = sabs[:, :6]
    posx, posy, posz = M[:, 0], M[:, 1], M[:, 2]
    thetaM, psiM, VM = M[:, 3], M[:, 4], M[:, 5]

    r = np.sqrt(posx**2 + posy**2 + posz**2 + 1e-6)
    q_y = np.arcsin(np.clip(posy / r, -1.0, 1.0))
    q_z = -np.arctan2(posz, posx)

    rdot = VM * (
        np.cos(thetaM) * np.cos(q_y) * np.cos(q_z - psiM)
        + np.sin(thetaM) * np.sin(q_y)
    )
    term_M_qy = VM * (
        np.cos(thetaM) * np.sin(q_y) * np.cos(q_z - psiM)
        - np.sin(thetaM) * np.cos(q_y)
    )
    qd_y = -term_M_qy / r
    cos_qy = np.sqrt(np.cos(q_y) ** 2 + 1e-10)
    qd_z = (-VM * np.cos(thetaM) * np.sin(q_z - psiM)) / (r * cos_qy)

    Q = np.sqrt(qd_y**2 + qd_z**2 * np.cos(q_y) ** 2 + 1e-10)
    tgo = np.where(rdot < -1e-3, -r / rdot, r / 1e-3)
    Qt = np.clip(Q * tgo, -30.0, 30.0)
    zT = np.cosh(Qt) * r + np.sinh(Qt) / Q * rdot
    return np.abs(zT).astype(np.float32)


def _compute_reward(
    states: np.ndarray,          # [N, 13]
    next_states: np.ndarray,     # [N, 13]
    next_state_abs: np.ndarray,  # [N, 18]
    zT0_all: np.ndarray,         # [N]
    a_M_norm: float = 0.0,       # 无加速度幅值信息时设 0
    k1: float = 1.0,
    kT: float = 1.0,
    kE: float = 1e-6,
    dt: float = 0.1,
    zT0_floor: float = 25.0,     # zT0 下限（与 zT_reach 对齐），防止小 zT0 导致比值爆炸
    ratio_cap: float = 5.0,      # zT/zT0 比值上限，与 env_step.m 保持一致（已从30降到5）
) -> np.ndarray:
    """
    与 env_step.m 保持一致的 reward 计算（近似版，无 a_M 幅值）。
    目标项使用下一步绝对状态计算 zT，并按每条轨迹起点 zT0 归一化。

    zT0 下限取 max(zT0, zT0_floor=25m)，与 env_reset.m / env_step.m 对齐。
    比值 zT/zT0 上限 clip 到 ratio_cap=30，防止平方后 float32 溢出为 -inf。
    """
    # 列索引（对应 extract_expert_transitions.m 定义）
    # 0:r_MD1  1:rdot_MD1  2:qy_D1M  3:qz_D1M
    # 4:r_MD2  5:rdot_MD2  6:qy_D2M  7:qz_D2M
    # 8:r_MT   9:qy_MT    10:qz_MT  11:V_M  12:tgo_norm
    r_MD1_n = np.maximum(next_states[:, 0], 1.0)
    r_MD2_n = np.maximum(next_states[:, 4], 1.0)
    zT_n    = _target_zem_from_state_abs(next_state_abs)

    zT0_safe = np.maximum(zT0_all, zT0_floor)
    ratio    = np.minimum(zT_n / zT0_safe, ratio_cap)

    r_evade  = k1  * (np.log10(r_MD1_n) + np.log10(r_MD2_n))
    r_target = -kT * ratio ** 2
    reward   = (r_evade + r_target).astype(np.float32)
    return reward


def load_expert_buffer(
    replay_buffer: ReplayBuffer,
    mat_path: str = DEFAULT_EXPERT_PATH,
    reward_kwargs: Optional[dict] = None,
    max_steps: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """
    从 expert_lib_1000.mat 读取所有 (s, a, s', r, done) 并写入 replay_buffer。

    参数
    ------
    replay_buffer : SB3 ReplayBuffer（已通过 model.replay_buffer 获取）
    mat_path      : expert_lib_1000.mat 路径
    reward_kwargs : 传给 _compute_reward 的额外关键字参数
    max_steps     : 最多预填充步数（None=全部）
    verbose       : 是否打印进度

    返回
    ------
    int: 实际写入的 transition 数量
    """
    if not Path(mat_path).exists():
        raise FileNotFoundError(
            f"专家库文件不存在：{mat_path}\n"
            "请先运行 MATLAB 的 generate_ps_expert_library 生成专家轨迹，"
            "或修改 mat_path 参数指向正确路径。"
        )

    if verbose:
        print(f"[load_expert_buffer] 加载专家库：{mat_path}")

    # ── 读取 .mat 文件（v7 / v7.3 均支持，不修改原文件）────────────────────
    results = _load_results(mat_path)
    expert_sa: np.ndarray = _get_expert_sa(results)
    # expert_sa shape: [N_total, 17]
    # 列: 0-12=state, 13=w(丢弃), 14=gamma, 15=wT, 16=wE

    N_total = len(expert_sa)
    if verbose:
        print(f"[load_expert_buffer] 共 {N_total} 条 transition")

    states_all  = expert_sa[:, :13]        # [N, 13]
    gamma_all   = expert_sa[:, 14]         # [N]
    wT_all      = expert_sa[:, 15]         # [N]
    wE_all      = expert_sa[:, 16]         # [N]
    actions_all = _norm_actions(gamma_all, wT_all, wE_all)  # [N, 3]

    # ── 提取轨迹级 episode 边界（利用 trajectories 结构体）─────────────────
    # 每条轨迹对应 results.trajectories(k).transitions，用于确定 done 标志
    trajs = _iter_trajectories(results)

    # 建立每步的 episode done 标志（各轨迹最后一步置 True）
    dones_all = np.zeros(N_total, dtype=bool)
    cursor = 0
    for traj in trajs:
        n_k = _traj_n_steps(traj)
        if n_k <= 0:
            continue
        if cursor + n_k <= N_total:
            dones_all[cursor + n_k - 1] = True
        cursor += n_k
        if cursor >= N_total:
            break

    # ── 拼接绝对状态（与 expert_sa 行序一致，用于 ZEM reward）──────────────
    state_abs_chunks: list[np.ndarray] = []
    zT0_all = np.ones(N_total, dtype=np.float32)
    cursor = 0
    for traj in trajs:
        n_k = _traj_n_steps(traj)
        if n_k <= 0:
            continue
        sabs = _get_transition_field(traj, "state_abs")
        if sabs is None:
            raise ValueError(
                "专家库缺少 trajectories(k).transitions.state_abs，"
                "无法按 zT 计算 BC reward。"
            )
        sabs = np.asarray(sabs, dtype=np.float32)
        end_idx = min(cursor + n_k, N_total)
        n_use = end_idx - cursor
        chunk = sabs[:n_use]
        state_abs_chunks.append(chunk)
        zT0_all[cursor:end_idx] = _target_zem_from_state_abs(chunk[0:1])[0]
        cursor = end_idx
        if cursor >= N_total:
            break

    if not state_abs_chunks:
        raise ValueError("未能从专家库 trajectories 中读取任何 state_abs。")
    state_abs_all = np.vstack(state_abs_chunks)

    # ── 对齐 next_state（相邻步，episode 边界重复最后一步）──────────────────
    next_states_all = np.concatenate([states_all[1:], states_all[-1:]], axis=0)
    next_state_abs_all = np.concatenate(
        [state_abs_all[1:], state_abs_all[-1:]], axis=0
    )
    # episode 边界处 next_state 用自身填充（done=True 时 critic target 不使用）
    done_idx = np.where(dones_all)[0]
    for idx in done_idx:
        next_states_all[idx] = states_all[idx]
        next_state_abs_all[idx] = state_abs_all[idx]

    # ── 计算 reward ─────────────────────────────────────────────────────────
    rw_kw = reward_kwargs or {}
    rewards_all = _compute_reward(
        states_all,
        next_states_all,
        next_state_abs_all,
        zT0_all,
        **rw_kw,
    )

    # ── 写入 ReplayBuffer ────────────────────────────────────────────────────
    N_write = N_total if max_steps is None else min(max_steps, N_total - 1)
    written = 0
    for i in range(N_write - 1):
        # SB3 ReplayBuffer.add(obs, next_obs, action, reward, done, infos)
        replay_buffer.add(
            obs=states_all[i],
            next_obs=next_states_all[i],
            action=actions_all[i],
            reward=np.array([rewards_all[i]]),
            done=np.array([dones_all[i]]),
            infos=[{}],
        )
        written += 1

    if verbose:
        print(
            f"[load_expert_buffer] 预填充完成：写入 {written} 条 transition，"
            f"ReplayBuffer 当前大小 {replay_buffer.size()} / {replay_buffer.buffer_size}"
        )
    return written


if __name__ == "__main__":
    """独立测试：验证读取格式是否正确（不启动 MATLAB）。"""
    import sys

    mat = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPERT_PATH
    r   = _load_results(mat)
    sa  = _get_expert_sa(r)
    print(f"expert_sa shape : {sa.shape}")
    print(f"state  range    : {sa[:, :13].min():.3f} ~ {sa[:, :13].max():.3f}")
    print(f"gamma  range    : {sa[:, 14].min():.3f} ~ {sa[:, 14].max():.3f}")
    print(f"wT     range    : {sa[:, 15].min():.1f} ~ {sa[:, 15].max():.1f}")
    print(f"wE     range    : {sa[:, 16].min():.1f} ~ {sa[:, 16].max():.1f}")

    trajs = _iter_trajectories(r)
    sabs_list = []
    for traj in trajs:
        sabs = _get_transition_field(traj, "state_abs")
        if sabs is not None:
            sabs_list.append(np.asarray(sabs, dtype=np.float64))
    if sabs_list:
        sabs_all = np.vstack(sabs_list)
        zT_all = _target_zem_from_state_abs(sabs_all)
        print(f"zT     range    : {zT_all.min():.1f} ~ {zT_all.max():.1f} m")
    else:
        print("zT     range    : (无 state_abs，跳过)")
