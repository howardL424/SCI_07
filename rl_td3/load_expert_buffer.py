"""
load_expert_buffer.py — 从伪谱专家库预填充 SB3 TD3 ReplayBuffer（BC 热启动）。

expert_lib_1000.mat 格式（由 generate_ps_expert_library.m 生成）：
    results.expert_sa  [N_total_steps × 17]
        第 1-13 列  = state  (13维态势)
        第 14 列    = w      (固定 1e4，丢弃)
        第 15 列    = gamma
        第 16 列    = wT
        第 17 列    = wE

终端奖励：读取 precompute_expert_terminals.py 生成的 cache，
仅在每条轨迹最后一步追加 R1 + bonus（与在线 env 对齐）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
from stable_baselines3.common.buffers import ReplayBuffer


DEFAULT_EXPERT_PATH = r"E:\matlab\result_0714\expert_lib_1000.mat"
DEFAULT_TERMINAL_CACHE = str(Path(__file__).parent / "expert_terminal_cache.npz")


@dataclass
class ExpertLoadResult:
    n_written: int
    states: np.ndarray   # [N, 13] 写入 buffer 的专家状态（供 TD3BC）
    actions: np.ndarray  # [N, 3]


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
        sa = np.asarray(results["expert_sa"], dtype=np.float32)
    else:
        sa = np.array(results.expert_sa, dtype=np.float32)
    # HDF5 有时是 (17, N)
    if sa.ndim == 2 and sa.shape[0] == 17 and sa.shape[1] != 17:
        sa = sa.T
    return sa


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
    wTn = np.clip((np.log10(np.clip(wT, 1e2, 1e5)) - 3.5) / 1.5, -1.0, 1.0)
    wEn = np.clip(np.log10(np.clip(wE, 1e2, 1e4)) - 3.0, -1.0, 1.0)
    return np.stack([gn, wTn, wEn], axis=-1).astype(np.float32)


def _soft_sat_log_evade(
    r_AD: np.ndarray,
    dstar: float = 20.0,
    kill_radius: float = 6.0,
) -> np.ndarray:
    """与 env_step.m soft_sat_log_evade 一致。"""
    r = np.asarray(r_AD, dtype=np.float64)
    out = np.zeros_like(r, dtype=np.float64)
    ok = r > kill_radius
    r_eff = np.minimum(r[ok], dstar)
    out[ok] = np.log10(np.maximum(r_eff - 5.0, 1e-6))
    return out


def _compute_reward(
    states: np.ndarray,
    next_states: np.ndarray,
    next_state_abs: np.ndarray,
    zT0_all: np.ndarray,
    proc_coef: float = 0.05,
    dstar: float = 20.0,
    kill_radius: float = 6.0,
    **_unused,
) -> np.ndarray:
    """与 env_step.m 过程奖励一致。"""
    del states, next_state_abs, zT0_all, _unused
    r_MD1_n = next_states[:, 0]
    r_MD2_n = next_states[:, 4]
    R2 = _soft_sat_log_evade(r_MD1_n, dstar=dstar, kill_radius=kill_radius)
    R3 = _soft_sat_log_evade(r_MD2_n, dstar=dstar, kill_radius=kill_radius)
    return (proc_coef * (R2 + R3)).astype(np.float32)


def _load_terminal_cache(cache_path: str, mat_path: str, verbose: bool = True) -> dict:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"专家终端奖励缓存不存在：{cache_path}\n"
            "请先运行：\n"
            "  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "
            "rl_td3/precompute_expert_terminals.py\n"
            "生成 expert_terminal_cache.npz 后再训练。"
        )
    data = np.load(path, allow_pickle=True)
    cached_expert = str(data["expert_path"]) if "expert_path" in data.files else ""
    if cached_expert:
        try:
            same = Path(cached_expert).resolve() == Path(mat_path).resolve()
        except OSError:
            same = cached_expert == mat_path
        if not same and verbose:
            print(
                f"[load_expert_buffer] 警告: 终端缓存来自\n  {cached_expert}\n"
                f"  当前专家库为\n  {mat_path}\n  仍将使用该缓存。"
            )
    return {k: data[k] for k in data.files}


def load_expert_buffer(
    replay_buffer: ReplayBuffer,
    mat_path: str = DEFAULT_EXPERT_PATH,
    reward_kwargs: Optional[dict] = None,
    max_steps: Optional[int] = None,
    terminal_cache: str = DEFAULT_TERMINAL_CACHE,
    require_terminal_cache: bool = True,
    verbose: bool = True,
) -> ExpertLoadResult:
    """
    从 expert_lib 读取 (s,a,s',r,done) 写入 replay_buffer。
    若提供终端缓存，在每条轨迹最后一步追加 R1+bonus。

    返回 ExpertLoadResult(n_written, states, actions) 供 TD3BC 使用。
    """
    if not Path(mat_path).exists():
        raise FileNotFoundError(
            f"专家库文件不存在：{mat_path}\n"
            "请先运行 MATLAB 的 generate_ps_expert_library 生成专家轨迹，"
            "或修改 mat_path 参数指向正确路径。"
        )

    if verbose:
        print(f"[load_expert_buffer] 加载专家库：{mat_path}")

    results = _load_results(mat_path)
    expert_sa: np.ndarray = _get_expert_sa(results)

    N_total = len(expert_sa)
    if verbose:
        print(f"[load_expert_buffer] 共 {N_total} 条 transition")

    states_all  = expert_sa[:, :13]
    gamma_all   = expert_sa[:, 14]
    wT_all      = expert_sa[:, 15]
    wE_all      = expert_sa[:, 16]
    actions_all = _norm_actions(gamma_all, wT_all, wE_all)

    trajs = _iter_trajectories(results)

    dones_all = np.zeros(N_total, dtype=bool)
    traj_id_of_step = -np.ones(N_total, dtype=np.int32)
    cursor = 0
    traj_meta_n = []
    for traj in trajs:
        n_k = _traj_n_steps(traj)
        if n_k <= 0:
            continue
        traj_idx = len(traj_meta_n)
        traj_meta_n.append(n_k)
        end = min(cursor + n_k, N_total)
        if end > cursor:
            traj_id_of_step[cursor:end] = traj_idx
            dones_all[end - 1] = True
        cursor = end
        if cursor >= N_total:
            break

    next_states_all = np.concatenate([states_all[1:], states_all[-1:]], axis=0)
    done_idx = np.where(dones_all)[0]
    for idx in done_idx:
        next_states_all[idx] = states_all[idx]

    zT0_all = np.ones(N_total, dtype=np.float32)
    dummy_abs = np.zeros((N_total, 18), dtype=np.float32)
    rw_kw = reward_kwargs or {}
    rewards_all = _compute_reward(
        states_all,
        next_states_all,
        dummy_abs,
        zT0_all,
        **rw_kw,
    )

    # ── 追加终端 R1/+100 ───────────────────────────────────────────────────
    if require_terminal_cache or Path(terminal_cache).exists():
        cache = _load_terminal_cache(terminal_cache, mat_path, verbose=verbose)
        n_steps_c = np.asarray(cache["n_steps"]).ravel()
        R1_c = np.asarray(cache["R1"], dtype=np.float32).ravel()
        bonus_c = np.asarray(cache["bonus"], dtype=np.float32).ravel()
        n_cache = len(n_steps_c)
        n_traj = len(traj_meta_n)
        if n_cache != n_traj:
            raise ValueError(
                f"终端缓存轨迹数 ({n_cache}) 与专家库有效轨迹数 ({n_traj}) 不一致。"
                "请重新运行 precompute_expert_terminals.py。"
            )
        n_term = 0
        for idx in done_idx:
            tid = int(traj_id_of_step[idx])
            if tid < 0 or tid >= n_cache:
                continue
            rewards_all[idx] = rewards_all[idx] + R1_c[tid] + bonus_c[tid]
            n_term += 1
        if verbose:
            print(
                f"[load_expert_buffer] 已为 {n_term} 个 episode 末步追加终端奖励 "
                f"(R1 med={np.median(R1_c):.2f}, bonus>0={(bonus_c>0).mean()*100:.1f}%)"
            )
    elif verbose:
        print("[load_expert_buffer] 未加载终端缓存，BC 末步仅含过程奖励。")

    N_write = N_total if max_steps is None else min(max_steps, N_total)
    # 与旧逻辑一致：最多写到 N_write-1 的开区间风格；这里改为写 min(N_write, N_total) 条
    n_target = N_total if max_steps is None else min(max_steps, N_total)
    written = 0
    for i in range(n_target):
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

    return ExpertLoadResult(
        n_written=written,
        states=states_all[:written].copy(),
        actions=actions_all[:written].copy(),
    )


if __name__ == "__main__":
    import sys

    mat = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPERT_PATH
    r   = _load_results(mat)
    sa  = _get_expert_sa(r)
    print(f"expert_sa shape : {sa.shape}")
    print(f"state  range    : {sa[:, :13].min():.3f} ~ {sa[:, :13].max():.3f}")
    print(f"gamma  range    : {sa[:, 14].min():.3f} ~ {sa[:, 14].max():.3f}")
    print(f"wT     range    : {sa[:, 15].min():.1f} ~ {sa[:, 15].max():.1f}")
    print(f"wE     range    : {sa[:, 16].min():.1f} ~ {sa[:, 16].max():.1f}")

    next_states = np.concatenate([sa[1:, :13], sa[-1:, :13]], axis=0)
    rew = _compute_reward(sa[:, :13], next_states, np.zeros((len(sa), 18)), np.ones(len(sa)))
    print(f"proc_rew range  : {rew.min():.4f} ~ {rew.max():.4f}  mean={rew.mean():.4f}")

    cache_path = Path(DEFAULT_TERMINAL_CACHE)
    if cache_path.exists():
        c = np.load(cache_path)
        print(f"terminal cache  : {cache_path}  n_traj={len(c['n_steps'])}")
        print(f"  R1 med={np.median(c['R1']):.2f}  bonus>0={(c['bonus']>0).mean()*100:.1f}%")
    else:
        print(f"terminal cache  : 缺失 ({cache_path})")
