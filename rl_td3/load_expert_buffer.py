"""
load_expert_buffer.py — 从专家物理 transition 预填充 SB3 ReplayBuffer。

新库（phys_transition_v1）：直接读 trajectories.transitions，无需 rebuild_*。
旧库：读 expert_transitions_aligned.mat 派生表。

奖励：由 reward_projection 按物理事实廉价投影（非缓存 reward 列真值）。
终端：读 stage3 sidecar 的 r_minT 等事实，按当前 reward_version 投影 R1/bonus。
扁平 expert_sa 仅 QA，不参与轨迹切分。
观测写入 buffer 前使用 obs_norm.normalize_obs。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.io
from stable_baselines3.common.buffers import ReplayBuffer

from obs_norm import normalize_obs, overflow_ratio
from reward_projection import (
    DEFAULT_REWARD_CFG,
    RewardConfig,
    project_transition_rewards,
)


RL_TD3_ROOT = Path(__file__).resolve().parent
DATA_DIR = RL_TD3_ROOT / "data"

DEFAULT_EXPERT_PATH = r"E:\matlab\result_0714\expert_lib_1000.mat"
DEFAULT_TERMINAL_CACHE = str(DATA_DIR / "expert_terminal_cache.npz")
DEFAULT_ALIGNED_PATH = str(DATA_DIR / "expert_transitions_aligned.mat")


@dataclass
class ExpertLoadResult:
    n_written: int
    states: np.ndarray   # [N, 13] 归一化后（供 Actor BC；默认 both_ok）
    actions: np.ndarray  # [N, 3]
    sample_summary: Optional[dict] = None
    train_traj_ids: Optional[np.ndarray] = None
    val_traj_ids: Optional[np.ndarray] = None
    replay_step_indices: Optional[np.ndarray] = None
    actor_step_indices: Optional[np.ndarray] = None


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
        t_arr = getattr(trans, "t", None)
    if t_arr is None:
        if isinstance(trans, dict):
            st = trans.get("state")
        else:
            st = getattr(trans, "state", None)
        if st is None:
            return 0
        arr = np.asarray(st)
        if arr.ndim == 2 and arr.shape[0] == 13 and arr.shape[1] != 13:
            return int(arr.shape[1])
        return int(arr.shape[0]) if arr.ndim >= 1 else 0
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


def _as_NxD(arr, D: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"期望 2D 数组，得到 {a.shape}")
    if a.shape[1] == D:
        return a
    if a.shape[0] == D:
        return a.T
    raise ValueError(f"无法解释为 (N,{D}): {a.shape}")


def _norm_actions(gamma: np.ndarray, wT: np.ndarray, wE: np.ndarray) -> np.ndarray:
    """将物理权重参数归一化到 [-1,1]³（默认当前 DEFAULT_ACTION_MAPPING）。"""
    from action_mapping import DEFAULT_ACTION_MAPPING, phys_to_action_norm

    an, _ = phys_to_action_norm(gamma, wT, wE, version=DEFAULT_ACTION_MAPPING)
    return np.asarray(an, dtype=np.float32)


def _soft_sat_log_evade(
    r_AD: np.ndarray,
    dstar: float = 20.0,
    kill_radius: float = 6.0,
) -> np.ndarray:
    """兼容旧调用；实现见 reward_projection.soft_sat_log_evade。"""
    from reward_projection import soft_sat_log_evade

    return np.asarray(
        soft_sat_log_evade(r_AD, dstar=dstar, kill_radius=kill_radius),
        dtype=np.float64,
    )


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
    """兼容旧调用；新代码请用 reward_projection.project_process_rewards。"""
    del states, next_state_abs, zT0_all, _unused
    from reward_projection import RewardConfig, project_process_rewards

    cfg = RewardConfig(proc_coef=proc_coef, dstar=dstar, kill_radius=kill_radius)
    # 旧接口不含拦截惩罚（对齐数据逐步已含）；此处仅过程项
    ns = np.asarray(next_states, dtype=np.float64)
    step_int = np.zeros(len(ns), dtype=bool)
    return project_process_rewards(ns, step_int, cfg=cfg)


def _field(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def load_aligned_transitions(aligned_path: str) -> dict:
    """加载 rebuild_expert_transitions 输出的 aligned 结构。"""
    path = Path(aligned_path)
    if not path.exists():
        raise FileNotFoundError(
            f"对齐专家数据不存在：{aligned_path}\n"
            "请先运行：\n"
            "  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "
            "rl_td3/tools/rebuild_expert_data.py\n"
            "或在 MATLAB 中调用 rebuild_expert_transitions。"
        )
    try:
        raw = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        aligned = raw["aligned"]
    except NotImplementedError:
        import mat73
        data = mat73.loadmat(str(path))
        aligned = data["aligned"]

    def arr(name, dtype=None):
        v = np.asarray(_field(aligned, name))
        if dtype is not None:
            v = v.astype(dtype)
        return v

    out = {
        "expert_path": str(_field(aligned, "expert_path", "")),
        "states": arr("states", np.float64),
        "actions": arr("actions", np.float32),
        "next_states": arr("next_states", np.float64),
        "reward_cached": arr("rewards", np.float32).ravel(),  # 派生，非权威
        "rewards": arr("rewards", np.float32).ravel(),  # 占位；加载时重投影覆盖
        "dones": arr("dones").astype(bool).ravel(),
        "traj_ids": arr("traj_ids", np.int32).ravel(),
        "n_steps": arr("n_steps", np.int32).ravel(),
        "n_steps_recorded": arr("n_steps_recorded", np.int32).ravel()
        if _field(aligned, "n_steps_recorded") is not None
        else None,
        "intercepted": arr("intercepted", np.int8).ravel(),
        "terminal_abs": arr("terminal_abs", np.float64),
        "t_end": arr("t_end", np.float64).ravel(),
        "source_idx": arr("source_idx", np.int32).ravel(),
        "r_min1": arr("r_min1", np.float64).ravel(),
        "r_min2": arr("r_min2", np.float64).ravel(),
        "n_actions_clipped": int(np.asarray(_field(aligned, "n_actions_clipped", 0)).ravel()[0]),
        "action_mapping_version": str(
            _field(aligned, "action_mapping_version", "td3_box_log_v1")
        ),
    }
    if out["terminal_abs"].ndim == 1:
        out["terminal_abs"] = out["terminal_abs"].reshape(1, -1)
    if out["terminal_abs"].shape[1] != 18 and out["terminal_abs"].shape[0] == 18:
        out["terminal_abs"] = out["terminal_abs"].T
    out["source_mode"] = "aligned"
    return out


def expert_lib_has_phys_schema(mat_path: str) -> bool:
    """检测 expert_lib 是否已含 phys_transition_v1（next_state / terminal_abs）。"""
    results = _load_results(mat_path)
    # 库级版本字段
    ver = _field(results, "schema_version", None)
    if ver is not None and str(ver).startswith("phys_transition"):
        return True
    for traj in _iter_trajectories(results):
        ok = _field(traj, "ok_use", False)
        if not ok:
            continue
        if _get_transition_field(traj, "next_state") is None:
            continue
        if _get_transition_field(traj, "terminal_abs") is None:
            continue
        if _get_transition_field(traj, "action_norm") is None:
            continue
        return True
    return False


def _scalar_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size == 0:
        return default
    return float(a[0])


def _traj_intercepted(trans_intercepted, n: int) -> bool:
    if trans_intercepted is None:
        return False
    a = np.asarray(trans_intercepted).astype(bool).ravel()
    if a.size == 0:
        return False
    return bool(np.any(a[:n]))


def load_phys_transitions_from_expert_lib(expert_path: str, *, load_abs: bool = False) -> dict:
    """
    从新 schema expert_lib 直接读取物理 transition（无需 rebuild_*）。
    返回字段与 load_aligned_transitions 对齐；rewards 占位，由 reward_projection 填充。

    可撤回性：默认 load_abs=False，返回字段与历史完全一致；
    16D 实验显式传 load_abs=True 时才额外返回 state_abs / next_state_abs（[N,18]）。
    """
    results = _load_results(expert_path)
    trajs = _iter_trajectories(results)

    states_l: list[np.ndarray] = []
    actions_l: list[np.ndarray] = []
    next_l: list[np.ndarray] = []
    step_int_l: list[np.ndarray] = []
    dones_l: list[np.ndarray] = []
    traj_ids_l: list[np.ndarray] = []
    n_steps_l: list[int] = []
    intercepted_l: list[int] = []
    terminal_abs_l: list[np.ndarray] = []
    t_end_l: list[float] = []
    source_idx_l: list[int] = []
    r_min1_l: list[float] = []
    r_min2_l: list[float] = []
    state_abs_l: list[np.ndarray] = []
    next_abs_l: list[np.ndarray] = []
    n_clip_total = 0
    tid = 0

    for src_i, traj in enumerate(trajs):
        ok = bool(_field(traj, "ok_use", False))
        if not ok:
            continue
        st = _get_transition_field(traj, "state")
        an = _get_transition_field(traj, "action_norm")
        ns = _get_transition_field(traj, "next_state")
        dn = _get_transition_field(traj, "done")
        it = _get_transition_field(traj, "intercepted")
        term = _get_transition_field(traj, "terminal_abs")
        t_end = _get_transition_field(traj, "t_end")
        if st is None or an is None or ns is None or term is None:
            continue

        st = _as_NxD(st, 13)
        an = _as_NxD(an, 3)
        ns = _as_NxD(ns, 13)
        n = min(len(st), len(an), len(ns))
        if n <= 0:
            continue
        st, an, ns = st[:n], an[:n], ns[:n]

        if dn is None:
            done = np.zeros(n, dtype=bool)
            done[-1] = True
        else:
            done = np.asarray(dn).astype(bool).ravel()[:n]
            if done.size < n:
                pad = np.zeros(n, dtype=bool)
                pad[: done.size] = done
                pad[-1] = True
                done = pad
            done[-1] = True

        step_int = np.zeros(n, dtype=bool)
        if it is not None:
            raw_it = np.asarray(it).astype(bool).ravel()
            step_int[: min(n, raw_it.size)] = raw_it[:n]

        term_abs = np.asarray(term, dtype=np.float64).ravel()
        if term_abs.size != 18:
            nsa = _get_transition_field(traj, "next_state_abs")
            if nsa is not None:
                nsa = _as_NxD(nsa, 18)
                term_abs = nsa[n - 1].ravel()
            else:
                raise ValueError(f"轨迹 source_idx={src_i+1} terminal_abs 非 18 维")

        states_l.append(st)
        actions_l.append(an.astype(np.float32))
        next_l.append(ns)
        step_int_l.append(step_int)
        dones_l.append(done)
        if load_abs:
            sabs = _get_transition_field(traj, "state_abs")
            nabs = _get_transition_field(traj, "next_state_abs")
            if sabs is None or nabs is None:
                raise ValueError(
                    f"轨迹 source_idx={src_i+1} 缺少 state_abs/next_state_abs，"
                    "无法派生 16D Q；请确认专家库为新 extract 生成。"
                )
            state_abs_l.append(_as_NxD(sabs, 18)[:n])
            next_abs_l.append(_as_NxD(nabs, 18)[:n])
        traj_ids_l.append(np.full(n, tid, dtype=np.int32))
        n_steps_l.append(n)
        intercepted_l.append(1 if _traj_intercepted(it, n) else 0)
        terminal_abs_l.append(term_abs.astype(np.float64))
        t_end_l.append(_scalar_float(t_end, default=float(n) * 0.1))
        source_idx_l.append(src_i + 1)
        r_min1_l.append(_scalar_float(_get_transition_field(traj, "r_min1"), default=np.inf))
        r_min2_l.append(_scalar_float(_get_transition_field(traj, "r_min2"), default=np.inf))
        n_clip = _get_transition_field(traj, "n_actions_clipped")
        if n_clip is not None:
            n_clip_total += int(np.asarray(n_clip).ravel()[0])
        tid += 1

    if tid == 0:
        raise ValueError(
            f"专家库无 phys_transition 可用轨迹：{expert_path}\n"
            "请确认 generate_ps_expert_library 已用新 extract 生成。"
        )

    n_total = sum(n_steps_l)
    out = {
        "expert_path": str(Path(expert_path).resolve()),
        "states": np.concatenate(states_l, axis=0),
        "actions": np.concatenate(actions_l, axis=0),
        "next_states": np.concatenate(next_l, axis=0),
        "step_intercepted": np.concatenate(step_int_l, axis=0),
        "reward_cached": None,
        "rewards": np.zeros(n_total, dtype=np.float32),  # 占位；加载时重投影
        "dones": np.concatenate(dones_l, axis=0),
        "traj_ids": np.concatenate(traj_ids_l, axis=0),
        "n_steps": np.asarray(n_steps_l, dtype=np.int32),
        "n_steps_recorded": np.asarray(n_steps_l, dtype=np.int32),
        "intercepted": np.asarray(intercepted_l, dtype=np.int8),
        "terminal_abs": np.stack(terminal_abs_l, axis=0),
        "t_end": np.asarray(t_end_l, dtype=np.float64),
        "source_idx": np.asarray(source_idx_l, dtype=np.int32),
        "r_min1": np.asarray(r_min1_l, dtype=np.float64),
        "r_min2": np.asarray(r_min2_l, dtype=np.float64),
        "n_actions_clipped": int(n_clip_total),
        "source_mode": "expert_lib",
        "schema_version": str(_field(results, "schema_version", "phys_transition_v1")),
        "stage2_dynamics_version": str(
            _field(results, "stage2_dynamics_version", "stage2_rk4_eq47_v1")
        ),
        "action_mapping_version": str(
            _field(results, "action_mapping_version", "td3_box_log_v1")
        ),
    }
    if load_abs:
        out["state_abs"] = np.concatenate(state_abs_l, axis=0)
        out["next_state_abs"] = np.concatenate(next_abs_l, axis=0)
    return out


def apply_action_mapping(
    data: dict,
    target_version: str,
    *,
    source_version: Optional[str] = None,
    w: float = 1e4,
) -> dict:
    """
    将 data['actions'] 从源映射确定性变换到 target_version（不改 s'/done）。
    旧 aligned / 0714 库默认源版本 td3_box_log_v1。
    """
    from action_mapping import remap_action_norm

    src = source_version or str(data.get("action_mapping_version", "td3_box_log_v1"))
    acts = np.asarray(data["actions"], dtype=np.float32)
    if src == target_version:
        data = dict(data)
        data["action_mapping_version"] = target_version
        data["actions_source_version"] = src
        return data
    remapped = remap_action_norm(acts, from_version=src, to_version=target_version, w=w)
    data = dict(data)
    data["actions"] = remapped
    data["actions_source_version"] = src
    data["action_mapping_version"] = target_version
    return data


def load_transition_data(
    expert_path: str,
    aligned_path: Optional[str] = None,
    *,
    load_abs: bool = False,
) -> dict:
    """
    统一入口：新库优先直接读物理 transition；否则读 aligned 派生表。
    load_abs=True 时新库额外返回 state_abs / next_state_abs（16D Q 派生用）；
    旧 aligned 路径无绝对态，此时直接报错（避免静默错一拍）。
    """
    if expert_lib_has_phys_schema(expert_path):
        return load_phys_transitions_from_expert_lib(expert_path, load_abs=load_abs)
    if load_abs:
        raise ValueError(
            "load_abs=True 需要新 schema 专家库（含 state_abs/next_state_abs），"
            f"但 {expert_path} 未检出 phys_transition schema。"
        )
    if aligned_path is None:
        aligned_path = DEFAULT_ALIGNED_PATH
    return load_aligned_transitions(aligned_path)


def load_terminal_facts(
    expert_path: str,
    aligned_path: Optional[str] = None,
) -> dict:
    """
    仅取 stage3 sidecar 所需末态事实（n_steps/terminal_abs/t_end/r_min*/intercepted）。
    新库不经 rebuild；旧库经 aligned。
    """
    data = load_transition_data(expert_path, aligned_path=aligned_path)
    return {
        "expert_path": data["expert_path"],
        "source_mode": data.get("source_mode", "aligned"),
        "n_steps": data["n_steps"],
        "intercepted": data["intercepted"],
        "terminal_abs": data["terminal_abs"],
        "t_end": data["t_end"],
        "r_min1": data["r_min1"],
        "r_min2": data["r_min2"],
        "source_idx": data.get("source_idx"),
        "schema_version": data.get("schema_version"),
        "stage2_dynamics_version": data.get("stage2_dynamics_version"),
        "aligned_path": (
            str(Path(aligned_path).resolve())
            if aligned_path and data.get("source_mode") == "aligned"
            else ""
        ),
    }


def validate_expert_sa_vs_trajectories(
    mat_path: str,
    rtol: float = 1e-5,
    atol: float = 1e-4,
) -> dict:
    """用扁平 expert_sa 校验 trajectories 记录（非重放结果）。"""
    results = _load_results(mat_path)
    expert_sa = _get_expert_sa(results)
    trajs = _iter_trajectories(results)

    chunks = []
    n_valid = 0
    for traj in trajs:
        n_k = _traj_n_steps(traj)
        if n_k <= 0:
            continue
        st = _get_transition_field(traj, "state")
        ac = _get_transition_field(traj, "action")
        if st is None or ac is None:
            raise ValueError(f"有效轨迹缺少 state/action（已处理 {n_valid} 条）")
        st = _as_NxD(st, 13)
        ac = _as_NxD(ac, 4)
        n_use = min(n_k, len(st), len(ac))
        chunks.append(np.concatenate([st[:n_use], ac[:n_use]], axis=1))
        n_valid += 1

    if not chunks:
        raise ValueError("专家库中无有效 transitions")

    rebuilt = np.concatenate(chunks, axis=0).astype(np.float64)
    if rebuilt.shape[0] != expert_sa.shape[0]:
        raise ValueError(
            f"expert_sa 步数 ({expert_sa.shape[0]}) 与 trajectories 拼接 "
            f"({rebuilt.shape[0]}) 不一致"
        )
    if not np.allclose(rebuilt, expert_sa.astype(np.float64), rtol=rtol, atol=atol):
        diff = np.max(np.abs(rebuilt - expert_sa.astype(np.float64)))
        raise ValueError(f"expert_sa 与 trajectories 内容不一致，max|diff|={diff:.3e}")
    return {"n_traj": n_valid, "n_steps": int(rebuilt.shape[0])}


def check_aligned_consistency(
    data: dict,
    cache: Optional[dict] = None,
    expert_path: Optional[str] = None,
) -> list[str]:
    """
    一致性检查。返回警告列表；遇到硬错误直接 raise。
    硬错误：NaN、动作越界、轨迹步数不匹配、拦截仍有 R1、cache 路径不匹配。
    """
    warnings: list[str] = []
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    rewards = data["rewards"]
    dones = data["dones"]
    traj_ids = data["traj_ids"]
    n_steps = data["n_steps"]
    intercepted = data["intercepted"]

    n_traj = len(n_steps)
    n_total = int(states.shape[0])
    if int(n_steps.sum()) != n_total:
        raise ValueError(
            f"n_steps 之和 ({int(n_steps.sum())}) 与扁平步数 ({n_total}) 不一致"
        )
    if len(intercepted) != n_traj:
        raise ValueError("intercepted 长度与轨迹数不一致")

    for name, arr in (
        ("states", states),
        ("actions", actions),
        ("next_states", next_states),
        ("rewards", rewards),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} 含 NaN/Inf")

    if np.any(np.abs(actions) > 1.0 + 1e-5):
        raise ValueError(
            f"动作超出 [-1,1]：max|a|={np.max(np.abs(actions)):.4f}"
        )

    # done 必须落在每条轨迹末步
    cursor = 0
    for tid in range(n_traj):
        n_k = int(n_steps[tid])
        seg = dones[cursor : cursor + n_k]
        if n_k <= 0:
            raise ValueError(f"轨迹 {tid} n_steps=0")
        if not bool(seg[-1]):
            raise ValueError(f"轨迹 {tid} 末步 done=False")
        if n_k > 1 and np.any(seg[:-1]):
            raise ValueError(f"轨迹 {tid} 在非末步出现 done=True")
        if not np.all(traj_ids[cursor : cursor + n_k] == tid):
            raise ValueError(f"轨迹 {tid} traj_ids 不连续")
        cursor += n_k

    if expert_path and data.get("expert_path"):
        try:
            same = Path(data["expert_path"]).resolve() == Path(expert_path).resolve()
        except OSError:
            same = data["expert_path"] == expert_path
        if not same:
            raise ValueError(
                f"对齐数据来自\n  {data['expert_path']}\n"
                f"与当前专家库\n  {expert_path}\n不匹配"
            )

    if cache is not None:
        cached_expert = str(cache.get("expert_path", ""))
        if expert_path and cached_expert:
            try:
                same = Path(cached_expert).resolve() == Path(expert_path).resolve()
            except OSError:
                same = cached_expert == expert_path
            if not same:
                raise ValueError(
                    f"终端缓存来自\n  {cached_expert}\n"
                    f"与当前专家库\n  {expert_path}\n不匹配"
                )
        n_cache = len(np.asarray(cache["n_steps"]).ravel())
        if n_cache != n_traj:
            raise ValueError(
                f"终端缓存轨迹数 ({n_cache}) 与对齐数据 ({n_traj}) 不一致"
            )
        R1 = np.asarray(cache["R1"], dtype=np.float64).ravel()
        bonus = np.asarray(cache["bonus"], dtype=np.float64).ravel()
        bad = (intercepted.astype(bool)) & ((R1 > 0) | (bonus > 0))
        if np.any(bad):
            raise ValueError(
                f"拦截轨迹仍含 R1/bonus：{int(np.sum(bad))} 条"
            )

    ov = overflow_ratio(normalize_obs(states))
    if ov > 0.05:
        warnings.append(f"归一化后 |obs|>1 比例较高：{ov*100:.2f}%")
    return warnings


def _load_terminal_cache(
    cache_path: str,
    mat_path: str,
    *,
    require_path_match: bool = True,
    verbose: bool = True,
) -> dict:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"专家终端奖励缓存不存在：{cache_path}\n"
            "请先运行：\n"
            "  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe "
            "rl_td3/tools/precompute_expert_terminals.py\n"
            "生成 expert_terminal_cache.npz 后再训练。"
        )
    data = np.load(path, allow_pickle=True)
    cached_expert = str(data["expert_path"]) if "expert_path" in data.files else ""
    if cached_expert:
        try:
            same = Path(cached_expert).resolve() == Path(mat_path).resolve()
        except OSError:
            same = cached_expert == mat_path
        if not same:
            msg = (
                f"终端缓存来自\n  {cached_expert}\n"
                f"当前专家库为\n  {mat_path}"
            )
            if require_path_match:
                raise ValueError(msg + "\n路径必须匹配，请重新预计算。")
            if verbose:
                print(f"[load_expert_buffer] 警告: {msg}\n  仍将使用该缓存。")
    return {k: data[k] for k in data.files}

def load_expert_buffer(
    replay_buffer: ReplayBuffer,
    mat_path: str = DEFAULT_EXPERT_PATH,
    aligned_path: str = DEFAULT_ALIGNED_PATH,
    reward_kwargs: Optional[dict] = None,
    max_steps: Optional[int] = None,
    terminal_cache: str = DEFAULT_TERMINAL_CACHE,
    require_terminal_cache: bool = True,
    validate_expert_sa: bool = True,
    verbose: bool = True,
    reward_cfg: Optional[RewardConfig] = None,
    *,
    sample_mode: str = "balanced",
    sample_seed: int = 0,
    val_ratio: float = 0.1,
    n_tgo_bins: int = 4,
    actor_exclude_intercepted: bool = True,
    actor_eligible: str = "both_ok",
    replay_include_intercepted: bool = True,
    load_ok_modes: bool = False,
    sample_plan_out: Optional[str] = None,
    action_mapping: Optional[str] = None,
) -> ExpertLoadResult:
    """
    读取专家 (s,a,s',r,done) 写入 replay_buffer。
    新库：直接读物理 transition；旧库：读 aligned。
    奖励由 reward_projection 按物理事实 + sidecar r_minT 投影（不把缓存 reward 当真值）。
    reward_kwargs 可覆盖 RewardConfig 字段（如 proc_coef、bonus_both）。
    action_mapping：目标动作坐标（默认 ratio_log_box_v1）；源标签确定性重标。

    采样（sample_mode）：
      - balanced（默认）：按轨迹均衡 + tgo 分层 + 强制末步；train/val 按轨划分
      - prefix：旧行为，扁平表前缀截取（仅兼容对照）
    Actor BC / TD3 专家 MSE 默认 actor_eligible=both_ok；Replay 仍可含拦截 / stage3 失败。
    """
    from action_mapping import DEFAULT_ACTION_MAPPING
    from expert_sampling import (
        ACTOR_ELIGIBLE_MODES,
        build_expert_sample_plan,
        load_ok_modes_from_expert,
        resolve_actor_eligible,
    )

    target_amap = action_mapping or DEFAULT_ACTION_MAPPING
    actor_eligible = resolve_actor_eligible(
        actor_eligible,
        actor_exclude_intercepted=actor_exclude_intercepted,
    )
    if actor_eligible not in ACTOR_ELIGIBLE_MODES:
        raise ValueError(
            f"未知 actor_eligible={actor_eligible!r}，支持 {ACTOR_ELIGIBLE_MODES}"
        )

    cfg = reward_cfg or DEFAULT_REWARD_CFG
    if reward_kwargs:
        from reward_projection import with_reward_overrides

        cfg = with_reward_overrides(cfg, **reward_kwargs)

    if not Path(mat_path).exists():
        raise FileNotFoundError(
            f"专家库文件不存在：{mat_path}\n"
            "请先运行 MATLAB 的 generate_ps_expert_library 生成专家轨迹，"
            "或修改 mat_path 参数指向正确路径。"
        )

    use_phys = expert_lib_has_phys_schema(mat_path)
    if verbose:
        print(f"[load_expert_buffer] 专家库：{mat_path}")
        if use_phys:
            print("[load_expert_buffer] 检测到 phys_transition schema → 直接加载（跳过 rebuild）")
        else:
            print(f"[load_expert_buffer] 旧库 → 对齐数据：{aligned_path}")
        print(f"[load_expert_buffer] reward_version={cfg.version}")
        print(f"[load_expert_buffer] action_mapping(target)={target_amap}")
        print(f"[load_expert_buffer] actor_eligible={actor_eligible}")

    if validate_expert_sa:
        meta = validate_expert_sa_vs_trajectories(mat_path)
        if verbose:
            print(
                f"[load_expert_buffer] expert_sa 校验通过（QA）："
                f"{meta['n_traj']} 轨 / {meta['n_steps']} 步"
            )

    if use_phys:
        data = load_phys_transitions_from_expert_lib(mat_path)
    else:
        data = load_aligned_transitions(aligned_path)
        if data["expert_path"]:
            try:
                same = Path(data["expert_path"]).resolve() == Path(mat_path).resolve()
            except OSError:
                same = data["expert_path"] == mat_path
            if not same:
                raise ValueError(
                    f"对齐数据 expert_path 与当前专家库不匹配：\n"
                    f"  aligned: {data['expert_path']}\n"
                    f"  expert : {mat_path}"
                )

    src_map = str(data.get("action_mapping_version", "td3_box_log_v1"))
    data = apply_action_mapping(data, target_amap, source_version=src_map)
    if verbose and src_map != target_amap:
        print(f"[load_expert_buffer] 动作重标：{src_map} → {target_amap}")

    states_phys = data["states"]
    next_phys = data["next_states"]
    actions_all = data["actions"].astype(np.float32)
    dones_all = data["dones"].astype(bool)
    n_steps = data["n_steps"]
    N_total = len(states_phys)

    cache = None
    if require_terminal_cache or Path(terminal_cache).exists():
        cache = _load_terminal_cache(
            terminal_cache, mat_path, require_path_match=True, verbose=verbose
        )
    elif verbose:
        print("[load_expert_buffer] 未加载终端缓存，BC 末步仅含过程奖励。")

    # 由物理事实投影（权威）；缓存 reward 列仅作对照
    proj = project_transition_rewards(data, terminal_cache=cache, cfg=cfg)
    rewards_all = proj["rewards"]
    data["rewards"] = rewards_all  # 供一致性检查

    if verbose and cache is not None:
        R1_c = proj["R1"]
        bonus_c = proj["bonus"]
        n_term = int(np.sum(~data["intercepted"].astype(bool)))
        print(
            f"[load_expert_buffer] 投影终端：未拦截 {n_term} 条  "
            f"R1 med={np.median(R1_c):.2f}, bonus>0={(bonus_c > 0).mean()*100:.1f}%  "
            f"version={proj['reward_version']}"
        )
        cached = proj.get("reward_cached")
        if cached is not None and len(cached) == len(rewards_all):
            # 旧 aligned 缓存过程奖可能与重投影略有差；仅统计
            diff = np.max(np.abs(cached - rewards_all))
            if diff > 1e-3 and verbose:
                print(
                    f"[load_expert_buffer] 提示: reward_cached 与重投影 max|diff|={diff:.3e} "
                    f"（以投影为准）"
                )

    warns = check_aligned_consistency(data, cache=cache, expert_path=mat_path)
    for w in warns:
        print(f"[load_expert_buffer] 警告: {w}")

    states_n = normalize_obs(states_phys)
    next_n = normalize_obs(next_phys)
    if verbose:
        print(
            f"[load_expert_buffer] 观测归一化 |x|>1 比例："
            f"s={overflow_ratio(states_n)*100:.2f}%  "
            f"s'={overflow_ratio(next_n)*100:.2f}%"
        )

    budget = N_total if max_steps is None else min(int(max_steps), N_total)
    sample_summary: Optional[dict] = None
    train_traj_ids: Optional[np.ndarray] = None
    val_traj_ids: Optional[np.ndarray] = None
    replay_idx: Optional[np.ndarray] = None
    actor_idx: Optional[np.ndarray] = None

    if sample_mode == "prefix":
        replay_idx = np.arange(budget, dtype=np.int64)
        actor_idx = replay_idx.copy()
        traj_ids_flat = np.asarray(data["traj_ids"], dtype=np.int32).ravel()
        if actor_eligible == "both_ok":
            if cache is None or "r_minT" not in cache:
                raise ValueError("prefix + actor_eligible=both_ok 需要 terminal_cache.r_minT")
            intercepted = np.asarray(data["intercepted"], dtype=bool).ravel()
            r_min1 = np.asarray(data["r_min1"], dtype=np.float64).ravel()
            r_min2 = np.asarray(data["r_min2"], dtype=np.float64).ravel()
            r_minT = np.asarray(cache["r_minT"], dtype=np.float64).ravel()
            tids = traj_ids_flat[replay_idx]
            both_ok = (
                (~intercepted[tids])
                & (r_min1[tids] >= float(cfg.kill_radius))
                & (r_min2[tids] >= float(cfg.kill_radius))
                & np.isfinite(r_minT[tids])
                & (r_minT[tids] <= float(cfg.strike_ok))
            )
            actor_idx = replay_idx[both_ok]
        elif actor_eligible == "non_intercepted" or actor_exclude_intercepted:
            intercepted = np.asarray(data["intercepted"], dtype=bool).ravel()
            keep = ~intercepted[traj_ids_flat[replay_idx]]
            actor_idx = replay_idx[keep]
        sample_summary = {
            "sample_mode": "prefix",
            "actor_eligible": actor_eligible,
            "n_replay_steps": int(len(replay_idx)),
            "n_actor_steps": int(len(actor_idx)),
            "seed": int(sample_seed),
        }
        if verbose:
            print(
                f"[load_expert_buffer] 采样模式=prefix：Replay {len(replay_idx)}，"
                f"Actor {len(actor_idx)}（eligible={actor_eligible}）"
            )
    elif sample_mode == "balanced":
        ok_modes = None
        if load_ok_modes:
            if verbose:
                print("[load_expert_buffer] 加载 ok_mode（ok/debug）元数据…")
            ok_modes = load_ok_modes_from_expert(mat_path, data["source_idx"])
        plan = build_expert_sample_plan(
            data,
            budget,
            seed=sample_seed,
            val_ratio=val_ratio,
            n_tgo_bins=n_tgo_bins,
            force_terminal=True,
            actor_exclude_intercepted=actor_exclude_intercepted,
            actor_eligible=actor_eligible,
            replay_include_intercepted=replay_include_intercepted,
            terminal_cache=cache,
            ok_modes=ok_modes,
            kill_radius=float(cfg.kill_radius),
            strike_ok=float(cfg.strike_ok),
        )
        replay_idx = plan.step_indices
        actor_idx = plan.actor_step_indices
        train_traj_ids = plan.train_traj_ids
        val_traj_ids = plan.val_traj_ids
        sample_summary = dict(plan.summary)
        sample_summary["sample_mode"] = "balanced"
        if sample_plan_out:
            plan.save(sample_plan_out)
            if verbose:
                print(f"[load_expert_buffer] 采样计划已保存：{sample_plan_out}")
        if verbose:
            s = sample_summary
            print(
                f"[load_expert_buffer] 采样模式=balanced seed={sample_seed}："
                f"Replay {s['n_replay_steps']} / Actor {s['n_actor_steps']}；"
                f"train轨 {s['n_train_traj']} val轨 {s['n_val_traj']}；"
                f"均步/已用轨={s['steps_per_used_traj_mean']:.1f} "
                f"max={s['steps_per_used_traj_max']}；"
                f"actor_eligible={s.get('actor_eligible')} "
                f"stage3_fail_frac={s.get('actor_stage3_fail_step_frac')}"
            )
    else:
        raise ValueError(f"未知 sample_mode={sample_mode!r}（支持 balanced|prefix）")

    assert replay_idx is not None and actor_idx is not None
    written = 0
    for i in replay_idx:
        ii = int(i)
        replay_buffer.add(
            obs=states_n[ii],
            next_obs=next_n[ii],
            action=actions_all[ii],
            reward=np.array([rewards_all[ii]]),
            done=np.array([bool(dones_all[ii])]),
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
        states=states_n[actor_idx].copy(),
        actions=actions_all[actor_idx].copy(),
        sample_summary=sample_summary,
        train_traj_ids=train_traj_ids,
        val_traj_ids=val_traj_ids,
        replay_step_indices=replay_idx,
        actor_step_indices=actor_idx,
    )


if __name__ == "__main__":
    import sys

    mat = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXPERT_PATH
    print(f"校验 expert_sa ↔ trajectories：{mat}")
    meta = validate_expert_sa_vs_trajectories(mat)
    print(f"  OK  n_traj={meta['n_traj']}  n_steps={meta['n_steps']}")

    aligned = Path(DEFAULT_ALIGNED_PATH)
    if aligned.exists():
        data = load_aligned_transitions(str(aligned))
        warns = check_aligned_consistency(data, expert_path=mat)
        print(
            f"对齐数据：{aligned}  traj={len(data['n_steps'])}  "
            f"steps={len(data['states'])}  "
            f"intercept={(data['intercepted']>0).mean()*100:.1f}%"
        )
        for w in warns:
            print(f"  警告: {w}")
    else:
        print(f"对齐数据缺失：{aligned}")
