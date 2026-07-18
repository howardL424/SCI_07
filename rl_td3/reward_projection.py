"""
reward_projection.py — 由已存物理事实廉价投影奖励（可版本化）。

权威物理量：next_state 距离、intercepted、r_minT / r_min1 / r_min2。
缓存中的 reward / R1 / bonus 均为派生，换公式时只重投影，不重放 stage2/stage3。

失效边界见计划 §1.5：仅改阈值/系数 → 只跑本模块；改动力学才重建 transition。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """与 env_step.m / stage2_process_reward.m / evasion_env 终端语义对齐。"""

    proc_coef: float = 0.05
    dstar: float = 20.0
    kill_radius: float = 6.0
    intercept_pen: float = 50.0  # 使用时取负
    r1_num: float = 80.0
    r1_den_offset: float = 5.0
    r1_ceil: float = 25.0
    strike_ok: float = 10.0
    bonus_both: float = 100.0
    version: str = "r1_bonus_v1"


DEFAULT_REWARD_CFG = RewardConfig()
REWARD_VERSION = DEFAULT_REWARD_CFG.version


def soft_sat_log_evade(
    r_AD: np.ndarray | float,
    *,
    dstar: float = 20.0,
    kill_radius: float = 6.0,
) -> np.ndarray | float:
    """与 MATLAB soft_sat_log_evade 一致：r≤kill→0；否则 log10(min(r,d*)-5)。"""
    scalar = np.isscalar(r_AD) or (isinstance(r_AD, np.ndarray) and r_AD.ndim == 0)
    r = np.asarray(r_AD, dtype=np.float64)
    out = np.zeros_like(r, dtype=np.float64)
    ok = r > kill_radius
    r_eff = np.minimum(r[ok], dstar)
    out[ok] = np.log10(np.maximum(r_eff - 5.0, 1e-6))
    if scalar:
        return float(out.ravel()[0])
    return out


def infer_step_intercepted(
    next_states: np.ndarray,
    *,
    kill_radius: float = 6.0,
) -> np.ndarray:
    """由 next_state 距离事实推断逐步拦截（与 stage2_step 判据一致）。"""
    ns = np.asarray(next_states, dtype=np.float64)
    return (np.minimum(ns[:, 0], ns[:, 4]) <= kill_radius)


def project_process_rewards(
    next_states: np.ndarray,
    step_intercepted: Optional[np.ndarray] = None,
    cfg: RewardConfig = DEFAULT_REWARD_CFG,
) -> np.ndarray:
    """
    逐步过程奖励（含拦截 -intercept_pen）。
    不依赖已存 reward 列；权威为 next_state 距离 + intercepted。
    """
    ns = np.asarray(next_states, dtype=np.float64)
    if step_intercepted is None:
        step_int = infer_step_intercepted(ns, kill_radius=cfg.kill_radius)
    else:
        step_int = np.asarray(step_intercepted, dtype=bool).ravel()
        if step_int.size != len(ns):
            raise ValueError("step_intercepted 长度与 next_states 不一致")

    R2 = soft_sat_log_evade(ns[:, 0], dstar=cfg.dstar, kill_radius=cfg.kill_radius)
    R3 = soft_sat_log_evade(ns[:, 4], dstar=cfg.dstar, kill_radius=cfg.kill_radius)
    rew = (cfg.proc_coef * (R2 + R3)).astype(np.float32)
    rew[step_int] = np.float32(-cfg.intercept_pen)
    return rew


def terminal_R1(
    r_minT: float,
    cfg: RewardConfig = DEFAULT_REWARD_CFG,
) -> float:
    """R1 = r1_num/(r1_den_offset+r_minT) if r_minT≤r1_ceil else 0。"""
    if not np.isfinite(r_minT):
        return 0.0
    if r_minT <= cfg.r1_ceil:
        return float(cfg.r1_num / (cfg.r1_den_offset + r_minT))
    return 0.0


def project_terminal(
    r_minT: float,
    r_min1: float,
    r_min2: float,
    *,
    intercepted: bool = False,
    cfg: RewardConfig = DEFAULT_REWARD_CFG,
) -> tuple[float, float, int]:
    """
    由 stage3/stage2 事实投影终端增量。
    返回 (R1, bonus, both_ok)。拦截轨迹强制全 0。
    """
    if intercepted:
        return 0.0, 0.0, 0
    if not np.isfinite(r_minT):
        return 0.0, 0.0, 0
    R1 = terminal_R1(r_minT, cfg=cfg)
    evade_ok = (r_min1 >= cfg.kill_radius) and (r_min2 >= cfg.kill_radius)
    strike_ok = r_minT <= cfg.strike_ok
    both_ok = 1 if (evade_ok and strike_ok) else 0
    bonus = float(cfg.bonus_both) if both_ok else 0.0
    return R1, bonus, both_ok


def project_terminal_batch(
    r_minT: np.ndarray,
    r_min1: np.ndarray,
    r_min2: np.ndarray,
    intercepted: np.ndarray,
    cfg: RewardConfig = DEFAULT_REWARD_CFG,
) -> dict[str, np.ndarray]:
    """批量投影终端派生量（可代替 sidecar 中的 R1/bonus 列做重算）。"""
    n = len(r_minT)
    R1 = np.zeros(n, dtype=np.float64)
    bonus = np.zeros(n, dtype=np.float64)
    both_ok = np.zeros(n, dtype=np.int8)
    for i in range(n):
        r1, b, ok = project_terminal(
            float(r_minT[i]),
            float(r_min1[i]),
            float(r_min2[i]),
            intercepted=bool(intercepted[i]),
            cfg=cfg,
        )
        R1[i], bonus[i], both_ok[i] = r1, b, ok
    return {"R1": R1, "bonus": bonus, "both_ok": both_ok, "reward_version": cfg.version}


def append_terminal_to_rewards(
    rewards: np.ndarray,
    n_steps: np.ndarray,
    intercepted: np.ndarray,
    R1: np.ndarray,
    bonus: np.ndarray,
) -> np.ndarray:
    """仅在未拦截轨迹末步追加 R1+bonus。"""
    out = np.asarray(rewards, dtype=np.float32).copy()
    cursor = 0
    for tid, n_k in enumerate(np.asarray(n_steps, dtype=np.int32).ravel()):
        n_k = int(n_k)
        end = cursor + n_k - 1
        if not bool(intercepted[tid]):
            out[end] = out[end] + np.float32(R1[tid]) + np.float32(bonus[tid])
        cursor += n_k
    return out


def project_transition_rewards(
    data: Mapping[str, Any],
    terminal_cache: Optional[Mapping[str, Any]] = None,
    cfg: RewardConfig = DEFAULT_REWARD_CFG,
    *,
    use_cache_R1_bonus: bool = False,
) -> dict[str, Any]:
    """
    对扁平 transition 表投影完整逐步奖励。

    - 过程项：始终由 next_states (+ 可选逐步 intercepted) 重算。
    - 终端项：默认由 cache 的 r_minT/r_min1/r_min2 重投影；
      use_cache_R1_bonus=True 时直接用缓存 R1/bonus（旧行为）。
    - data['rewards'] 若存在，保留为 reward_cached（非权威）。

    返回 dict: rewards, reward_cached, reward_version, R1, bonus, both_ok
    """
    next_states = np.asarray(data["next_states"], dtype=np.float64)
    n_steps = np.asarray(data["n_steps"], dtype=np.int32).ravel()
    traj_int = np.asarray(data["intercepted"], dtype=bool).ravel()

    step_int = None
    if "step_intercepted" in data and data["step_intercepted"] is not None:
        step_int = np.asarray(data["step_intercepted"], dtype=bool).ravel()
    else:
        step_int = infer_step_intercepted(next_states, kill_radius=cfg.kill_radius)

    process = project_process_rewards(next_states, step_int, cfg=cfg)

    reward_cached = None
    if data.get("reward_cached") is not None:
        reward_cached = np.asarray(data["reward_cached"], dtype=np.float32).ravel()
    elif data.get("source_mode") == "aligned" and data.get("rewards") is not None:
        # 仅旧 aligned 派生表把 rewards 列当作 cached；新库占位 zeros 不算
        reward_cached = np.asarray(data["rewards"], dtype=np.float32).ravel()

    R1 = np.zeros(len(n_steps), dtype=np.float64)
    bonus = np.zeros(len(n_steps), dtype=np.float64)
    both_ok = np.zeros(len(n_steps), dtype=np.int8)

    if terminal_cache is not None:
        if use_cache_R1_bonus and "R1" in terminal_cache and "bonus" in terminal_cache:
            R1 = np.asarray(terminal_cache["R1"], dtype=np.float64).ravel()
            bonus = np.asarray(terminal_cache["bonus"], dtype=np.float64).ravel()
            if "both_ok" in terminal_cache:
                both_ok = np.asarray(terminal_cache["both_ok"], dtype=np.int8).ravel()
            else:
                both_ok = ((R1 > 0) & (bonus > 0)).astype(np.int8)
        else:
            # 权威：r_minT + 距离事实 → 按当前 cfg 投影
            r_minT = np.asarray(terminal_cache["r_minT"], dtype=np.float64).ravel()
            r_min1 = np.asarray(
                terminal_cache.get("r_min1", data["r_min1"]), dtype=np.float64
            ).ravel()
            r_min2 = np.asarray(
                terminal_cache.get("r_min2", data["r_min2"]), dtype=np.float64
            ).ravel()
            term = project_terminal_batch(r_minT, r_min1, r_min2, traj_int, cfg=cfg)
            R1, bonus, both_ok = term["R1"], term["bonus"], term["both_ok"]

        # 硬约束：拦截不得有正终端
        bad = traj_int & ((R1 > 0) | (bonus > 0))
        if np.any(bad):
            raise ValueError(f"拦截轨迹仍含正终端奖励：{int(np.sum(bad))} 条")

        rewards = append_terminal_to_rewards(process, n_steps, traj_int, R1, bonus)
    else:
        rewards = process

    return {
        "rewards": rewards.astype(np.float32),
        "reward_cached": reward_cached,
        "reward_version": cfg.version,
        "R1": R1,
        "bonus": bonus,
        "both_ok": both_ok,
    }


def with_reward_overrides(
    cfg: RewardConfig = DEFAULT_REWARD_CFG,
    **kwargs,
) -> RewardConfig:
    """返回覆盖部分字段后的新配置（换奖励实验用，不碰物理数据）。"""
    return replace(cfg, **kwargs)
