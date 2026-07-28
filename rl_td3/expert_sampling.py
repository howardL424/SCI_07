"""
expert_sampling.py — 按轨迹均衡、可复现的专家步采样。

目标（对照预算约 99k）：
  - 禁止扁平表前缀截取 / 禁止长轨迹支配样本
  - 按轨迹均分配额；轨迹内按 tgo（步序）分层；强制含末步
  - train/val 按轨迹 ID 划分，避免相邻步泄漏
  - Replay 与 Actor BC 共用同一采样映射；Actor 默认仅 both_ok（stage3 质量门）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


ACTOR_ELIGIBLE_MODES = ("both_ok", "non_intercepted", "all")


def resolve_actor_eligible(
    actor_eligible: Optional[str] = None,
    *,
    actor_exclude_intercepted: bool = True,
) -> str:
    """
    解析 Actor 标签资格。
    优先使用 actor_eligible；若为 None，则按旧开关映射：
      exclude_intercepted=True  → non_intercepted
      exclude_intercepted=False → all
    新默认由调用方显式传 both_ok。
    """
    if actor_eligible is not None:
        mode = str(actor_eligible).strip().lower()
        if mode not in ACTOR_ELIGIBLE_MODES:
            raise ValueError(
                f"未知 actor_eligible={actor_eligible!r}，支持 {ACTOR_ELIGIBLE_MODES}"
            )
        return mode
    return "non_intercepted" if actor_exclude_intercepted else "all"


@dataclass
class TrajQuality:
    """单条轨迹质量元数据（诊断用；不先验硬删 debug）。"""

    traj_id: int
    source_idx: int
    n_steps: int
    t_end: float
    intercepted: bool
    ok_mode: str  # 'ok' | 'debug' | 'unknown'
    evade_ok: bool
    strike_ok: bool
    both_ok: bool
    r_min1: float
    r_min2: float
    r_minT: float


def actor_eligible_mask(
    qualities: list[TrajQuality],
    mode: str,
) -> np.ndarray:
    """轨迹级 Actor 资格掩码（len = n_traj）。"""
    mode = resolve_actor_eligible(mode)
    n = len(qualities)
    if mode == "all":
        return np.ones(n, dtype=bool)
    if mode == "non_intercepted":
        return np.asarray([not q.intercepted for q in qualities], dtype=bool)
    # both_ok
    return np.asarray([bool(q.both_ok) for q in qualities], dtype=bool)


@dataclass
class ExpertSamplePlan:
    """一次可复现采样的索引与划分结果。"""

    step_indices: np.ndarray  # 扁平 transition 全局下标（Replay 用）
    actor_step_indices: np.ndarray  # Actor BC 训练用（默认可去掉拦截轨）
    val_actor_step_indices: np.ndarray  # Actor BC 验证用（val 轨全步，默认可去拦截）
    train_traj_ids: np.ndarray
    val_traj_ids: np.ndarray
    traj_offsets: np.ndarray  # 每条轨迹在扁平表中的起点
    n_steps: np.ndarray
    qualities: list[TrajQuality] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        q = self.qualities
        np.savez_compressed(
            path,
            step_indices=self.step_indices.astype(np.int64),
            actor_step_indices=self.actor_step_indices.astype(np.int64),
            val_actor_step_indices=self.val_actor_step_indices.astype(np.int64),
            train_traj_ids=self.train_traj_ids.astype(np.int32),
            val_traj_ids=self.val_traj_ids.astype(np.int32),
            traj_offsets=self.traj_offsets.astype(np.int64),
            n_steps=self.n_steps.astype(np.int32),
            traj_id=np.asarray([x.traj_id for x in q], dtype=np.int32),
            source_idx=np.asarray([x.source_idx for x in q], dtype=np.int32),
            intercepted=np.asarray([int(x.intercepted) for x in q], dtype=np.int8),
            ok_mode=np.asarray([x.ok_mode for x in q], dtype=object),
            evade_ok=np.asarray([int(x.evade_ok) for x in q], dtype=np.int8),
            strike_ok=np.asarray([int(x.strike_ok) for x in q], dtype=np.int8),
            both_ok=np.asarray([int(x.both_ok) for x in q], dtype=np.int8),
            r_min1=np.asarray([x.r_min1 for x in q], dtype=np.float64),
            r_min2=np.asarray([x.r_min2 for x in q], dtype=np.float64),
            r_minT=np.asarray([x.r_minT for x in q], dtype=np.float64),
            t_end=np.asarray([x.t_end for x in q], dtype=np.float64),
            summary_json=np.asarray([_summary_to_json(self.summary)], dtype=object),
        )


def _summary_to_json(summary: dict) -> str:
    import json

    def _conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    return json.dumps({k: _conv(v) for k, v in summary.items()}, ensure_ascii=False, indent=2)


def traj_offsets_from_n_steps(n_steps: np.ndarray) -> np.ndarray:
    n_steps = np.asarray(n_steps, dtype=np.int64).ravel()
    off = np.zeros(len(n_steps) + 1, dtype=np.int64)
    np.cumsum(n_steps, out=off[1:])
    return off


def collect_steps_for_trajs(
    traj_ids: np.ndarray,
    offsets: np.ndarray,
    *,
    intercepted: Optional[np.ndarray] = None,
    exclude_intercepted: bool = False,
    eligible: Optional[np.ndarray] = None,
) -> np.ndarray:
    """收集给定轨迹上的全部全局步下标（可选 exclude 拦截，或按 eligible 掩码）。"""
    traj_ids = np.asarray(traj_ids, dtype=np.int32).ravel()
    chunks: list[np.ndarray] = []
    for tid in traj_ids:
        tid_i = int(tid)
        if eligible is not None:
            if not bool(eligible[tid_i]):
                continue
        elif exclude_intercepted and intercepted is not None and bool(intercepted[tid_i]):
            continue
        start = int(offsets[tid_i])
        end = int(offsets[tid_i + 1])
        if end > start:
            chunks.append(np.arange(start, end, dtype=np.int64))
    if not chunks:
        return np.array([], dtype=np.int64)
    return np.concatenate(chunks)


def split_traj_ids(
    n_traj: int,
    *,
    val_ratio: float = 0.1,
    seed: int = 0,
    stratify: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    按轨迹划分 train/val。stratify 为每轨类别（如 intercepted），尽量分层。
    """
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio 须在 [0,1): {val_ratio}")
    rng = np.random.default_rng(seed)
    ids = np.arange(n_traj, dtype=np.int32)
    if n_traj == 0:
        return ids, ids.copy()
    if val_ratio == 0.0 or n_traj < 2:
        return ids, np.array([], dtype=np.int32)

    n_val = max(1, int(round(n_traj * val_ratio)))
    n_val = min(n_val, n_traj - 1)

    if stratify is None:
        perm = rng.permutation(ids)
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])

    stratify = np.asarray(stratify).ravel()
    if len(stratify) != n_traj:
        raise ValueError("stratify 长度须等于轨迹数")
    train_l: list[int] = []
    val_l: list[int] = []
    for label in np.unique(stratify):
        group = ids[stratify == label]
        g_val = max(1, int(round(len(group) * val_ratio))) if len(group) >= 2 else 0
        g_val = min(g_val, len(group) - 1) if len(group) >= 2 else 0
        perm = rng.permutation(group)
        val_l.extend(perm[:g_val].tolist())
        train_l.extend(perm[g_val:].tolist())
    # 若分层后 val 数量偏差过大，从 train 补齐/削减
    val_ids = np.asarray(val_l, dtype=np.int32)
    train_ids = np.asarray(train_l, dtype=np.int32)
    if len(val_ids) < n_val and len(train_ids) > 1:
        need = n_val - len(val_ids)
        take = rng.choice(train_ids, size=min(need, len(train_ids) - 1), replace=False)
        val_ids = np.concatenate([val_ids, take])
        train_ids = np.setdiff1d(train_ids, take, assume_unique=False)
    elif len(val_ids) > n_val:
        drop = rng.choice(val_ids, size=len(val_ids) - n_val, replace=False)
        val_ids = np.setdiff1d(val_ids, drop, assume_unique=False)
        train_ids = np.concatenate([train_ids, drop])
    return np.sort(train_ids), np.sort(val_ids)


def _sample_within_traj(
    n_k: int,
    quota: int,
    rng: np.random.Generator,
    *,
    n_tgo_bins: int = 4,
    force_terminal: bool = True,
) -> np.ndarray:
    """在单条轨迹内采样局部步下标 [0, n_k)。"""
    if n_k <= 0 or quota <= 0:
        return np.array([], dtype=np.int64)
    take = min(int(quota), n_k)
    if take >= n_k:
        return np.arange(n_k, dtype=np.int64)

    chosen: set[int] = set()
    if force_terminal:
        chosen.add(n_k - 1)
        take_rest = take - 1
    else:
        take_rest = take

    if take_rest <= 0:
        return np.asarray(sorted(chosen), dtype=np.int64)

    # 候选：除已选末步外
    candidates = np.arange(0, n_k - 1 if force_terminal else n_k, dtype=np.int64)
    if candidates.size == 0:
        return np.asarray(sorted(chosen), dtype=np.int64)

    n_bins = max(1, min(int(n_tgo_bins), candidates.size))
    # 按步序近似 tgo 分层（早→晚）
    edges = np.linspace(0, candidates.size, n_bins + 1, dtype=np.int64)
    # 配额在各层均分，余数随机
    base = take_rest // n_bins
    rem = take_rest % n_bins
    bin_quota = np.full(n_bins, base, dtype=np.int64)
    if rem > 0:
        bin_quota[rng.choice(n_bins, size=rem, replace=False)] += 1

    picked: list[int] = []
    leftover = 0
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        pool = candidates[lo:hi]
        q = int(bin_quota[b]) + leftover
        if pool.size == 0:
            leftover = q
            continue
        if q >= pool.size:
            picked.extend(pool.tolist())
            leftover = q - pool.size
        elif q > 0:
            picked.extend(rng.choice(pool, size=q, replace=False).tolist())
            leftover = 0
        else:
            leftover = 0
    # 仍不足：从剩余候选补齐
    if len(picked) < take_rest:
        remain = np.setdiff1d(candidates, np.asarray(picked, dtype=np.int64), assume_unique=False)
        need = take_rest - len(picked)
        if remain.size > 0 and need > 0:
            picked.extend(
                rng.choice(remain, size=min(need, remain.size), replace=False).tolist()
            )
    chosen.update(picked[:take_rest])
    # 若仍不足（极端），再放宽
    while len(chosen) < take:
        remain = [i for i in range(n_k) if i not in chosen]
        if not remain:
            break
        chosen.add(int(rng.choice(remain)))
    return np.asarray(sorted(chosen), dtype=np.int64)


def allocate_traj_quotas(
    n_steps: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    traj_ids: Optional[np.ndarray] = None,
) -> dict[int, int]:
    """
    对给定轨迹集合硬均分预算：每轨目标 ≈ budget/n，再与长度取 min。
    短轨取不满时**不把余量无限灌给长轨**（避免长轨支配）；实际步数可能略低于 budget。
    返回 {traj_id: quota}。
    """
    n_steps = np.asarray(n_steps, dtype=np.int64).ravel()
    if traj_ids is None:
        traj_ids = np.arange(len(n_steps), dtype=np.int32)
    else:
        traj_ids = np.asarray(traj_ids, dtype=np.int32).ravel()

    if len(traj_ids) == 0 or budget <= 0:
        return {}

    active = [int(tid) for tid in traj_ids if int(n_steps[tid]) > 0]
    if not active:
        return {int(tid): 0 for tid in traj_ids}

    n_a = len(active)
    base = int(budget) // n_a
    rem = int(budget) % n_a
    bonus_set: set[int] = set()
    if rem > 0:
        bonus_set = set(int(x) for x in rng.choice(active, size=rem, replace=False))

    quotas = {int(tid): 0 for tid in traj_ids}
    for tid in active:
        target = base + (1 if tid in bonus_set else 0)
        quotas[tid] = min(int(n_steps[tid]), target)
    return quotas


def sample_balanced_step_indices(
    n_steps: np.ndarray,
    budget: int,
    *,
    seed: int = 0,
    traj_ids: Optional[np.ndarray] = None,
    n_tgo_bins: int = 4,
    force_terminal: bool = True,
) -> np.ndarray:
    """
    按轨迹均衡采样，返回扁平全局步下标（升序）。
    """
    n_steps = np.asarray(n_steps, dtype=np.int64).ravel()
    offsets = traj_offsets_from_n_steps(n_steps)
    rng = np.random.default_rng(seed)

    if traj_ids is None:
        traj_ids = np.arange(len(n_steps), dtype=np.int32)
    else:
        traj_ids = np.asarray(traj_ids, dtype=np.int32).ravel()

    total_avail = int(sum(int(n_steps[t]) for t in traj_ids))
    budget = int(min(budget, total_avail))
    if budget <= 0:
        return np.array([], dtype=np.int64)

    quotas = allocate_traj_quotas(n_steps, budget, rng, traj_ids=traj_ids)
    indices: list[np.ndarray] = []
    for tid, q in quotas.items():
        if q <= 0:
            continue
        local = _sample_within_traj(
            int(n_steps[tid]),
            q,
            rng,
            n_tgo_bins=n_tgo_bins,
            force_terminal=force_terminal,
        )
        indices.append(offsets[tid] + local)

    if not indices:
        return np.array([], dtype=np.int64)
    out = np.concatenate(indices)
    out.sort()
    # 去重（理论上不应重复）
    out = np.unique(out)
    return out.astype(np.int64)


def build_traj_qualities(
    data: dict,
    terminal_cache: Optional[dict] = None,
    *,
    kill_radius: float = 6.0,
    strike_ok: float = 10.0,
    ok_modes: Optional[np.ndarray] = None,
) -> list[TrajQuality]:
    """从对齐数据 + sidecar 的 r_minT 构建轨迹质量表；both_ok 由当前阈值重算。"""
    n_steps = np.asarray(data["n_steps"], dtype=np.int32).ravel()
    n_traj = len(n_steps)
    intercepted = np.asarray(data["intercepted"], dtype=bool).ravel()
    t_end = np.asarray(data.get("t_end", np.zeros(n_traj)), dtype=np.float64).ravel()
    source_idx = np.asarray(
        data.get("source_idx", np.arange(1, n_traj + 1)), dtype=np.int32
    ).ravel()
    r_min1 = np.asarray(data["r_min1"], dtype=np.float64).ravel()
    r_min2 = np.asarray(data["r_min2"], dtype=np.float64).ravel()

    if terminal_cache is not None and "r_minT" in terminal_cache:
        r_minT = np.asarray(terminal_cache["r_minT"], dtype=np.float64).ravel()
    else:
        r_minT = np.full(n_traj, np.nan)

    if ok_modes is None:
        modes = np.asarray(["unknown"] * n_traj, dtype=object)
    else:
        modes = np.asarray(ok_modes, dtype=object).ravel()
        if len(modes) != n_traj:
            raise ValueError("ok_modes 长度与轨迹数不一致")

    out: list[TrajQuality] = []
    for i in range(n_traj):
        # both_ok 始终由原始事实 + 当前阈值重算，不信任 sidecar 缓存的 both_ok
        evade = (float(r_min1[i]) >= kill_radius) and (float(r_min2[i]) >= kill_radius)
        strike = bool(np.isfinite(r_minT[i]) and float(r_minT[i]) <= strike_ok)
        both_i = (not bool(intercepted[i])) and evade and strike
        out.append(
            TrajQuality(
                traj_id=i,
                source_idx=int(source_idx[i]) if i < len(source_idx) else i + 1,
                n_steps=int(n_steps[i]),
                t_end=float(t_end[i]) if i < len(t_end) else float(n_steps[i]) * 0.1,
                intercepted=bool(intercepted[i]),
                ok_mode=str(modes[i]),
                evade_ok=bool(evade) and not bool(intercepted[i]),
                strike_ok=bool(strike) and not bool(intercepted[i]),
                both_ok=bool(both_i),
                r_min1=float(r_min1[i]),
                r_min2=float(r_min2[i]),
                r_minT=float(r_minT[i]) if np.isfinite(r_minT[i]) else float("nan"),
            )
        )
    return out


def load_ok_modes_from_expert(
    expert_path: str,
    source_idx: np.ndarray,
) -> np.ndarray:
    """
    从 expert_lib 读取 stageB.ok_mode（ok/debug）。
    按 aligned 的 source_idx（1-based）对齐到轨迹顺序。
    """
    from load_expert_buffer import _field, _iter_trajectories, _load_results

    results = _load_results(expert_path)
    trajs = _iter_trajectories(results)
    modes_by_src: dict[int, str] = {}
    for i, traj in enumerate(trajs):
        src = i + 1
        stage_b = _field(traj, "stageB", None)
        mode = _field(stage_b, "ok_mode", None) if stage_b is not None else None
        if mode is None:
            # 兼容：仅有 ok 布尔
            ok_flag = _field(stage_b, "ok", None) if stage_b is not None else None
            if ok_flag is True:
                mode = "ok"
            elif bool(_field(traj, "ok_use", False)):
                mode = "debug"
            else:
                mode = "fail"
        modes_by_src[src] = str(mode)

    source_idx = np.asarray(source_idx, dtype=np.int32).ravel()
    out = []
    for s in source_idx:
        out.append(modes_by_src.get(int(s), "unknown"))
    return np.asarray(out, dtype=object)


def build_expert_sample_plan(
    data: dict,
    budget: int,
    *,
    seed: int = 0,
    val_ratio: float = 0.1,
    n_tgo_bins: int = 4,
    force_terminal: bool = True,
    actor_exclude_intercepted: bool = True,
    actor_eligible: str = "both_ok",
    replay_include_intercepted: bool = True,
    terminal_cache: Optional[dict] = None,
    ok_modes: Optional[np.ndarray] = None,
    kill_radius: float = 6.0,
    strike_ok: float = 10.0,
) -> ExpertSamplePlan:
    """
    构建可复现专家采样计划。

    - train/val 按轨迹划分（分层：拦截与否）
    - 仅在 train 轨迹上按均衡策略采 `budget` 步写入 Replay
    - Actor BC / TD3 专家 MSE 默认只取 both_ok 轨迹上的已采步（actor_eligible）
    """
    n_steps = np.asarray(data["n_steps"], dtype=np.int32).ravel()
    intercepted = np.asarray(data["intercepted"], dtype=bool).ravel()
    n_traj = len(n_steps)
    offsets = traj_offsets_from_n_steps(n_steps)

    eligible_mode = resolve_actor_eligible(
        actor_eligible,
        actor_exclude_intercepted=actor_exclude_intercepted,
    )

    qualities = build_traj_qualities(
        data,
        terminal_cache,
        kill_radius=kill_radius,
        strike_ok=strike_ok,
        ok_modes=ok_modes,
    )
    eligible = actor_eligible_mask(qualities, eligible_mode)

    train_ids, val_ids = split_traj_ids(
        n_traj,
        val_ratio=val_ratio,
        seed=seed,
        stratify=intercepted.astype(np.int8),
    )

    if replay_include_intercepted:
        replay_pool = train_ids
    else:
        replay_pool = train_ids[~intercepted[train_ids]]

    step_indices = sample_balanced_step_indices(
        n_steps,
        budget,
        seed=seed,
        traj_ids=replay_pool,
        n_tgo_bins=n_tgo_bins,
        force_terminal=force_terminal,
    )

    # Actor 训练：同一批 Replay step 中按 actor_eligible 过滤
    traj_of_step = np.searchsorted(offsets[1:], step_indices, side="right")
    actor_steps = step_indices[eligible[traj_of_step]]

    # Actor 验证：val 轨全部步（按同一 eligible），避免相邻步泄漏到 train
    val_actor_steps = collect_steps_for_trajs(
        val_ids,
        offsets,
        eligible=eligible,
    )

    # 统计
    def _count_modes(ids: np.ndarray) -> dict[str, int]:
        c: dict[str, int] = {}
        for tid in ids:
            m = qualities[int(tid)].ok_mode
            c[m] = c.get(m, 0) + 1
        return c

    traj_of_all = np.searchsorted(offsets[1:], step_indices, side="right")
    per_traj_counts = np.bincount(traj_of_all, minlength=n_traj)

    def _actor_intercept_frac(steps: np.ndarray) -> float:
        if len(steps) == 0:
            return 0.0
        tids = np.searchsorted(offsets[1:], steps, side="right")
        return float(intercepted[tids].mean())

    def _actor_stage3_fail_frac(steps: np.ndarray) -> float:
        """未拦截但非 both_ok（即 stage3 打击失败）步占比。"""
        if len(steps) == 0:
            return 0.0
        tids = np.searchsorted(offsets[1:], steps, side="right")
        both = np.asarray([qualities[int(t)].both_ok for t in tids], dtype=bool)
        inter = intercepted[tids]
        # stage3-fail：安全完成 stage2 但未 both_ok
        fail = (~inter) & (~both)
        return float(fail.mean())

    # val 轨上专家标签成功率（诊断基线；含拦截轨时 both_ok 应为 0）
    val_q = [qualities[int(t)] for t in val_ids]
    val_non_int = [q for q in val_q if not q.intercepted]

    summary = {
        "seed": int(seed),
        "budget_requested": int(budget),
        "n_replay_steps": int(len(step_indices)),
        "n_actor_steps": int(len(actor_steps)),
        "n_val_actor_steps": int(len(val_actor_steps)),
        "budget_filled_frac": float(len(step_indices) / max(budget, 1)),
        "n_traj_total": int(n_traj),
        "n_train_traj": int(len(train_ids)),
        "n_val_traj": int(len(val_ids)),
        "n_replay_traj_used": int(np.sum(per_traj_counts > 0)),
        "actor_eligible": eligible_mode,
        "actor_exclude_intercepted": bool(
            eligible_mode in ("both_ok", "non_intercepted")
        ),
        "replay_include_intercepted": bool(replay_include_intercepted),
        "force_terminal": bool(force_terminal),
        "n_tgo_bins": int(n_tgo_bins),
        "val_ratio": float(val_ratio),
        "intercept_rate_all": float(intercepted.mean()) if n_traj else 0.0,
        "intercept_rate_train": float(intercepted[train_ids].mean()) if len(train_ids) else 0.0,
        "actor_intercept_step_frac": _actor_intercept_frac(actor_steps),
        "actor_stage3_fail_step_frac": _actor_stage3_fail_frac(actor_steps),
        "val_actor_intercept_step_frac": _actor_intercept_frac(val_actor_steps),
        "val_actor_stage3_fail_step_frac": _actor_stage3_fail_frac(val_actor_steps),
        "n_actor_eligible_train": int(np.sum(eligible[train_ids])),
        "steps_per_used_traj_mean": float(
            per_traj_counts[per_traj_counts > 0].mean()
            if np.any(per_traj_counts > 0)
            else 0.0
        ),
        "steps_per_used_traj_max": int(
            per_traj_counts.max() if len(per_traj_counts) else 0
        ),
        "ok_mode_train": _count_modes(train_ids),
        "ok_mode_val": _count_modes(val_ids),
        "both_ok_train": int(sum(1 for t in train_ids if qualities[int(t)].both_ok)),
        "both_ok_val": int(sum(1 for t in val_ids if qualities[int(t)].both_ok)),
        "evade_ok_train": int(sum(1 for t in train_ids if qualities[int(t)].evade_ok)),
        "strike_ok_train": int(sum(1 for t in train_ids if qualities[int(t)].strike_ok)),
        "expert_both_ok_rate_val": (
            float(sum(1 for q in val_q if q.both_ok) / len(val_q)) if val_q else 0.0
        ),
        "expert_both_ok_rate_val_non_intercept": (
            float(sum(1 for q in val_non_int if q.both_ok) / len(val_non_int))
            if val_non_int
            else 0.0
        ),
    }

    return ExpertSamplePlan(
        step_indices=step_indices,
        actor_step_indices=actor_steps,
        val_actor_step_indices=val_actor_steps,
        train_traj_ids=train_ids,
        val_traj_ids=val_ids,
        traj_offsets=offsets,
        n_steps=n_steps,
        qualities=qualities,
        summary=summary,
    )


def plan_to_dict(plan: ExpertSamplePlan) -> dict[str, Any]:
    return {
        "summary": plan.summary,
        "n_replay_steps": len(plan.step_indices),
        "n_actor_steps": len(plan.actor_step_indices),
        "n_val_actor_steps": len(plan.val_actor_step_indices),
        "qualities": [asdict(q) for q in plan.qualities],
    }


def load_sample_plan(path: str | Path) -> ExpertSamplePlan:
    """从 npz 恢复采样计划（兼容旧文件无 val_actor_step_indices）。"""
    import json

    path = Path(path)
    data = np.load(path, allow_pickle=True)
    offsets = np.asarray(data["traj_offsets"], dtype=np.int64)
    n_steps = np.asarray(data["n_steps"], dtype=np.int32)
    val_traj_ids = np.asarray(data["val_traj_ids"], dtype=np.int32)
    intercepted = np.asarray(data["intercepted"], dtype=bool).ravel()

    summary: dict[str, Any] = {}
    if "summary_json" in data.files:
        raw = data["summary_json"]
        text = str(raw.ravel()[0]) if hasattr(raw, "ravel") else str(raw)
        summary = json.loads(text)

    if "val_actor_step_indices" in data.files:
        val_actor = np.asarray(data["val_actor_step_indices"], dtype=np.int64)
    else:
        mode = str(summary.get("actor_eligible", ""))
        if not mode:
            mode = resolve_actor_eligible(
                None,
                actor_exclude_intercepted=bool(
                    summary.get("actor_exclude_intercepted", True)
                ),
            )
        both_ok = np.asarray(data["both_ok"], dtype=bool).ravel()
        eligible = actor_eligible_mask(
            [
                TrajQuality(
                    traj_id=i,
                    source_idx=i + 1,
                    n_steps=int(n_steps[i]),
                    t_end=0.0,
                    intercepted=bool(intercepted[i]),
                    ok_mode="unknown",
                    evade_ok=False,
                    strike_ok=False,
                    both_ok=bool(both_ok[i]),
                    r_min1=0.0,
                    r_min2=0.0,
                    r_minT=float("nan"),
                )
                for i in range(len(n_steps))
            ],
            mode,
        )
        val_actor = collect_steps_for_trajs(val_traj_ids, offsets, eligible=eligible)

    n_q = len(np.asarray(data["traj_id"]).ravel())
    qualities: list[TrajQuality] = []
    for i in range(n_q):
        qualities.append(
            TrajQuality(
                traj_id=int(data["traj_id"][i]),
                source_idx=int(data["source_idx"][i]),
                n_steps=int(n_steps[i]),
                t_end=float(data["t_end"][i]),
                intercepted=bool(intercepted[i]),
                ok_mode=str(data["ok_mode"][i]),
                evade_ok=bool(data["evade_ok"][i]),
                strike_ok=bool(data["strike_ok"][i]),
                both_ok=bool(data["both_ok"][i]),
                r_min1=float(data["r_min1"][i]),
                r_min2=float(data["r_min2"][i]),
                r_minT=float(data["r_minT"][i]),
            )
        )

    return ExpertSamplePlan(
        step_indices=np.asarray(data["step_indices"], dtype=np.int64),
        actor_step_indices=np.asarray(data["actor_step_indices"], dtype=np.int64),
        val_actor_step_indices=val_actor,
        train_traj_ids=np.asarray(data["train_traj_ids"], dtype=np.int32),
        val_traj_ids=val_traj_ids,
        traj_offsets=offsets,
        n_steps=n_steps,
        qualities=qualities,
        summary=summary,
    )


def bc_arrays_from_plan(
    states_norm: np.ndarray,
    actions: np.ndarray,
    plan: ExpertSamplePlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按采样计划切出 Actor BC 的 train/val (s,a)。"""
    tr = plan.actor_step_indices
    va = plan.val_actor_step_indices
    return (
        np.asarray(states_norm[tr], dtype=np.float32),
        np.asarray(actions[tr], dtype=np.float32),
        np.asarray(states_norm[va], dtype=np.float32),
        np.asarray(actions[va], dtype=np.float32),
    )
