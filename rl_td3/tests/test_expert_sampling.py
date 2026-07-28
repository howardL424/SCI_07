"""
test_expert_sampling.py — 轨迹均衡采样单测（合成数据，不依赖 MATLAB）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from expert_sampling import (  # noqa: E402
    allocate_traj_quotas,
    build_expert_sample_plan,
    sample_balanced_step_indices,
    split_traj_ids,
    traj_offsets_from_n_steps,
)


def _synth_data(n_traj: int = 40, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    # 故意制造长短不均：多数短轨 + 少数超长轨
    n_steps = np.concatenate(
        [
            rng.integers(20, 40, size=n_traj - 4),
            np.array([200, 250, 300, 400], dtype=np.int32),
        ]
    ).astype(np.int32)
    rng.shuffle(n_steps)
    intercepted = rng.random(n_traj) < 0.25
    # 未拦截中约 20% 为 stage3 打击失败（r_minT 过大 → both_ok=False）
    non_int = ~intercepted
    stage3_fail = np.zeros(n_traj, dtype=bool)
    fail_cand = np.where(non_int)[0]
    if len(fail_cand) > 0:
        n_fail = max(1, int(round(0.2 * len(fail_cand))))
        stage3_fail[
            rng.choice(fail_cand, size=min(n_fail, len(fail_cand)), replace=False)
        ] = True
    total = int(n_steps.sum())
    states = rng.normal(size=(total, 13))
    traj_ids = np.concatenate(
        [np.full(int(n_steps[i]), i, dtype=np.int32) for i in range(n_traj)]
    )
    r_minT = np.full(n_traj, 5.0, dtype=np.float64)
    r_minT[intercepted] = np.nan
    r_minT[stage3_fail] = 50.0  # > strike_ok
    return {
        "states": states,
        "actions": rng.uniform(-1, 1, size=(total, 3)).astype(np.float32),
        "next_states": states.copy(),
        "dones": np.zeros(total, dtype=bool),  # 测试采样不依赖
        "traj_ids": traj_ids,
        "n_steps": n_steps,
        "intercepted": intercepted.astype(np.int8),
        "t_end": n_steps.astype(np.float64) * 0.1,
        "source_idx": np.arange(1, n_traj + 1, dtype=np.int32),
        "r_min1": np.where(intercepted, 3.0, 15.0),
        "r_min2": np.where(intercepted, 3.0, 15.0),
        "r_minT": r_minT,
    }


def test_split_disjoint():
    train, val = split_traj_ids(100, val_ratio=0.1, seed=0, stratify=np.zeros(100, dtype=int))
    assert len(train) + len(val) == 100
    assert len(np.intersect1d(train, val)) == 0
    train2, val2 = split_traj_ids(100, val_ratio=0.1, seed=0, stratify=np.zeros(100, dtype=int))
    assert np.array_equal(train, train2) and np.array_equal(val, val2)


def test_quota_not_dominated_by_long():
    n_steps = np.array([30] * 20 + [500, 500], dtype=np.int32)
    rng = np.random.default_rng(0)
    quotas = allocate_traj_quotas(n_steps, budget=1000, rng=rng)
    # 硬均分：每轨目标 ≈ 1000/22 ≈ 45，短轨被长度截到 30，长轨不超过目标+1
    long_q = max(quotas[20], quotas[21])
    assert long_q <= 46
    assert all(quotas[i] <= 30 for i in range(20))
    assert sum(quotas.values()) < 1000  # 短轨截断 → 略低于预算（刻意）
    assert sum(quotas.values()) >= 20 * 30  # 至少吃满短轨


def test_force_terminal_and_reproducible():
    n_steps = np.array([50, 80, 120, 40], dtype=np.int32)
    off = traj_offsets_from_n_steps(n_steps)
    idx1 = sample_balanced_step_indices(n_steps, 100, seed=42, n_tgo_bins=4)
    idx2 = sample_balanced_step_indices(n_steps, 100, seed=42, n_tgo_bins=4)
    assert np.array_equal(idx1, idx2)
    # 每条被采到的轨必须含末步
    for tid in range(len(n_steps)):
        local = idx1[(idx1 >= off[tid]) & (idx1 < off[tid + 1])] - off[tid]
        if len(local) == 0:
            continue
        assert (n_steps[tid] - 1) in set(local.tolist())


def test_plan_actor_excludes_intercept():
    data = _synth_data(48, seed=2)
    both_ok = (
        (~data["intercepted"].astype(bool))
        & (data["r_min1"] >= 6)
        & (data["r_minT"] <= 10)
    ).astype(np.int8)
    cache = {
        "r_minT": data["r_minT"],
        "both_ok": both_ok,
    }
    plan = build_expert_sample_plan(
        data,
        budget=800,
        seed=0,
        val_ratio=0.1,
        actor_exclude_intercepted=True,
        actor_eligible="non_intercepted",
        replay_include_intercepted=True,
        terminal_cache=cache,
        ok_modes=np.asarray(["ok"] * len(data["n_steps"]), dtype=object),
    )
    assert len(plan.step_indices) <= 800
    assert len(plan.step_indices) > 0
    assert len(np.intersect1d(plan.train_traj_ids, plan.val_traj_ids)) == 0
    off = plan.traj_offsets
    actor_traj = np.searchsorted(off[1:], plan.actor_step_indices, side="right")
    assert not np.any(data["intercepted"].astype(bool)[actor_traj])
    # Replay 应能包含拦截轨（若 train 中有）
    replay_traj = np.searchsorted(off[1:], plan.step_indices, side="right")
    if np.any(data["intercepted"].astype(bool)[plan.train_traj_ids]):
        assert np.any(data["intercepted"].astype(bool)[replay_traj])
    assert plan.summary["n_actor_steps"] <= plan.summary["n_replay_steps"]
    # val Actor 步来自 val 轨，且与 train 轨无交集
    assert "n_val_actor_steps" in plan.summary
    assert len(plan.val_actor_step_indices) == plan.summary["n_val_actor_steps"]
    if len(plan.val_actor_step_indices) > 0:
        val_actor_traj = np.searchsorted(
            off[1:], plan.val_actor_step_indices, side="right"
        )
        assert len(np.intersect1d(val_actor_traj, plan.train_traj_ids)) == 0
        assert not np.any(data["intercepted"].astype(bool)[val_actor_traj])


def test_plan_actor_eligible_both_ok_zero_stage3_fail():
    """P1：actor_eligible=both_ok 时 Actor 标签中 stage3-fail 占比应为 0。"""
    data = _synth_data(48, seed=2)
    # 故意写入错误的缓存 both_ok，确认采样按事实重算、不信任缓存列
    cache = {
        "r_minT": data["r_minT"],
        "both_ok": np.ones(len(data["n_steps"]), dtype=np.int8),
    }
    both_ok_truth = (
        (~data["intercepted"].astype(bool))
        & (data["r_min1"] >= 6)
        & np.isfinite(data["r_minT"])
        & (data["r_minT"] <= 10)
    )
    assert int(both_ok_truth.sum()) < int((~data["intercepted"].astype(bool)).sum())
    plan = build_expert_sample_plan(
        data,
        budget=800,
        seed=0,
        val_ratio=0.1,
        actor_eligible="both_ok",
        replay_include_intercepted=True,
        terminal_cache=cache,
        ok_modes=np.asarray(["ok"] * len(data["n_steps"]), dtype=object),
    )
    assert plan.summary["actor_eligible"] == "both_ok"
    assert plan.summary["actor_stage3_fail_step_frac"] == 0.0
    assert plan.summary["actor_intercept_step_frac"] == 0.0
    off = plan.traj_offsets
    actor_traj = np.searchsorted(off[1:], plan.actor_step_indices, side="right")
    assert len(actor_traj) > 0
    assert np.all(both_ok_truth[actor_traj])
    # Replay 仍可含非 both_ok（拦截或 stage3 失败）
    replay_traj = np.searchsorted(off[1:], plan.step_indices, side="right")
    assert np.any(~both_ok_truth[replay_traj])


def test_prefix_vs_balanced_coverage():
    """均衡采样覆盖的轨迹数应明显多于同预算前缀截取。"""
    data = _synth_data(60, seed=3)
    n_steps = data["n_steps"]
    budget = 900
    # 前缀覆盖轨数
    cum = np.cumsum(n_steps)
    n_prefix_traj = int(np.searchsorted(cum, budget, side="left") + 1)
    idx = sample_balanced_step_indices(n_steps, budget, seed=0)
    off = traj_offsets_from_n_steps(n_steps)
    bal_traj = np.unique(np.searchsorted(off[1:], idx, side="right"))
    assert len(bal_traj) > n_prefix_traj


if __name__ == "__main__":
    test_split_disjoint()
    test_quota_not_dominated_by_long()
    test_force_terminal_and_reproducible()
    test_plan_actor_excludes_intercept()
    test_plan_actor_eligible_both_ok_zero_stage3_fail()
    test_prefix_vs_balanced_coverage()
    print("OK: test_expert_sampling")
