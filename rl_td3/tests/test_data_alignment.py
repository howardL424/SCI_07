"""轻量单测：obs 归一化、对齐一致性、新 schema / 奖励重投影（不依赖长时 IPOPT）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_expert_buffer import (
    check_aligned_consistency,
    expert_lib_has_phys_schema,
    load_phys_transitions_from_expert_lib,
)
from obs_norm import OBS_NORM_VERSION, OBS_SCALE, denormalize_obs, normalize_obs, overflow_ratio
from reward_projection import (
    DEFAULT_REWARD_CFG,
    project_transition_rewards,
    with_reward_overrides,
)


def test_obs_norm_roundtrip():
    x = np.array(
        [1e4, -500.0, 0.5, 1.0, 2e4, 100.0, -0.2, 0.3, 8e3, 0.1, -1.0, 800.0, 0.4],
        dtype=np.float64,
    )
    xn = normalize_obs(x)
    assert xn.shape == (13,)
    assert np.allclose(denormalize_obs(xn), x, rtol=1e-6)
    assert overflow_ratio(xn) == 0.0
    big = normalize_obs(x * 10)
    assert overflow_ratio(big) > 0.0
    assert OBS_SCALE.shape == (13,)
    assert OBS_NORM_VERSION == "obs_scale_0714_v1"


def test_check_aligned_consistency_ok():
    n1, n2 = 3, 2
    states = np.random.randn(n1 + n2, 13)
    actions = np.clip(np.random.randn(n1 + n2, 3), -1, 1).astype(np.float32)
    data = {
        "states": states,
        "actions": actions,
        "next_states": states.copy(),
        "rewards": np.zeros(n1 + n2, dtype=np.float32),
        "dones": np.array([False, False, True, False, True]),
        "traj_ids": np.array([0, 0, 0, 1, 1], dtype=np.int32),
        "n_steps": np.array([n1, n2], dtype=np.int32),
        "intercepted": np.array([0, 1], dtype=np.int8),
        "expert_path": "",
    }
    cache = {
        "expert_path": "",
        "n_steps": np.array([n1, n2], dtype=np.int32),
        "R1": np.array([1.0, 0.0]),
        "bonus": np.array([100.0, 0.0]),
    }
    warns = check_aligned_consistency(data, cache=cache)
    assert isinstance(warns, list)


def test_check_intercept_with_R1_raises():
    data = {
        "states": np.zeros((1, 13)),
        "actions": np.zeros((1, 3), dtype=np.float32),
        "next_states": np.zeros((1, 13)),
        "rewards": np.array([-50.0], dtype=np.float32),
        "dones": np.array([True]),
        "traj_ids": np.array([0], dtype=np.int32),
        "n_steps": np.array([1], dtype=np.int32),
        "intercepted": np.array([1], dtype=np.int8),
        "expert_path": "",
    }
    cache = {
        "expert_path": "",
        "n_steps": np.array([1], dtype=np.int32),
        "R1": np.array([5.0]),
        "bonus": np.array([0.0]),
    }
    try:
        check_aligned_consistency(data, cache=cache)
        raised = False
    except ValueError:
        raised = True
    assert raised, "拦截轨迹含 R1 应报错"


def test_last_step_next_state_and_single_done():
    """末步 next_state 存在；每轨仅末步 done；拦截轨无正终端投影。"""
    n = 4
    next_states = np.zeros((n, 13), dtype=np.float64)
    next_states[:, 0] = 20.0
    next_states[:, 4] = 20.0
    next_states[-1, 0] = 5.0  # 末步拦截
    next_states[-1, 4] = 8.0
    data = {
        "states": np.zeros((n, 13)),
        "actions": np.zeros((n, 3), dtype=np.float32),
        "next_states": next_states,
        "dones": np.array([False, False, False, True]),
        "traj_ids": np.zeros(n, dtype=np.int32),
        "n_steps": np.array([n], dtype=np.int32),
        "intercepted": np.array([1], dtype=np.int8),
        "r_min1": np.array([5.0]),
        "r_min2": np.array([8.0]),
        "terminal_abs": np.zeros((1, 18)),
        "expert_path": "",
    }
    cache = {
        "r_minT": np.array([np.nan]),
        "r_min1": np.array([5.0]),
        "r_min2": np.array([8.0]),
        "R1": np.array([0.0]),
        "bonus": np.array([0.0]),
        "n_steps": np.array([n], dtype=np.int32),
    }
    proj = project_transition_rewards(data, cache, cfg=DEFAULT_REWARD_CFG)
    assert proj["rewards"][-1] == -50.0
    assert proj["R1"][0] == 0.0 and proj["bonus"][0] == 0.0
    data2 = {**data, "rewards": proj["rewards"]}
    check_aligned_consistency(data2, cache=cache)


def test_phys_smoke_schema_if_present():
    lib = Path(__file__).resolve().parent.parent / "data" / "expert_lib_phys_smoke.mat"
    if not lib.exists():
        print("SKIP: expert_lib_phys_smoke.mat missing")
        return
    assert expert_lib_has_phys_schema(str(lib))
    data = load_phys_transitions_from_expert_lib(str(lib))
    assert data["source_mode"] == "expert_lib"
    assert data["next_states"].shape[1] == 13
    assert str(data.get("schema_version", "")).startswith("phys_transition")
    cursor = 0
    for n_k in data["n_steps"]:
        n_k = int(n_k)
        seg = data["dones"][cursor : cursor + n_k]
        assert bool(seg[-1])
        if n_k > 1:
            assert not np.any(seg[:-1])
        cursor += n_k


def test_reproject_changes_coef():
    n = 3
    ns = np.tile([20.0, 0, 0, 0, 20.0, 0, 0, 0, 0, 0, 0, 0, 0], (n, 1))
    data = {
        "next_states": ns,
        "n_steps": np.array([n], dtype=np.int32),
        "intercepted": np.array([0], dtype=np.int8),
        "r_min1": np.array([20.0]),
        "r_min2": np.array([20.0]),
    }
    cache = {"r_minT": np.array([8.0]), "r_min1": np.array([20.0]), "r_min2": np.array([20.0])}
    a = project_transition_rewards(data, cache, cfg=DEFAULT_REWARD_CFG)
    b = project_transition_rewards(data, cache, cfg=with_reward_overrides(proc_coef=0.1))
    assert abs(b["rewards"][0] / a["rewards"][0] - 2.0) < 1e-5


if __name__ == "__main__":
    test_obs_norm_roundtrip()
    test_check_aligned_consistency_ok()
    test_check_intercept_with_R1_raises()
    test_last_step_next_state_and_single_done()
    test_reproject_changes_coef()
    test_phys_smoke_schema_if_present()
    from test_reward_projection import (
        test_process_and_intercept,
        test_reproject_coef_without_matlab,
        test_soft_sat_fixed_samples,
        test_terminal_and_intercept_zero,
    )

    test_soft_sat_fixed_samples()
    test_process_and_intercept()
    test_terminal_and_intercept_zero()
    test_reproject_coef_without_matlab()
    print("OK: test_data_alignment (schema + reward + invariants)")
