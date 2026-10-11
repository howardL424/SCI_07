"""
test_bc_actor.py — 纯 Actor BC 轻量单测（合成数据，不依赖 MATLAB）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bc_actor import (  # noqa: E402
    BcActor,
    compute_bc_metrics,
    copy_bc_weights_to_td3_actor,
    copy_bc_weights_to_td3_actors,
    evaluate_bc,
    load_bc_checkpoint,
    save_bc_checkpoint,
    train_bc_epoch,
    validate_bc_checkpoint,
)
from obs_norm import OBS_NORM_VERSION, OBS_SCHEMA_VERSION  # noqa: E402
from expert_sampling import (  # noqa: E402
    bc_arrays_from_plan,
    build_expert_sample_plan,
    load_sample_plan,
)


def test_metrics_and_fit(tmp_path: Path | None = None):
    rng = np.random.default_rng(0)
    # 可学映射：a = tanh(W s) 近似
    states = rng.normal(size=(2000, 13)).astype(np.float32)
    w = rng.normal(size=(13, 3)).astype(np.float32) * 0.1
    actions = np.tanh(states @ w).astype(np.float32)

    device = torch.device("cpu")
    actor = BcActor(obs_dim=13, act_dim=3, net_arch=(64, 64)).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    rng_t = np.random.default_rng(1)
    for _ in range(40):
        train_bc_epoch(
            actor, opt, states[:1600], actions[:1600],
            batch_size=128, device=device, rng=rng_t,
        )
    m = evaluate_bc(actor, states[1600:], actions[1600:], device=device)
    assert m.mse < 0.05, f"MSE too high: {m.mse}"
    assert m.n == 400

    # 贴边统计形状
    pred = np.ones((10, 3), dtype=np.float32)
    tgt = np.zeros((10, 3), dtype=np.float32)
    br = compute_bc_metrics(pred, tgt, boundary_thresh=0.99)
    assert br.boundary_rate == 1.0

    out = Path(tmp_path) if tmp_path else ROOT / "data" / "_bc_test_ckpt.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_bc_checkpoint(out, actor, metrics=m.to_dict(), epoch=1)
    actor2, payload = load_bc_checkpoint(out, device=device)
    assert payload["epoch"] == 1
    m2 = evaluate_bc(actor2, states[1600:], actions[1600:], device=device)
    assert abs(m2.mse - m.mse) < 1e-8
    if tmp_path is None and out.exists():
        out.unlink()


def test_plan_bc_arrays_and_roundtrip(tmp_path: Path | None = None):
    rng = np.random.default_rng(4)
    n_traj = 30
    n_steps = rng.integers(20, 40, size=n_traj).astype(np.int32)
    total = int(n_steps.sum())
    intercepted = rng.random(n_traj) < 0.2
    data = {
        "states": rng.normal(size=(total, 13)),
        "actions": rng.uniform(-1, 1, size=(total, 3)).astype(np.float32),
        "next_states": rng.normal(size=(total, 13)),
        "dones": np.zeros(total, dtype=bool),
        "traj_ids": np.concatenate(
            [np.full(int(n_steps[i]), i, dtype=np.int32) for i in range(n_traj)]
        ),
        "n_steps": n_steps,
        "intercepted": intercepted.astype(np.int8),
        "t_end": n_steps.astype(np.float64) * 0.1,
        "source_idx": np.arange(1, n_traj + 1, dtype=np.int32),
        "r_min1": np.where(intercepted, 3.0, 12.0),
        "r_min2": np.where(intercepted, 3.0, 12.0),
        "r_minT": np.where(intercepted, np.nan, 8.0),
    }
    plan = build_expert_sample_plan(data, budget=500, seed=0, val_ratio=0.2)
    assert len(plan.val_actor_step_indices) > 0
    s_tr, a_tr, s_va, a_va = bc_arrays_from_plan(
        data["states"].astype(np.float32), data["actions"], plan
    )
    assert len(s_tr) == len(a_tr) == len(plan.actor_step_indices)
    assert len(s_va) == len(a_va) == len(plan.val_actor_step_indices)

    out = Path(tmp_path) / "plan.npz" if tmp_path else ROOT / "data" / "_bc_plan_test.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    plan.save(out)
    plan2 = load_sample_plan(out)
    assert np.array_equal(plan.actor_step_indices, plan2.actor_step_indices)
    assert np.array_equal(plan.val_actor_step_indices, plan2.val_actor_step_indices)
    if tmp_path is None and out.exists():
        out.unlink()


def test_copy_to_sb3_shapes():
    """BcActor 与 SB3 TD3 Actor 的 mu 层形状一致时可拷贝。"""
    import gymnasium as gym
    from stable_baselines3 import TD3

    # 用与 Evasion 同维的假环境空间
    class _BoxEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-10, high=10, shape=(13,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(13, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(13, dtype=np.float32), 0.0, True, False, {}

    env = _BoxEnv()
    model = TD3("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), verbose=0)
    bc = BcActor(obs_dim=13, act_dim=3, net_arch=(256, 256))
    # 扰动 BC 权重再拷入
    with torch.no_grad():
        for p in bc.parameters():
            p.add_(0.1)
    copy_bc_weights_to_td3_actor(model.actor, bc)
    # 抽一层比对
    assert torch.allclose(
        model.actor.mu[0].weight.data.cpu(), bc.mu[0].weight.data.cpu()
    )


def test_warmstart_actor_target_outputs_and_metadata():
    """Actor/target 均须与 BC 输出一致，schema 或映射不匹配时拒绝加载。"""
    import gymnasium as gym
    from stable_baselines3 import TD3

    class _BoxEnv(gym.Env):
        observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(13,), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(13, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(13, dtype=np.float32), 0.0, True, False, {}

    model = TD3(
        "MlpPolicy",
        _BoxEnv(),
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        device="cpu",
    )
    bc = BcActor(obs_dim=13, act_dim=3, net_arch=(256, 256))
    payload = {
        "obs_dim": 13,
        "act_dim": 3,
        "net_arch": [256, 256],
        "config": {
            "action_mapping": "ratio_log_box_v1",
            "obs_schema_version": OBS_SCHEMA_VERSION,
            "obs_norm_version": OBS_NORM_VERSION,
        },
    }
    validate_bc_checkpoint(
        payload,
        obs_dim=13,
        act_dim=3,
        net_arch=(256, 256),
        action_mapping="ratio_log_box_v1",
    )
    copy_bc_weights_to_td3_actors(model.actor, model.actor_target, bc)
    obs = torch.randn(32, 13)
    with torch.no_grad():
        expected = bc(obs)
        assert torch.allclose(model.actor(obs), expected)
        assert torch.allclose(model.actor_target(obs), expected)

    bad = dict(payload)
    bad["config"] = dict(payload["config"], action_mapping="td3_box_log_v1")
    try:
        validate_bc_checkpoint(
            bad,
            obs_dim=13,
            act_dim=3,
            net_arch=(256, 256),
            action_mapping="ratio_log_box_v1",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("不兼容 action_mapping 应被拒绝")


if __name__ == "__main__":
    test_metrics_and_fit()
    test_plan_bc_arrays_and_roundtrip()
    test_copy_to_sb3_shapes()
    test_warmstart_actor_target_outputs_and_metadata()
    print("OK: test_bc_actor")
