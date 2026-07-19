"""
test_action_mapping.py — 比值动作映射往返、旧→新标签变换、相对权重等价。
不依赖 MATLAB。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from action_mapping import (  # noqa: E402
    ACTION_MAPPING_RATIO,
    ACTION_MAPPING_TD3_BOX,
    W_FIXED,
    action_norm_to_phys,
    action_norm_to_ratios,
    old_box_feasible_from_ratio_action,
    phys_to_action_norm,
    remap_action_norm,
)
from load_expert_buffer import apply_action_mapping  # noqa: E402


def test_td3_box_roundtrip():
    rng = np.random.default_rng(0)
    gamma = rng.uniform(-3, 3, size=200)
    wT = 10 ** rng.uniform(2, 5, size=200)
    wE = 10 ** rng.uniform(2, 4, size=200)
    an, clipped = phys_to_action_norm(
        gamma, wT, wE, version=ACTION_MAPPING_TD3_BOX
    )
    g2, wt2, we2 = action_norm_to_phys(an, version=ACTION_MAPPING_TD3_BOX)
    an2, _ = phys_to_action_norm(g2, wt2, we2, version=ACTION_MAPPING_TD3_BOX)
    assert np.max(np.abs(an - an2)) < 1e-6
    # 未裁剪样本：物理往返应接近
    keep = ~clipped
    if np.any(keep):
        assert np.max(np.abs(gamma[keep] - g2[keep])) < 1e-5


def test_ratio_roundtrip_and_old_box():
    rng = np.random.default_rng(1)
    # 在旧盒内采样 → 比值应 100% 可行
    gamma = rng.uniform(-3, 3, size=500)
    wT = 10 ** rng.uniform(2, 5, size=500)
    wE = 10 ** rng.uniform(2, 4, size=500)
    an, clipped = phys_to_action_norm(
        gamma, wT, wE, version=ACTION_MAPPING_RATIO, w=W_FIXED
    )
    assert float(np.mean(clipped)) < 0.01  # 盒内几乎不裁
    g2, wt2, we2 = action_norm_to_phys(an, version=ACTION_MAPPING_RATIO, w=W_FIXED)
    an2, _ = phys_to_action_norm(g2, wt2, we2, version=ACTION_MAPPING_RATIO, w=W_FIXED)
    assert np.max(np.abs(an - an2)) < 1e-6
    feas = old_box_feasible_from_ratio_action(an, w=W_FIXED)
    assert float(np.mean(feas)) > 0.999


def test_remap_preserves_guidance_ratios():
    """旧 action_norm → 比值后，式(47) 相对权重 (r1,r2,rT) 应一致。"""
    rng = np.random.default_rng(2)
    n = 300
    # 均匀旧归一化动作
    old = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    new = remap_action_norm(
        old, from_version=ACTION_MAPPING_TD3_BOX, to_version=ACTION_MAPPING_RATIO
    )
    r1a, r2a, rTa = action_norm_to_ratios(old, version=ACTION_MAPPING_TD3_BOX)
    r1b, r2b, rTb = action_norm_to_ratios(new, version=ACTION_MAPPING_RATIO)
    err = max(
        float(np.max(np.abs(r1a - r1b))),
        float(np.max(np.abs(r2a - r2b))),
        float(np.max(np.abs(rTa - rTb))),
    )
    # log10 往返量化约 1e-4 量级，对 a_M 可忽略
    assert err < 1e-4, f"相对权重误差过大: {err}"
    # γ 维在比值动作中为第 3 维，应等于旧第 1 维
    assert np.max(np.abs(old[:, 0] - new[:, 2])) < 1e-6


def test_apply_action_mapping_on_dict():
    rng = np.random.default_rng(3)
    data = {
        "actions": rng.uniform(-1, 1, size=(50, 3)).astype(np.float32),
        "action_mapping_version": ACTION_MAPPING_TD3_BOX,
        "states": rng.normal(size=(50, 13)),
    }
    out = apply_action_mapping(data, ACTION_MAPPING_RATIO)
    assert out["action_mapping_version"] == ACTION_MAPPING_RATIO
    assert out["actions_source_version"] == ACTION_MAPPING_TD3_BOX
    assert out["actions"].shape == (50, 3)
    # 幂等
    out2 = apply_action_mapping(out, ACTION_MAPPING_RATIO)
    assert np.allclose(out["actions"], out2["actions"])


def test_inverse_remap_roundtrip_norm():
    rng = np.random.default_rng(4)
    old = rng.uniform(-1, 1, size=(100, 3)).astype(np.float32)
    new = remap_action_norm(
        old, from_version=ACTION_MAPPING_TD3_BOX, to_version=ACTION_MAPPING_RATIO
    )
    back = remap_action_norm(
        new, from_version=ACTION_MAPPING_RATIO, to_version=ACTION_MAPPING_TD3_BOX
    )
    assert np.max(np.abs(old - back)) < 1e-4


if __name__ == "__main__":
    test_td3_box_roundtrip()
    test_ratio_roundtrip_and_old_box()
    test_remap_preserves_guidance_ratios()
    test_apply_action_mapping_on_dict()
    test_inverse_remap_roundtrip_norm()
    print("OK: test_action_mapping")
