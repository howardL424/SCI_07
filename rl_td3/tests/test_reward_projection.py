"""奖励投影单测：公式一致性、换系数不调 MATLAB、可选 MATLAB 对照。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_projection import (
    DEFAULT_REWARD_CFG,
    infer_step_intercepted,
    project_process_rewards,
    project_terminal,
    project_transition_rewards,
    soft_sat_log_evade,
    terminal_R1,
    with_reward_overrides,
)


def test_soft_sat_fixed_samples():
    assert soft_sat_log_evade(5.0) == 0.0
    assert soft_sat_log_evade(6.0) == 0.0
    # r=20 → log10(20-5)=log10(15)
    assert abs(soft_sat_log_evade(20.0) - math.log10(15.0)) < 1e-12
    # r=100 软饱和到 d*=20
    assert abs(soft_sat_log_evade(100.0) - math.log10(15.0)) < 1e-12


def test_process_and_intercept():
    ns = np.array(
        [
            [20.0, 0, 0, 0, 20.0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5.0, 0, 0, 0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    rew = project_process_rewards(ns)
    expect0 = 0.05 * (math.log10(15.0) + math.log10(15.0))
    assert abs(rew[0] - expect0) < 1e-6
    assert rew[1] == -50.0
    assert infer_step_intercepted(ns).tolist() == [False, True]


def test_terminal_and_intercept_zero():
    R1, bonus, ok = project_terminal(5.0, 10.0, 10.0, intercepted=False)
    assert abs(R1 - 80.0 / 10.0) < 1e-9
    assert bonus == 100.0 and ok == 1
    R1b, bb, okb = project_terminal(5.0, 10.0, 10.0, intercepted=True)
    assert R1b == 0.0 and bb == 0.0 and okb == 0
    assert terminal_R1(30.0) == 0.0


def test_reproject_coef_without_matlab():
    """改 proc_coef / bonus → 只重投影，不需 MATLAB。"""
    n = 4
    next_states = np.tile(
        [15.0, 0, 0, 0, 18.0, 0, 0, 0, 0, 0, 0, 0, 0], (n, 1)
    ).astype(np.float64)
    data = {
        "next_states": next_states,
        "n_steps": np.array([n], dtype=np.int32),
        "intercepted": np.array([0], dtype=np.int8),
        "r_min1": np.array([15.0]),
        "r_min2": np.array([18.0]),
        "rewards": np.zeros(n, dtype=np.float32),
    }
    cache = {
        "r_minT": np.array([8.0]),
        "r_min1": np.array([15.0]),
        "r_min2": np.array([18.0]),
    }
    base = project_transition_rewards(data, cache, cfg=DEFAULT_REWARD_CFG)
    hot = project_transition_rewards(
        data, cache, cfg=with_reward_overrides(proc_coef=0.1, bonus_both=200.0)
    )
    # 过程项应约翻倍
    assert abs(hot["rewards"][0] / base["rewards"][0] - 2.0) < 1e-5
    # 末步含终端：bonus 200 vs 100
    assert abs(hot["bonus"][0] - 200.0) < 1e-9
    assert abs(base["bonus"][0] - 100.0) < 1e-9
    assert hot["reward_version"] == DEFAULT_REWARD_CFG.version


def test_python_matlab_process_reward():
    """固定样例：Python 投影 ↔ MATLAB stage2_process_reward。"""
    try:
        import matlab.engine
    except ImportError:
        print("SKIP: matlab.engine 不可用")
        return

    eng = matlab.engine.start_matlab()
    try:
        root = Path(__file__).resolve().parent.parent  # rl_td3/
        eng.addpath(str(root / "matlab"), nargout=0)
        eng.addpath(str(root.parent), nargout=0)  # GAIL/ for evasion_helpers
        core = eng.stage2_core_init({}, nargout=1)
        samples = [
            (20.0, 20.0, False),
            (5.0, 30.0, True),
            (12.0, 8.0, False),
            (6.0, 6.0, True),
        ]
        for r1, r2, intercepted in samples:
            fact = eng.struct()
            fact["r_MD1"] = float(r1)
            fact["r_MD2"] = float(r2)
            fact["r_MT"] = 1000.0
            fact["intercepted"] = bool(intercepted)
            fact["stage2_done"] = False
            fact["terminated"] = bool(intercepted)
            fact["done"] = bool(intercepted)
            r_ml = float(eng.stage2_process_reward(core, fact, nargout=1))
            ns = np.array([[r1, 0, 0, 0, r2, 0, 0, 0, 0, 0, 0, 0, 0]])
            step_int = np.array([intercepted])
            r_py = float(project_process_rewards(ns, step_int)[0])
            assert abs(r_ml - r_py) < 1e-6, f"mismatch {r1},{r2},{intercepted}: {r_ml} vs {r_py}"
        print("OK: Python <-> MATLAB process reward")
    finally:
        eng.quit()


if __name__ == "__main__":
    test_soft_sat_fixed_samples()
    test_process_and_intercept()
    test_terminal_and_intercept_zero()
    test_reproject_coef_without_matlab()
    test_python_matlab_process_reward()
    print("OK: test_reward_projection")
