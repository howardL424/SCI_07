"""Phase C-E 配对键与统一缩放域参数 NRMSE 单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl_td3"))
sys.path.insert(0, str(ROOT / "ident" / "scripts"))

from evaluate_td3 import _theta_scaled_nrmse  # noqa: E402
from run_p0_4_evaluation import (  # noqa: E402
    _mcnemar_exact_p,
    _p95_cvar95,
    validate_strict_pairing,
)


def test_scaled_nrmse_zero_and_componentwise_normalization() -> None:
    truth = [
        np.asarray([1e-4, 0.2, -0.1, 2, 2e-4, 0.1, 0.3, -1, 1e-4, 0.2, 0.1]),
        np.asarray([2e-4, 0.1, -0.2, 1, 1e-4, 0.2, 0.2, -2, 2e-4, 0.1, 0.2]),
    ]
    zero, components, n_steps = _theta_scaled_nrmse(
        truth, truth, r0=1e4, a0=50, denominator_floor=1.0
    )
    assert zero == 0.0
    np.testing.assert_array_equal(components, np.zeros(11))
    assert n_steps == 2

    estimates = [row.copy() for row in truth]
    estimates[0][0] += 1e-4
    value, components, _ = _theta_scaled_nrmse(
        estimates, truth, r0=1e4, a0=50, denominator_floor=1.0
    )
    assert value > 0
    assert components[0] > 0
    np.testing.assert_array_equal(components[1:], np.zeros(10))


def test_scaled_nrmse_burn_in_and_nan_causal_first_step() -> None:
    nan_row = np.full(11, np.nan)
    truth_row = np.ones(11)
    estimate_row = 1.1 * truth_row
    value, _, n_steps = _theta_scaled_nrmse(
        [nan_row, estimate_row],
        [nan_row, truth_row],
        r0=1.0,
        a0=1.0,
        start_step=1,
        denominator_floor=1.0,
    )
    assert np.isclose(value, 0.1)
    assert n_steps == 1


def test_strict_pairing_uses_actual_noise_seed() -> None:
    rows = [
        {"local_idx": 0, "row_idx": 3, "noise_seed": 100},
        {"local_idx": 1, "row_idx": 7, "noise_seed": 101},
    ]
    result = validate_strict_pairing(
        {
            "td3__analytic_noisy": rows,
            "td3__sparse_ffrls": [dict(row) for row in rows],
            "bc_td3__analytic_noisy": [dict(row) for row in rows],
            "bc_td3__sparse_ffrls": [dict(row) for row in rows],
        }
    )
    assert result["passed"] is True
    assert result["n_paired"] == 2
    assert result["pair_keys"] == [[0, 3, 100], [1, 7, 101]]

    mismatched = [dict(row) for row in rows]
    mismatched[1]["noise_seed"] = 999
    try:
        validate_strict_pairing({"a": rows, "b": mismatched})
    except RuntimeError as error:
        assert "严格配对失败" in str(error)
    else:
        raise AssertionError("不同实际 noise_seed 未触发严格配对失败")


def test_exact_mcnemar_and_tail_metrics() -> None:
    assert _mcnemar_exact_p(0, 0) == 1.0
    assert np.isclose(_mcnemar_exact_p(5, 0), 0.0625)
    assert np.isclose(_mcnemar_exact_p(3, 1), 0.625)

    tail = _p95_cvar95(list(range(1, 21)))
    assert tail["n"] == 20
    assert np.isclose(tail["p95"], 19.05)
    assert tail["cvar95"] == 20.0


if __name__ == "__main__":
    test_scaled_nrmse_zero_and_componentwise_normalization()
    test_scaled_nrmse_burn_in_and_nan_causal_first_step()
    test_strict_pairing_uses_actual_noise_seed()
    test_exact_mcnemar_and_tail_metrics()
    print("P0-4 evaluation tests: 4 passed")
