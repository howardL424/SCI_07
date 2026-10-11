"""全过程能耗换算与 MC 落盘字段单测（不依赖 MATLAB）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_td3 import (  # noqa: E402
    STAGE2_DT,
    accel_energy_kj,
    accel_integral,
    energy_int_to_kj,
    save_mc_metrics,
)


def test_constant_accel_matches_closed_form():
    a = 10.0
    n = 11
    t = np.linspace(0.0, 1.0, n)
    a_sq = np.full(n, a * a)
    ju = accel_integral(a_sq, t=t)
    assert abs(ju - a * a * 1.0) < 1e-9
    e_kj = accel_energy_kj(a_sq, t=t)
    assert abs(e_kj - 0.5 * a * a * 1.0 / 1000.0) < 1e-12


def test_single_sample_uses_dt():
    ju = accel_integral([4.0], dt=STAGE2_DT)
    assert abs(ju - 4.0 * STAGE2_DT) < 1e-12
    assert abs(energy_int_to_kj(ju) - 0.5 * ju / 1000.0) < 1e-15


def test_nan_and_empty():
    assert accel_integral([]) == 0.0
    assert accel_integral([np.nan, np.nan]) == 0.0
    assert np.isnan(energy_int_to_kj(float("nan")))


def test_save_mc_metrics_writes_energy(tmp_path: Path):
    results = [
        {
            "r_MD1_min": 20.0,
            "r_MD2_min": 18.0,
            "r_minT": 4.0,
            "evade_ok": True,
            "strike_ok": True,
            "both_ok": True,
            "intercepted": False,
            "row_idx": 1,
            "local_idx": 0,
            "a_energy": 200.0,
            "energy_int": 240.0,
            "energy_kJ": 0.12,
            "energy_stage2_kJ": 0.10,
            "energy_stage3_kJ": 0.02,
        }
    ]
    path = tmp_path / "mc_metrics.npz"
    save_mc_metrics(results, path)
    data = np.load(path)
    assert float(data["energy_kJ"][0]) == 0.12
    assert float(data["energy_stage2_kJ"][0]) == 0.10
    assert float(data["energy_stage3_kJ"][0]) == 0.02
    assert float(data["energy_int"][0]) == 240.0


if __name__ == "__main__":
    test_constant_accel_matches_closed_form()
    test_single_sample_uses_dt()
    test_nan_and_empty()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_save_mc_metrics_writes_energy(Path(td))
    print("ok")
