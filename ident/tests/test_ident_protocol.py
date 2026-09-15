"""参数辨识配置与 MATLAB/Python 85维诊断协议测试。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

IDENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(IDENT))

from ident_cfg import (  # noqa: E402
    IDENT_CLI_DEFAULTS,
    add_ident_args,
    matlab_cfg_from_args,
)
from ident_protocol import IDENT_INFO_LEN, parse_ident_diag  # noqa: E402


def test_ident_defaults_and_removed_kf_covariance_fields():
    assert IDENT_CLI_DEFAULTS["sigma_r"] == 10.0
    assert IDENT_CLI_DEFAULTS["sigma_rdot"] == 0.5
    assert IDENT_CLI_DEFAULTS["rls_P_trace_max"] == 1.0
    assert "kf_sigma_r" not in IDENT_CLI_DEFAULTS
    assert "kf_sigma_rdot" not in IDENT_CLI_DEFAULTS

    parser = add_ident_args(argparse.ArgumentParser())
    cfg = matlab_cfg_from_args(parser.parse_args([]))
    assert cfg["sigma_r"] == 10.0
    assert cfg["sigma_rdot"] == 0.5
    assert "kf_sigma_r" not in cfg
    assert "kf_sigma_rdot" not in cfg


def test_ident_protocol_85_slices_and_causal_aliases():
    vec = np.arange(IDENT_INFO_LEN, dtype=np.float64)
    vec[57:68] = np.nan  # 首拍 causal 真值允许为 NaN
    diag = parse_ident_diag(vec)

    assert diag is not None
    np.testing.assert_array_equal(diag["r_err"], vec[9:12])
    np.testing.assert_array_equal(diag["rdot_err"], vec[12:15])
    np.testing.assert_array_equal(diag["rddot_err"], vec[15:18])
    np.testing.assert_array_equal(diag["theta_err_causal"], vec[18:29])
    np.testing.assert_array_equal(diag["theta_hat"], vec[46:57])
    np.testing.assert_array_equal(diag["theta_true_current"], vec[68:79])
    np.testing.assert_array_equal(diag["frozen_model_residual"], vec[79:82])
    np.testing.assert_array_equal(diag["qdot_err_command"], vec[82:85])
    np.testing.assert_allclose(
        diag["theta_true"], diag["theta_true_causal"], equal_nan=True
    )
    np.testing.assert_array_equal(
        diag["theta_err_command"], vec[46:57] - vec[68:79]
    )


def test_base_protocol_has_no_ident_diag():
    assert parse_ident_diag(np.zeros(9)) is None


if __name__ == "__main__":
    test_ident_defaults_and_removed_kf_covariance_fields()
    test_ident_protocol_85_slices_and_causal_aliases()
    test_base_protocol_has_no_ident_diag()
    print("protocol tests: 3 passed")
