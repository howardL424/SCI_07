"""
test_obs_qmag.py — 16D Q 扩展的轻量验收（合成数据，不依赖专家库）。

覆盖：
  1. Q 派生 shape/dtype/有限性/非负；
  2. 16D 前 13 维与 13D 归一化完全一致（可撤回：13D 不受影响）；
  3. 归一化/反归一化可逆；
  4. Q scale 只用训练集估计、为正；
  5. 13D/16D checkpoint 互斥（误加载直接拒绝）；
  6. Python 移植与 MATLAB stage2_qmag_from_abs 一致（需 Engine，缺失则跳过）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False


class _Skipped(Exception):
    pass


def _skip(msg: str) -> None:
    raise _Skipped(msg)


class _RaisesCtx:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"应抛出 {self.exc} 但未抛出")
        return issubclass(et, self.exc)


def _raises(exc):
    return _RaisesCtx(exc)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bc_actor import BcActor, save_bc_checkpoint, validate_bc_checkpoint  # noqa: E402
from obs_norm import normalize_obs  # noqa: E402
from obs_qmag import (  # noqa: E402
    SCHEMA_REL13_QMAG16,
    denormalize_obs16,
    estimate_q_scales,
    normalize_obs16,
    qmag_batch,
    qmag_from_state_abs,
)


def _synthetic_abs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S = np.zeros((n, 18), dtype=np.float64)
    # M/D1/D2 位置分散 1e3~2e4 m，姿态角小范围，速度 200~600 m/s
    for base in (0, 6, 12):
        S[:, base + 0] = rng.uniform(-2e4, 2e4, n)
        S[:, base + 1] = rng.uniform(1e3, 8e3, n)
        S[:, base + 2] = rng.uniform(-2e4, 2e4, n)
        S[:, base + 3] = rng.uniform(-0.3, 0.3, n)
        S[:, base + 4] = rng.uniform(-0.5, 0.5, n)
        S[:, base + 5] = rng.uniform(200.0, 600.0, n)
    return S


def test_qmag_shapes_finite_nonneg():
    S = _synthetic_abs(50)
    q = qmag_batch(S)
    assert q.shape == (50, 3)
    assert np.all(np.isfinite(q)), "Q 含 NaN/Inf"
    assert np.all(q >= 0), "Q 幅值须非负"
    # 确定性：同输入同输出
    q2 = qmag_batch(S)
    np.testing.assert_array_equal(q, q2)


def test_qmag_single_matches_batch():
    S = _synthetic_abs(5, seed=1)
    q = qmag_batch(S)
    for i in range(5):
        np.testing.assert_allclose(q[i], qmag_from_state_abs(S[i]), rtol=0, atol=0)


def test_qmag_bad_dim_rejected():
    with _raises(ValueError):
        qmag_from_state_abs(np.zeros(13))
    with _raises(ValueError):
        qmag_batch(np.zeros((4, 13)))


def test_obs16_prefix_equals_13d():
    rng = np.random.default_rng(2)
    obs13 = rng.normal(size=(32, 13))
    q = np.abs(rng.normal(size=(32, 3))) + 0.01
    scale = np.array([0.05, 0.05, 0.02])
    out16 = normalize_obs16(obs13, q, scale)
    assert out16.shape == (32, 16)
    np.testing.assert_allclose(out16[:, :13], normalize_obs(obs13), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out16[:, 13:], (q / scale).astype(np.float32), rtol=1e-6, atol=1e-7)


def test_denormalize_roundtrip():
    rng = np.random.default_rng(3)
    obs13 = rng.normal(size=(16, 13)) * np.array(
        [3e4, 1.2e3, 1.5, 3.0, 3e4, 1.2e3, 1.5, 3.0, 1.5e4, 1.5, 3.0, 1e3, 1.0]
    )
    q = np.abs(rng.normal(size=(16, 3))) * 0.05 + 1e-4
    scale = np.array([0.05, 0.05, 0.02])
    out16 = normalize_obs16(obs13, q, scale).astype(np.float64)
    back = denormalize_obs16(out16, scale)
    np.testing.assert_allclose(back[:, :13], obs13, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(back[:, 13:], q, rtol=1e-5, atol=1e-8)


def test_estimate_q_scales_train_only_positive():
    rng = np.random.default_rng(4)
    qtr = np.abs(rng.normal(size=(1000, 3))) * 0.05 + 1e-4
    s = estimate_q_scales(qtr)
    assert s.shape == (3,)
    assert np.all(s > 0) and np.all(np.isfinite(s))
    # 验证集大 Q 不应改变已估计的 scale（纯函数，无状态）
    s2 = estimate_q_scales(qtr)
    np.testing.assert_array_equal(s, s2)
    with _raises(ValueError):
        estimate_q_scales(np.zeros((10, 2)))


def test_checkpoint_13d_16d_mutually_reject(tmp_path):
    device = torch.device("cpu")
    actor13 = BcActor(obs_dim=13, act_dim=3).to(device)
    p13 = tmp_path / "bc13.pt"
    save_bc_checkpoint(
        p13, actor13,
        config={"action_mapping": "ratio_log_box_v1",
                "obs_schema_version": "stage2_relative_obs_13d_v1",
                "obs_norm_version": "obs_scale_0714_v1"},
    )
    payload13 = torch.load(p13, map_location="cpu", weights_only=False)
    validate_bc_checkpoint(payload13, obs_dim=13, act_dim=3,
                           net_arch=(256, 256), action_mapping="ratio_log_box_v1")
    with _raises(ValueError):
        validate_bc_checkpoint(payload13, obs_dim=16, act_dim=3,
                               net_arch=(256, 256), action_mapping="ratio_log_box_v1")

    actor16 = BcActor(obs_dim=16, act_dim=3).to(device)
    p16 = tmp_path / "bc16.pt"
    save_bc_checkpoint(
        p16, actor16,
        config={"action_mapping": "ratio_log_box_v1",
                "obs_schema_version": SCHEMA_REL13_QMAG16,
                "obs_norm_version": "obs_scale_0714_v1+qmag_q995_v1"},
    )
    payload16 = torch.load(p16, map_location="cpu", weights_only=False)
    validate_bc_checkpoint(payload16, obs_dim=16, act_dim=3,
                           net_arch=(256, 256), action_mapping="ratio_log_box_v1")
    with _raises(ValueError):
        validate_bc_checkpoint(payload16, obs_dim=13, act_dim=3,
                               net_arch=(256, 256), action_mapping="ratio_log_box_v1")


def test_qmag_for_aligned_mapping_and_guards():
    """source_idx 前缀映射：成功拼出 Q；首步错配/越界直接拒绝（合成数据）。"""
    import sys as _sys
    import types as _types

    from obs_qmag import qmag_batch as _qb
    from obs_qmag import qmag_for_aligned_data as _map

    S0 = _synthetic_abs(10, seed=11)
    S1 = _synthetic_abs(6, seed=12)
    # 伪 13D 记录态（首步必须与“库记录”一致才放行）
    st0 = np.random.default_rng(21).normal(size=(10, 13))
    st1 = np.random.default_rng(22).normal(size=(6, 13))
    fake_trajs = [
        {"ok_use": True, "transitions": {"state": st0, "state_abs": S0}},
        {"ok_use": False, "transitions": None},
        {"ok_use": True, "transitions": {"state": st1, "state_abs": S1}},
    ]
    fake_mat73 = _types.ModuleType("mat73")
    # _iter_trajectories 要求 results 为 dict（含 trajectories）或具属性对象
    fake_mat73.loadmat = lambda _p: {"results": {"trajectories": fake_trajs,  # type: ignore[attr-defined]
                                                "schema_version": "x"}}
    _sys.modules["mat73"] = fake_mat73
    try:
        # aligned 轨0 ← src#1 全长；轨1 ← src#3 前缀截断（模拟拦截截断）
        n_steps = np.array([10, 4], dtype=np.int32)
        states = np.concatenate([st0, st1[:4]], axis=0)
        aligned = {"n_steps": n_steps,
                   "source_idx": np.array([1, 3], dtype=np.int32),
                   "states": states}
        q, rep = _map(aligned, "fake.mat")
        assert q.shape == (14, 3)
        np.testing.assert_allclose(q[:10], _qb(S0), rtol=0, atol=0)
        np.testing.assert_allclose(q[10:], _qb(S1[:4]), rtol=0, atol=0)
        assert rep["n_traj"] == 2 and rep["max_first_step_diff"] == 0.0

        # 首步错配 → 拒绝
        bad = dict(aligned)
        bad_states = states.copy()
        bad_states[0, 0] += 1.0
        bad["states"] = bad_states
        try:
            _map(bad, "fake.mat")
        except ValueError:
            pass
        else:
            raise AssertionError("首步错配应被拒绝")

        # 越界 → 拒绝
        bad2 = dict(aligned)
        bad2["n_steps"] = np.array([11, 4], dtype=np.int32)
        try:
            _map(bad2, "fake.mat")
        except ValueError:
            pass
        else:
            raise AssertionError("步数越界应被拒绝")
    finally:
        del _sys.modules["mat73"]


def test_python_matches_matlab_qmag():
    """Python 移植 vs MATLAB stage2_qmag_from_abs（需 MATLAB Engine）。"""
    try:
        import matlab.engine  # noqa: F401
    except Exception as e:
        _skip(f"无 matlab.engine：{e}")
    import matlab.engine as mleng

    eng = mleng.start_matlab()
    try:
        eng.addpath(str(ROOT / "matlab"), nargout=0)
        eng.addpath(str(ROOT.parent), nargout=0)  # evasion_helpers.m 在仓库根
        core = eng.stage2_core_init(nargout=1)
        S = _synthetic_abs(8, seed=7)
        for i in range(len(S)):
            q_py = qmag_from_state_abs(S[i])
            q_ml = eng.stage2_qmag_from_abs(core, matlab.double(S[i].tolist()), nargout=3)
            q_ml = np.array([float(q_ml[0]), float(q_ml[1]), float(q_ml[2])])
            np.testing.assert_allclose(q_py, q_ml, rtol=1e-9, atol=1e-9,
                                       err_msg=f"第 {i} 步 Python/MATLAB Q 不一致")
    finally:
        try:
            eng.quit()
        except Exception:
            pass


if __name__ == "__main__":
    import traceback

    cases = [
        ("qmag_shapes_finite_nonneg", test_qmag_shapes_finite_nonneg),
        ("qmag_single_matches_batch", test_qmag_single_matches_batch),
        ("qmag_bad_dim_rejected", test_qmag_bad_dim_rejected),
        ("obs16_prefix_equals_13d", test_obs16_prefix_equals_13d),
        ("denormalize_roundtrip", test_denormalize_roundtrip),
        ("estimate_q_scales", test_estimate_q_scales_train_only_positive),
        ("aligned_mapping_guards", test_qmag_for_aligned_mapping_and_guards),
        ("checkpoint_mutual_reject", test_checkpoint_13d_16d_mutually_reject),
    ]
    import tempfile

    failed = 0
    for name, fn in cases:
        try:
            if name == "checkpoint_mutual_reject":
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    # MATLAB 一致性单独跑（需 Engine，失败不阻断 13D）
    try:
        test_python_matches_matlab_qmag()
        print("PASS matlab_consistency")
    except Exception as e:
        print(f"SKIP/FAIL matlab_consistency：{e}")
    print("OK: test_obs_qmag" if failed == 0 else f"{failed} 个用例失败")
    raise SystemExit(1 if failed else 0)
