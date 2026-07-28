"""
scan_samples_warmstart.py — 扫描初态池：env_reset / warm-start 能否进入 stage2。

用法（仓库根目录）：
    E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/scan_samples_warmstart.py
    E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/scan_samples_warmstart.py \\
      --samples samples_clean_test.mat samples_clean_val.mat
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matlab.engine
import numpy as np
import scipy.io

_RL_TD3 = Path(__file__).resolve().parent.parent
_ROOT = _RL_TD3.parent
sys.path.insert(0, str(_RL_TD3))

from evasion_env import (  # noqa: E402
    GAIL_ROOT,
    MATLAB_DIR,
    EXPERT_DIR,
    SAMPLES_TEST_FILE,
    SAMPLES_VAL_FILE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="扫描 samples 初态 warm-start 成功率")
    p.add_argument(
        "--samples",
        nargs="+",
        default=[SAMPLES_TEST_FILE, SAMPLES_VAL_FILE],
        help="要扫描的 mat（默认 test + val）",
    )
    return p.parse_args()


def load_nonzero_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sc = scipy.io.loadmat(str(path), squeeze_me=True)
    samples_all = np.asarray(sc["samples_clean"], dtype=np.float64)
    if samples_all.ndim != 2:
        raise ValueError(f"{path}: samples_clean 须为 2D，收到 {samples_all.shape}")
    valid = np.linalg.norm(samples_all, axis=1) > 1e-12
    row_ids = np.flatnonzero(valid).astype(np.int64)
    return samples_all[valid], row_ids


def scan_file(eng, path: Path) -> dict:
    samples, row_ids = load_nonzero_rows(path)
    n = samples.shape[0]
    ok_flags = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    print(f"\n=== {path.name} ===")
    print(f"  非零行: {n}（mat 原始行号范围 "
          f"{int(row_ids[0])}..{int(row_ids[-1])}）")

    for i in range(n):
        row_ml = matlab.double(samples[i].tolist())
        *_, ok_ml = eng.env_reset(row_ml, {}, nargout=5)
        ok_flags[i] = bool(ok_ml)
        if (i + 1) % 50 == 0 or i + 1 == n:
            n_ok = int(ok_flags[: i + 1].sum())
            elapsed = time.perf_counter() - t0
            print(
                f"  [{i+1:3d}/{n}] ok={n_ok}  fail={i+1-n_ok}  "
                f"elapsed={elapsed:.1f}s"
            )

    fail_local = np.flatnonzero(~ok_flags)
    fail_rows = row_ids[fail_local] if fail_local.size else np.array([], dtype=np.int64)
    n_ok = int(ok_flags.sum())
    elapsed = time.perf_counter() - t0
    print(
        f"  结果: ok={n_ok}/{n} ({100*n_ok/n:.1f}%)  "
        f"fail={n-n_ok}  用时 {elapsed:.1f}s"
    )
    if fail_rows.size:
        preview = ", ".join(str(int(x)) for x in fail_rows[:20])
        more = f" …(+{fail_rows.size-20})" if fail_rows.size > 20 else ""
        print(f"  失败原始行号: [{preview}{more}]")
        print(f"  失败 local_idx: {fail_local.tolist()[:20]}"
              + (f" …(+{fail_local.size-20})" if fail_local.size > 20 else ""))

    return {
        "file": str(path),
        "n_nonzero": n,
        "n_ok": n_ok,
        "n_fail": int(n - n_ok),
        "fail_row_ids": fail_rows.tolist(),
        "fail_local_idx": fail_local.tolist(),
        "elapsed_s": elapsed,
    }


def main() -> None:
    args = parse_args()
    paths = []
    for s in args.samples:
        p = Path(s)
        if not p.is_absolute():
            p = _ROOT / p
        if not p.is_file():
            print(f"[scan] 找不到文件: {p}")
            sys.exit(1)
        paths.append(p)

    print("[scan] 启动 MATLAB Engine…")
    eng = matlab.engine.start_matlab()
    eng.addpath(MATLAB_DIR, nargout=0)
    eng.addpath(GAIL_ROOT, nargout=0)
    eng.addpath(EXPERT_DIR, nargout=0)
    print("[scan] MATLAB 就绪")

    summaries = []
    try:
        for path in paths:
            summaries.append(scan_file(eng, path))
    finally:
        eng.quit()

    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    for s in summaries:
        print(
            f"  {Path(s['file']).name}: "
            f"ok {s['n_ok']}/{s['n_nonzero']}  fail {s['n_fail']}"
        )
        if s["n_ok"] >= 300:
            print("    → 已够固定取 300 局（无需加采）")
        elif s["n_ok"] > 0:
            print(f"    → 距 300 还差 {300 - s['n_ok']}；建议加采后再预筛")
        else:
            print("    → 全部失败，请检查 warm-start / 路径配置")


if __name__ == "__main__":
    main()
