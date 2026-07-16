"""
precompute_expert_terminals.py — 对专家库每条 stage2 轨迹末端跑 stage3 PN，
缓存 R1/+100，供 load_expert_buffer 写入最后一步 reward。

用法：
    E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/precompute_expert_terminals.py
    ... --expert E:/matlab/result_0702/expert_lib_1000.mat
    ... --out rl_td3/expert_terminal_cache.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matlab.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evasion_env import (
    BONUS_BOTH,
    EXPERT_DIR,
    GAIL_ROOT,
    KILL_R,
    MATLAB_DIR,
    STRIKE_OK,
    terminal_R1,
)
from load_expert_buffer import (
    DEFAULT_EXPERT_PATH,
    _get_transition_field,
    _iter_trajectories,
    _load_results,
    _traj_n_steps,
)

DEFAULT_OUT = str(Path(__file__).parent / "expert_terminal_cache.npz")


def _as_Nx18(sabs: np.ndarray) -> np.ndarray:
    sabs = np.asarray(sabs, dtype=np.float64)
    if sabs.ndim != 2:
        raise ValueError(f"state_abs 维数异常: {sabs.shape}")
    if sabs.shape[1] == 18:
        return sabs
    if sabs.shape[0] == 18:
        return sabs.T
    raise ValueError(f"state_abs 形状无法解释为 (N,18): {sabs.shape}")


def _as_Nx13(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2:
        raise ValueError(f"state 维数异常: {state.shape}")
    if state.shape[1] == 13:
        return state
    if state.shape[0] == 13:
        return state.T
    raise ValueError(f"state 形状无法解释为 (N,13): {state.shape}")


def main() -> None:
    p = argparse.ArgumentParser(description="预计算专家轨迹终端 R1/+100")
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT_PATH)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--max-traj", type=int, default=None, help="最多处理条数（调试用）")
    p.add_argument(
        "--dt",
        type=float,
        default=1e-2,
        help="stage3 积分步长（默认 1e-2，比在线 1e-3 快约 10 倍；要更高保真可改 1e-3）",
    )
    args = p.parse_args()

    mat_path = Path(args.expert)
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    print(f"[precompute] 加载专家库: {mat_path}")
    results = _load_results(str(mat_path))
    trajs = _iter_trajectories(results)

    print("[precompute] 启动 MATLAB Engine…")
    eng = matlab.engine.start_matlab()
    eng.addpath(MATLAB_DIR, nargout=0)
    eng.addpath(GAIL_ROOT, nargout=0)
    eng.addpath(EXPERT_DIR, nargout=0)

    n_steps_list = []
    r_minT_list = []
    R1_list = []
    bonus_list = []
    r_min1_list = []
    r_min2_list = []
    ok_list = []

    n_done = 0
    for i, traj in enumerate(trajs):
        n_k = _traj_n_steps(traj)
        if n_k <= 0:
            continue
        if args.max_traj is not None and n_done >= args.max_traj:
            break

        sabs = _get_transition_field(traj, "state_abs")
        state = _get_transition_field(traj, "state")
        if sabs is None or state is None:
            print(f"[precompute] 跳过 traj#{i}: 缺少 state/state_abs")
            continue

        sabs_arr = _as_Nx18(sabs)
        state_arr = _as_Nx13(state)
        n_use = min(n_k, len(sabs_arr), len(state_arr))
        if n_use <= 0:
            continue

        r_min1 = float(np.min(state_arr[:n_use, 0]))
        r_min2 = float(np.min(state_arr[:n_use, 4]))
        last_abs = sabs_arr[n_use - 1]

        # Tev≈n_use*0.1，stage3 从规避末时刻起步
        t0 = float(n_use) * 0.1
        sabs_ml = matlab.double(last_abs.tolist())
        cfg = {"dt": float(args.dt)}
        r_minT_ml, reached_ml, diverged_ml, _n_ml = eng.stage3_rollout(
            sabs_ml, t0, cfg, nargout=4
        )
        r_minT = float(r_minT_ml)
        R1 = terminal_R1(r_minT)
        evade_ok = (r_min1 >= KILL_R) and (r_min2 >= KILL_R)
        strike_ok = r_minT <= STRIKE_OK
        bonus = BONUS_BOTH if (evade_ok and strike_ok) else 0.0

        n_steps_list.append(n_use)
        r_minT_list.append(r_minT)
        R1_list.append(R1)
        bonus_list.append(bonus)
        r_min1_list.append(r_min1)
        r_min2_list.append(r_min2)
        ok_list.append(1 if (evade_ok and strike_ok) else 0)
        n_done += 1

        if n_done % 20 == 0 or n_done == 1:
            print(
                f"  [{n_done}] traj#{i} n={n_use}  r_minT={r_minT:.3f}  "
                f"R1={R1:.2f} bonus={bonus:.0f}  "
                f"reached={bool(reached_ml)} diverged={bool(diverged_ml)}"
            )

    eng.quit()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        expert_path=str(mat_path.resolve()),
        n_steps=np.asarray(n_steps_list, dtype=np.int32),
        r_minT=np.asarray(r_minT_list, dtype=np.float64),
        R1=np.asarray(R1_list, dtype=np.float64),
        bonus=np.asarray(bonus_list, dtype=np.float64),
        r_min1=np.asarray(r_min1_list, dtype=np.float64),
        r_min2=np.asarray(r_min2_list, dtype=np.float64),
        both_ok=np.asarray(ok_list, dtype=np.int8),
    )

    R1_a = np.asarray(R1_list)
    bonus_a = np.asarray(bonus_list)
    rT_a = np.asarray(r_minT_list)
    print(f"\n[precompute] 完成 {n_done} 条轨迹 → {out}")
    print(f"  r_minT  med={np.median(rT_a):.3f}  ≤10m={(rT_a<=10).mean()*100:.1f}%")
    print(f"  R1      med={np.median(R1_a):.2f}  >0={(R1_a>0).mean()*100:.1f}%")
    print(f"  bonus>0 比例={(bonus_a>0).mean()*100:.1f}%")
    print(f"  终端增量 med={np.median(R1_a+bonus_a):.2f}")


if __name__ == "__main__":
    main()
