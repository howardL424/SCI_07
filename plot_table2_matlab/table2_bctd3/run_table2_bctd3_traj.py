"""
run_table2_bctd3_traj.py — Table 2 初值 + BC-TD3 整轨求解（Python 驱动，MATLAB Engine）。

当 MATLAB 侧 py 接口不可用时，由 main_table2_bctd3.m 设 UsePythonDriver=true 调用本脚本。

用法:
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe \\
    plot_table2_matlab/table2_bctd3/run_table2_bctd3_traj.py \\
    --model rl_td3/checkpoints/training_500k_20260728_1455/best/best_model \\
    --out plot_table2_matlab/table2_bctd3/record_table2_bctd3.mat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matlab.engine
import numpy as np
import scipy.io

_THIS = Path(__file__).resolve().parent          # .../plot_table2_matlab/table2_bctd3
_PLOT = _THIS.parent                             # .../plot_table2_matlab
_GAIL = _PLOT.parent                             # .../GAIL
_RL_TD3 = _GAIL / "rl_td3"
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_RL_TD3))

from action_mapping import action_norm_to_phys  # noqa: E402
from obs_norm import normalize_obs  # noqa: E402
from td3_policy_bridge import Td3PolicyBridge, _normalize_model_path  # noqa: E402


def _fact_get(fact, name: str):
    if isinstance(fact, dict):
        return fact[name]
    return getattr(fact, name)

# Table 2
M0 = np.array([10000.0, 6000.0, 3500.0, np.deg2rad(-20), np.deg2rad(155), 500.0])
D10 = np.array([1000.0, 0.0, 0.0, np.deg2rad(40), np.deg2rad(-20), 600.0])
D20 = np.array([-1000.0, 0.0, 0.0, np.deg2rad(40), np.deg2rad(-20), 600.0])
T_POS = np.array([0.0, 0.0, 0.0])


def _ml_vec(a: np.ndarray):
    return matlab.double(np.asarray(a, dtype=np.float64).ravel().tolist())


def _as_col(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=str(
            _RL_TD3
            / "checkpoints"
            / "training_500k_20260728_1455"
            / "best"
            / "best_model"
        ),
    )
    p.add_argument(
        "--out",
        default=str(_THIS / "record_table2_bctd3.mat"),
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model_path = str(_normalize_model_path(args.model))
    bridge = Td3PolicyBridge(model_path, device=args.device)

    print("[table2] 启动 MATLAB Engine …", flush=True)
    eng = matlab.engine.start_matlab()
    eng.addpath(str(_RL_TD3 / "matlab"), nargout=0)
    eng.addpath(str(_GAIL / "ident" / "matlab"), nargout=0)
    eng.addpath(str(_GAIL), nargout=0)
    eng.addpath(str(_GAIL / "expert_trajectory"), nargout=0)

    cfg_ws = {
        "KM": 3.0,
        "KD1": 4.0,
        "KD2": 4.0,
        "dt": 1e-3,
        "t_max": 60.0,
        "r_safe": 8000.0,
        "g0": 9.81,
        "T": matlab.double(T_POS.tolist()),
        "w": 1e4,
        "gamma": 0.0,
        "wE": 1.44e3,
        "wT": 1e3,
        "dstar": 20.0,
    }
    cfg_s2 = {
        "dt": 0.1,
        "dstar": 20.0,
        "KD1": 4.0,
        "KD2": 4.0,
        "KM": 3.0,
        "g0": 9.81,
        "kill_radius": 6.0,
        "w": 1e4,
        "action_mapping_version": "ratio_log_box_v1",
        "coeff_mode": "analytic_truth",
    }

    print("[table2] stage1 warm-start …", flush=True)
    tau = eng.simulate_warmstart_trajectory(
        _ml_vec(M0), _ml_vec(D10), _ml_vec(D20), cfg_ws, nargout=1
    )
    stage = np.asarray(tau["stage"], dtype=np.float64).ravel()
    idx2 = np.flatnonzero(stage == 2)
    if idx2.size == 0:
        raise RuntimeError("warm-start 未进入 stage==2")
    i0 = int(idx2[0])
    i_e = int(idx2[-1])
    t_arr = np.asarray(tau["t"], dtype=np.float64).ravel()
    t_pen0 = float(t_arr[i0])
    Tev = max(float(t_arr[i_e] - t_arr[i0]), 1.0)

    M = np.asarray(tau["M"], dtype=np.float64)
    D1 = np.asarray(tau["D1"], dtype=np.float64)
    D2 = np.asarray(tau["D2"], dtype=np.float64)
    if M.shape[0] != 6:
        M, D1, D2 = M.T, D1.T, D2.T
    state_abs = np.concatenate([M[:, i0], D1[:, i0], D2[:, i0]])
    print(
        f"[table2] pen start t={t_pen0:.3f}s Tev={Tev:.3f}s "
        f"r_MD1={float(np.asarray(tau['r_MD1']).ravel()[i0]):.1f} "
        f"r_MD2={float(np.asarray(tau['r_MD2']).ravel()[i0]):.1f}",
        flush=True,
    )

    # stage1 降采样记录
    ds = max(1, int(round(0.1 / 1e-3)))
    idx1 = np.arange(0, i0 + 1, ds)
    r_MD1 = np.asarray(tau["r_MD1"], dtype=np.float64).ravel()
    r_MD2 = np.asarray(tau["r_MD2"], dtype=np.float64).ravel()
    r_MT = np.asarray(tau["r_MT"], dtype=np.float64).ravel()
    a_M = np.asarray(tau["a_M"], dtype=np.float64)
    if a_M.shape[0] != 2:
        a_M = a_M.T

    rec_t = [t_arr[idx1]]
    rec_pos_M = [M[0:3, idx1]]
    rec_pos_D1 = [D1[0:3, idx1]]
    rec_pos_D2 = [D2[0:3, idx1]]
    rec_a_M = [a_M[:, idx1]]
    rec_r_MT = [r_MT[idx1]]
    rec_r_MD1 = [r_MD1[idx1]]
    rec_r_MD2 = [r_MD2[idx1]]
    rec_stage = [stage[idx1]]
    rec_act = [np.full((3, idx1.size), np.nan)]
    rec_gamma = [np.full(idx1.size, np.nan)]
    rec_wT = [np.full(idx1.size, np.nan)]
    rec_wE = [np.full(idx1.size, np.nan)]

    core = eng.stage2_core_init(cfg_s2, nargout=1)
    t = 0.0
    terminated = False
    intercepted = False
    stage2_done = False
    r_min1 = np.inf
    r_min2 = np.inf
    n_s2 = 0

    print("[table2] stage2 BC-TD3 …", flush=True)
    while not terminated:
        obs_ml = eng.stage2_obs_from_abs(
            core, _ml_vec(state_abs), float(t), float(Tev), nargout=1
        )
        obs_phys = _as_col(obs_ml)
        obs_n = normalize_obs(obs_phys)
        action = bridge.predict(obs_n)

        obs_next, sabs_next, t_new, fact = eng.stage2_step(
            core,
            _ml_vec(state_abs),
            float(t),
            float(Tev),
            _ml_vec(action),
            nargout=4,
        )
        del obs_next
        state_abs = _as_col(sabs_next)
        t = float(t_new)
        n_s2 += 1

        r1 = float(_fact_get(fact, "r_MD1"))
        r2 = float(_fact_get(fact, "r_MD2"))
        rT = float(_fact_get(fact, "r_MT"))
        a_sat = _as_col(_fact_get(fact, "a_M_sat"))
        intercepted = bool(_fact_get(fact, "intercepted"))
        stage2_done = bool(_fact_get(fact, "stage2_done"))
        terminated = bool(_fact_get(fact, "terminated"))

        gamma, wT, wE = action_norm_to_phys(
            action, version="ratio_log_box_v1", w=1e4
        )
        gamma = float(np.asarray(gamma).reshape(-1)[0])
        wT = float(np.asarray(wT).reshape(-1)[0])
        wE = float(np.asarray(wE).reshape(-1)[0])

        t_abs = t_pen0 + t
        rec_t.append(np.array([t_abs]))
        rec_pos_M.append(state_abs[0:3].reshape(3, 1))
        rec_pos_D1.append(state_abs[6:9].reshape(3, 1))
        rec_pos_D2.append(state_abs[12:15].reshape(3, 1))
        rec_a_M.append(a_sat.reshape(2, 1))
        rec_r_MT.append(np.array([rT]))
        rec_r_MD1.append(np.array([r1]))
        rec_r_MD2.append(np.array([r2]))
        rec_stage.append(np.array([2.0]))
        rec_act.append(action.reshape(3, 1))
        rec_gamma.append(np.array([gamma]))
        rec_wT.append(np.array([wT]))
        rec_wE.append(np.array([wE]))

        r_min1 = min(r_min1, r1)
        r_min2 = min(r_min2, r2)

    print(
        f"[table2] stage2 done steps={n_s2} intercepted={intercepted} "
        f"stage2_done={stage2_done} r_min1={r_min1:.4f} r_min2={r_min2:.4f}",
        flush=True,
    )

    r_minT = np.nan
    if stage2_done and not intercepted:
        print("[table2] stage3 PN …", flush=True)
        cfg_s3 = {"dt": 1e-3, "traj_sample_dt": 0.1, "T": matlab.double(T_POS.tolist())}
        r_minT_ml, reached, diverged, n3, traj3 = eng.stage3_rollout(
            _ml_vec(state_abs), float(t), cfg_s3, nargout=5
        )
        r_minT = float(r_minT_ml)
        print(
            f"[table2] stage3 r_minT={r_minT:.4f} reached={bool(reached)} "
            f"diverged={bool(diverged)} steps={int(n3)}",
            flush=True,
        )
        t3 = np.asarray(
            traj3["t"] if isinstance(traj3, dict) else traj3.t, dtype=np.float64
        ).ravel()
        if t3.size:
            pos_M3 = np.asarray(
                traj3["pos_M"] if isinstance(traj3, dict) else traj3.pos_M,
                dtype=np.float64,
            )
            pos_D13 = np.asarray(
                traj3["pos_D1"] if isinstance(traj3, dict) else traj3.pos_D1,
                dtype=np.float64,
            )
            pos_D23 = np.asarray(
                traj3["pos_D2"] if isinstance(traj3, dict) else traj3.pos_D2,
                dtype=np.float64,
            )
            a_M3 = np.asarray(
                traj3["a_M"] if isinstance(traj3, dict) else traj3.a_M,
                dtype=np.float64,
            )
            if pos_M3.shape[0] != 3:
                pos_M3, pos_D13, pos_D23 = pos_M3.T, pos_D13.T, pos_D23.T
            if a_M3.ndim == 1:
                a_M3 = a_M3.reshape(2, -1)
            elif a_M3.shape[0] != 2:
                a_M3 = a_M3.T
            r_MT3 = np.asarray(
                traj3["r_MT"] if isinstance(traj3, dict) else traj3.r_MT,
                dtype=np.float64,
            ).ravel()
            r_MD13 = np.asarray(
                traj3["r_MD1"] if isinstance(traj3, dict) else traj3.r_MD1,
                dtype=np.float64,
            ).ravel()
            r_MD23 = np.asarray(
                traj3["r_MD2"] if isinstance(traj3, dict) else traj3.r_MD2,
                dtype=np.float64,
            ).ravel()
            t3_abs = t_pen0 + t3
            i0s = 1 if abs(t3_abs[0] - rec_t[-1][-1]) < 1e-9 else 0
            if i0s < t3.size:
                sel = slice(i0s, None)
                n3s = t3_abs[sel].size
                rec_t.append(t3_abs[sel])
                rec_pos_M.append(pos_M3[:, sel])
                rec_pos_D1.append(pos_D13[:, sel])
                rec_pos_D2.append(pos_D23[:, sel])
                rec_a_M.append(a_M3[:, sel])
                rec_r_MT.append(r_MT3[sel])
                rec_r_MD1.append(r_MD13[sel])
                rec_r_MD2.append(r_MD23[sel])
                rec_stage.append(np.full(n3s, 3.0))
                rec_act.append(np.full((3, n3s), np.nan))
                rec_gamma.append(np.full(n3s, np.nan))
                rec_wT.append(np.full(n3s, np.nan))
                rec_wE.append(np.full(n3s, np.nan))

    eng.quit()

    t_all = np.concatenate(rec_t).astype(np.float64)
    a_M_all = np.concatenate(rec_a_M, axis=1)
    stage_all = np.concatenate(rec_stage).astype(np.float64)

    # 规避段能耗: Energy = 0.5 * ∫ ||a_M||² dt / 1000 [kJ]（stage==2）
    idx_evade = np.flatnonzero(stage_all == 2)
    if idx_evade.size >= 2:
        a_cmd = np.sqrt(a_M_all[0, idx_evade] ** 2 + a_M_all[1, idx_evade] ** 2)
        energy = 0.5 * float(np.trapz(a_cmd**2, t_all[idx_evade])) / 1000.0
        t_pen_end = float(t_all[idx_evade[-1]])
    else:
        energy = 0.0
        t_pen_end = float(t_pen0)

    record = {
        "t": t_all,
        "pos_M": np.concatenate(rec_pos_M, axis=1),
        "pos_D1": np.concatenate(rec_pos_D1, axis=1),
        "pos_D2": np.concatenate(rec_pos_D2, axis=1),
        "a_M": a_M_all,
        "r_MT": np.concatenate(rec_r_MT).astype(np.float64),
        "r_MD1": np.concatenate(rec_r_MD1).astype(np.float64),
        "r_MD2": np.concatenate(rec_r_MD2).astype(np.float64),
        "stage": stage_all,
        "action_norm": np.concatenate(rec_act, axis=1),
        "gamma": np.concatenate(rec_gamma).astype(np.float64),
        "wT": np.concatenate(rec_wT).astype(np.float64),
        "wE": np.concatenate(rec_wE).astype(np.float64),
        "M0": M0.reshape(1, 6),
        "D10": D10.reshape(1, 6),
        "D20": D20.reshape(1, 6),
        "T": T_POS.reshape(1, 3),
        "t_pen0": float(t_pen0),
        "t_pen_end": t_pen_end,
        "Tev": float(Tev),
        "r_min1": float(r_min1),
        "r_min2": float(r_min2),
        "r_minT": float(r_minT) if np.isfinite(r_minT) else np.nan,
        "Energy": float(energy),
        "intercepted": bool(intercepted),
        "stage2_done": bool(stage2_done),
        "model_path": model_path,
        "n_stage2": int(n_s2),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(out), {"record": record}, do_compression=True)
    print(
        f"[table2] saved {out}\n"
        f"  evade_ok={r_min1 >= 6 and r_min2 >= 6} "
        f"strike_ok={np.isfinite(r_minT) and r_minT <= 10}\n"
        f"  r_min1={r_min1:.4f} r_min2={r_min2:.4f} r_minT={r_minT}\n"
        f"  Energy={energy:.4f} kJ  (t∈[{t_pen0:.3f}, {t_pen_end:.3f}] s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
