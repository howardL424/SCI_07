"""
MC 落盘 + 分层挑选代表轨迹（不画图，只存画图数据）。

落盘范围：
  - stage2：位置 / 惯性加速度 / RL 动作参数（与原先一致）
  - stage3：PN 段降采样位置与加速度（约 0.1 s）；无 stage3 时为空数组
动作参数 gamma/wT/wE 仍仅 stage2。

分层（各至多 1 条）:
  1) both_ok：r_minT 最接近成功子集中位数
  2) evade_ok 且打击失败：r_minT ∈ [15, 50] m（无则取该区间外最近候选）
  3) B 型：若前两档均为 A 型（段内动作近乎常数），再从全库挑 1 条 B 型

用法（仓库根目录）:
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe ^
    rl_td3/tools/dump_mc_representative_trajs.py ^
    --model rl_td3/checkpoints/training_300k_td3bc_0711/best/best_model ^
    --action-mapping td3_box_log_v1 --obs-mode raw ^
    --episodes 100 --seed 0

产物默认:
  {out-dir}/mc_summary.json
  {out-dir}/mc_episode_index.npz
  {out-dir}/selected_trajs.npz
  {out-dir}/selected_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import TD3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_mapping import (  # noqa: E402
    ACTION_MAPPING_TD3_BOX,
    action_norm_to_phys,
)
from evasion_env import EvasionEnv, KILL_R, STRIKE_OK  # noqa: E402
from td3_bc import TD3BC  # noqa: E402

BOUND_THR = 0.99
TYPE_A_STD = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MC 落盘并分层挑选代表轨迹")
    p.add_argument(
        "--model",
        type=str,
        default=str(
            ROOT / "checkpoints" / "training_300k_td3bc_0711" / "best" / "best_model"
        ),
    )
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--action-mapping",
        type=str,
        default=ACTION_MAPPING_TD3_BOX,
        help="0711 请用 td3_box_log_v1；新训默认 ratio_log_box_v1",
    )
    p.add_argument(
        "--obs-mode",
        type=str,
        choices=("raw", "scaled"),
        default="raw",
        help="raw=旧模型物理观测(0711)；scaled=OBS_SCALE 归一化(新训)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="默认：{model_dir}/rep_trajs_seed{seed}",
    )
    p.add_argument("--fail-rminT-lo", type=float, default=15.0)
    p.add_argument("--fail-rminT-hi", type=float, default=50.0)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟：episodes=8，写出 smoke 子目录",
    )
    return p.parse_args()


def _body_to_inertial_accel(ay: float, az: float, theta: float, psi: float, V: float) -> np.ndarray:
    """由法向指令 (a_y,a_z) 与姿态还原惯性系加速度（V 常值运动学）。"""
    if abs(V) < 1e-6:
        return np.zeros(3, dtype=np.float64)
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    cos_ps, sin_ps = np.cos(psi), np.sin(psi)
    thdot = ay / V
    cth = cos_th if abs(cos_th) > 1e-8 else (1e-8 if cos_th >= 0 else -1e-8)
    psidot = -az / (V * cth)
    ax = V * (-sin_th * cos_ps * thdot - cos_th * sin_ps * psidot)
    ay_i = V * (cos_th * thdot)
    az_i = V * (sin_th * sin_ps * thdot - cos_th * cos_ps * psidot)
    return np.array([ax, ay_i, az_i], dtype=np.float64)


def _classify_action_type(action_norm: np.ndarray) -> dict[str, Any]:
    an = np.asarray(action_norm, dtype=np.float64)
    if an.ndim != 2 or an.shape[0] == 0:
        return {
            "action_type": "unknown",
            "std_n": [np.nan, np.nan, np.nan],
            "bound_frac": [np.nan, np.nan, np.nan],
            "is_type_a": False,
            "is_type_b": False,
        }
    std_n = an.std(axis=0)
    bound_frac = (np.abs(an) >= BOUND_THR).mean(axis=0)
    is_type_a = bool(np.all(std_n < TYPE_A_STD))
    is_type_b = (not is_type_a) and bool(np.all(bound_frac >= 0.99))
    # 若未全贴边但仍有明显段内翻转，也标 B（宽松）
    if not is_type_a and not is_type_b and float(std_n.max()) >= 0.1:
        is_type_b = True
    return {
        "action_type": "A" if is_type_a else ("B" if is_type_b else "other"),
        "std_n": std_n.tolist(),
        "bound_frac": bound_frac.tolist(),
        "is_type_a": is_type_a,
        "is_type_b": is_type_b,
    }


def run_episode_record(
    env: EvasionEnv,
    model: TD3,
    *,
    seed: int | None,
    action_mapping: str,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    s0 = np.asarray(env._state_abs, dtype=np.float64).ravel().copy()

    t_list = [0.0]
    pos_M = [s0[0:3].copy()]
    pos_D1 = [s0[6:9].copy()]
    pos_D2 = [s0[12:15].copy()]
    att_M = [s0[3:6].copy()]  # theta, psi, V

    action_norm_l: list[np.ndarray] = []
    gamma_l: list[float] = []
    wT_l: list[float] = []
    wE_l: list[float] = []
    a_body_l: list[np.ndarray] = []
    a_body_raw_l: list[np.ndarray] = []
    a_inertial_l: list[np.ndarray] = []
    r_MD1_l: list[float] = []
    r_MD2_l: list[float] = []
    r_MT_l: list[float] = []

    total_reward = 0.0
    r_MD1_min = np.inf
    r_MD2_min = np.inf
    r_minT = np.nan
    intercepted = False
    evade_ok = False
    strike_ok = False
    stage2_done = False
    diverged = False
    reached_pn = False
    n_steps_stage3 = 0
    stage3_traj: dict[str, np.ndarray] | None = None
    step = 0

    terminated = False
    while not terminated:
        s_before = np.asarray(env._state_abs, dtype=np.float64).ravel()
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64).ravel()
        gamma, wT, wE = action_norm_to_phys(action, version=action_mapping)
        g = float(np.asarray(gamma).ravel()[0])
        wt = float(np.asarray(wT).ravel()[0])
        we = float(np.asarray(wE).ravel()[0])

        obs, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        step += 1

        a_sat = np.asarray(info.get("a_M_sat", [np.nan, np.nan]), dtype=np.float64).ravel()
        a_raw = np.asarray(info.get("a_M_raw", a_sat), dtype=np.float64).ravel()
        a_i = _body_to_inertial_accel(
            float(a_sat[0]), float(a_sat[1]), float(s_before[3]), float(s_before[4]), float(s_before[5])
        )

        s_after = np.asarray(env._state_abs, dtype=np.float64).ravel()
        t_list.append(float(info["t"]))
        pos_M.append(s_after[0:3].copy())
        pos_D1.append(s_after[6:9].copy())
        pos_D2.append(s_after[12:15].copy())
        att_M.append(s_after[3:6].copy())

        action_norm_l.append(action.copy())
        gamma_l.append(g)
        wT_l.append(wt)
        wE_l.append(we)
        a_body_l.append(a_sat[:2].copy())
        a_body_raw_l.append(a_raw[:2].copy())
        a_inertial_l.append(a_i)
        r_MD1_l.append(float(info["r_MD1"]))
        r_MD2_l.append(float(info["r_MD2"]))
        r_MT_l.append(float(info["r_MT"]))

        r_MD1_min = min(r_MD1_min, float(info["r_MD1"]))
        r_MD2_min = min(r_MD2_min, float(info["r_MD2"]))
        intercepted = intercepted or bool(info["intercepted"])
        stage2_done = stage2_done or bool(info.get("stage2_done", False))
        if np.isfinite(info.get("r_minT", np.nan)):
            r_minT = float(info["r_minT"])
            evade_ok = bool(info.get("evade_ok", False))
            strike_ok = bool(info.get("strike_ok", False))
            diverged = bool(info.get("diverged", False))
            reached_pn = bool(info.get("reached_pn", False))
            n_steps_stage3 = int(info.get("n_steps_stage3", 0))
            if isinstance(info.get("stage3_traj"), dict):
                stage3_traj = info["stage3_traj"]

    if not evade_ok:
        evade_ok = (r_MD1_min >= KILL_R) and (r_MD2_min >= KILL_R)
    if not strike_ok and np.isfinite(r_minT):
        strike_ok = r_minT <= STRIKE_OK

    an = np.asarray(action_norm_l, dtype=np.float64) if action_norm_l else np.zeros((0, 3))
    cls = _classify_action_type(an)

    s3 = _normalize_stage3_traj(stage3_traj)
    a_s3_body = s3["a_M_body"]
    a_s3_inertial = np.zeros((a_s3_body.shape[0], 3), dtype=np.float64)
    for i in range(a_s3_body.shape[0]):
        a_s3_inertial[i] = _body_to_inertial_accel(
            float(a_s3_body[i, 0]),
            float(a_s3_body[i, 1]),
            float(s3["att_M"][i, 0]),
            float(s3["att_M"][i, 1]),
            float(s3["att_M"][i, 2]),
        )

    return {
        "steps": step,
        "total_reward": total_reward,
        "r_MD1_min": float(r_MD1_min),
        "r_MD2_min": float(r_MD2_min),
        "r_minT": float(r_minT) if np.isfinite(r_minT) else float("nan"),
        "intercepted": bool(intercepted),
        "evade_ok": bool(evade_ok),
        "strike_ok": bool(strike_ok),
        "both_ok": bool(evade_ok and strike_ok),
        "stage2_done": bool(stage2_done),
        "diverged": bool(diverged),
        "reached_pn": bool(reached_pn),
        "n_steps_stage3": int(n_steps_stage3),
        **cls,
        "t": np.asarray(t_list, dtype=np.float64),
        "pos_M": np.asarray(pos_M, dtype=np.float64),
        "pos_D1": np.asarray(pos_D1, dtype=np.float64),
        "pos_D2": np.asarray(pos_D2, dtype=np.float64),
        "att_M": np.asarray(att_M, dtype=np.float64),
        "action_norm": an,
        "gamma": np.asarray(gamma_l, dtype=np.float64),
        "wT": np.asarray(wT_l, dtype=np.float64),
        "wE": np.asarray(wE_l, dtype=np.float64),
        "a_M_body": np.asarray(a_body_l, dtype=np.float64),
        "a_M_body_raw": np.asarray(a_body_raw_l, dtype=np.float64),
        "a_M_inertial": np.asarray(a_inertial_l, dtype=np.float64),
        "r_MD1": np.asarray(r_MD1_l, dtype=np.float64),
        "r_MD2": np.asarray(r_MD2_l, dtype=np.float64),
        "r_MT": np.asarray(r_MT_l, dtype=np.float64),
        "t_s3": s3["t"],
        "pos_M_s3": s3["pos_M"],
        "pos_D1_s3": s3["pos_D1"],
        "pos_D2_s3": s3["pos_D2"],
        "att_M_s3": s3["att_M"],
        "a_M_body_s3": a_s3_body,
        "a_M_inertial_s3": a_s3_inertial,
        "r_MD1_s3": s3["r_MD1"],
        "r_MD2_s3": s3["r_MD2"],
        "r_MT_s3": s3["r_MT"],
    }


def _normalize_stage3_traj(traj: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    empty = {
        "t": np.zeros(0, dtype=np.float64),
        "pos_M": np.zeros((0, 3), dtype=np.float64),
        "pos_D1": np.zeros((0, 3), dtype=np.float64),
        "pos_D2": np.zeros((0, 3), dtype=np.float64),
        "att_M": np.zeros((0, 3), dtype=np.float64),
        "a_M_body": np.zeros((0, 2), dtype=np.float64),
        "r_MT": np.zeros(0, dtype=np.float64),
        "r_MD1": np.zeros(0, dtype=np.float64),
        "r_MD2": np.zeros(0, dtype=np.float64),
    }
    if not isinstance(traj, dict):
        return empty
    out = dict(empty)
    for key in empty:
        if key not in traj:
            continue
        arr = np.asarray(traj[key], dtype=np.float64)
        if key.startswith("pos_") or key == "att_M":
            arr = arr.reshape(-1, 3) if arr.size else empty[key]
        elif key == "a_M_body":
            arr = arr.reshape(-1, 2) if arr.size else empty[key]
        else:
            arr = arr.reshape(-1)
        out[key] = arr
    return out


def _summary_row(ep: int, rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "ep": ep,
        "steps": int(rec["steps"]),
        "total_reward": float(rec["total_reward"]),
        "r_MD1_min": float(rec["r_MD1_min"]),
        "r_MD2_min": float(rec["r_MD2_min"]),
        "r_minT": float(rec["r_minT"]),
        "intercepted": bool(rec["intercepted"]),
        "evade_ok": bool(rec["evade_ok"]),
        "strike_ok": bool(rec["strike_ok"]),
        "both_ok": bool(rec["both_ok"]),
        "diverged": bool(rec.get("diverged", False)),
        "reached_pn": bool(rec.get("reached_pn", False)),
        "n_steps_stage3": int(rec.get("n_steps_stage3", 0)),
        "action_type": rec["action_type"],
        "is_type_a": bool(rec["is_type_a"]),
        "is_type_b": bool(rec["is_type_b"]),
        "std_n": rec["std_n"],
        "bound_frac": rec["bound_frac"],
    }


def pick_success_median(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [r for r in rows if r["both_ok"] and np.isfinite(r["r_minT"])]
    if not ok:
        return None
    med = float(np.median([r["r_minT"] for r in ok]))
    best = min(ok, key=lambda r: (abs(r["r_minT"] - med), r["ep"]))
    return {
        "tier": "success_median",
        "ep": int(best["ep"]),
        "reason": f"both_ok, r_minT={best['r_minT']:.4f} closest to median={med:.4f} (n={len(ok)})",
        "target_median_r_minT": med,
        "row": best,
    }


def pick_evade_fail_strike(
    rows: list[dict[str, Any]],
    lo: float,
    hi: float,
    *,
    exclude_eps: set[int],
) -> dict[str, Any] | None:
    cands = [
        r
        for r in rows
        if r["evade_ok"]
        and (not r["strike_ok"])
        and (not r["intercepted"])
        and np.isfinite(r["r_minT"])
        and int(r["ep"]) not in exclude_eps
    ]
    if not cands:
        return None
    in_band = [r for r in cands if lo <= r["r_minT"] <= hi]
    if in_band:
        # 取区间中点最近
        mid = 0.5 * (lo + hi)
        best = min(in_band, key=lambda r: (abs(r["r_minT"] - mid), r["ep"]))
        reason = (
            f"evade_ok & strike fail, r_minT={best['r_minT']:.4f} in [{lo},{hi}] "
            f"(n_band={len(in_band)})"
        )
        fallback = False
    else:
        # 落到区间外：优先 >STRIKE_OK 的失败里最接近 [lo,hi] 中点
        mid = 0.5 * (lo + hi)
        best = min(cands, key=lambda r: (abs(r["r_minT"] - mid), r["ep"]))
        reason = (
            f"FALLBACK: no r_minT in [{lo},{hi}]; closest evade-fail "
            f"r_minT={best['r_minT']:.4f} (n_fail={len(cands)})"
        )
        fallback = True
    return {
        "tier": "evade_fail_strike",
        "ep": int(best["ep"]),
        "reason": reason,
        "fallback": fallback,
        "row": best,
    }


def pick_type_b(
    rows: list[dict[str, Any]],
    *,
    exclude_eps: set[int],
    prefer_both_ok: bool = True,
) -> dict[str, Any] | None:
    b = [r for r in rows if r["is_type_b"] and int(r["ep"]) not in exclude_eps]
    if not b:
        return None
    if prefer_both_ok:
        prefer = [r for r in b if r["both_ok"]]
        pool = prefer if prefer else b
    else:
        pool = b
    # 优先段内 std 更大、步数更接近探测里的短轨(~11)或与已选区分明显
    best = max(pool, key=lambda r: (float(np.max(r["std_n"])), -abs(r["steps"] - 11), -r["ep"]))
    return {
        "tier": "type_b",
        "ep": int(best["ep"]),
        "reason": (
            f"type B action (std_n={best['std_n']}), steps={best['steps']}, "
            f"both_ok={best['both_ok']}"
        ),
        "row": best,
    }


def select_tiers(
    rows: list[dict[str, Any]],
    *,
    fail_lo: float,
    fail_hi: float,
) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []
    used: set[int] = set()

    s = pick_success_median(rows)
    if s is not None:
        picks.append(s)
        used.add(int(s["ep"]))

    f = pick_evade_fail_strike(rows, fail_lo, fail_hi, exclude_eps=used)
    if f is not None:
        picks.append(f)
        used.add(int(f["ep"]))

    need_b = True
    if picks:
        need_b = all(bool(p["row"]["is_type_a"]) for p in picks)
    if need_b:
        b = pick_type_b(rows, exclude_eps=used)
        if b is not None:
            picks.append(b)
            used.add(int(b["ep"]))
        else:
            picks.append(
                {
                    "tier": "type_b",
                    "ep": None,
                    "reason": "requested but no type-B episode found in MC pool",
                    "row": None,
                }
            )
    else:
        picks.append(
            {
                "tier": "type_b",
                "ep": None,
                "reason": "skipped: at least one prior pick is already non-A",
                "row": None,
            }
        )
    return picks


def _pack_selected(prefix: str, rec: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in (
        "t",
        "pos_M",
        "pos_D1",
        "pos_D2",
        "att_M",
        "action_norm",
        "gamma",
        "wT",
        "wE",
        "a_M_body",
        "a_M_body_raw",
        "a_M_inertial",
        "r_MD1",
        "r_MD2",
        "r_MT",
        "t_s3",
        "pos_M_s3",
        "pos_D1_s3",
        "pos_D2_s3",
        "att_M_s3",
        "a_M_body_s3",
        "a_M_inertial_s3",
        "r_MD1_s3",
        "r_MD2_s3",
        "r_MT_s3",
    ):
        out[f"{prefix}/{key}"] = np.asarray(rec[key])
    # 逐步时间（与动作/加速度对齐，长度=steps）；动作仍仅 stage2
    t_full = np.asarray(rec["t"], dtype=np.float64)
    out[f"{prefix}/t_step"] = t_full[1:] if t_full.size else t_full
    out[f"{prefix}/t_switch"] = np.asarray(
        [float(t_full[-1]) if t_full.size else np.nan], dtype=np.float64
    )
    out[f"{prefix}/n_steps_stage3"] = np.asarray(
        [int(rec.get("n_steps_stage3", 0))], dtype=np.int32
    )
    out[f"{prefix}/diverged"] = np.asarray(
        [bool(rec.get("diverged", False))], dtype=np.bool_
    )
    out[f"{prefix}/reached_pn"] = np.asarray(
        [bool(rec.get("reached_pn", False))], dtype=np.bool_
    )
    a_body = np.asarray(rec["a_M_body"], dtype=np.float64)
    if a_body.ndim == 2 and a_body.shape[0]:
        out[f"{prefix}/a_M_norm"] = np.linalg.norm(a_body, axis=1)
        out[f"{prefix}/a_M_inertial_norm"] = np.linalg.norm(
            np.asarray(rec["a_M_inertial"], dtype=np.float64), axis=1
        )
    else:
        out[f"{prefix}/a_M_norm"] = np.zeros(0, dtype=np.float64)
        out[f"{prefix}/a_M_inertial_norm"] = np.zeros(0, dtype=np.float64)
    return out


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.episodes = min(args.episodes, 8)

    model_path = args.model
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    model_zip = model_path + ".zip"
    if not Path(model_zip).exists():
        raise SystemExit(f"[dump] 找不到模型: {model_zip}")

    model_dir = Path(model_path).resolve().parent
    if Path(model_path).name == "best_model":
        # .../best/best_model → 跑目录用 checkpoints/.../
        model_dir = model_dir.parent
    out_dir = Path(args.out_dir) if args.out_dir else (
        model_dir / f"rep_trajs_seed{args.seed}{'_smoke' if args.smoke else ''}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    print(f"[dump] model={model_zip}")
    print(f"[dump] action_mapping={args.action_mapping}  obs_mode={args.obs_mode}")
    print(f"[dump] episodes={args.episodes}  seed={args.seed}")
    print(f"[dump] out_dir={out_dir}")

    env = EvasionEnv(
        action_mapping=args.action_mapping,
        obs_normalize=(args.obs_mode == "scaled"),
        record_stage3_traj=True,
    )
    try:
        model = TD3BC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for ep in range(args.episodes):
        ep_seed = args.seed if ep == 0 else None
        rec = run_episode_record(
            env, model, seed=ep_seed, action_mapping=args.action_mapping
        )
        records.append(rec)
        row = _summary_row(ep, rec)
        rows.append(row)
        if (ep + 1) % 10 == 0 or args.smoke:
            ok_n = sum(1 for r in rows if r["both_ok"])
            rT = row["r_minT"]
            rT_s = f"{rT:.2f}" if np.isfinite(rT) else "nan"
            print(
                f"  [ep {ep+1:3d}/{args.episodes}] both_ok={100*ok_n/(ep+1):.1f}% "
                f"type={row['action_type']} steps={row['steps']} "
                f"r_minT={rT_s}m"
            )

    env.close()

    n = len(rows)
    both_ok = sum(1 for r in rows if r["both_ok"])
    evade_ok = sum(1 for r in rows if r["evade_ok"])
    strike_ok = sum(1 for r in rows if r["strike_ok"])
    intercepted = sum(1 for r in rows if r["intercepted"])
    n_a = sum(1 for r in rows if r["is_type_a"])
    n_b = sum(1 for r in rows if r["is_type_b"])

    summary = {
        "model": model_zip,
        "action_mapping": args.action_mapping,
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "episodes": n,
        "rates": {
            "evade_ok": evade_ok / n,
            "strike_ok": strike_ok / n,
            "both_ok": both_ok / n,
            "intercepted": intercepted / n,
            "type_a": n_a / n,
            "type_b": n_b / n,
        },
        "counts": {
            "evade_ok": evade_ok,
            "strike_ok": strike_ok,
            "both_ok": both_ok,
            "intercepted": intercepted,
            "type_a": n_a,
            "type_b": n_b,
        },
        "selection_rules": {
            "success": "both_ok, r_minT closest to median of successes",
            "evade_fail_strike": f"evade_ok & not strike_ok, prefer r_minT in [{args.fail_rminT_lo},{args.fail_rminT_hi}]",
            "type_b": "only if all prior picks are type A; else skipped",
            "type_a_def": f"all action_norm std < {TYPE_A_STD}",
            "type_b_def": "not type A, and (all |a|>=0.99 or max std_n>=0.1)",
        },
        "field_notes": {
            "pos_*": "stage2 only: N+1 x 3, [x,y,z] after reset then each step (inertial)",
            "pos_*_s3": "stage3 PN traj (downsampled ~0.1s): M x 3; may be empty if intercepted",
            "plot_convention_expert": "plot3(x, z, y) as in generate_ps_expert_library.plot_trajectory",
            "a_M_body": "stage2 N x 2, saturated lateral cmd [a_y, a_z] used by updateState",
            "a_M_inertial": "stage2 N x 3, reconstructed inertial accel from (a_y,a_z,theta,psi,V)",
            "a_M_inertial_s3": "stage3 M x 3, same reconstruction from PN a_M + att_M",
            "gamma_wT_wE": "stage2 only N, physical weights via action_mapping",
            "t": "stage2 N+1 (includes t=0); t_step length N aligns with actions/accel",
            "t_s3": "stage3 times continuing from stage2 end (same clock as t)",
            "t_switch": "stage2 end time; vertical marker between stage2 and stage3",
            "trajectory_scope": "3D/accel plots should stitch stage2 + stage3; actions remain stage2-only",
        },
    }

    picks = select_tiers(rows, fail_lo=args.fail_rminT_lo, fail_hi=args.fail_rminT_hi)
    selected_meta: list[dict[str, Any]] = []
    packed: dict[str, np.ndarray] = {}

    for p in picks:
        ep = p.get("ep")
        meta = {
            "tier": p["tier"],
            "ep": ep,
            "reason": p["reason"],
            "fallback": bool(p.get("fallback", False)),
        }
        if ep is None or p.get("row") is None:
            selected_meta.append(meta)
            print(f"[select] {p['tier']}: (none) — {p['reason']}")
            continue
        rec = records[int(ep)]
        meta.update(
            {
                "steps": int(rec["steps"]),
                "both_ok": bool(rec["both_ok"]),
                "evade_ok": bool(rec["evade_ok"]),
                "strike_ok": bool(rec["strike_ok"]),
                "intercepted": bool(rec["intercepted"]),
                "diverged": bool(rec.get("diverged", False)),
                "reached_pn": bool(rec.get("reached_pn", False)),
                "n_steps_stage3": int(rec.get("n_steps_stage3", 0)),
                "r_MD1_min": float(rec["r_MD1_min"]),
                "r_MD2_min": float(rec["r_MD2_min"]),
                "r_minT": float(rec["r_minT"]),
                "action_type": rec["action_type"],
                "std_n": rec["std_n"],
                "bound_frac": rec["bound_frac"],
                "total_reward": float(rec["total_reward"]),
            }
        )
        selected_meta.append(meta)
        prefix = f"{p['tier']}_ep{int(ep):03d}"
        packed.update(_pack_selected(prefix, rec))
        packed[f"{prefix}/ep"] = np.asarray([int(ep)], dtype=np.int32)
        print(
            f"[select] {p['tier']}: ep={ep} type={rec['action_type']} "
            f"steps={rec['steps']} r_minT={rec['r_minT']:.3f} — {p['reason']}"
        )

    summary["selected"] = selected_meta

    # 全量 episode 索引（不含逐步大数组）
    index = {
        "ep": np.asarray([r["ep"] for r in rows], dtype=np.int32),
        "steps": np.asarray([r["steps"] for r in rows], dtype=np.int32),
        "total_reward": np.asarray([r["total_reward"] for r in rows], dtype=np.float64),
        "r_MD1_min": np.asarray([r["r_MD1_min"] for r in rows], dtype=np.float64),
        "r_MD2_min": np.asarray([r["r_MD2_min"] for r in rows], dtype=np.float64),
        "r_minT": np.asarray([r["r_minT"] for r in rows], dtype=np.float64),
        "intercepted": np.asarray([r["intercepted"] for r in rows], dtype=np.bool_),
        "evade_ok": np.asarray([r["evade_ok"] for r in rows], dtype=np.bool_),
        "strike_ok": np.asarray([r["strike_ok"] for r in rows], dtype=np.bool_),
        "both_ok": np.asarray([r["both_ok"] for r in rows], dtype=np.bool_),
        "diverged": np.asarray([r.get("diverged", False) for r in rows], dtype=np.bool_),
        "reached_pn": np.asarray([r.get("reached_pn", False) for r in rows], dtype=np.bool_),
        "n_steps_stage3": np.asarray(
            [r.get("n_steps_stage3", 0) for r in rows], dtype=np.int32
        ),
        "is_type_a": np.asarray([r["is_type_a"] for r in rows], dtype=np.bool_),
        "is_type_b": np.asarray([r["is_type_b"] for r in rows], dtype=np.bool_),
        "std_n": np.asarray([r["std_n"] for r in rows], dtype=np.float64),
        "bound_frac": np.asarray([r["bound_frac"] for r in rows], dtype=np.float64),
    }

    summary_path = out_dir / "mc_summary.json"
    meta_path = out_dir / "selected_meta.json"
    index_path = out_dir / "mc_episode_index.npz"
    traj_path = out_dir / "selected_trajs.npz"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(selected_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(index_path, **index)
    if packed:
        np.savez_compressed(traj_path, **packed)
    else:
        np.savez_compressed(traj_path, _empty=np.array([0]))

    print("\n[dump] MC rates:")
    for k, v in summary["rates"].items():
        print(f"  {k:12s} {100*v:5.1f}%")
    print(f"[dump] wrote {summary_path}")
    print(f"[dump] wrote {meta_path}")
    print(f"[dump] wrote {index_path}")
    print(f"[dump] wrote {traj_path}")
    print("[dump] 未画图；下一步用 selected_trajs.npz + selected_meta.json 绘图即可。")


if __name__ == "__main__":
    main()
