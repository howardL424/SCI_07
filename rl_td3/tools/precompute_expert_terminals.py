"""
precompute_expert_terminals.py — stage3 终端结果 sidecar（可独立于生成主库运行）。

旧库：读 expert_transitions_aligned.mat（需先 rebuild）。
新库：直接读 expert_lib 中 transitions.terminal_abs / t_end / r_min*（无需 rebuild）。

权威字段：r_minT、reached、diverged、n_steps_stage3、stage3_dt、r_min1/r_min2。
派生字段（带 reward_version）：R1、bonus、both_ok — 不可代替 r_minT。
拦截轨迹跳过 stage3，R1=bonus=0。

用法：
    # 旧库（aligned）
    E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/precompute_expert_terminals.py `
      --expert E:\\matlab\\result_0714\\expert_lib_1000.mat `
      --aligned rl_td3/data/expert_transitions_aligned.mat `
      --dt 1e-3 --out rl_td3/data/expert_terminal_cache.npz

    # 新库（phys schema，可省略 --aligned）
    ... --expert path/to/new_expert_lib.mat --dt 1e-3 --out ...

    # 断点续跑
    ... --resume
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matlab.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evasion_env import (
    EXPERT_DIR,
    GAIL_ROOT,
    MATLAB_DIR,
)
from load_expert_buffer import (
    DEFAULT_ALIGNED_PATH,
    DEFAULT_EXPERT_PATH,
    DEFAULT_TERMINAL_CACHE,
    expert_lib_has_phys_schema,
    load_aligned_transitions,
    load_terminal_facts,
)
from reward_projection import (
    REWARD_VERSION,
    project_terminal,
)

STAGE3_VERSION = "pn_rollout_v1"


def _fingerprint(path: Path) -> str:
    """可追溯指纹：resolved path + size + mtime（大文件不做全量 hash）。"""
    st = path.stat()
    raw = f"{path.resolve()}|size={st.st_size}|mtime_ns={st.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] + "|" + raw


def _load_facts(expert: Path, aligned: Path | None, prefer: str) -> dict:
    if prefer == "expert_lib" or (
        prefer == "auto" and expert_lib_has_phys_schema(str(expert))
    ):
        facts = load_terminal_facts(str(expert), aligned_path=None)
        print(f"[precompute] 源模式=expert_lib（新 schema，跳过 rebuild）")
        return facts

    if aligned is None or not aligned.exists():
        raise FileNotFoundError(
            f"旧库需要对齐数据：{aligned}\n"
            "请先运行 rl_td3/tools/rebuild_expert_data.py，或使用已含 "
            "phys_transition 的新 expert_lib。"
        )
    data = load_aligned_transitions(str(aligned))
    if data["expert_path"]:
        try:
            same = Path(data["expert_path"]).resolve() == expert.resolve()
        except OSError:
            same = data["expert_path"] == str(expert)
        if not same:
            raise ValueError(
                f"对齐数据来自 {data['expert_path']}，与 --expert {expert} 不匹配"
            )
    print(f"[precompute] 源模式=aligned（旧库派生）: {aligned}")
    return {
        "expert_path": str(expert.resolve()),
        "source_mode": "aligned",
        "n_steps": data["n_steps"],
        "intercepted": data["intercepted"],
        "terminal_abs": data["terminal_abs"],
        "t_end": data["t_end"],
        "r_min1": data["r_min1"],
        "r_min2": data["r_min2"],
        "source_idx": data.get("source_idx"),
        "schema_version": data.get("schema_version"),
        "stage2_dynamics_version": data.get("stage2_dynamics_version"),
        "aligned_path": str(aligned.resolve()),
    }


def _try_resume(out: Path, expert_fp: str, dt: float, n_use: int) -> dict | None:
    if not out.exists():
        return None
    prev = np.load(out, allow_pickle=True)
    files = set(prev.files)
    need = {"expert_fingerprint", "stage3_dt", "r_minT", "n_steps", "intercepted"}
    if not need.issubset(files):
        print(f"[precompute] 已有 {out} 但缺续跑字段，将重算")
        return None
    if str(prev["expert_fingerprint"]) != expert_fp:
        print("[precompute] 指纹不匹配，将重算")
        return None
    if float(prev["stage3_dt"]) != float(dt):
        print("[precompute] stage3_dt 不匹配，将重算")
        return None
    n_prev = len(np.asarray(prev["n_steps"]).ravel())
    if n_prev > n_use:
        print(f"[precompute] 缓存条数 {n_prev} > 目标 {n_use}，将重算")
        return None
    print(f"[precompute] 续跑：已有 {n_prev}/{n_use} 条，从下一条继续")
    return {k: prev[k] for k in prev.files}


def main() -> None:
    p = argparse.ArgumentParser(
        description="stage3 终端 sidecar：旧库读 aligned，新库直接读 expert_lib"
    )
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT_PATH)
    p.add_argument(
        "--aligned",
        type=str,
        default=DEFAULT_ALIGNED_PATH,
        help="旧库对齐 mat；新 schema 库可忽略",
    )
    p.add_argument("--out", type=str, default=DEFAULT_TERMINAL_CACHE)
    p.add_argument("--max-traj", type=int, default=None, help="最多处理条数（调试）")
    p.add_argument(
        "--dt",
        type=float,
        default=1e-3,
        help="stage3 积分步长（正式与在线一致默认 1e-3）",
    )
    p.add_argument(
        "--source",
        type=str,
        choices=("auto", "aligned", "expert_lib"),
        default="auto",
        help="auto=有 phys schema 则走专家库，否则 aligned",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="若 --out 已存在且指纹/dt 匹配，则从已算条数续跑",
    )
    args = p.parse_args()

    mat_path = Path(args.expert)
    aligned_path = Path(args.aligned) if args.aligned else None
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    prefer = args.source
    if prefer == "aligned":
        facts = _load_facts(mat_path, aligned_path, prefer="aligned")
    elif prefer == "expert_lib":
        facts = _load_facts(mat_path, aligned_path, prefer="expert_lib")
    else:
        facts = _load_facts(mat_path, aligned_path, prefer="auto")

    n_traj = len(facts["n_steps"])
    n_use = n_traj if args.max_traj is None else min(n_traj, args.max_traj)
    expert_fp = _fingerprint(mat_path)
    out = Path(args.out)

    # 续跑状态
    start_tid = 0
    n_steps_list: list[int] = []
    r_minT_list: list[float] = []
    R1_list: list[float] = []
    bonus_list: list[float] = []
    r_min1_list: list[float] = []
    r_min2_list: list[float] = []
    ok_list: list[int] = []
    intercepted_list: list[int] = []
    reached_list: list[int] = []
    diverged_list: list[int] = []
    n_s3_list: list[int] = []
    skipped_stage3 = 0

    if args.resume:
        prev = _try_resume(out, expert_fp, args.dt, n_use)
        if prev is not None:
            start_tid = len(np.asarray(prev["n_steps"]).ravel())
            n_steps_list = list(np.asarray(prev["n_steps"], dtype=np.int32).ravel())
            r_minT_list = list(np.asarray(prev["r_minT"], dtype=np.float64).ravel())
            R1_list = list(np.asarray(prev["R1"], dtype=np.float64).ravel())
            bonus_list = list(np.asarray(prev["bonus"], dtype=np.float64).ravel())
            r_min1_list = list(np.asarray(prev["r_min1"], dtype=np.float64).ravel())
            r_min2_list = list(np.asarray(prev["r_min2"], dtype=np.float64).ravel())
            ok_list = list(np.asarray(prev["both_ok"], dtype=np.int8).ravel())
            intercepted_list = list(
                np.asarray(prev["intercepted"], dtype=np.int8).ravel()
            )
            if "reached" in prev:
                reached_list = list(np.asarray(prev["reached"], dtype=np.int8).ravel())
            else:
                reached_list = [0] * start_tid
            if "diverged" in prev:
                diverged_list = list(np.asarray(prev["diverged"], dtype=np.int8).ravel())
            else:
                diverged_list = [0] * start_tid
            if "n_steps_stage3" in prev:
                n_s3_list = list(
                    np.asarray(prev["n_steps_stage3"], dtype=np.int32).ravel()
                )
            else:
                n_s3_list = [0] * start_tid
            skipped_stage3 = int(np.sum(np.asarray(intercepted_list) > 0))

    if start_tid >= n_use:
        print(f"[precompute] 已完成 {n_use} 条，无需再算 → {out}")
        return

    print("[precompute] 启动 MATLAB Engine…")
    eng = matlab.engine.start_matlab()
    eng.addpath(MATLAB_DIR, nargout=0)
    eng.addpath(GAIL_ROOT, nargout=0)
    eng.addpath(EXPERT_DIR, nargout=0)

    try:
        for tid in range(start_tid, n_use):
            n_k = int(facts["n_steps"][tid])
            intercepted = bool(facts["intercepted"][tid])
            r_min1 = float(facts["r_min1"][tid])
            r_min2 = float(facts["r_min2"][tid])
            t0 = float(facts["t_end"][tid])
            last_abs = np.asarray(facts["terminal_abs"][tid], dtype=np.float64).ravel()

            if intercepted:
                r_minT = np.nan
                R1 = 0.0
                bonus = 0.0
                both_ok = 0
                reached = 0
                diverged = 0
                n_s3 = 0
                skipped_stage3 += 1
            else:
                sabs_ml = matlab.double(last_abs.tolist())
                cfg = {"dt": float(args.dt)}
                r_minT_ml, reached_ml, diverged_ml, n_ml = eng.stage3_rollout(
                    sabs_ml, t0, cfg, nargout=4
                )
                r_minT = float(r_minT_ml)
                reached = 1 if bool(reached_ml) else 0
                diverged = 1 if bool(diverged_ml) else 0
                n_s3 = int(n_ml)
                R1, bonus, both_ok = project_terminal(
                    r_minT, r_min1, r_min2, intercepted=False
                )
                # 拦截不得写入正终端奖励（已由跳过保证）
                if (tid + 1) % 20 == 0 or tid == 0 or tid == start_tid:
                    print(
                        f"  [{tid+1}/{n_use}] n={n_k}  r_minT={r_minT:.3f}  "
                        f"R1={R1:.2f} bonus={bonus:.0f}  "
                        f"reached={reached} diverged={diverged}"
                    )

            n_steps_list.append(n_k)
            r_minT_list.append(r_minT)
            R1_list.append(R1)
            bonus_list.append(bonus)
            r_min1_list.append(r_min1)
            r_min2_list.append(r_min2)
            ok_list.append(both_ok)
            intercepted_list.append(1 if intercepted else 0)
            reached_list.append(reached)
            diverged_list.append(diverged)
            n_s3_list.append(n_s3)

            # 周期性落盘，便于中断恢复
            if args.resume and ((tid + 1) % 50 == 0 or tid + 1 == n_use):
                _save_sidecar(
                    out,
                    mat_path=mat_path,
                    aligned_path=facts.get("aligned_path", ""),
                    expert_fp=expert_fp,
                    dt=args.dt,
                    facts=facts,
                    n_steps_list=n_steps_list,
                    r_minT_list=r_minT_list,
                    R1_list=R1_list,
                    bonus_list=bonus_list,
                    r_min1_list=r_min1_list,
                    r_min2_list=r_min2_list,
                    ok_list=ok_list,
                    intercepted_list=intercepted_list,
                    reached_list=reached_list,
                    diverged_list=diverged_list,
                    n_s3_list=n_s3_list,
                )
    finally:
        eng.quit()

    _save_sidecar(
        out,
        mat_path=mat_path,
        aligned_path=facts.get("aligned_path", ""),
        expert_fp=expert_fp,
        dt=args.dt,
        facts=facts,
        n_steps_list=n_steps_list,
        r_minT_list=r_minT_list,
        R1_list=R1_list,
        bonus_list=bonus_list,
        r_min1_list=r_min1_list,
        r_min2_list=r_min2_list,
        ok_list=ok_list,
        intercepted_list=intercepted_list,
        reached_list=reached_list,
        diverged_list=diverged_list,
        n_s3_list=n_s3_list,
    )

    R1_a = np.asarray(R1_list)
    bonus_a = np.asarray(bonus_list)
    rT_a = np.asarray(r_minT_list, dtype=np.float64)
    valid = np.isfinite(rT_a)
    print(f"\n[precompute] 完成 {n_use} 条轨迹 → {out}")
    print(f"  源模式: {facts.get('source_mode')}")
    print(f"  跳过 stage3（拦截）: {skipped_stage3}")
    if valid.any():
        print(
            f"  r_minT  med={np.median(rT_a[valid]):.3f}  "
            f"≤10m={(rT_a[valid] <= 10).mean() * 100:.1f}%"
        )
    print(f"  R1      med={np.median(R1_a):.2f}  >0={(R1_a > 0).mean() * 100:.1f}%")
    print(f"  bonus>0 比例={(bonus_a > 0).mean() * 100:.1f}%")
    print(f"  stage3 dt={args.dt}  version={STAGE3_VERSION}")
    print(f"  reward_version={REWARD_VERSION}（派生，不可代替 r_minT）")


def _save_sidecar(
    out: Path,
    *,
    mat_path: Path,
    aligned_path: str,
    expert_fp: str,
    dt: float,
    facts: dict,
    n_steps_list,
    r_minT_list,
    R1_list,
    bonus_list,
    r_min1_list,
    r_min2_list,
    ok_list,
    intercepted_list,
    reached_list,
    diverged_list,
    n_s3_list,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # 拦截轨迹禁止正终端奖励
    inter = np.asarray(intercepted_list, dtype=bool)
    R1_a = np.asarray(R1_list, dtype=np.float64)
    bonus_a = np.asarray(bonus_list, dtype=np.float64)
    if np.any(inter & ((R1_a > 0) | (bonus_a > 0))):
        raise RuntimeError("内部错误：拦截轨迹写入了正 R1/bonus")

    np.savez_compressed(
        out,
        expert_path=str(mat_path.resolve()),
        aligned_path=str(aligned_path or ""),
        expert_fingerprint=expert_fp,
        source_mode=str(facts.get("source_mode", "")),
        schema_version=str(facts.get("schema_version") or ""),
        stage2_dynamics_version=str(facts.get("stage2_dynamics_version") or ""),
        stage3_version=STAGE3_VERSION,
        stage3_dt=np.float64(dt),
        reward_version=REWARD_VERSION,
        n_steps=np.asarray(n_steps_list, dtype=np.int32),
        r_minT=np.asarray(r_minT_list, dtype=np.float64),
        reached=np.asarray(reached_list, dtype=np.int8),
        diverged=np.asarray(diverged_list, dtype=np.int8),
        n_steps_stage3=np.asarray(n_s3_list, dtype=np.int32),
        R1=R1_a,
        bonus=bonus_a,
        r_min1=np.asarray(r_min1_list, dtype=np.float64),
        r_min2=np.asarray(r_min2_list, dtype=np.float64),
        both_ok=np.asarray(ok_list, dtype=np.int8),
        intercepted=np.asarray(intercepted_list, dtype=np.int8),
    )


if __name__ == "__main__":
    main()
