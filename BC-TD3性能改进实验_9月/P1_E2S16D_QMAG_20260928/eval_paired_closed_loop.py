"""固定 150 IC 上评估 13D/16D 纯 BC 策略（只评估，不训练）。

建议并行运行两个臂：
  ...python.exe eval_paired_closed_loop.py --arm 13D
  ...python.exe eval_paired_closed_loop.py --arm 16D

每个进程只启动一个 MATLAB Engine，并在 seed 间复用。16D 每个 seed
切换到其训练 Actor 子集估计的 q_scale。初态集合与 P1/E0/E1/E1V 相同：
samples_clean_val.mat 中 order_seed=0 置换后的前 150 个有效初态。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
RL = ROOT / "rl_td3"
sys.path.insert(0, str(RL))

from action_mapping import DEFAULT_ACTION_MAPPING  # noqa: E402
from bc_actor import load_bc_checkpoint, predict_actions  # noqa: E402
from evasion_env import EvasionEnv, SAMPLES_VAL_FILE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("13D", "16D"), required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 21])
    p.add_argument("--n-ic", type=int, default=150)
    p.add_argument("--order-seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--output", type=str, default=None)
    return p.parse_args()


def fixed_local_indices(pool: int, n_ic: int, order_seed: int) -> list[int]:
    order = np.random.default_rng(order_seed).permutation(pool)
    n = pool if n_ic <= 0 else min(int(n_ic), pool)
    return [int(v) for v in order[:n]]


def rollout(env: EvasionEnv, actor, local_idx: int, device: torch.device,
            max_steps: int) -> dict:
    obs, reset_info = env.reset(options={"local_idx": int(local_idx)})
    total_reward = 0.0
    last_info: dict = {}
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = predict_actions(
            actor, np.asarray(obs, dtype=np.float32).reshape(1, -1),
            device=device,
        )[0]
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        step += 1
        if step >= max_steps:
            raise RuntimeError(
                f"local_idx={local_idx} 超过 max_steps={max_steps} 未终止"
            )
    return {
        "local_idx": int(local_idx),
        "row_idx": int(reset_info["row_idx"]),
        "both_ok": bool(last_info.get("both_ok", False)),
        "evade_ok": bool(last_info.get("evade_ok", False)),
        "strike_ok": bool(last_info.get("strike_ok", False)),
        "intercepted": bool(last_info.get("intercepted", False)),
        "r_min1": float(last_info.get("r_min1", np.nan)),
        "r_min2": float(last_info.get("r_min2", np.nan)),
        "r_minT": float(last_info.get("r_minT", np.nan)),
        "total_reward": total_reward,
        "steps": int(step),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "both_ok_rate": float(np.mean([r["both_ok"] for r in rows])),
        "evade_ok_rate": float(np.mean([r["evade_ok"] for r in rows])),
        "strike_ok_rate": float(np.mean([r["strike_ok"] for r in rows])),
        "intercepted_rate": float(np.mean([r["intercepted"] for r in rows])),
        "mean_reward": float(np.mean([r["total_reward"] for r in rows])),
    }


def main() -> None:
    args = parse_args()
    obs_dim = 13 if args.arm == "13D" else 16
    schema = "rel13_v1" if obs_dim == 13 else "rel13_qmag16_v1"
    first_scale = None
    if obs_dim == 16:
        first_scale = np.load(
            EXP / f"runs/ab16_s{args.seeds[0]}/q_scale_seed{args.seeds[0]}.npz"
        )["q_scale"]

    env = EvasionEnv(
        samples_file=SAMPLES_VAL_FILE,
        action_mapping=DEFAULT_ACTION_MAPPING,
        obs_normalize=True,
        actor_obs_mode="truth",
        obs_schema=schema,
        q_scale=first_scale,
    )
    local_indices = fixed_local_indices(
        env.n_samples, args.n_ic, args.order_seed
    )
    row_indices = [int(env.sample_row_ids[i]) for i in local_indices]
    print(
        f"[pair-loop] {args.arm}: {len(local_indices)}/{env.n_samples} IC, "
        f"row_idx前10={row_indices[:10]}",
        flush=True,
    )

    device = torch.device("cpu")
    result = {
        "arm": args.arm,
        "obs_schema": schema,
        "samples_file": SAMPLES_VAL_FILE,
        "n_ic_requested": int(args.n_ic),
        "order_seed": int(args.order_seed),
        "local_idx": local_indices,
        "row_idx": row_indices,
        "seeds": {},
    }
    try:
        for seed in args.seeds:
            if obs_dim == 16:
                q_scale = np.load(
                    EXP / f"runs/ab16_s{seed}/q_scale_seed{seed}.npz"
                )["q_scale"].astype(np.float64)
                env.q_scale = q_scale
            else:
                q_scale = None
            ckpt = (
                EXP
                / f"runs/ab{obs_dim}_s{seed}/bc_ab{obs_dim}_s{seed}_best.pt"
            )
            actor, payload = load_bc_checkpoint(ckpt, device=device)
            if actor.obs_dim != obs_dim:
                raise ValueError(
                    f"{ckpt} obs_dim={actor.obs_dim}，预期 {obs_dim}"
                )
            rows: list[dict] = []
            failures: list[dict] = []
            start = time.time()
            for i, local_idx in enumerate(local_indices, start=1):
                try:
                    rows.append(
                        rollout(env, actor, local_idx, device, args.max_steps)
                    )
                except RuntimeError as exc:
                    if "warm-start" not in str(exc):
                        raise
                    failures.append({
                        "local_idx": int(local_idx),
                        "row_idx": int(env.sample_row_ids[local_idx]),
                        "error": str(exc),
                    })
                if i % 25 == 0 or i == len(local_indices):
                    s = summarize(rows)
                    print(
                        f"[pair-loop] {args.arm} seed={seed}: {i}/"
                        f"{len(local_indices)}, valid={s.get('n', 0)}, "
                        f"both_ok={s.get('both_ok_rate', 0):.2%}",
                        flush=True,
                    )
            result["seeds"][str(seed)] = {
                "checkpoint": str(ckpt),
                "checkpoint_epoch": int(payload.get("epoch", -1)),
                "q_scale": None if q_scale is None else q_scale.tolist(),
                "elapsed_sec": float(time.time() - start),
                "summary": summarize(rows),
                "failures": failures,
                "episodes": rows,
            }
    finally:
        env.close()

    out = (
        Path(args.output)
        if args.output
        else EXP / f"paired_closed_loop_{args.arm.lower()}_{len(local_indices)}ic.json"
    )
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pair-loop] 写入 {out}", flush=True)


if __name__ == "__main__":
    main()
