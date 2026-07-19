"""
build_expert_set.py — 构建可复现、轨迹均衡的约 99k 专家采样计划。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/build_expert_set.py
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/build_expert_set.py `
    --budget 99000 --seed 0 --with-ok-mode `
    --out rl_td3/data/expert_set_99k.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from expert_sampling import (  # noqa: E402
    build_expert_sample_plan,
    load_ok_modes_from_expert,
)
from load_expert_buffer import (  # noqa: E402
    DEFAULT_ALIGNED_PATH,
    DEFAULT_EXPERT_PATH,
    DEFAULT_TERMINAL_CACHE,
    DATA_DIR,
    load_transition_data,
    _load_terminal_cache,
)
from reward_projection import DEFAULT_REWARD_CFG  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="构建轨迹均衡专家采样计划（约 99k）")
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT_PATH)
    p.add_argument("--aligned", type=str, default=DEFAULT_ALIGNED_PATH)
    p.add_argument("--terminal-cache", type=str, default=DEFAULT_TERMINAL_CACHE)
    p.add_argument("--budget", type=int, default=99_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--n-tgo-bins", type=int, default=4)
    p.add_argument(
        "--actor-include-intercepted",
        action="store_true",
        help="Actor BC 也纳入拦截轨迹（默认排除）",
    )
    p.add_argument(
        "--replay-exclude-intercepted",
        action="store_true",
        help="Replay 也不含拦截轨迹（默认含，供 Critic）",
    )
    p.add_argument(
        "--with-ok-mode",
        action="store_true",
        help="从 expert_lib 读取 ok/debug（较慢，但写入质量元数据）",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(DATA_DIR / "expert_set_99k.npz"),
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="摘要 JSON 路径（默认与 --out 同名 .json）",
    )
    args = p.parse_args()

    print(f"[build_expert_set] expert={args.expert}")
    print(f"[build_expert_set] aligned/phys via load_transition_data")
    data = load_transition_data(args.expert, aligned_path=args.aligned)
    cache = _load_terminal_cache(
        args.terminal_cache, args.expert, require_path_match=True, verbose=True
    )

    ok_modes = None
    if args.with_ok_mode:
        print("[build_expert_set] 加载 ok_mode …")
        ok_modes = load_ok_modes_from_expert(args.expert, data["source_idx"])
        uniq, cnt = np.unique(ok_modes, return_counts=True)
        print("  ok_mode 分布:", dict(zip(uniq.tolist(), cnt.tolist())))

    plan = build_expert_sample_plan(
        data,
        args.budget,
        seed=args.seed,
        val_ratio=args.val_ratio,
        n_tgo_bins=args.n_tgo_bins,
        force_terminal=True,
        actor_exclude_intercepted=not args.actor_include_intercepted,
        replay_include_intercepted=not args.replay_exclude_intercepted,
        terminal_cache=cache,
        ok_modes=ok_modes,
        kill_radius=float(DEFAULT_REWARD_CFG.kill_radius),
        strike_ok=float(DEFAULT_REWARD_CFG.strike_ok),
    )

    out = Path(args.out)
    plan.save(out)
    summary_path = Path(args.summary_json) if args.summary_json else out.with_suffix(".json")
    summary = dict(plan.summary)
    summary["expert_path"] = str(Path(args.expert).resolve())
    summary["aligned_or_source"] = data.get("source_mode")
    summary["terminal_cache"] = str(Path(args.terminal_cache).resolve())
    summary["out"] = str(out.resolve())
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[build_expert_set] 完成")
    for k in (
        "n_replay_steps",
        "n_actor_steps",
        "n_train_traj",
        "n_val_traj",
        "n_replay_traj_used",
        "steps_per_used_traj_mean",
        "steps_per_used_traj_max",
        "intercept_rate_train",
        "both_ok_train",
        "ok_mode_train",
    ):
        if k in summary:
            print(f"  {k}: {summary[k]}")
    print(f"  npz : {out}")
    print(f"  json: {summary_path}")


if __name__ == "__main__":
    main()
