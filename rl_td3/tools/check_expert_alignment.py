"""
check_expert_alignment.py — 对齐数据 / 终端缓存一致性检查入口。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/check_expert_alignment.py
  ... --expert ... --aligned ... --cache ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_expert_buffer import (
    DEFAULT_ALIGNED_PATH,
    DEFAULT_EXPERT_PATH,
    DEFAULT_TERMINAL_CACHE,
    check_aligned_consistency,
    load_aligned_transitions,
    validate_expert_sa_vs_trajectories,
)
from obs_norm import normalize_obs, overflow_ratio


def main() -> None:
    p = argparse.ArgumentParser(description="检查专家对齐数据一致性")
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT_PATH)
    p.add_argument("--aligned", type=str, default=DEFAULT_ALIGNED_PATH)
    p.add_argument("--cache", type=str, default=DEFAULT_TERMINAL_CACHE)
    p.add_argument("--no-cache", action="store_true", help="不检查终端缓存")
    p.add_argument("--skip-sa", action="store_true", help="跳过 expert_sa 校验（较慢）")
    args = p.parse_args()

    print(f"[check] expert ={args.expert}")
    if not args.skip_sa:
        meta = validate_expert_sa_vs_trajectories(args.expert)
        print(f"  expert_sa <-> trajectories OK  traj={meta['n_traj']} steps={meta['n_steps']}")
    else:
        print("  跳过 expert_sa 校验")

    data = load_aligned_transitions(args.aligned)
    cache = None
    if args.no_cache:
        print("[check] 跳过 cache 检查")
    elif Path(args.cache).exists():
        raw = np.load(args.cache, allow_pickle=True)
        cache = {k: raw[k] for k in raw.files}
        print(f"[check] cache  ={args.cache}")
    else:
        print(f"[check] cache  缺失（仅检查对齐数据）: {args.cache}")

    warns = check_aligned_consistency(data, cache=cache, expert_path=args.expert)
    states_n = normalize_obs(data["states"])
    print(
        f"[check] aligned OK  traj={len(data['n_steps'])}  "
        f"steps={len(data['states'])}  "
        f"intercept={(data['intercepted']>0).mean()*100:.1f}%  "
        f"clip_steps={data.get('n_actions_clipped', 'n/a')}"
    )
    print(f"  归一化 |obs|>1 比例={overflow_ratio(states_n)*100:.2f}%")
    done_idx = np.where(data["dones"])[0]
    print(f"  done 步数={len(done_idx)}（应等于轨迹数）")
    for w in warns:
        print(f"  警告: {w}")
    print("[check] 全部硬检查通过")


if __name__ == "__main__":
    main()
