"""
rebuild_expert_data.py — 调用 MATLAB rebuild_expert_transitions，生成对齐派生数据。

用法：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/rebuild_expert_data.py
  ... --max-traj 20          # 冒烟
  ... --expert E:/matlab/result_0714/expert_lib_1000.mat
  ... --out rl_td3/data/expert_transitions_aligned.mat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matlab.engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evasion_env import EXPERT_DIR, GAIL_ROOT, MATLAB_DIR
from load_expert_buffer import DEFAULT_ALIGNED_PATH, DEFAULT_EXPERT_PATH


def main() -> None:
    p = argparse.ArgumentParser(description="MATLAB 批处理重建专家 (s,a,s',r,done)")
    p.add_argument("--expert", type=str, default=DEFAULT_EXPERT_PATH)
    p.add_argument("--out", type=str, default=DEFAULT_ALIGNED_PATH)
    p.add_argument("--max-traj", type=int, default=None, help="最多有效轨迹数（冒烟）")
    args = p.parse_args()

    expert = Path(args.expert)
    out = Path(args.out)
    if not expert.exists():
        raise FileNotFoundError(expert)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[rebuild_expert_data] 启动 MATLAB Engine…")
    eng = matlab.engine.start_matlab()
    eng.addpath(MATLAB_DIR, nargout=0)
    eng.addpath(GAIL_ROOT, nargout=0)
    eng.addpath(EXPERT_DIR, nargout=0)

    opts = {"verbose": True}
    if args.max_traj is not None:
        opts["max_traj"] = float(args.max_traj)

    print(f"[rebuild_expert_data] expert={expert}")
    print(f"[rebuild_expert_data] out   ={out}")
    eng.rebuild_expert_transitions(str(expert.resolve()), str(out.resolve()), opts, nargout=1)
    eng.quit()
    print(f"[rebuild_expert_data] 完成 → {out}")


if __name__ == "__main__":
    main()
