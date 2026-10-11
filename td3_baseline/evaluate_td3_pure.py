"""
evaluate_td3_pure.py — 纯 TD3 对照评估薄封装。

转发 rl_td3/evaluate_td3.py 的同一 MC 协议（默认全量
samples_clean_test），便于与 BC-TD3 结果直接对比。加载优先 TD3BC，
失败回退 SB3 TD3。

用法：
    python td3_baseline/evaluate_td3_pure.py \\
      --model td3_baseline/checkpoints/.../best/best_model
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_RL_TD3_EVAL = Path(__file__).resolve().parent.parent / "rl_td3" / "evaluate_td3.py"


def main() -> None:
    if not _RL_TD3_EVAL.is_file():
        print(f"[evaluate] 错误：找不到 {_RL_TD3_EVAL}")
        sys.exit(1)
    # 以 __main__ 运行，复用 evaluate_td3 的 argparse / MC 主流程
    runpy.run_path(str(_RL_TD3_EVAL), run_name="__main__")


if __name__ == "__main__":
    main()
