"""
td3_policy_bridge.py — 供 MATLAB（或 CLI）加载 BC-TD3 / TD3 策略并推理。

本文件位于 GAIL/plot_table2_matlab/table2_bctd3/；依赖同仓库 rl_td3/td3_bc.py。

用法（MATLAB py）:
    bridge = py.td3_policy_bridge.Td3PolicyBridge(model_path)
    action = bridge.predict(obs_norm)   # obs_norm: 已 OBS_SCALE 归一化的 13 维

用法（CLI 单步）:
    python td3_policy_bridge.py --model .../best/best_model --obs obs.npy --out act.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

_THIS = Path(__file__).resolve().parent          # .../plot_table2_matlab/table2_bctd3
_PLOT = _THIS.parent                             # .../plot_table2_matlab
_GAIL = _PLOT.parent                             # .../GAIL
_RL_TD3 = _GAIL / "rl_td3"
for p in (_THIS, _RL_TD3):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stable_baselines3 import TD3  # noqa: E402

from td3_bc import TD3BC  # noqa: E402


def _normalize_model_path(model: Union[str, Path]) -> Path:
    p = Path(model)
    if p.suffix == ".zip":
        p = p.with_suffix("")
    return p


class Td3PolicyBridge:
    """加载 SB3 TD3/TD3BC checkpoint，对归一化观测做 deterministic predict。"""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",
        prefer_td3bc: bool = True,
    ):
        path = _normalize_model_path(model_path)
        zip_path = path.with_suffix(".zip")
        if not zip_path.exists():
            raise FileNotFoundError(f"找不到模型: {zip_path}")

        self.model_path = str(path)
        self.device = str(device)
        self.algo = "TD3BC" if prefer_td3bc else "TD3"

        try:
            if prefer_td3bc:
                self.model = TD3BC.load(str(path), device=self.device)
            else:
                self.model = TD3.load(str(path), device=self.device)
        except Exception:
            # zip 可能是裸 TD3
            self.model = TD3.load(str(path), device=self.device)
            self.algo = "TD3"

        print(
            f"[Td3PolicyBridge] loaded {self.algo}: {zip_path} (device={self.device})",
            flush=True,
        )

    def predict(self, obs) -> np.ndarray:
        """
        输入已归一化观测（13,）或（1,13），返回 action_norm（3,）float64 ∈[-1,1]。
        """
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.size != 13:
            raise ValueError(f"观测须为 13 维，收到 size={x.size}")
        action, _ = self.model.predict(x, deterministic=True)
        return np.asarray(action, dtype=np.float64).reshape(3)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="TD3/TD3BC 单步推理桥接")
    p.add_argument("--model", required=True, help="checkpoint 路径（可无 .zip）")
    p.add_argument("--obs", required=True, help="输入 obs.npy（已归一化 13 维）")
    p.add_argument("--out", required=True, help="输出 action.npy（3 维）")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    bridge = Td3PolicyBridge(args.model, device=args.device)
    obs = np.load(args.obs)
    act = bridge.predict(obs)
    np.save(args.out, act.astype(np.float64))
    print(f"[Td3PolicyBridge] action={act}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
