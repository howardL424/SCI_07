"""fixed_ic_eval.py — 训练内评估的固定初态协议（EvalCallback 采样量整改）。

SB3 的 EvalCallback 每次评估都让 env 重新随机抽初态，20 局时 `mean_reward`
基本只反映"这次抽到哪些初态"：平台期真实失败率约 6% 时，单次 20 局全成功的
概率约 29%，跑几十次评估几乎必然中一次，`best_model` 于是由噪声决定、换种子
不可复现。

这里把评估初态固定为初态池的一个有序子集，配合 `deterministic=True` 与确定性
的 MATLAB 仿真，使每次评估成为**无采样噪声**的配对比较：曲线上的差异只来自
策略本身，`best_model` 才有意义。

用法：
    eval_env = FixedICEvalEnv(Monitor(EvasionEnv(samples_file=...)), n_ic=150)
    eval_cb = FixedICEvalCallback(eval_env=eval_env, n_eval_episodes=len(
        eval_env.ic_local_idx), eval_freq=25_000, ...)
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class FixedICEvalEnv(gym.Wrapper):
    """按固定顺序遍历初态池子集的评估环境包装。

    - `n_ic=0` 取整个池；否则用 `order_seed` 的定序置换取前 `n_ic` 个（子集也
      随机但可复现，避免按文件顺序取前 N 个引入的结构性偏差）。
    - 每次 `reset()` 依次推进游标；`reset_ic_cursor()` 由 `FixedICEvalCallback`
      在每轮评估开始前调用，保证每轮覆盖同一组初态。
    - 外部显式传入 `options={"local_idx": ...}`（如 MC 评估脚本）时原样透传。

    注：必须包在 `Monitor` **外层**。gymnasium 1.x 已移除 `Wrapper` 的属性透传，
    而 SB3 `DummyVecEnv.env_method` 直接对最外层 env 取属性。
    """

    def __init__(
        self,
        env: gym.Env,
        n_ic: int = 0,
        order_seed: int = 0,
        verbose: int = 1,
    ):
        super().__init__(env)
        pool = int(env.unwrapped.n_samples)
        order = np.random.default_rng(order_seed).permutation(pool)
        n = pool if not n_ic else min(int(n_ic), pool)
        self._ic_list: list[int] = [int(i) for i in order[:n]]
        self._order_seed = int(order_seed)
        self._cursor = 0
        self._dropped: set[int] = set()
        if verbose >= 1:
            print(
                f"[FixedICEvalEnv] 固定评估初态 {n}/{pool} 个"
                f"（order_seed={order_seed}），row_idx 前 10: "
                f"{self.ic_row_idx[:10]}"
            )

    @property
    def ic_local_idx(self) -> list[int]:
        """本次评估使用的库内下标（已滤全零行后的编号）。"""
        return list(self._ic_list)

    @property
    def ic_row_idx(self) -> list[int]:
        """对应初态 mat 的原始行号（便于与 MC 结果对照）。"""
        row_ids = self.env.unwrapped.sample_row_ids
        return [int(row_ids[i]) for i in self._ic_list]

    @property
    def order_seed(self) -> int:
        """定序置换所用种子（决定 n_ic<池容量 时取哪些初态）。"""
        return self._order_seed

    @property
    def dropped_local_idx(self) -> list[int]:
        """warm-start 失败而被剔除的下标。"""
        return sorted(self._dropped)

    def reset_ic_cursor(self) -> None:
        """游标归零：下一轮评估从固定集合的第一个初态重新开始。"""
        self._cursor = 0

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if "local_idx" in options:
            return self.env.reset(seed=seed, options=options)

        n = len(self._ic_list)
        for _ in range(n):
            local_idx = self._ic_list[self._cursor % n]
            self._cursor += 1
            if local_idx in self._dropped:
                continue
            options["local_idx"] = local_idx
            try:
                return self.env.reset(seed=seed, options=options)
            except RuntimeError as e:
                if "warm-start" not in str(e):
                    raise
                self._dropped.add(local_idx)
                print(
                    f"[FixedICEvalEnv] local_idx={local_idx} warm-start 失败，"
                    f"已从评估集合剔除（剩余 {n - len(self._dropped)} 个）"
                )
        raise RuntimeError(
            "[FixedICEvalEnv] 固定评估集合内所有初态 warm-start 均失败，"
            "请检查初态池与 warm-start 配置。"
        )


class FixedICEvalCallback(EvalCallback):
    """每轮评估前复位初态游标，使各轮评估遍历完全相同的初态集合。

    评估环境不是 `FixedICEvalEnv` 时退化为原生 `EvalCallback` 行为。
    """

    def __init__(self, *args, **kwargs):
        self.evaluate_at_start = bool(kwargs.pop("evaluate_at_start", False))
        super().__init__(*args, **kwargs)
        self._metric_buffers = self._empty_metric_buffers()
        self.evaluations_metrics: list[dict[str, list[bool]]] = []

    @staticmethod
    def _empty_metric_buffers() -> dict[str, list[bool]]:
        return {
            "both_ok": [],
            "evade_ok": [],
            "strike_ok": [],
            "intercepted": [],
        }

    def _log_success_callback(self, locals_, globals_) -> None:
        del globals_
        if not locals_["done"]:
            return
        info = locals_["info"]
        row = {
            "both_ok": bool(info.get("both_ok", False)),
            "evade_ok": bool(info.get("evade_ok", False)),
            "strike_ok": bool(info.get("strike_ok", False)),
            "intercepted": bool(info.get("intercepted", False)),
        }
        for key, value in row.items():
            self._metric_buffers[key].append(value)
        # 继续写入 SB3 标准 successes 字段；本项目主成功判据即 both_ok。
        self._is_success_buffer.append(row["both_ok"])

    def _save_success_metrics(self) -> None:
        snapshot = {k: list(v) for k, v in self._metric_buffers.items()}
        self.evaluations_metrics.append(snapshot)
        rates = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in snapshot.items()
        }
        if self.verbose >= 1:
            print(
                "Validation metrics: "
                f"both_ok={rates['both_ok']:.2%}, "
                f"evade_ok={rates['evade_ok']:.2%}, "
                f"strike_ok={rates['strike_ok']:.2%}, "
                f"intercepted={rates['intercepted']:.2%}"
            )
        if self.log_path is None:
            return
        out_dir = Path(self.log_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timesteps": np.asarray(self.evaluations_timesteps, dtype=np.int64),
        }
        for key in self._metric_buffers:
            rows = [m[key] for m in self.evaluations_metrics]
            payload[key] = np.asarray(rows, dtype=bool)
            payload[f"{key}_rate"] = np.asarray(
                [float(np.mean(row)) if row else 0.0 for row in rows],
                dtype=np.float64,
            )
        np.savez(out_dir / "success_metrics.npz", **payload)
        summary = [
            {
                "timestep": int(self.evaluations_timesteps[i]),
                **{
                    f"{key}_rate": float(payload[f"{key}_rate"][i])
                    for key in self._metric_buffers
                },
            }
            for i in range(len(self.evaluations_timesteps))
        ]
        (out_dir / "success_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _on_training_start(self) -> None:
        super()._on_training_start()
        if not self.evaluate_at_start:
            return
        self._metric_buffers = self._empty_metric_buffers()
        try:
            self.eval_env.env_method("reset_ic_cursor")
        except AttributeError:
            pass
        EvalCallback._on_step(self)
        self._save_success_metrics()

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if is_eval_step:
            self._metric_buffers = self._empty_metric_buffers()
            try:
                self.eval_env.env_method("reset_ic_cursor")
            except AttributeError:
                pass
        continue_training = super()._on_step()
        if is_eval_step:
            self._save_success_metrics()
        return continue_training


def save_ic_pool(env: FixedICEvalEnv, path: str | Path) -> Path:
    """把固定评估初态清单落盘，供复现与跨 run 对照。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples_file": str(getattr(env.unwrapped, "samples_file", "")),
        "n_ic": len(env.ic_local_idx),
        "order_seed": env.order_seed,
        "local_idx": env.ic_local_idx,
        "row_idx": env.ic_row_idx,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[FixedICEvalEnv] 评估初态清单已写入 {path}")
    return path
