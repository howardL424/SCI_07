"""
train_td3_pure.py — 纯在线 TD3 对照训练入口（无专家预填、无 BC 正则）。

与 rl_td3/train_td3.py（BC-TD3）对齐同一环境、奖励与主超参；算法类为 SB3 原生 TD3。

用法：
    python td3_baseline/train_td3_pure.py --smoke
    python td3_baseline/train_td3_pure.py --save-dir td3_baseline/checkpoints/run1 --name td3_pure
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

_ROOT = Path(__file__).resolve().parent
_RL_TD3 = _ROOT.parent / "rl_td3"
sys.path.insert(0, str(_RL_TD3))

from action_mapping import (  # noqa: E402
    ACTION_MAPPING_RATIO,
    ACTION_MAPPING_TD3_BOX,
    DEFAULT_ACTION_MAPPING,
)
from evasion_env import EvasionEnv, SAMPLES_FILE, SAMPLES_VAL_FILE  # noqa: E402
from fixed_ic_eval import (  # noqa: E402
    FixedICEvalCallback,
    FixedICEvalEnv,
    save_ic_pool,
)


def set_global_seed(seed: int) -> None:
    """固定 NumPy / PyTorch / CUDA / SB3 随机性。"""
    import random

    from stable_baselines3.common.utils import set_random_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())


# 与 rl_td3/train_td3.py DEFAULTS 对齐（去掉 BC 相关项）
DEFAULTS = dict(
    total_steps=300_000,
    learning_rate=1e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    buffer_size=300_000,
    learning_starts=1_000,
    train_freq=(1, "step"),
    policy_delay=3,
    net_arch=[256, 256],
    noise_sigma=0.1,
    eval_freq=25_000,
    n_eval_episodes=150,
    eval_ic_seed=0,
    save_dir="td3_baseline/checkpoints",
    run_name="td3_pure",
    grad_clip=10.0,
)

SMOKE_DEFAULTS = dict(
    total_steps=5_000,
    save_dir="td3_baseline/checkpoints_smoke",
    run_name="td3_pure_smoke",
    eval_freq=1_000,
    n_eval=5,
)

LOG_DIR = _ROOT / "logs"


class _Tee:
    """同时将 write 转发到多个流（终端 + 日志文件）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return getattr(self.streams[0], "isatty", lambda: False)()


@contextmanager
def tee_terminal_log(log_path: Path):
    """将 stdout/stderr 同时写入 log_path，退出时恢复。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)
    try:
        yield log_path
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


def _uses_production_paths(save_dir: str, run_name: str) -> bool:
    return save_dir == DEFAULTS["save_dir"] and run_name == DEFAULTS["run_name"]


def resolve_run_config(args: argparse.Namespace) -> argparse.Namespace:
    """应用 --smoke 预设，并在可能覆盖正式产物时给出警告。"""
    if args.smoke:
        if args.steps == DEFAULTS["total_steps"]:
            args.steps = SMOKE_DEFAULTS["total_steps"]
        if args.save_dir == DEFAULTS["save_dir"]:
            args.save_dir = SMOKE_DEFAULTS["save_dir"]
        if args.name == DEFAULTS["run_name"]:
            args.name = SMOKE_DEFAULTS["run_name"]
        # 冒烟步数远小于正式 eval_freq，需下调才能真正走一遍评估路径
        if args.eval_freq == DEFAULTS["eval_freq"]:
            args.eval_freq = SMOKE_DEFAULTS["eval_freq"]
        if args.n_eval == DEFAULTS["n_eval_episodes"]:
            args.n_eval = SMOKE_DEFAULTS["n_eval"]
    elif args.steps != DEFAULTS["total_steps"] and _uses_production_paths(
        args.save_dir, args.name
    ):
        print(
            "[train] 警告: 非默认步数仍写入正式目录 "
            f"{DEFAULTS['save_dir']} / {DEFAULTS['run_name']}，"
            "可能与进行中的正式训练互相覆盖。"
        )
        print(
            "[train] 建议: 并行冒烟请使用  python td3_baseline/train_td3_pure.py --smoke  "
            "或显式 --save-dir / --name。"
        )
    return args


class GradClipCallback(BaseCallback):
    """元素级梯度裁剪（±max_grad）；与 BC-TD3 训练入口一致。"""

    def __init__(self, max_grad: float = 10.0, verbose: int = 0):
        super().__init__(verbose)
        self.max_grad = max_grad
        self._hooks: list = []

    def _on_training_start(self) -> None:
        for param in self.model.policy.parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda g, mg=self.max_grad: g.clamp_(-mg, mg)
                )
                self._hooks.append(hook)
        if self.verbose >= 1:
            n = len(self._hooks)
            print(
                f"[GradClipCallback] 已为 {n} 个参数张量注册梯度裁剪钩子 (±{self.max_grad})"
            )

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="纯在线 TD3 对照训练（无专家 / 无 BC）")
    p.add_argument("--steps", type=int, default=DEFAULTS["total_steps"])
    p.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--buf", type=int, default=DEFAULTS["buffer_size"])
    p.add_argument("--noise", type=float, default=DEFAULTS["noise_sigma"])
    p.add_argument("--save-dir", type=str, default=DEFAULTS["save_dir"])
    p.add_argument("--name", type=str, default=DEFAULTS["run_name"])
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "冒烟测试：默认 5000 步，checkpoint 写入 "
            f"{SMOKE_DEFAULTS['save_dir']}"
        ),
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="终端输出日志路径（默认 td3_baseline/logs/{name}_{时间戳}.log）",
    )
    p.add_argument("--no-log", action="store_true", help="不写入终端输出日志")
    p.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULTS["grad_clip"],
        help="梯度元素裁剪幅度 ±grad_clip（默认 10.0，设为 0 禁用）",
    )
    p.add_argument(
        "--n-eval",
        type=int,
        default=DEFAULTS["n_eval_episodes"],
        help=(
            f"每次评估的 episode 数（默认 {DEFAULTS['n_eval_episodes']}，"
            "0=遍历整个评估初态池）"
        ),
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULTS["eval_freq"],
        help=f"每隔多少步评估一次（默认 {DEFAULTS['eval_freq']:,}）",
    )
    p.add_argument(
        "--eval-ic-seed",
        type=int,
        default=DEFAULTS["eval_ic_seed"],
        help="固定评估子集的定序置换种子（决定取哪 n_eval 个 val 初态）",
    )
    p.add_argument(
        "--eval-random-ic",
        action="store_true",
        help="回退旧协议：每局重新随机抽评估初态（复现历史 run 时用）",
    )
    p.add_argument(
        "--train-samples",
        type=str,
        default=SAMPLES_FILE,
        help="训练 env 初态池（默认 samples_clean.mat）",
    )
    p.add_argument(
        "--eval-samples",
        type=str,
        default=SAMPLES_VAL_FILE,
        help="EvalCallback 初态池（默认 samples_clean_val.mat）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="统一训练种子（NumPy/PyTorch/SB3/环境）",
    )
    p.add_argument(
        "--action-mapping",
        type=str,
        default=DEFAULT_ACTION_MAPPING,
        choices=[ACTION_MAPPING_RATIO, ACTION_MAPPING_TD3_BOX],
        help="动作映射（默认 ratio_log_box_v1）",
    )
    return resolve_run_config(p.parse_args())


def make_env(
    action_mapping: str = DEFAULT_ACTION_MAPPING,
    samples_file: str = SAMPLES_FILE,
) -> EvasionEnv:
    env = EvasionEnv(samples_file=samples_file, action_mapping=action_mapping)
    return Monitor(env)


def _run_training(args: argparse.Namespace, save_dir: Path) -> None:
    set_global_seed(args.seed)

    print("=" * 60)
    print("  训练方案: 纯在线 TD3（无专家预填、无 BC 正则）")
    print(f"  算法类  : stable_baselines3.TD3")
    print(f"  运行模式: {'冒烟测试' if args.smoke else '正式训练'}")
    print(f"  总步数  : {args.steps:,}")
    print(f"  学习率  : {args.lr}（Actor / Critic 统一）")
    print(f"  运行名称: {args.name}")
    print(f"  seed    : {args.seed}")
    print(f"  动作映射: {args.action_mapping}")
    print(f"  learning_starts: {DEFAULTS['learning_starts']:,}")
    print(
        f"  评估 ep 数: {args.n_eval if args.n_eval else '整池'}"
        f"（每 {args.eval_freq:,} 步，"
        f"{'随机抽初态' if args.eval_random_ic else '固定初态遍历'}）"
    )
    print(f"  训练初态: {args.train_samples}")
    print(f"  评估初态: {args.eval_samples}（EvalCallback / best_model）")
    print(f"  梯度裁剪: {'禁用' if args.grad_clip == 0 else f'±{args.grad_clip}'}")
    print(f"  保存目录: {save_dir}")
    print("=" * 60)

    env = make_env(args.action_mapping, samples_file=args.train_samples)
    # 独立评估环境：初态只取 val-IC，用于曲线与选 best
    eval_env = make_env(args.action_mapping, samples_file=args.eval_samples)
    env.reset(seed=args.seed)
    n_eval_episodes = args.n_eval or int(eval_env.unwrapped.n_samples)
    if args.eval_random_ic:
        eval_env.reset(seed=args.seed + 1)
        print(f"[train] 评估协议: 每次随机抽 {n_eval_episodes} 个 val 初态（旧协议）")
    else:
        eval_env = FixedICEvalEnv(
            eval_env, n_ic=args.n_eval, order_seed=args.eval_ic_seed
        )
        n_eval_episodes = len(eval_env.ic_local_idx)
        save_ic_pool(eval_env, save_dir / "eval_logs" / "eval_ic_pool.json")
        print(
            f"[train] 评估协议: 每 {args.eval_freq:,} 步遍历同一组 "
            f"{n_eval_episodes} 个 val 初态（deterministic，无采样噪声）"
        )

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise * np.ones(n_actions),
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        buffer_size=args.buf,
        batch_size=args.batch,
        gamma=DEFAULTS["gamma"],
        tau=DEFAULTS["tau"],
        train_freq=DEFAULTS["train_freq"],
        policy_delay=DEFAULTS["policy_delay"],
        action_noise=action_noise,
        learning_starts=DEFAULTS["learning_starts"],
        policy_kwargs=dict(net_arch=DEFAULTS["net_arch"]),
        verbose=1,
        tensorboard_log=str(save_dir / "tb_logs"),
        seed=args.seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(save_dir),
        name_prefix=args.name,
        verbose=1,
    )
    eval_cb = FixedICEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=args.eval_freq,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "eval_logs"),
        deterministic=True,
        verbose=1,
    )

    callbacks = [checkpoint_cb, eval_cb]
    if args.grad_clip > 0:
        callbacks.append(GradClipCallback(max_grad=args.grad_clip, verbose=1))

    print(f"\n[train] 开始纯 TD3 训练，共 {args.steps:,} 步…")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        log_interval=10,
        reset_num_timesteps=True,
    )

    final_path = str(save_dir / args.name)
    model.save(final_path)
    print(f"\n[train] 训练完成，最终模型已保存至 {final_path}.zip")

    env.close()
    eval_env.close()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.no_log:
        _run_training(args, save_dir)
        return

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{args.name}_{ts}.log"

    with tee_terminal_log(log_path):
        print(f"[train] 终端输出同时写入 {log_path}")
        _run_training(args, save_dir)


if __name__ == "__main__":
    main()
