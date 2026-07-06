"""
train_td3.py — BC 热启动 + TD3 精调训练入口

训练范式：
    1. 用专家库 expert_lib_1000.mat 预填充 ReplayBuffer（行为克隆热启动）
    2. TD3 在线与仿真环境交互，精调策略网络

用法：
    python train_td3.py                          # 使用全部默认参数
    python train_td3.py --no-bc                  # 不做 BC 预填充（纯 TD3）
    python train_td3.py --steps 200000           # 自定义训练步数
    python train_td3.py --expert E:/path/to.mat  # 指定专家库路径
    python train_td3.py --smoke                  # 冒烟测试（独立 checkpoint，不与正式训练冲突）

终端输出默认同时写入 rl_td3/logs（终端输出记录）/{run_name}_{时间戳}.log
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
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

# ── 确保 rl_td3/ 目录在 Python 路径上 ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from evasion_env import EvasionEnv
from load_expert_buffer import load_expert_buffer, DEFAULT_EXPERT_PATH


# ── 训练超参数（合理默认值，可通过 argparse 覆盖）────────────────────────────
DEFAULTS = dict(
    total_steps     = 500_000,    # TD3 在线交互总步数
    learning_rate   = 3e-4,
    batch_size      = 256,
    gamma           = 0.99,
    tau             = 0.005,      # 目标网络软更新系数
    buffer_size     = 300_000,    # ReplayBuffer 容量
    learning_starts = 1_000,      # 多少步后开始更新（BC 预填充后可设为 0）
    train_freq      = (1, "step"),
    policy_delay    = 2,          # TD3 延迟策略更新频率
    net_arch        = [256, 256], # Actor / Critic MLP 结构
    noise_sigma     = 0.1,        # 探索噪声标准差
    eval_freq       = 10_000,     # 每隔多少步评估一次
    n_eval_episodes = 5,
    save_dir        = "rl_td3/checkpoints",
    run_name        = "td3_evasion_bc",
    grad_clip       = 10.0,       # 梯度元素裁剪幅度（±grad_clip），防止 Critic 发散
    bc_ratio        = 0.33,       # BC 预填充占 buffer 的比例（降低 BC 锚定效应）
)

# 冒烟测试专用路径（与正式训练 rl_td3/checkpoints 完全隔离）
SMOKE_DEFAULTS = dict(
    total_steps = 5_000,
    save_dir    = "rl_td3/checkpoints_smoke",
    run_name    = "td3_smoke",
)

LOG_DIR = Path(__file__).parent / "logs（终端输出记录）"


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
    """应用 --smoke 预设，并在可能覆盖正式训练产物时给出警告。"""
    if args.smoke:
        if args.steps == DEFAULTS["total_steps"]:
            args.steps = SMOKE_DEFAULTS["total_steps"]
        if args.save_dir == DEFAULTS["save_dir"]:
            args.save_dir = SMOKE_DEFAULTS["save_dir"]
        if args.name == DEFAULTS["run_name"]:
            args.name = SMOKE_DEFAULTS["run_name"]
    elif args.steps != DEFAULTS["total_steps"] and _uses_production_paths(
        args.save_dir, args.name
    ):
        print(
            "[train] 警告: 非默认步数仍写入正式目录 "
            f"{DEFAULTS['save_dir']} / {DEFAULTS['run_name']}，"
            "可能与进行中的正式训练互相覆盖。"
        )
        print(
            "[train] 建议: 并行冒烟请使用  python train_td3.py --smoke  "
            "或显式 --save-dir / --name。"
        )
    return args


class GradClipCallback(BaseCallback):
    """
    通过 PyTorch backward hook 对所有策略参数施加元素级梯度裁剪（±max_grad）。
    在 SB3 2.x 中 TD3 无原生 max_grad_norm 参数，故以此 callback 替代。
    注：元素级裁剪（clamp）比全局 norm 裁剪更保守，适合 Critic 发散的早期稳定阶段。
    """

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
            print(f"[GradClipCallback] 已为 {n} 个参数张量注册梯度裁剪钩子 (±{self.max_grad})")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BC + TD3 制导律权重训练")
    p.add_argument("--steps",    type=int,   default=DEFAULTS["total_steps"])
    p.add_argument("--lr",       type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--batch",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--buf",      type=int,   default=DEFAULTS["buffer_size"])
    p.add_argument("--noise",    type=float, default=DEFAULTS["noise_sigma"])
    p.add_argument("--expert",   type=str,   default=DEFAULT_EXPERT_PATH)
    p.add_argument("--no-bc",    action="store_true", help="不做专家库 BC 预填充")
    p.add_argument("--save-dir", type=str,   default=DEFAULTS["save_dir"])
    p.add_argument("--name",     type=str,   default=DEFAULTS["run_name"])
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "冒烟测试模式：默认 5000 步，checkpoint 写入 "
            f"{SMOKE_DEFAULTS['save_dir']}（不与正式训练冲突）"
        ),
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="终端输出日志路径（默认 rl_td3/logs（终端输出记录）/{name}_{时间戳}.log）",
    )
    p.add_argument("--no-log",    action="store_true", help="不写入终端输出日志")
    p.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULTS["grad_clip"],
        help="梯度元素裁剪幅度 ±grad_clip（默认 10.0，设为 0 禁用）",
    )
    p.add_argument(
        "--bc-ratio",
        type=float,
        default=DEFAULTS["bc_ratio"],
        help="BC 预填充占 buffer_size 的比例（默认 0.33，降低 BC 锚定效应）",
    )
    return resolve_run_config(p.parse_args())


def make_env() -> EvasionEnv:
    """创建并用 Monitor 包装环境，记录每个 episode 的 reward 和长度。"""
    env = EvasionEnv()
    return Monitor(env)


def _run_training(args: argparse.Namespace, save_dir: Path) -> None:
    bc_max_steps = int(DEFAULTS["buffer_size"] * args.bc_ratio) if not args.no_bc else 0
    print("=" * 60)
    print(f"  训练方案: BC 热启动 + TD3 精调")
    print(f"  运行模式: {'冒烟测试' if args.smoke else '正式训练'}")
    print(f"  总步数  : {args.steps:,}")
    print(f"  运行名称: {args.name}")
    print(f"  BC 预填充: {'否' if args.no_bc else f'是（最多 {bc_max_steps:,} 步，占 buffer {args.bc_ratio:.0%}）'}")
    print(f"  专家库  : {args.expert}")
    print(f"  梯度裁剪: {'禁用' if args.grad_clip == 0 else f'±{args.grad_clip}'}")
    print(f"  保存目录: {save_dir}")
    print("=" * 60)

    # ── 创建环境 ─────────────────────────────────────────────────────────────
    env      = make_env()
    eval_env = make_env()      # 独立评估环境（避免干扰训练 episode）

    # ── 探索噪声（TD3 用高斯噪声代替 DDPG 的 OU 噪声）────────────────────
    n_actions = env.action_space.shape[0]   # = 3
    action_noise = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = args.noise * np.ones(n_actions),
    )

    # ── 构建 TD3 模型 ─────────────────────────────────────────────────────────
    model = TD3(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = args.lr,
        buffer_size     = args.buf,
        batch_size      = DEFAULTS["batch_size"],
        gamma           = DEFAULTS["gamma"],
        tau             = DEFAULTS["tau"],
        train_freq      = DEFAULTS["train_freq"],
        policy_delay    = DEFAULTS["policy_delay"],
        action_noise    = action_noise,
        learning_starts = 0 if not args.no_bc else DEFAULTS["learning_starts"],
        policy_kwargs   = dict(net_arch=DEFAULTS["net_arch"]),
        verbose         = 1,
        tensorboard_log = str(save_dir / "tb_logs"),
    )

    # ── BC 预填充 ReplayBuffer ────────────────────────────────────────────────
    if not args.no_bc:
        try:
            n_written = load_expert_buffer(
                model.replay_buffer,
                mat_path    = args.expert,
                max_steps   = bc_max_steps,      # BC 退火：只填充 buffer 的 bc_ratio 份
                verbose     = True,
            )
            print(f"[train] BC 预填充完成，写入 {n_written} 条 transition")
        except FileNotFoundError as e:
            print(f"[train] 警告: {e}")
            print("[train] 将跳过 BC 预填充，改为纯 TD3 训练。")

    # ── 回调函数 ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = str(save_dir),
        name_prefix = args.name,
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env            = eval_env,
        n_eval_episodes     = DEFAULTS["n_eval_episodes"],
        eval_freq           = DEFAULTS["eval_freq"],
        best_model_save_path= str(save_dir / "best"),
        log_path            = str(save_dir / "eval_logs"),
        deterministic       = True,
        verbose             = 1,
    )

    # ── 梯度裁剪 callback ─────────────────────────────────────────────────────
    callbacks = [checkpoint_cb, eval_cb]
    if args.grad_clip > 0:
        callbacks.append(GradClipCallback(max_grad=args.grad_clip, verbose=1))

    # ── 开始训练 ─────────────────────────────────────────────────────────────
    print(f"\n[train] 开始 TD3 训练，共 {args.steps:,} 步…")
    model.learn(
        total_timesteps = args.steps,
        callback        = callbacks,
        log_interval    = 10,
        reset_num_timesteps = True,
    )

    # ── 保存最终模型 ─────────────────────────────────────────────────────────
    final_path = str(save_dir / args.name)
    model.save(final_path)
    print(f"\n[train] 训练完成，最终模型已保存至 {final_path}.zip")

    env.close()
    eval_env.close()


def main():
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
