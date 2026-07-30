"""
train_td3.py — BC 预填充 + 衰减 BC 正则 + TD3 精调训练入口

训练范式：
    1. 用专家库预填充 ReplayBuffer（末步含 R1/+100，需 expert_terminal_cache.npz）
    2. TD3BC：actor 损失加衰减 λ‖π(s)-a_E‖²
    3. 在线与仿真环境交互精调

用法：
    python train_td3.py                          # 默认 300k 步
    python train_td3.py --no-bc                  # 不做 BC 预填充
    python train_td3.py --steps 200000           # 自定义训练步数
    python train_td3.py --expert E:/path/to.mat  # 指定专家库路径
    python train_td3.py --smoke                  # 冒烟测试

终端输出默认同时写入 rl_td3/logs/{run_name}_{时间戳}.log
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

# ── 确保 rl_td3/ 目录在 Python 路径上 ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from evasion_env import EvasionEnv, SAMPLES_FILE, SAMPLES_VAL_FILE
from fixed_ic_eval import FixedICEvalCallback, FixedICEvalEnv, save_ic_pool
from load_expert_buffer import (
    DATA_DIR,
    DEFAULT_ALIGNED_PATH,
    DEFAULT_EXPERT_PATH,
    DEFAULT_TERMINAL_CACHE,
    load_expert_buffer,
)
from action_mapping import DEFAULT_ACTION_MAPPING, ACTION_MAPPING_RATIO, ACTION_MAPPING_TD3_BOX
from td3_bc import TD3BC


def set_global_seed(seed: int) -> None:
    """固定 NumPy / PyTorch / CUDA / SB3 随机性（训练可复现入口）。"""
    import random

    from stable_baselines3.common.utils import set_random_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())


# ── 训练超参数（合理默认值，可通过 argparse 覆盖）────────────────────────────
DEFAULTS = dict(
    total_steps     = 300_000,    # TD3 在线交互总步数
    learning_rate   = 1e-4,   # Actor / Critic 统一学习率（SB3 TD3 单 lr 参数）
    batch_size      = 256,
    gamma           = 0.99,
    tau             = 0.005,      # 目标网络软更新系数
    buffer_size     = 300_000,    # ReplayBuffer 容量
    learning_starts = 1_000,      # 多少步后开始更新（BC 预填充后可设为 0）
    train_freq      = (1, "step"),
    policy_delay    = 3,          # TD3 延迟策略更新频率（2→3，给 Critic 更多收敛时间）
    net_arch        = [256, 256], # Actor / Critic MLP 结构
    noise_sigma     = 0.1,        # 探索噪声标准差
    eval_freq       = 25_000,     # 每隔多少步评估一次（低频 + 大样本）
    n_eval_episodes = 150,        # 每次评估遍历的固定 val 初态数（0=整池）
    eval_ic_seed    = 0,          # 固定评估子集的定序置换种子
    save_dir        = "rl_td3/checkpoints",
    run_name        = "td3_evasion_bc",
    grad_clip       = 10.0,       # 梯度元素裁剪幅度（±grad_clip），防止 Critic 发散
    bc_ratio        = 0.33,       # BC 预填充占 buffer 的比例（降低 BC 锚定效应）
    bc_lambda0      = 1.0,        # 衰减 BC 正则初始系数 λ0
    bc_decay_steps  = 200_000,    # λ 线性衰减到 0 的步数
)

# 冒烟测试专用路径（与正式训练 rl_td3/checkpoints 完全隔离）
SMOKE_DEFAULTS = dict(
    total_steps = 5_000,
    save_dir    = "rl_td3/checkpoints_smoke",
    run_name    = "td3_smoke",
    eval_freq   = 1_000,
    n_eval      = 5,
)

LOG_DIR = Path(__file__).parent / "logs"


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
    p.add_argument(
        "--aligned",
        type=str,
        default=DEFAULT_ALIGNED_PATH,
        help="MATLAB rebuild_expert_transitions 生成的对齐 transition 文件",
    )
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
        help="终端输出日志路径（默认 rl_td3/logs/{name}_{时间戳}.log）",
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
    p.add_argument(
        "--n-eval",
        type=int,
        default=DEFAULTS["n_eval_episodes"],
        help=(
            f"每次评估的 episode 数（默认 {DEFAULTS['n_eval_episodes']}，"
            "0=遍历整个评估初态池；样本越大 best_model 越可信）"
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
        "--bc-lambda0",
        type=float,
        default=DEFAULTS["bc_lambda0"],
        help="衰减 BC 正则初始 λ0（默认 1.0；0 关闭）",
    )
    p.add_argument(
        "--bc-decay-steps",
        type=int,
        default=DEFAULTS["bc_decay_steps"],
        help="BC 正则 λ 线性衰减到 0 的步数（默认 200000）",
    )
    p.add_argument(
        "--terminal-cache",
        type=str,
        default=DEFAULT_TERMINAL_CACHE,
        help="专家终端 R1/+100 缓存 npz（由 precompute_expert_terminals.py 生成）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="统一训练种子（NumPy/PyTorch/SB3/环境）；采样默认与此相同",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="专家采样种子（默认=--seed）",
    )
    p.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["balanced", "prefix"],
        help="专家预填充采样：balanced=轨迹均衡（默认）；prefix=旧前缀截取",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="按轨迹划分 validation 比例（采样元数据；纯 BC 验证用）",
    )
    p.add_argument(
        "--actor-include-intercepted",
        action="store_true",
        help="Actor BC 专家标签包含拦截轨迹（等价于 --actor-eligible all）",
    )
    p.add_argument(
        "--actor-eligible",
        type=str,
        default="both_ok",
        choices=["both_ok", "non_intercepted", "all"],
        help="Actor/TD3 专家 MSE 轨迹资格（默认 both_ok）",
    )
    p.add_argument(
        "--save-sample-plan",
        type=str,
        default=None,
        help="可选：保存采样计划 npz（默认不写；可用 rl_td3/data/expert_set_99k.npz）",
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
    """创建并用 Monitor 包装环境，记录每个 episode 的 reward 和长度。"""
    env = EvasionEnv(samples_file=samples_file, action_mapping=action_mapping)
    return Monitor(env)


def _run_training(args: argparse.Namespace, save_dir: Path) -> None:
    # 以实际 buffer 容量计算预填预算（不再写死 DEFAULTS buffer_size）
    bc_max_steps = int(args.buf * args.bc_ratio) if not args.no_bc else 0
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    set_global_seed(args.seed)

    print("=" * 60)
    print(f"  训练方案: BC 预填充 + 衰减 BC 正则 + TD3")
    print(f"  运行模式: {'冒烟测试' if args.smoke else '正式训练'}")
    print(f"  总步数  : {args.steps:,}")
    print(f"  学习率  : {args.lr}（Actor / Critic 统一）")
    print(f"  运行名称: {args.name}")
    print(f"  seed    : {args.seed}（sample_seed={sample_seed}）")
    print(f"  BC 预填充: {'否' if args.no_bc else f'是（最多 {bc_max_steps:,} 步，占 buffer {args.bc_ratio:.0%}）'}")
    print(f"  采样模式: {args.sample_mode}（actor_eligible={getattr(args, 'actor_eligible', 'both_ok')}；include_intercepted={args.actor_include_intercepted}）")
    print(f"  动作映射: {args.action_mapping}")
    print(f"  BC 正则  : λ0={args.bc_lambda0}, 衰减步数={args.bc_decay_steps:,}")
    print(f"  终端缓存: {args.terminal_cache}")
    print(f"  专家库  : {args.expert}")
    print(f"  对齐数据: {args.aligned}")
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

    # ── 创建环境 ─────────────────────────────────────────────────────────────
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

    # ── 探索噪声（TD3 用高斯噪声代替 DDPG 的 OU 噪声）────────────────────
    n_actions = env.action_space.shape[0]   # = 3
    action_noise = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = args.noise * np.ones(n_actions),
    )

    # ── 构建 TD3BC 模型 ───────────────────────────────────────────────────────
    model = TD3BC(
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
        bc_lambda0      = args.bc_lambda0,
        bc_decay_steps  = args.bc_decay_steps,
        seed            = args.seed,
    )

    # ── BC 预填充 ReplayBuffer + 注入专家数据集 ───────────────────────────────
    if not args.no_bc:
        try:
            plan_out = args.save_sample_plan
            if plan_out is None and args.sample_mode == "balanced" and not args.smoke:
                # 正式跑默认落一份采样计划，便于对照 BC / 复现
                plan_out = str(DATA_DIR / f"expert_set_{bc_max_steps}_seed{sample_seed}.npz")
            result = load_expert_buffer(
                model.replay_buffer,
                mat_path=args.expert,
                aligned_path=args.aligned,
                max_steps=bc_max_steps,
                terminal_cache=args.terminal_cache,
                verbose=True,
                sample_mode=args.sample_mode,
                sample_seed=sample_seed,
                val_ratio=args.val_ratio,
                actor_exclude_intercepted=not args.actor_include_intercepted,
                actor_eligible=(
                    "all" if args.actor_include_intercepted else args.actor_eligible
                ),
                sample_plan_out=plan_out,
                action_mapping=args.action_mapping,
            )
            model.set_expert_dataset(result.states, result.actions)
            print(
                f"[train] BC 预填充完成，Replay 写入 {result.n_written} 条；"
                f"Actor BC 标签 {len(result.states)} 条；"
                f"已注入专家数据集供衰减 BC 正则"
            )
            if result.sample_summary:
                s = result.sample_summary
                print(
                    f"[train] 采样摘要: mode={s.get('sample_mode')} "
                    f"train轨={s.get('n_train_traj')} val轨={s.get('n_val_traj')} "
                    f"both_ok_train={s.get('both_ok_train')} "
                    f"actor_eligible={s.get('actor_eligible')} "
                    f"stage3_fail_frac={s.get('actor_stage3_fail_step_frac')}"
                )
        except FileNotFoundError as e:
            print(f"[train] 警告: {e}")
            print("[train] 将跳过 BC 预填充；若 λ0>0 且无专家集，BC 正则不生效。")

    # ── 回调函数 ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = str(save_dir),
        name_prefix = args.name,
        verbose     = 1,
    )
    eval_cb = FixedICEvalCallback(
        eval_env            = eval_env,
        n_eval_episodes     = n_eval_episodes,
        eval_freq           = args.eval_freq,
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
    print(f"\n[train] 开始 TD3BC 训练，共 {args.steps:,} 步…")
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
