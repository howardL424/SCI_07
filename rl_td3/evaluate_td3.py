"""
evaluate_td3.py — 加载训练好的 TD3/TD3BC 模型进行蒙特卡洛评估。

用法：
    # 默认：测完全部 samples_clean_test.mat 有效初态
    python evaluate_td3.py --model rl_td3/checkpoints/.../best/best_model

    # 旧协议：随机抽 N 轮（有放回）
    python evaluate_td3.py --model path/to/model --episodes 100 --seed 0

    # 0711/0715 等旧 checkpoint（raw 观测 + td3_box_log_v1）
    python evaluate_td3.py --model .../best/best_model --obs-mode raw --action-mapping td3_box_log_v1

默认初态池为 samples_clean_test.mat；--episodes 省略或 0 表示测全池。
终端输出默认 tee 到模型所在 training 目录：
    {training_dir}/monte_carlo_eval_{best|final|...}_seed{seed}_{时间戳}.log
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

sys.path.insert(0, str(Path(__file__).parent))
from action_mapping import DEFAULT_ACTION_MAPPING
from evasion_env import EvasionEnv, KILL_R, SAMPLES_TEST_FILE, STRIKE_OK
from td3_bc import TD3BC


class _Tee:
    """同时写入多个流（终端 + 日志文件）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

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


def _normalize_model_path(model: str) -> Path:
    p = Path(model)
    if p.suffix == ".zip":
        p = p.with_suffix("")
    return p


def resolve_mc_log_path(
    model: str,
    seed: int,
    log_file: str | None = None,
) -> Path:
    """默认日志：{training_dir}/monte_carlo_eval_{tag}_seed{seed}_{ts}.log。

    training_dir / tag 推断：
      .../best/best_model  → training_dir=.../ , tag=best
      .../{name}.zip       → training_dir=.../ , tag=final
      .../{name}_N_steps   → training_dir=.../ , tag={name}_N_steps
    """
    if log_file:
        return Path(log_file)

    model_path = _normalize_model_path(model)
    stem = model_path.name
    if model_path.parent.name == "best" and stem == "best_model":
        training_dir = model_path.parent.parent
        tag = "best"
    else:
        training_dir = model_path.parent
        tag = "final" if not stem.endswith("_steps") else stem

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return training_dir / f"monte_carlo_eval_{tag}_seed{seed}_{ts}.log"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TD3 制导律权重训练模型评估")
    p.add_argument("--model", type=str, required=True, help="模型路径（不含 .zip）")
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="随机抽 N 轮（有放回）；0=测完全部初态池（默认）",
    )
    p.add_argument(
        "--samples",
        type=str,
        default=SAMPLES_TEST_FILE,
        help="初态池 mat（默认 samples_clean_test.mat）",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", help="打印每步信息（调试用）")
    p.add_argument(
        "--obs-mode",
        choices=["scaled", "raw"],
        default="scaled",
        help="scaled=OBS_SCALE（当前默认）；raw=兼容 0711/0715 等旧模型",
    )
    p.add_argument(
        "--action-mapping",
        type=str,
        default=DEFAULT_ACTION_MAPPING,
        help="动作映射版本（旧模型用 td3_box_log_v1）",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "终端输出日志路径（默认写入模型所在 training 目录："
            "monte_carlo_eval_{tag}_seed{seed}_{时间戳}.log）"
        ),
    )
    p.add_argument("--no-log", action="store_true", help="不写入终端输出日志")
    return p.parse_args()


def run_episode(
    env: EvasionEnv,
    model: TD3,
    render: bool = False,
    seed: int | None = None,
    local_idx: int | None = None,
):
    """运行一个 episode，返回统计指标。

    - local_idx 非 None：定点测该初态（全量 test 模式）
    - 否则随机抽行；seed 非 None 时传入 env.reset(seed=...)（随机模式首轮）
    warm-start 失败时返回 None（仅定点模式；随机模式由 env 内部重试）。
    """
    try:
        if local_idx is not None:
            obs, info0 = env.reset(
                seed=seed, options={"local_idx": int(local_idx)}
            )
        else:
            obs, info0 = env.reset(seed=seed) if seed is not None else env.reset()
    except RuntimeError as e:
        if local_idx is not None and "warm-start" in str(e):
            return None
        raise

    terminated = False
    total_reward = 0.0
    r_MD1_min = np.inf
    r_MD2_min = np.inf
    r_minT = np.nan
    intercepted = False
    evade_ok = False
    strike_ok = False
    step = 0

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)

        total_reward += reward
        r_MD1_min = min(r_MD1_min, info["r_MD1"])
        r_MD2_min = min(r_MD2_min, info["r_MD2"])
        intercepted = intercepted or info["intercepted"]
        if np.isfinite(info.get("r_minT", np.nan)):
            r_minT = float(info["r_minT"])
            evade_ok = bool(info.get("evade_ok", False))
            strike_ok = bool(info.get("strike_ok", False))
        step += 1

        if render:
            ver = getattr(env, "action_mapping", None)
            kwargs = {"version": ver} if ver else {}
            gamma, wT, wE = EvasionEnv.denorm_action(action, **kwargs)
            print(
                f"  step={step:3d} | r_MD1={info['r_MD1']:7.1f}m "
                f"r_MD2={info['r_MD2']:7.1f}m r_MT={info['r_MT']:8.1f}m "
                f"| γ={gamma:5.2f} wT={wT:.1e} wE={wE:.1e} "
                f"| r={reward:7.3f}"
            )

    if not evade_ok:
        evade_ok = (r_MD1_min >= KILL_R) and (r_MD2_min >= KILL_R)
    if not strike_ok and np.isfinite(r_minT):
        strike_ok = r_minT <= STRIKE_OK

    return {
        "total_reward": total_reward,
        "r_MD1_min": r_MD1_min,
        "r_MD2_min": r_MD2_min,
        "r_minT": r_minT,
        "intercepted": intercepted,
        "evade_ok": evade_ok,
        "strike_ok": strike_ok,
        "both_ok": evade_ok and strike_ok,
        "steps": step,
        "row_idx": int(info0.get("row_idx", -1)),
        "local_idx": int(info0.get("local_idx", -1)),
    }


def print_stats(label: str, results: list[dict]):
    n = len(results)
    if n == 0:
        print(f"\n[evaluate] 无有效 episode 可统计（label={label}）")
        return

    evade_ok = [r for r in results if r["evade_ok"]]
    strike_ok = [r for r in results if r["strike_ok"]]
    both_ok = [r for r in results if r["both_ok"]]
    intercepted = [r for r in results if r["intercepted"]]

    d1_arr = np.array([r["r_MD1_min"] for r in results])
    d2_arr = np.array([r["r_MD2_min"] for r in results])
    rT_arr = np.array([r["r_minT"] for r in results], dtype=float)
    rw_arr = np.array([r["total_reward"] for r in results])
    finite_T = np.isfinite(rT_arr)

    print(f"\n{'='*60}")
    print(f"  评估结果：{label}  （{n} 轮）")
    print(f"{'='*60}")
    print(f"  双防均规避成功  (d1,d2≥{KILL_R:.0f}m)       : {len(evade_ok):3d}/{n} = {100*len(evade_ok)/n:.1f}%")
    print(f"  打击成功        (r_minT≤{STRIKE_OK:.0f}m)     : {len(strike_ok):3d}/{n} = {100*len(strike_ok)/n:.1f}%")
    print(f"  规避+打击 全部达标                  : {len(both_ok):3d}/{n} = {100*len(both_ok)/n:.1f}%")
    print(f"  被拦截          (r_MD<{KILL_R:.0f}m)          : {len(intercepted):3d}/{n} = {100*len(intercepted)/n:.1f}%")
    print(f"  最小脱靶距离 D1  mean±std (m)       : {d1_arr.mean():.1f} ± {d1_arr.std():.1f}")
    print(f"  最小脱靶距离 D2  mean±std (m)       : {d2_arr.mean():.1f} ± {d2_arr.std():.1f}")
    if finite_T.any():
        print(
            f"  终端脱靶 r_minT  mean±std (m)       : "
            f"{rT_arr[finite_T].mean():.2f} ± {rT_arr[finite_T].std():.2f}"
            f"  (n={finite_T.sum()})"
        )
    else:
        print("  终端脱靶 r_minT  : (无有效样本，可能全部被拦截)")
    print(f"  累计奖励         mean±std            : {rw_arr.mean():.2f} ± {rw_arr.std():.2f}")


def _run_eval(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path_zip = model_path + ".zip"
    else:
        model_path_zip = model_path
        model_path = model_path[:-4]

    if not Path(model_path_zip).exists():
        print(f"[evaluate] 错误：找不到模型文件 {model_path_zip}")
        sys.exit(1)

    print(f"[evaluate] 加载模型：{model_path_zip}")
    print(f"[evaluate] obs_mode={args.obs_mode}  action_mapping={args.action_mapping}")
    print(f"[evaluate] 初态池：{args.samples}")
    env = EvasionEnv(
        samples_file=args.samples,
        action_mapping=args.action_mapping,
        obs_normalize=(args.obs_mode == "scaled"),
    )
    try:
        model = TD3BC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    sweep_all = args.episodes <= 0
    if sweep_all:
        n_plan = env.n_samples
        mode_desc = f"全量 test-IC（{n_plan} 个有效初态）"
    else:
        n_plan = args.episodes
        mode_desc = f"随机 {n_plan} 轮（seed={args.seed}）"
        print(f"[evaluate] 固定种子 seed={args.seed}（首轮 reset 注入 env.np_random）")

    print(f"\n[evaluate] 开始评估：{mode_desc}…")
    results = []
    n_skip = 0
    for ep in range(n_plan):
        if sweep_all:
            ep_seed = args.seed if ep == 0 else None
            ep_result = run_episode(
                env,
                model,
                render=(args.render and ep == 0),
                seed=ep_seed,
                local_idx=ep,
            )
            if ep_result is None:
                n_skip += 1
                print(f"  [skip local_idx={ep}] warm-start 无有效规避段")
                continue
        else:
            ep_seed = args.seed if ep == 0 else None
            ep_result = run_episode(
                env, model, render=(args.render and ep == 0), seed=ep_seed
            )

        results.append(ep_result)
        done_n = len(results)
        if done_n % 10 == 0 or ep + 1 == n_plan:
            done_so_far = [r for r in results if r["both_ok"]]
            rT = ep_result["r_minT"]
            rT_str = f"{rT:.2f}m" if np.isfinite(rT) else "nan"
            print(
                f"  [done {done_n:3d} | scanned {ep+1:3d}/{n_plan}] "
                f"双防+打击成功率 {100*len(done_so_far)/done_n:.1f}%  "
                f"r_MD1_min={ep_result['r_MD1_min']:.1f}m  "
                f"r_minT={rT_str}"
            )

    if sweep_all and n_skip:
        print(f"[evaluate] warm-start 跳过 {n_skip}/{n_plan} 个初态")

    label = "TD3 (全量 test-IC)" if sweep_all else "TD3 (随机 MC)"
    print_stats(label, results)

    # 动作分布探针：取池内前 min(20,n) 个有效初态（定点，可复现）
    n_probe = min(20, env.n_samples)
    print(f"\n[evaluate] 探针前 {n_probe} 个初态，输出策略网络动作分布…")
    gammas, wTs, wEs = [], [], []
    for i in range(n_probe):
        try:
            obs, _ = env.reset(seed=args.seed if i == 0 else None, options={"local_idx": i})
        except RuntimeError:
            print(f"    [skip local_idx={i}] warm-start 失败")
            continue
        action, _ = model.predict(obs, deterministic=True)
        ver = getattr(env, "action_mapping", None)
        kwargs = {"version": ver} if ver else {}
        gamma, wT, wE = EvasionEnv.denorm_action(action, **kwargs)
        gammas.append(gamma)
        wTs.append(wT)
        wEs.append(wE)
        print(f"    γ={gamma:6.3f}  wT={wT:.2e}  wE={wE:.2e}")

    if gammas:
        print(f"\n  γ  均值={np.mean(gammas):.3f}  范围=[{np.min(gammas):.3f}, {np.max(gammas):.3f}]")
        print(f"  wT 均值={np.mean(wTs):.2e}  范围=[{np.min(wTs):.2e}, {np.max(wTs):.2e}]")
        print(f"  wE 均值={np.mean(wEs):.2e}  范围=[{np.min(wEs):.2e}, {np.max(wEs):.2e}]")

    env.close()


def main():
    args = parse_args()
    if args.no_log:
        _run_eval(args)
        return

    log_path = resolve_mc_log_path(args.model, args.seed, args.log_file)
    with tee_terminal_log(log_path):
        print(f"[evaluate] 终端输出同时写入 {log_path}")
        _run_eval(args)


if __name__ == "__main__":
    main()
