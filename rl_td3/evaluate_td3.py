"""
evaluate_td3.py — 加载训练好的 TD3/TD3BC 模型进行蒙特卡洛评估。

用法：
    python evaluate_td3.py --model rl_td3/checkpoints/.../best/best_model
    python evaluate_td3.py --model path/to/model --episodes 100 --seed 0

--seed：仅首轮 env.reset(seed=...) 固定初态抽样序列；策略用 deterministic=True。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

sys.path.insert(0, str(Path(__file__).parent))
from evasion_env import EvasionEnv, KILL_R, STRIKE_OK
from td3_bc import TD3BC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TD3 制导律权重训练模型评估")
    p.add_argument("--model",    type=str, required=True, help="模型路径（不含 .zip）")
    p.add_argument("--episodes", type=int, default=100,   help="蒙特卡洛评估轮次")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--render",   action="store_true",     help="打印每步信息（调试用）")
    return p.parse_args()


def run_episode(
    env: EvasionEnv,
    model: TD3,
    render: bool = False,
    seed: int | None = None,
):
    """运行一个 episode，返回统计指标。

    seed 非 None 时传入 env.reset(seed=...)，用于可复现评估的首轮初始化；
    后续轮次不传 seed，沿用同一 Gymnasium RNG 序列。
    """
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
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
        "r_MD1_min":    r_MD1_min,
        "r_MD2_min":    r_MD2_min,
        "r_minT":       r_minT,
        "intercepted":  intercepted,
        "evade_ok":     evade_ok,
        "strike_ok":    strike_ok,
        "both_ok":      evade_ok and strike_ok,
        "steps":        step,
    }


def print_stats(label: str, results: list[dict]):
    n = len(results)
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


def main():
    args = parse_args()
    # 与 Gymnasium env.np_random 对齐；动作分布探针也依赖同一序列
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
    print(f"[evaluate] 固定种子 seed={args.seed}（首轮 reset 注入 env.np_random）")
    env = EvasionEnv()
    try:
        model = TD3BC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    print(f"\n[evaluate] 开始 {args.episodes} 轮蒙特卡洛评估…")
    results = []
    for ep in range(args.episodes):
        # 仅首轮传 seed，后续沿同一 RNG 抽 samples_clean 行
        ep_seed = args.seed if ep == 0 else None
        ep_result = run_episode(
            env, model, render=(args.render and ep == 0), seed=ep_seed
        )
        results.append(ep_result)
        if (ep + 1) % 10 == 0:
            done_so_far = [r for r in results if r["both_ok"]]
            rT = ep_result["r_minT"]
            rT_str = f"{rT:.2f}m" if np.isfinite(rT) else "nan"
            print(
                f"  [ep {ep+1:3d}/{args.episodes}] "
                f"双防+打击成功率 {100*len(done_so_far)/(ep+1):.1f}%  "
                f"r_MD1_min={ep_result['r_MD1_min']:.1f}m  "
                f"r_minT={rT_str}"
            )

    print_stats("TD3 (BC热启动)", results)

    # 动作分布探针：用同一 seed 重启 RNG，与 MC 序列解耦、可复现
    print("\n[evaluate] 抽样 20 个初态，输出策略网络动作分布…")
    gammas, wTs, wEs = [], [], []
    for i in range(20):
        obs, _ = env.reset(seed=args.seed) if i == 0 else env.reset()
        action, _ = model.predict(obs, deterministic=True)
        ver = getattr(env, "action_mapping", None)
        kwargs = {"version": ver} if ver else {}
        gamma, wT, wE = EvasionEnv.denorm_action(action, **kwargs)
        gammas.append(gamma)
        wTs.append(wT)
        wEs.append(wE)
        print(f"    γ={gamma:6.3f}  wT={wT:.2e}  wE={wE:.2e}")

    print(f"\n  γ  均值={np.mean(gammas):.3f}  范围=[{np.min(gammas):.3f}, {np.max(gammas):.3f}]")
    print(f"  wT 均值={np.mean(wTs):.2e}  范围=[{np.min(wTs):.2e}, {np.max(wTs):.2e}]")
    print(f"  wE 均值={np.mean(wEs):.2e}  范围=[{np.min(wEs):.2e}, {np.max(wEs):.2e}]")

    env.close()


if __name__ == "__main__":
    main()
