"""
evaluate_td3.py — 加载训练好的 TD3 模型进行蒙特卡洛评估，并与 BP+LM 基线对比。

用法：
    python evaluate_td3.py --model rl_td3/checkpoints/td3_evasion_bc
    python evaluate_td3.py --model path/to/model --episodes 200 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

sys.path.insert(0, str(Path(__file__).parent))
from evasion_env import EvasionEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TD3 制导律权重训练模型评估")
    p.add_argument("--model",    type=str, required=True, help="模型路径（不含 .zip）")
    p.add_argument("--episodes", type=int, default=100,   help="蒙特卡洛评估轮次")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--render",   action="store_true",     help="打印每步信息（调试用）")
    return p.parse_args()


def run_episode(env: EvasionEnv, model: TD3, render: bool = False):
    """运行一个 episode，返回统计指标。"""
    obs, _ = env.reset()
    terminated = False
    total_reward = 0.0
    r_MD1_min = np.inf
    r_MD2_min = np.inf
    r_MT_final = np.inf
    intercepted = False
    reached     = False
    step = 0

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)

        total_reward += reward
        r_MD1_min = min(r_MD1_min, info["r_MD1"])
        r_MD2_min = min(r_MD2_min, info["r_MD2"])
        r_MT_final = info["r_MT"]
        intercepted = intercepted or info["intercepted"]
        reached     = reached or info["reached"]
        step += 1

        if render:
            gamma, wT, wE = EvasionEnv.denorm_action(action)
            print(
                f"  step={step:3d} | r_MD1={info['r_MD1']:7.1f}m "
                f"r_MD2={info['r_MD2']:7.1f}m r_MT={info['r_MT']:8.1f}m "
                f"| γ={gamma:5.2f} wT={wT:.1e} wE={wE:.1e} "
                f"| r={reward:7.3f}"
            )

    return {
        "total_reward": total_reward,
        "r_MD1_min":    r_MD1_min,
        "r_MD2_min":    r_MD2_min,
        "r_MT_final":   r_MT_final,
        "intercepted":  intercepted,
        "reached":      reached,
        "steps":        step,
    }


def print_stats(label: str, results: list[dict]):
    n = len(results)
    r_evade_ok = [
        r for r in results if r["r_MD1_min"] >= 20 and r["r_MD2_min"] >= 20
    ]
    reached_ok = [r for r in results if r["reached"]]
    both_ok    = [
        r for r in results
        if r["r_MD1_min"] >= 20 and r["r_MD2_min"] >= 20 and r["reached"]
    ]
    intercepted = [r for r in results if r["intercepted"]]

    d1_arr = np.array([r["r_MD1_min"]  for r in results])
    d2_arr = np.array([r["r_MD2_min"]  for r in results])
    rT_arr = np.array([r["r_MT_final"] for r in results])
    rw_arr = np.array([r["total_reward"] for r in results])

    print(f"\n{'='*60}")
    print(f"  评估结果：{label}  （{n} 轮）")
    print(f"{'='*60}")
    print(f"  双防均规避成功  (d1,d2≥20m)        : {len(r_evade_ok):3d}/{n} = {100*len(r_evade_ok)/n:.1f}%")
    print(f"  到达目标        (r_MT≤50m)          : {len(reached_ok):3d}/{n} = {100*len(reached_ok)/n:.1f}%")
    print(f"  规避+到达 全部达标                  : {len(both_ok):3d}/{n} = {100*len(both_ok)/n:.1f}%")
    print(f"  被拦截          (r_MD<6m)           : {len(intercepted):3d}/{n} = {100*len(intercepted)/n:.1f}%")
    print(f"  最小脱靶距离 D1  mean±std (m)       : {d1_arr.mean():.1f} ± {d1_arr.std():.1f}")
    print(f"  最小脱靶距离 D2  mean±std (m)       : {d2_arr.mean():.1f} ± {d2_arr.std():.1f}")
    print(f"  终端目标距离     mean±std (m)        : {rT_arr.mean():.1f} ± {rT_arr.std():.1f}")
    print(f"  累计奖励         mean±std            : {rw_arr.mean():.2f} ± {rw_arr.std():.2f}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ── 加载模型 ─────────────────────────────────────────────────────────────
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
    env   = EvasionEnv()
    model = TD3.load(model_path, env=env)

    # ── 蒙特卡洛评估 ─────────────────────────────────────────────────────────
    print(f"\n[evaluate] 开始 {args.episodes} 轮蒙特卡洛评估…")
    results = []
    for ep in range(args.episodes):
        ep_result = run_episode(env, model, render=(args.render and ep == 0))
        results.append(ep_result)
        if (ep + 1) % 10 == 0:
            done_so_far = [r for r in results
                           if r["r_MD1_min"] >= 20 and r["r_MD2_min"] >= 20
                           and r["reached"]]
            print(
                f"  [ep {ep+1:3d}/{args.episodes}] "
                f"双防+到达成功率 {100*len(done_so_far)/(ep+1):.1f}%  "
                f"r_MD1_min={ep_result['r_MD1_min']:.1f}m  "
                f"r_MT_final={ep_result['r_MT_final']:.1f}m"
            )

    print_stats("TD3 (BC热启动)", results)

    # ── 输出动作分布（便于论文分析）────────────────────────────────────────
    print("\n[evaluate] 抽样 20 个初态，输出策略网络动作分布…")
    env.reset(seed=args.seed)
    gammas, wTs, wEs = [], [], []
    for _ in range(20):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        gamma, wT, wE = EvasionEnv.denorm_action(action)
        gammas.append(gamma); wTs.append(wT); wEs.append(wE)
        print(f"    γ={gamma:6.3f}  wT={wT:.2e}  wE={wE:.2e}")

    print(f"\n  γ  均值={np.mean(gammas):.3f}  范围=[{np.min(gammas):.3f}, {np.max(gammas):.3f}]")
    print(f"  wT 均值={np.mean(wTs):.2e}  范围=[{np.min(wTs):.2e}, {np.max(wTs):.2e}]")
    print(f"  wE 均值={np.mean(wEs):.2e}  范围=[{np.min(wEs):.2e}, {np.max(wEs):.2e}]")

    env.close()


if __name__ == "__main__":
    main()
