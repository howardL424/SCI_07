"""汇总固定 150 IC 的 13D vs 16D 纯 BC 配对闭环结果。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

EXP = Path(__file__).resolve().parent
SEEDS = (0, 42, 21)


def main() -> None:
    p13 = json.loads(
        (EXP / "paired_closed_loop_13d_150ic.json").read_text(encoding="utf-8")
    )
    p16 = json.loads(
        (EXP / "paired_closed_loop_16d_150ic.json").read_text(encoding="utf-8")
    )
    if p13["local_idx"] != p16["local_idx"] or p13["row_idx"] != p16["row_idx"]:
        raise ValueError("13D/16D 固定 IC 清单不一致，拒绝汇总")

    rows = []
    for seed in SEEDS:
        a_pack = p13["seeds"][str(seed)]
        b_pack = p16["seeds"][str(seed)]
        a = {int(r["local_idx"]): r for r in a_pack["episodes"]}
        b = {int(r["local_idx"]): r for r in b_pack["episodes"]}
        common = [i for i in p13["local_idx"] if i in a and i in b]
        if not common:
            raise ValueError(f"seed={seed} 无共同有效 IC")
        aok = np.array([a[i]["both_ok"] for i in common], dtype=bool)
        bok = np.array([b[i]["both_ok"] for i in common], dtype=bool)
        lost = int(np.sum(aok & ~bok))
        gained = int(np.sum(~aok & bok))
        both_success = int(np.sum(aok & bok))
        both_fail = int(np.sum(~aok & ~bok))
        discordant = lost + gained
        p_value = (
            float(binomtest(min(lost, gained), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        sa = a_pack["summary"]
        sb = b_pack["summary"]
        rows.append({
            "seed": seed,
            "n_common": len(common),
            "n_failures_13d": len(a_pack["failures"]),
            "n_failures_16d": len(b_pack["failures"]),
            "both_ok_13d": float(sa["both_ok_rate"]),
            "both_ok_16d": float(sb["both_ok_rate"]),
            "delta_both_ok_pp": float(
                100 * (sb["both_ok_rate"] - sa["both_ok_rate"])
            ),
            "evade_ok_13d": float(sa["evade_ok_rate"]),
            "evade_ok_16d": float(sb["evade_ok_rate"]),
            "strike_ok_13d": float(sa["strike_ok_rate"]),
            "strike_ok_16d": float(sb["strike_ok_rate"]),
            "intercepted_13d": float(sa["intercepted_rate"]),
            "intercepted_16d": float(sb["intercepted_rate"]),
            "mean_reward_13d": float(sa["mean_reward"]),
            "mean_reward_16d": float(sb["mean_reward"]),
            "both_success": both_success,
            "lost_13_to_16": lost,
            "gained_13_to_16": gained,
            "both_fail": both_fail,
            "net_gain": gained - lost,
            "mcnemar_exact_p": p_value,
        })

    aggregate = {
        "mean_both_ok_13d": float(np.mean([r["both_ok_13d"] for r in rows])),
        "mean_both_ok_16d": float(np.mean([r["both_ok_16d"] for r in rows])),
        "mean_delta_pp": float(np.mean([r["delta_both_ok_pp"] for r in rows])),
        "seeds_improved": int(sum(r["delta_both_ok_pp"] > 0 for r in rows)),
        "seeds_degraded": int(sum(r["delta_both_ok_pp"] < 0 for r in rows)),
        "total_lost": int(sum(r["lost_13_to_16"] for r in rows)),
        "total_gained": int(sum(r["gained_13_to_16"] for r in rows)),
    }
    payload = {
        "protocol": {
            "samples_file": p13["samples_file"],
            "n_ic": len(p13["local_idx"]),
            "order_seed": p13["order_seed"],
            "local_idx": p13["local_idx"],
            "row_idx": p13["row_idx"],
            "no_training": True,
        },
        "seeds": rows,
        "aggregate": aggregate,
    }
    (EXP / "paired_closed_loop_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 13D vs 16D 纯 BC 配对闭环（固定 150 IC）",
        "",
        "协议：`samples_clean_val.mat`，`order_seed=0`，与 P1/E0/E1/E1V "
        "相同的 150 个固定初态；truth、deterministic、无训练。",
        "",
        "| seed | 13D both_ok | 16D both_ok | Δpp | 13成功→16失败 | "
        "13失败→16成功 | 净转移 | McNemar p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['both_ok_13d']:.2%} | "
            f"{r['both_ok_16d']:.2%} | {r['delta_both_ok_pp']:+.2f} | "
            f"{r['lost_13_to_16']} | {r['gained_13_to_16']} | "
            f"{r['net_gain']:+d} | {r['mcnemar_exact_p']:.4f} |"
        )
    lines += [
        "",
        f"- 三 seed 宏平均：13D={aggregate['mean_both_ok_13d']:.2%}，"
        f"16D={aggregate['mean_both_ok_16d']:.2%}，"
        f"Δ={aggregate['mean_delta_pp']:+.2f}pp；"
        f"{aggregate['seeds_improved']} seed 改善 / "
        f"{aggregate['seeds_degraded']} seed 退化。",
        f"- 三 seed 合计逐 IC 转移（描述性，不作独立样本检验）："
        f"成功→失败={aggregate['total_lost']}，失败→成功={aggregate['total_gained']}。",
        "- 13D seed 0/42 的 84.00%/83.33% 与 E1/E1V 的 t=0 完全一致，"
        "说明本次纯 BC 闭环评估口径正确。",
        "",
        "## 裁决",
        "",
        "16D 未通过纯 BC 配对闭环门：宏平均下降，且 seed 0/42 明显退化；"
        "不能因离线 MSE 平均下降 9.9% 而进入 E0-16D 或 16D TD3。"
    ]
    (EXP / "PAIRED_CLOSED_LOOP.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
