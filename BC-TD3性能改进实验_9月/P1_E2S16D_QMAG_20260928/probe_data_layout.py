"""probe_data_layout.py — 只读探查：aligned 与 expert_lib 的轨/步对应关系。

回答：
  1. aligned（含 n_steps/traj_ids/source_idx？）总量；
  2. expert_lib 内 transitions.state_abs 是否存在、形状；
  3. lib 是否含 action_norm/next_state_abs/terminal_abs/ok_use（phys 三件套）；
  4. 两侧按轨步数是否一致（决定 16D Q 能否按 flat 顺序直拼）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io

GAIL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GAIL_ROOT / "rl_td3"))

ALIGNED = str(GAIL_ROOT / "rl_td3" / "data" / "expert_transitions_aligned.mat")
EXPERT = r"E:\matlab\result_0714\expert_lib_1000.mat"


def main() -> None:
    print("[probe] 加载 aligned（mat73）…")
    import mat73

    al = mat73.loadmat(ALIGNED)["aligned"]

    def f(name):
        return getattr(al, name, None)

    for name in ["states", "actions", "next_states", "dones", "traj_ids",
                 "n_steps", "intercepted", "terminal_abs", "t_end",
                 "source_idx", "r_min1", "r_min2", "expert_path",
                 "action_mapping_version"]:
        v = al.get(name) if isinstance(al, dict) else getattr(al, name, None)
        if v is None:
            print(f"[aligned] {name}: 缺失")
            continue
        a = np.asarray(v)
        print(f"[aligned] {name}: shape={a.shape} dtype={a.dtype}")
    try:
        ep = al.get("expert_path") if isinstance(al, dict) else getattr(al, "expert_path", "")
        print("[aligned] expert_path =", str(np.asarray(ep).ravel()))
    except Exception as e:
        print("[aligned] expert_path 读取失败：", e)

    print("[probe] 加载 expert_lib（mat73，约2分钟）…")
    import mat73

    d = mat73.loadmat(EXPERT)
    r = d["results"]
    print("[lib] results keys:", sorted(r.keys())[:20])
    tr = r["trajectories"]
    print("[lib] trajectories 类型/长度:", type(tr), len(tr))
    items = tr if isinstance(tr, list) else [tr[k] for k in sorted(
        tr.keys(), key=lambda x: int(x) if str(x).isdigit() else x)][:5]
    for i, t in enumerate(items[:5]):
        keys = sorted(t.keys()) if isinstance(t, dict) else type(t)
        print(f"[lib] traj[{i}] keys: {keys if isinstance(keys, list) else keys}")
        if isinstance(t, dict):
            for k in ["ok_use", "source_idx", "t_end", "r_min1", "r_min2"]:
                if k in t:
                    print(f"    {k} = {t[k]}")
            trans = t.get("transitions", {})
            if isinstance(trans, dict):
                for k, v in trans.items():
                    a = np.asarray(v)
                    print(f"    trans.{k}: shape={a.shape} dtype={a.dtype}")
            else:
                print("    transitions 类型:", type(trans))


if __name__ == "__main__":
    main()
