"""
split_samples_kuoda.py — 把 samples_clean_kuoda.mat 的非全 0 行随机划成
train / val / test，且不覆盖仓库根目录原有的 samples_clean*.mat。

约定（对应旧池 8:1:1 且 val/test 仍各 300）：
  - 验证、测试各取 300 行；
  - 剩余全部归训练（有效行不是正好 3000 时，比例会略偏离 8:1:1）。

产物（默认写项目根目录）：
  samples_clean_kuoda_train.mat
  samples_clean_kuoda_val.mat
  samples_clean_kuoda_test.mat
  samples_split_kuoda.json          # 种子、原始行号，便于复现

用法（仓库根目录）：
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/split_samples_kuoda.py
  E:\\Python\\venvs\\rl_pytorch\\Scripts\\python.exe rl_td3/tools/split_samples_kuoda.py --seed 0 --dry-run

训练时显式改初态池（默认仍是旧的 samples_clean*.mat）：
  --train-samples samples_clean_kuoda_train.mat
  --eval-samples  samples_clean_kuoda_val.mat
评测：
  --samples samples_clean_kuoda_test.mat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

REPO = Path(__file__).resolve().parents[2]
PROTECTED = {
    "samples_clean.mat",
    "samples_clean_val.mat",
    "samples_clean_test.mat",
    "samples_clean_kuoda.mat",
}
VALID_NORM_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="随机划分 samples_clean_kuoda.mat（val/test 各 300，其余训练）"
    )
    p.add_argument(
        "--src",
        type=str,
        default=str(REPO / "samples_clean_kuoda.mat"),
        help="扩大范围后的未划分初态池",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO),
        help="输出目录（默认项目根）",
    )
    p.add_argument("--n-val", type=int, default=300, help="验证集有效行数")
    p.add_argument("--n-test", type=int, default=300, help="测试集有效行数")
    p.add_argument("--seed", type=int, default=0, help="划分随机种子")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印划分规模，不写文件",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="输出已存在时覆盖",
    )
    return p.parse_args()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def load_valid_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sc = scipy.io.loadmat(str(path), squeeze_me=True)
    if "samples_clean" not in sc:
        raise KeyError(f"{path}: 缺少 samples_clean")
    samples_all = np.asarray(sc["samples_clean"], dtype=np.float64)
    if samples_all.ndim != 2:
        raise ValueError(
            f"{path}: samples_clean 须为 2D，收到 shape={samples_all.shape}"
        )
    valid = np.linalg.norm(samples_all, axis=1) > VALID_NORM_EPS
    row_ids = np.flatnonzero(valid).astype(np.int64)
    return samples_all[valid], row_ids


def split_indices(
    n_valid: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    need = n_val + n_test
    if n_val < 0 or n_test < 0:
        raise ValueError("n_val / n_test 不能为负")
    if n_valid < need:
        raise ValueError(
            f"有效行 {n_valid} < val({n_val})+test({n_test})={need}，无法划分"
        )
    perm = np.random.default_rng(seed).permutation(n_valid)
    val_loc = perm[:n_val]
    test_loc = perm[n_val : n_val + n_test]
    train_loc = perm[n_val + n_test :]
    return train_loc.astype(np.int64), val_loc.astype(np.int64), test_loc.astype(np.int64)


def _assert_safe_out(path: Path) -> None:
    if path.name in PROTECTED:
        raise SystemExit(
            f"[split] 拒绝写入受保护文件 {path.name}（会覆盖原池）。"
            "请使用默认的 samples_clean_kuoda_{train,val,test}.mat 命名。"
        )


def save_split_mat(path: Path, samples: np.ndarray) -> None:
    _assert_safe_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(
        str(path),
        {"samples_clean": np.asarray(samples, dtype=np.float64)},
        do_compression=True,
        oned_as="column",
    )


def main() -> None:
    args = parse_args()
    src = _resolve(args.src)
    out_dir = _resolve(args.out_dir)
    if not src.is_file():
        raise SystemExit(f"[split] 找不到源文件: {src}")

    samples, row_ids = load_valid_rows(src)
    n_valid, dim = samples.shape
    train_loc, val_loc, test_loc = split_indices(
        n_valid, args.n_val, args.n_test, args.seed
    )

    splits = {
        "train": train_loc,
        "val": val_loc,
        "test": test_loc,
    }
    out_mats = {
        name: out_dir / f"samples_clean_kuoda_{name}.mat" for name in splits
    }
    manifest_path = out_dir / "samples_split_kuoda.json"

    n_train = int(len(train_loc))
    ratio = (n_train / n_valid, args.n_val / n_valid, args.n_test / n_valid)
    print(f"[split] src     = {src}")
    print(f"[split] valid   = {n_valid}  dim={dim}  (源矩阵非全 0 行)")
    print(
        f"[split] train/val/test = {n_train}/{args.n_val}/{args.n_test}"
        f"  比例 {ratio[0]:.3f}:{ratio[1]:.3f}:{ratio[2]:.3f}"
        f"  seed={args.seed}"
    )
    print("[split] 输出:")
    for name, path in out_mats.items():
        print(f"         {name:5s} -> {path}")
    print(f"         meta  -> {manifest_path}")

    union = np.concatenate([train_loc, val_loc, test_loc])
    if int(np.unique(union).size) != n_valid:
        raise RuntimeError("划分有重叠或缺行")

    if args.dry_run:
        print("[split] dry-run，未写文件")
        return

    for name, path in out_mats.items():
        if path.exists() and not args.force:
            raise SystemExit(
                f"[split] 已存在 {path.name}，如确认覆盖请加 --force"
            )

    for name, loc in splits.items():
        save_split_mat(out_mats[name], samples[loc])

    manifest = {
        "source": str(src),
        "seed": int(args.seed),
        "valid_criterion": f"row_l2_norm > {VALID_NORM_EPS}",
        "n_valid": int(n_valid),
        "dim": int(dim),
        "n_train": n_train,
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "ratio_train_val_test": [float(x) for x in ratio],
        "outputs": {k: str(v) for k, v in out_mats.items()},
        "original_row_idx": {
            name: row_ids[loc].astype(int).tolist() for name, loc in splits.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[split] 已写入划分产物")


if __name__ == "__main__":
    main()
