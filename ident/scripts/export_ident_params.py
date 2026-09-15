"""Export saved sparse FF-RLS parameter trajectories to readable files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PARAMETERS = (
    ("D1", "A21", 0),
    ("D1", "B1", 1),
    ("D1", "B2", 2),
    ("D1", "C1", 3),
    ("D2", "A21", 4),
    ("D2", "B1", 5),
    ("D2", "B2", 6),
    ("D2", "C1", 7),
    ("T", "A21", 8),
    ("T", "B1", 9),
    ("T", "B2", 10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出参数辨识逐拍结果")
    parser.add_argument("input", type=Path, help="shadow_diag_series.npz")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    return parser.parse_args()


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    out_dir = (args.out_dir or source.parent / "parameter_exports").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(source, allow_pickle=True)
    episode_ids = np.asarray(data["local_idx"], dtype=int)
    hats = data["theta_hat_series"]
    truths = data["theta_true_series"]  # 兼容名：因果真值 vartheta[k-1]
    truths_current = (
        data["theta_true_current_series"]
        if "theta_true_current_series" in data.files
        else truths
    )

    timeseries_path = out_dir / "ident_params_timeseries.csv"
    timeseries_fields = ["episode", "step", "time_s"]
    for channel, name, _ in PARAMETERS:
        prefix = f"{channel}_{name}"
        timeseries_fields.extend(
            (
                f"{prefix}_hat",
                f"{prefix}_true_causal",
                f"{prefix}_true_current",
                f"{prefix}_causal_error",
                f"{prefix}_command_error",
            )
        )

    with timeseries_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=timeseries_fields)
        writer.writeheader()
        for episode, hat_obj, true_obj, current_obj in zip(
            episode_ids, hats, truths, truths_current
        ):
            hat = np.asarray(hat_obj, dtype=float)
            true = np.asarray(true_obj, dtype=float)
            current = np.asarray(current_obj, dtype=float)
            if (
                hat.shape != true.shape
                or hat.shape != current.shape
                or hat.ndim != 2
                or hat.shape[1] != 11
            ):
                raise ValueError(
                    f"episode {episode}: theta shape mismatch "
                    f"{hat.shape} vs {true.shape} vs {current.shape}"
                )
            for step in range(hat.shape[0]):
                row: dict[str, float | int] = {
                    "episode": int(episode),
                    "step": step,
                    "time_s": step * args.dt,
                }
                for channel, name, index in PARAMETERS:
                    prefix = f"{channel}_{name}"
                    row[f"{prefix}_hat"] = float(hat[step, index])
                    row[f"{prefix}_true_causal"] = float(true[step, index])
                    row[f"{prefix}_true_current"] = float(current[step, index])
                    row[f"{prefix}_causal_error"] = float(
                        hat[step, index] - true[step, index]
                    )
                    row[f"{prefix}_command_error"] = float(
                        hat[step, index] - current[step, index]
                    )
                writer.writerow(row)

    tail_rows: list[dict[str, float | int | str]] = []
    final_rows: list[dict[str, float | int | str]] = []
    aggregate_rmse: dict[str, list[float]] = {
        f"{channel}_{name}": [] for channel, name, _ in PARAMETERS
    }
    aggregate_command_rmse: dict[str, list[float]] = {
        f"{channel}_{name}": [] for channel, name, _ in PARAMETERS
    }
    episode_summaries = []

    for episode, hat_obj, true_obj, current_obj in zip(
        episode_ids, hats, truths, truths_current
    ):
        hat = np.asarray(hat_obj, dtype=float)
        true = np.asarray(true_obj, dtype=float)
        current = np.asarray(current_obj, dtype=float)
        tail_count = max(5, int(0.2 * hat.shape[0]))
        hat_tail = hat[-tail_count:]
        true_tail = true[-tail_count:]
        current_tail = current[-tail_count:]
        episode_parameters = {}

        for channel, name, index in PARAMETERS:
            key = f"{channel}_{name}"
            error_tail = hat_tail[:, index] - true_tail[:, index]
            command_error_tail = hat_tail[:, index] - current_tail[:, index]
            rmse = float(np.sqrt(np.mean(error_tail**2)))
            command_rmse = float(np.sqrt(np.mean(command_error_tail**2)))
            mae = float(np.mean(np.abs(error_tail)))
            bias = float(np.mean(error_tail))
            tail_row = {
                "episode": int(episode),
                "channel": channel,
                "parameter": name,
                "tail_steps": tail_count,
                "hat_tail_mean": float(np.mean(hat_tail[:, index])),
                "true_tail_mean": float(np.mean(true_tail[:, index])),
                "true_current_tail_mean": float(np.mean(current_tail[:, index])),
                "tail_bias": bias,
                "tail_mae": mae,
                "tail_rmse": rmse,
                "command_tail_rmse": command_rmse,
            }
            final_row = {
                "episode": int(episode),
                "channel": channel,
                "parameter": name,
                "hat_final": float(hat[-1, index]),
                "true_final": float(true[-1, index]),
                "true_current_final": float(current[-1, index]),
                "final_error": float(hat[-1, index] - true[-1, index]),
                "command_final_error": float(hat[-1, index] - current[-1, index]),
            }
            tail_rows.append(tail_row)
            final_rows.append(final_row)
            aggregate_rmse[key].append(rmse)
            aggregate_command_rmse[key].append(command_rmse)
            episode_parameters[key] = {**tail_row, **final_row}

        episode_summaries.append(
            {
                "episode": int(episode),
                "steps": int(hat.shape[0]),
                "time_end_s": float((hat.shape[0] - 1) * args.dt),
                "parameters": episode_parameters,
            }
        )

    tail_path = out_dir / "ident_params_tail_summary.csv"
    with tail_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tail_rows[0]))
        writer.writeheader()
        writer.writerows(tail_rows)

    final_path = out_dir / "ident_params_final_values.csv"
    with final_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)

    summary = {
        "source": str(source),
        "episodes": int(len(episode_ids)),
        "dt_s": float(args.dt),
        "parameter_order": [f"{channel}_{name}" for channel, name, _ in PARAMETERS],
        "tail_rmse_across_episodes": {
            name: _finite_stats(np.asarray(values)) for name, values in aggregate_rmse.items()
        },
        "command_tail_rmse_across_episodes": {
            name: _finite_stats(np.asarray(values))
            for name, values in aggregate_command_rmse.items()
        },
        "per_episode": episode_summaries,
    }
    summary_path = out_dir / "ident_params_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )

    print(f"timeseries={timeseries_path}")
    print(f"tail_summary={tail_path}")
    print(f"final_values={final_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
