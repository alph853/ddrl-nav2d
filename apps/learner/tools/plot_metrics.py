#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_HEADER = [
    "timestamp",
    "event",
    "num_updates",
    "sim_id",
    "rollout_id",
    "steps",
    "episode_return",
    "episode_terminal",
    "last_step_terminal",
    "sum_reward",
    "mean_reward",
    "min_reward",
    "max_reward",
    "mean_speed",
    "mean_abs_steer",
    "batch_mean_return",
    "loss_total",
    "loss_actor",
    "loss_value",
    "loss_entropy",
    "loss_bc",
    "steps_in_update",
    "mean_rho",
    "max_rho",
    "rollouts_per_sec",
    "batches_per_sec",
    "buffer_size",
    "rollouts_dropped",
    "policy_version",
    "updates_since_checkpoint",
    "bc_coef",
    "expert_rollouts",
    "neural_rollouts",
    "rl_steps",
    "bc_steps",
    "behaviour_log_prob_mean",
    "behaviour_log_prob_min",
    "behaviour_log_prob_max",
    "action_speed_mean",
    "action_steer_mean",
    "raw_speed_mean",
    "raw_steer_mean",
    "reward_mean",
    "reward_min",
    "reward_max",
    "grad_norm",
    "max_abs_grad",
    "grad_params",
    "bc_speed_mae",
    "bc_steer_mae",
    "rollout_epoch",
    "rollout_epochs_per_update",
    "rollouts_processed",
    "rollout_steps_processed",
    "checkpoint_elapsed_sec",
]


ROLLING_FIELDS = {
    "episode_return",
    "mean_reward",
    "mean_speed",
    "mean_abs_steer",
    "batch_mean_return",
    "loss_total",
    "loss_actor",
    "loss_value",
    "loss_entropy",
    "loss_bc",
    "mean_rho",
    "max_rho",
    "rollouts_per_sec",
    "batches_per_sec",
    "steps_in_update",
    "updates_since_checkpoint",
    "bc_coef",
    "expert_rollouts",
    "neural_rollouts",
    "rl_steps",
    "bc_steps",
    "grad_norm",
    "max_abs_grad",
    "bc_speed_mae",
    "bc_steer_mae",
    "behaviour_log_prob_mean",
    "behaviour_log_prob_min",
    "behaviour_log_prob_max",
    "rollouts_processed",
    "rollout_steps_processed",
    "checkpoint_elapsed_sec",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        if "event" in first_line and "timestamp" in first_line:
            reader = csv.DictReader(handle)
        else:
            reader = csv.DictReader(handle, fieldnames=DEFAULT_HEADER)
        return [row for row in reader if row]


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def take_window(values: list[float], window: int) -> list[float]:
    if window <= 0:
        return values
    return values[-window:]


def rolling_mean(values: list[float], span: int) -> list[float]:
    if span <= 1 or not values:
        return values
    out: list[float] = []
    acc = 0.0
    for idx, value in enumerate(values):
        acc += value
        if idx >= span:
            acc -= values[idx - span]
            denom = span
        else:
            denom = idx + 1
        out.append(acc / denom)
    return out


def event_rows(rows: list[dict[str, str]], event: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("event") == event]


def training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    updates = event_rows(rows, "update")
    return updates if updates else event_rows(rows, "checkpoint")


def series(
    rows: list[dict[str, str]],
    field: str,
    *,
    window: int,
    smooth: int = 1,
) -> tuple[list[int], list[float]]:
    values = [to_float(row.get(field)) for row in rows]
    values = rolling_mean(values, smooth if field in ROLLING_FIELDS else 1)
    values = take_window(values, window)
    start = max(0, len(rows) - len(values))
    return list(range(start, start + len(values))), values


def field_available(rows: list[dict[str, str]], field: str) -> bool:
    return any(row.get(field) not in ("", None) for row in rows)


def plot_lines(
    ax,
    rows: list[dict[str, str]],
    fields: Iterable[tuple[str, str]],
    *,
    title: str,
    window: int,
    smooth: int,
    ylabel: str | None = None,
) -> None:
    ax.cla()
    ax.set_title(title)
    plotted = False
    for field, label in fields:
        if not field_available(rows, field):
            continue
        x, y = series(rows, field, window=window, smooth=smooth)
        if y:
            ax.plot(x, y, label=label)
            plotted = True
    if ylabel:
        ax.set_ylabel(ylabel)
    if plotted:
        ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)


def plot_counter_delta(
    ax,
    rows: list[dict[str, str]],
    field: str,
    *,
    title: str,
    window: int,
    label: str,
) -> None:
    ax.cla()
    ax.set_title(title)
    values = [to_float(row.get(field)) for row in rows]
    deltas = [0.0]
    for prev, cur in zip(values, values[1:]):
        deltas.append(max(0.0, cur - prev))
    deltas = take_window(deltas, window)
    start = max(0, len(values) - len(deltas))
    if deltas:
        ax.bar(list(range(start, start + len(deltas))), deltas, label=label, width=0.9)
        ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)


def plot_terminal_rate(
    ax,
    rollouts: list[dict[str, str]],
    *,
    window: int,
    smooth: int,
) -> None:
    ax.cla()
    ax.set_title("Terminal Rate")
    if not rollouts:
        return
    terminal = [to_float(row.get("episode_terminal")) for row in rollouts]
    last_step = [to_float(row.get("last_step_terminal")) for row in rollouts]
    terminal = rolling_mean(terminal, max(1, smooth))
    last_step = rolling_mean(last_step, max(1, smooth))
    terminal = take_window(terminal, window)
    last_step = take_window(last_step, window)
    start = max(0, len(rollouts) - len(terminal))
    x = list(range(start, start + len(terminal)))
    if terminal:
        ax.plot(x, terminal, label="episode_terminal")
    if last_step:
        ax.plot(x[-len(last_step) :], last_step, label="last_step_terminal")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)


def draw_dashboard(
    fig,
    axes,
    rows: list[dict[str, str]],
    *,
    csv_path: Path,
    window: int,
    smooth: int,
) -> None:
    rollouts = event_rows(rows, "rollout")
    updates = training_rows(rows)
    training_event = (
        "checkpoints"
        if updates and updates[0].get("event") == "checkpoint"
        else "updates"
    )
    fig.suptitle(
        f"Learner Metrics - {csv_path} ({len(rollouts)} rollouts, {len(updates)} {training_event})"
    )

    plot_lines(
        axes[0, 0],
        rollouts,
        [
            ("episode_return", "episode_return"),
            ("sum_reward", "sum_reward"),
        ],
        title="Rollout Returns",
        window=window,
        smooth=smooth,
    )
    plot_lines(
        axes[0, 1],
        rollouts,
        [
            ("mean_reward", "mean_reward"),
            ("min_reward", "min_reward"),
            ("max_reward", "max_reward"),
        ],
        title="Rollout Rewards",
        window=window,
        smooth=smooth,
    )
    plot_lines(
        axes[0, 2],
        rollouts,
        [
            ("mean_speed", "mean_speed"),
            ("mean_abs_steer", "mean_abs_steer"),
            ("steps", "steps"),
        ],
        title="Rollout Actions / Length",
        window=window,
        smooth=smooth,
    )

    plot_lines(
        axes[1, 0],
        updates,
        [
            ("loss_bc", "bc anchor"),
            ("loss_actor", "rl actor"),
            ("loss_value", "rl value"),
            ("loss_total", "total"),
        ],
        title="BC vs RL Loss",
        window=window,
        smooth=smooth,
        ylabel="loss",
    )
    plot_lines(
        axes[1, 1],
        updates,
        [
            ("bc_speed_mae", "speed_mae"),
            ("bc_steer_mae", "steer_mae"),
            ("bc_coef", "bc_coef"),
        ],
        title="BC Anchor Drift",
        window=window,
        smooth=smooth,
    )
    plot_lines(
        axes[1, 2],
        updates,
        [
            ("loss_entropy", "entropy"),
            ("mean_rho", "mean_rho"),
            ("max_rho", "max_rho"),
        ],
        title="RL Stability",
        window=window,
        smooth=smooth,
    )

    plot_lines(
        axes[2, 0],
        updates,
        [
            ("expert_rollouts", "expert_rollouts"),
            ("neural_rollouts", "neural_rollouts"),
            ("rl_steps", "rl_steps"),
            ("bc_steps", "bc_steps"),
        ],
        title="Expert / Neural Mix",
        window=window,
        smooth=smooth,
    )
    plot_lines(
        axes[2, 1],
        updates,
        [
            ("grad_norm", "grad_norm"),
            ("max_abs_grad", "max_abs_grad"),
            ("behaviour_log_prob_mean", "behaviour_log_prob_mean"),
            ("behaviour_log_prob_min", "behaviour_log_prob_min"),
        ],
        title="Gradients / Behaviour Log Prob",
        window=window,
        smooth=smooth,
    )
    plot_lines(
        axes[2, 2],
        updates,
        [
            ("batch_mean_return", "batch_mean_return"),
            ("rollouts_processed", "rollouts"),
            ("rollout_steps_processed", "rollout_steps"),
            ("checkpoint_elapsed_sec", "elapsed_sec"),
            ("rollouts_per_sec", "rollouts/sec"),
            ("rollouts_dropped", "rollouts_dropped"),
        ],
        title="Checkpoint Health",
        window=window,
        smooth=smooth,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learner_metrics.csv.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("learner_checkpoints/learner_metrics.csv"),
        help="Path to learner_metrics.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path for headless or --once mode. Defaults to <csv>.png.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Refresh interval in seconds for live mode.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Max points to show per panel. Use <=0 for all points.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling mean span for noisy scalar series.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render once and exit.",
    )
    parser.add_argument(
        "--counter-deltas",
        action="store_true",
        help="Plot per-update rollouts_dropped deltas instead of cumulative drops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.csv.with_suffix(".png")
    interactive = bool(os.environ.get("DISPLAY")) and not args.once

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), constrained_layout=True)
    if interactive:
        plt.ion()

    while True:
        rows = load_rows(args.csv)
        if rows:
            draw_dashboard(
                fig,
                axes,
                rows,
                csv_path=args.csv,
                window=int(args.window),
                smooth=max(1, int(args.smooth)),
            )
            if args.counter_deltas:
                updates = training_rows(rows)
                plot_counter_delta(
                    axes[2, 1],
                    updates,
                    "rollouts_dropped",
                    title="Dropped Rollouts Per Update",
                    window=int(args.window),
                    label="dropped_delta",
                )
            if interactive:
                plt.pause(max(float(args.refresh), 0.01))
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output, dpi=140)

        if args.once:
            break
        time.sleep(max(float(args.refresh), 0.01))


if __name__ == "__main__":
    main()
