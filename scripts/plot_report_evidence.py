#!/usr/bin/env python3
"""Render report figures from the saved step-four and step-five measurements."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "report_evidence"


def stage_figure() -> None:
    initial = json.loads((RESULTS / "step4_initial_stage_stats.json").read_text(encoding="utf-8"))["test"]
    retrained = json.loads((RESULTS / "step4_stage_stats.json").read_text(encoding="utf-8"))["test"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    for axis, key, title, denominator in zip(
        axes,
        ("positive_passing", "negative_passing"),
        ("Face crops retained", "Background crops passed"),
        (initial[0]["positive_entering"], initial[0]["negative_entering"]),
        strict=True,
    ):
        for label, rows, color in (("Initial", initial, "#687786"), ("After mining", retrained, "#c46542")):
            values = [row[key] for row in rows]
            axis.plot(x, values, marker="o", linewidth=2.2, label=label, color=color)
            for i, value in enumerate(values):
                axis.annotate(str(value), (i, value), xytext=(0, 7 if label == "Initial" else -15),
                              textcoords="offset points", ha="center", fontsize=8)
        axis.set_title(f"{title} (of {denominator})")
        axis.set_xticks(x, ["Stage 0", "Stage 1", "Stage 2"])
        axis.set_ylim(0, denominator * 1.12)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Crops")
    axes[0].legend(frameon=False)
    fig.suptitle("Cascade stages on reviewed test crops")
    fig.text(0.5, 0.01, "Crop classification only; not full-page IoU detection metrics.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUTPUT / "step4_stage_retention.png", dpi=180)
    plt.close(fig)


def search_figure() -> None:
    report = json.loads((RESULTS / "step5_comparison.json").read_text(encoding="utf-8"))
    grouped: dict[tuple[float, int], list[dict]] = {}
    for row in report["runs"]:
        grouped.setdefault((row["scale_factor"], row["step"]), []).append(row)
    factors = (1.1, 1.2, 1.3)
    x = np.arange(len(factors))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for step, offset, color in ((1, -0.18, "#687786"), (2, 0.18, "#c46542")):
        windows = [sum(row["total_windows"] for row in grouped[(factor, step)]) for factor in factors]
        seconds = [sum(row["total_seconds"] for row in grouped[(factor, step)]) for factor in factors]
        axes[0].bar(x + offset, np.array(windows) / 1000, width=0.34, label=f"step {step}", color=color)
        axes[1].bar(x + offset, seconds, width=0.34, label=f"step {step}", color=color)
    for axis, title, ylabel in zip(axes, ("Windows scanned", "Total time including NMS"),
                                   ("Thousands of windows", "Seconds"), strict=True):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [str(factor) for factor in factors])
        axis.set_xlabel("Pyramid scale factor")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Two fixed 320 x 320 validation crops")
    fig.text(0.5, 0.01, "Single-machine measurements; accuracy requires full-page evaluation.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUTPUT / "step5_scale_stride.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_figure()
    search_figure()
