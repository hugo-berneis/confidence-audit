"""Chart generation. PNG only -- no web dashboard, per CLAUDE.md's non-goals."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness.metrics.calibration import ReliabilityBin


def plot_reliability_diagram(
    bins: list[ReliabilityBin], title: str, output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    centers = [(b.lower + b.upper) / 2 for b in bins]
    accuracies = [b.accuracy for b in bins]
    width = (1.0 / len(bins) if bins else 0.1) * 0.9

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(centers, accuracies, width=width, edgecolor="black", label="accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
