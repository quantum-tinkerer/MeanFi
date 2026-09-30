"""Plot separate-build timing ratios, including variation between timing batches."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    rows = json.loads(source.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained", sharey=True)
    for ax, (family, representation, title) in zip(
        axes,
        [
            ("spectators", "tb", "One active band · tight binding"),
            ("spectators", "callable", "One active band · callable"),
            ("replicas", "tb", "More active bands · tight binding"),
        ],
    ):
        cases = [
            r
            for r in rows
            if r["family"] == family and r["representation"] == representation
        ]
        sizes = sorted({r["bands"] for r in cases})
        for variant, label, color in [
            ("previous", "Previous enclosure", "#cf7c32"),
            ("current", "Revised enclosure", "#167185"),
        ]:
            means, lower, upper = [], [], []
            for size in sizes:
                samples = {
                    v: np.array(
                        [
                            t
                            for r in cases
                            if r["bands"] == size and r["variant"] == v
                            for t in r["timing_samples"]
                        ]
                    )
                    for v in ("legacy", variant)
                }
                baseline = np.median(samples["legacy"])
                median = np.median(samples[variant]) / baseline
                means.append(median)
                lower.append(median - np.quantile(samples[variant], 0.1) / baseline)
                upper.append(np.quantile(samples[variant], 0.9) / baseline - median)
            ax.errorbar(
                sizes,
                means,
                yerr=[lower, upper],
                marker="o",
                ms=4,
                capsize=3,
                color=color,
                label=label,
            )
        ax.axhline(1, color=".4", lw=1, ls="--")
        ax.set(xscale="log", xlabel="Matrix size / band count", title=title)
        ax.set_xticks(
            [4, 12, 36, 96, 192] if family == "spectators" else [4, 12, 36, 96]
        )
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Time / historical estimator time")
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    fig.savefig(output)
    if output.suffix == ".svg":
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )
    fig.savefig(output.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
