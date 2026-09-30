"""Plot fixed-mesh matrix scaling from separate Release-build timings."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(paths, output):
    rows = [row for path in paths for row in json.loads(path.read_text())]
    sizes = sorted({row["bands"] for row in rows})
    samples = {
        (build, n): np.array(
            [
                time
                for row in rows
                if row["build"] == build and row["bands"] == n
                for time in row["timing_samples"]
            ]
        )
        for build in ("legacy", "ae3ce87", "optimized")
        for n in sizes
    }
    fig, (absolute, ratio) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    baseline = np.array([np.median(samples["legacy", n]) for n in sizes])
    for build, label, color in (
        ("legacy", "Old estimator", "#666666"),
        ("ae3ce87", "Before optimization", "#c47832"),
        ("optimized", "After optimization", "#167185"),
    ):
        times = np.array([np.median(samples[build, n]) for n in sizes])
        absolute.plot(sizes, times, "o-", ms=4, label=label, color=color)
        if build != "legacy":
            low, high = np.array(
                [np.quantile(samples[build, n], [0.1, 0.9]) for n in sizes]
            ).T
            ratio.errorbar(
                sizes,
                times / baseline,
                yerr=[(times - low) / baseline, (high - times) / baseline],
                marker="o",
                ms=4,
                capsize=3,
                label=label,
                color=color,
            )
    guide_sizes = np.array([192, 1024])
    guide = baseline[sizes.index(192)] * (guide_sizes / 192) ** 3
    absolute.plot(guide_sizes, guide, ":", color=".3", label="Cubic guide")
    absolute.set(yscale="log", ylabel="Seconds per integration")
    absolute.legend(frameon=False, fontsize=8)
    ratio.axhline(1, color=".4", ls="--", lw=1)
    ratio.set(ylabel="Time / old estimator time")
    ratio.set_ylim(bottom=0.8)
    for ax in (absolute, ratio):
        ax.set(xscale="log", xlabel="Bands (matrix dimension)")
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(alpha=0.15)
    fig.suptitle("Fixed physics and mesh: 15 vertices, 10 refinements", fontsize=11)
    fig.savefig(output)
    if output.suffix == ".svg":
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )
    fig.savefig(output.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
