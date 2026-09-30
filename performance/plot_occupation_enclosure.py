"""Plot the saved experiment using the existing documentation dependencies."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(native, paper, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    colors = {"legacy": "#777777", "quadratic": "#167d9a"}
    for method, color in colors.items():
        rows = [
            r for r in native["charge"] if r["method"] == method and "failure" not in r
        ]
        axes[0].scatter(
            [r["actual_error"] for r in rows],
            [r["estimated_error"] for r in rows],
            label=method,
            color=color,
            marker="o" if method == "quadratic" else "x",
            alpha=0.8,
        )
    axes[0].plot([1e-6, 1], [1e-6, 1], "k--", lw=1, label="estimate = error")
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Actual charge error",
        ylabel="Charge error indicator",
        title="Exact-reference charge checks",
    )
    axes[0].legend(fontsize=8)

    labels, ratios = [], []
    for old, new in zip(native["charge"][::2], native["charge"][1::2], strict=True):
        if old["target"] >= 1e-3 or "failure" in old or "failure" in new:
            continue
        labels.append(old["model"].replace("_", " "))
        ratios.append(new["seconds"] / old["seconds"])
    for old, new in zip(paper[::2], paper[1::2], strict=True):
        if (old["dimension"], old["target"]) not in ((1, 1e-3), (2, 1e-2)):
            continue
        if "failure" in old or "failure" in new:
            continue
        labels.append(f"paper 36, {old['dimension']}D")
        ratios.append(new["seconds"] / old["seconds"])
    axes[1].barh(
        labels, ratios, color=["#167d9a" if r < 1 else "#ad6947" for r in ratios]
    )
    axes[1].axvline(1, color="black", ls="--", lw=1)
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="Quadratic / legacy wall time", title="Matched targets; lower is faster"
    )

    rows = native["certification"]
    keys = ("legacy_gapped", "quadratic_gapped", "quadratic_depth6_gapped")
    false = [sum(r["pocket"] and r[key] for r in rows) for key in keys]
    missed = [sum(not r["pocket"] and not r[key] for r in rows) for key in keys]
    x = np.arange(3)
    axes[2].bar(
        x - 0.18,
        false,
        width=0.36,
        color="#ad6947",
        label="False gap claims / 45 pockets",
    )
    axes[2].bar(
        x + 0.18,
        missed,
        width=0.36,
        color="#167d9a",
        label="Unproved gaps / 45 gapped cases",
    )
    axes[2].set(
        xticks=x,
        xticklabels=[
            "Vertex test\nzero allowance",
            "Quadratic\ndepth 2",
            "Quadratic\ndepth 6",
        ],
        ylabel="Cases",
        title="Exact quadratic sign tests",
        ylim=(0, 54),
    )
    axes[2].legend(fontsize=8, loc="upper right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(
        json.loads(args.native_results.read_text()),
        json.loads(args.paper_results.read_text()),
        args.output,
    )
