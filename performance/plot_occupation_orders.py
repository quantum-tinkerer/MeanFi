"""Show the interpolation/probe distinction and the measured error orders."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    data = json.loads(source.read_text())["coupled"]
    fig, (nodes, orders) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    nodes.plot([0, 1, 0, 0], [0, 0, 1, 0], color=".65", lw=1)
    groups = {True: [], False: []}
    for i in range(5):
        for j in range(5 - i):
            alpha = (4 - i - j, i, j)
            groups[all(a % 2 == 0 for a in alpha)].append((i / 4, j / 4))
    for fitted, label, color in (
        (True, "6 quadratic interpolation nodes", "#167185"),
        (False, "9 additional validation probes", "#c47832"),
    ):
        points = np.array(groups[fitted])
        nodes.scatter(*points.T, color=color, s=45, label=label, zorder=3)
    nodes.set(
        aspect="equal",
        xlim=(-0.08, 1.08),
        ylim=(-0.08, 1.08),
        title="Triangle: fit a quadratic, check through degree four",
    )
    nodes.axis("off")
    nodes.legend(frameon=False, fontsize=8, loc="upper right")
    h = np.array([row["h"] for row in data])
    for name, label, color in (
        ("physical_charge_error", "Affine charge error (≈ h²)", "#666666"),
        ("model_allowance", "Full model allowance (≈ h³)", "#167185"),
        ("native_quartic_bound", "Squared-residual term (≈ h⁴)", "#c47832"),
    ):
        values = np.array([row[name] for row in data])
        orders.loglog(h, values / values[0], "o-", ms=4, label=label, color=color)
    orders.set(
        xlabel="Physical cell width h",
        ylabel="Error / value at h = 0.4",
        title="Coupled two-band example",
    )
    orders.set_xticks(h)
    orders.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    orders.legend(frameon=False, fontsize=8)
    orders.grid(alpha=0.15)
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
