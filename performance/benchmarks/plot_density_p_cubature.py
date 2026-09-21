"""Generate standalone figures from the density p-cubature benchmark records."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("performance/reports/density_p_cubature")
    )
    args = parser.parse_args()
    rows = json.loads((args.directory / "mesh_sweep.json").read_text())["records"]
    names = ["rotating_1d", "bulk_2d", "qwz_2d", "bulk_3d", "metal_2d"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
    for axis, name in zip(axes, names):
        for method, label, color in [
            ("h", "Density h-refinement", "#bc473c"),
            ("p", "Density p-refinement", "#1778a6"),
            ("centroid", "Vertices + centroid", "#b49c42"),
        ]:
            group = [r for r in rows if r["model"] == name and r["method"] == method]
            axis.loglog(
                [r["actual_error"] for r in group],
                [r["density_evaluations"] for r in group],
                "o-",
                label=label,
                color=color,
            )
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Actual max component error")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("New density diagonalizations")
    axes[0].legend(fontsize=7)
    for suffix in ["png", "pdf"]:
        fig.savefig(args.directory / f"work_vs_error.{suffix}", dpi=180)
    plt.close(fig)

    rows = json.loads((args.directory / "fixed_mesh.json").read_text())["records"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for axis, name in zip(axes, ["bulk_2d", "metal_2d"]):
        group = [r for r in rows if r["model"] == name and r["method"] == "p"]
        x = [r["tolerance"] for r in group]
        axis.loglog(x, [r["actual_error"] for r in group], "o-", label="Actual error")
        axis.loglog(
            x, [r["estimated_error"] for r in group], "s--", label="Cubature estimate"
        )
        axis.set_title(name.replace("_", " ") + " (fixed charge mesh)")
        axis.set_xlabel("Requested density tolerance")
        axis.set_ylabel("Max component error")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(args.directory / f"fixed_mesh.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
