from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from pyparsing.warnings import PyparsingDeprecationWarning


warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


_LEGACY_DENSITY_SCALING_PLOTS = (
    "wall_vs_unique_evals.png",
    "error_vs_unique_evals.png",
    "memory_vs_unique_evals.png",
    "wall_vs_ndof.png",
    "wall_per_unique_eval_vs_ndof.png",
    "memory_vs_ndof.png",
)

_METHOD_LABELS = {
    "adaptive_simplex": "AdaptiveSimplex",
    "uniform_grid": "UniformGrid",
}

_METHOD_STYLES = {
    "adaptive_simplex": {"linestyle": "-", "marker": "o"},
    "uniform_grid": {"linestyle": "-.", "marker": "^"},
}

_METHOD_COLORS = {
    "adaptive_simplex": "tab:blue",
    "uniform_grid": "tab:orange",
}


def _bytes_to_mib(value: np.ndarray | float) -> np.ndarray | float:
    if isinstance(value, np.ndarray):
        return value / (1024.0**2)
    return float(value) / (1024.0**2)


def _comparison_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (record.get("method"), record.get("ndof"))


def _comparison_label(record: dict[str, Any]) -> str:
    method = _METHOD_LABELS.get(record["method"], record["method"])
    return f"{method} (ndof={int(record['ndof'])})"


def _distinct_points(
    records: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, float] = {}
    for record in records:
        x_value = record.get(x_key)
        y_value = record.get(y_key)
        if x_value is None or y_value is None:
            continue
        x_float = float(x_value)
        y_float = float(y_value)
        if x_float <= 0.0 or y_float <= 0.0:
            continue
        best = grouped.get(x_float)
        if best is None or y_float < best:
            grouped[x_float] = y_float

    if not grouped:
        return np.array([]), np.array([])

    xs = np.array(sorted(grouped), dtype=float)
    ys = np.array([grouped[x] for x in xs], dtype=float)
    return xs, ys


def _pareto_frontier(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xs.size == 0:
        return xs, ys

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    frontier_x: list[float] = []
    frontier_y: list[float] = []
    best_cost = np.inf
    for x_value, y_value in zip(xs, ys, strict=False):
        if y_value < best_cost:
            frontier_x.append(float(x_value))
            frontier_y.append(float(y_value))
            best_cost = float(y_value)

    return np.array(frontier_x, dtype=float), np.array(frontier_y, dtype=float)


def _log_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float | None, float | None]:
    if xs.size < 2 or ys.size < 2:
        return None, None
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(slope), float(intercept)


def _interpolate_log_y_at_x(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    target_x: float,
) -> float | None:
    if xs.size == 0 or ys.size == 0 or target_x <= 0.0:
        return None
    if target_x < float(np.min(xs)) or target_x > float(np.max(xs)):
        return None
    log_xs = np.log(xs)
    log_ys = np.log(ys)
    log_y = np.interp(np.log(target_x), log_xs, log_ys)
    return float(np.exp(log_y))


def _cost_curves(
    records: list[dict[str, Any]],
    *,
    y_key: str,
) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray]]:
    grouped_records: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[_comparison_key(record)].append(record)

    curves: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for curve_key in sorted(grouped_records):
        curve = grouped_records[curve_key]
        xs, ys = _distinct_points(curve, x_key="density_matrix_error", y_key=y_key)
        if xs.size == 0:
            continue
        curves.append((curve[0], xs, ys))
    return curves


def _frontier_curves(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray]]:
    curves: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for sample_record, xs, ys in _cost_curves(records, y_key="wall_s"):
        frontier_x, frontier_y = _pareto_frontier(xs, ys)
        if frontier_x.size == 0:
            continue
        curves.append((sample_record, frontier_x, frontier_y))
    return curves


def _write_error_cost_plot(
    curves: list[tuple[dict[str, Any], np.ndarray, np.ndarray]],
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    y_transform=None,
) -> Path | None:
    if not curves:
        return None

    ndof_values = sorted({int(curve[0]["ndof"]) for curve in curves})
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(ndof_values), vmax=max(ndof_values))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    methods_in_plot: list[str] = []
    for sample_record, xs, ys in curves:
        method = str(sample_record["method"])
        style = _METHOD_STYLES.get(method, {"linestyle": "-", "marker": "o"})
        color = cmap(norm(int(sample_record["ndof"])))
        y_values = y_transform(ys) if y_transform is not None else ys
        ax.plot(
            xs,
            y_values,
            color=color,
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.8,
        )
        if method not in methods_in_plot:
            methods_in_plot.append(method)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Density Error")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=_METHOD_STYLES.get(method, {"linestyle": "-", "marker": "o"})["linestyle"],
            marker=_METHOD_STYLES.get(method, {"linestyle": "-", "marker": "o"})["marker"],
            linewidth=1.8,
            label=_METHOD_LABELS.get(method, method),
        )
        for method in methods_in_plot
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Method", fontsize="small")

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02, fraction=0.05)
    colorbar.set_label("ndof")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_density_comparison_plots(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    fixed_density_errors: tuple[float, ...] = (1e-2,),
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for legacy_name in _LEGACY_DENSITY_SCALING_PLOTS:
        legacy_path = output_path / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    legacy_fixed_error = output_path / "wall_vs_ndof_at_fixed_error.png"
    if legacy_fixed_error.exists():
        legacy_fixed_error.unlink()
    for legacy_path in output_path.glob("wall_vs_ndof_at_fixed_error_*.png"):
        legacy_path.unlink()

    wall_curves = _frontier_curves(records)
    rss_curves = _cost_curves(records, y_key="peak_rss_bytes")
    if not wall_curves:
        return []

    written: list[Path] = []
    wall_path = _write_error_cost_plot(
        wall_curves,
        output_path=output_path / "wall_vs_error.png",
        title="Wall Time vs Density Error",
        ylabel="Wall Time (s)",
    )
    if wall_path is not None:
        written.append(wall_path)

    rss_path = _write_error_cost_plot(
        rss_curves,
        output_path=output_path / "memory_vs_error.png",
        title="Peak RSS vs Density Error",
        ylabel="Peak RSS (MiB)",
        y_transform=_bytes_to_mib,
    )
    if rss_path is not None:
        written.append(rss_path)

    for fixed_density_error in fixed_density_errors:
        by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for sample_record, frontier_x, frontier_y in wall_curves:
            wall = _interpolate_log_y_at_x(
                frontier_x,
                frontier_y,
                target_x=fixed_density_error,
            )
            if wall is None:
                continue
            by_method[str(sample_record["method"])].append((int(sample_record["ndof"]), wall))

        if not by_method:
            continue

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for method in sorted(by_method):
            samples = sorted(by_method[method], key=lambda item: item[0])
            xs = np.array([sample[0] for sample in samples], dtype=float)
            ys = np.array([sample[1] for sample in samples], dtype=float)
            style = _METHOD_STYLES.get(method, {"linestyle": "-", "marker": "o"})
            color = _METHOD_COLORS.get(method, "black")
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8,
                label=_METHOD_LABELS.get(method, method),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Wall Time vs ndof at Density Error {fixed_density_error:.0e}")
        ax.set_xlabel("ndof")
        ax.set_ylabel("Wall Time (s)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")

        path = output_path / f"wall_vs_ndof_at_fixed_error_{fixed_density_error:.0e}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    return written
