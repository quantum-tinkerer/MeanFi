import json
from pathlib import Path
import sys

import numpy as np
import pytest

from meanfi.tests.helpers import BenchmarkResult, DenseReference
import performance.benchmarks.density_scaling as density_scaling
from performance._shared.plots import (
    _bytes_to_mib,
    _comparison_label,
    _cost_curves,
    _interpolate_log_y_at_x,
    _pareto_frontier,
    write_density_comparison_plots,
)


pytestmark = [
    pytest.mark.performance,
    pytest.mark.filterwarnings("ignore::pyparsing.warnings.PyparsingDeprecationWarning"),
]


def _sample_records() -> list[dict[str, object]]:
    return [
        {
            "scenario": "block_chain_zt_comparison",
            "method": "adaptive_simplex",
            "ndof": 4,
            "control_name": "density_matrix_tol",
            "control_value": 3e-3,
            "wall_s": 5e-3,
            "peak_rss_bytes": 3 * 1024**2,
            "unique_evals": 40,
            "density_matrix_error": 9e-4,
            "filling_error": 1e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "adaptive_simplex",
            "ndof": 4,
            "control_name": "density_matrix_tol",
            "control_value": 1e-2,
            "wall_s": 2e-3,
            "peak_rss_bytes": 2 * 1024**2,
            "unique_evals": 24,
            "density_matrix_error": 2e-2,
            "filling_error": 2e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "uniform_grid",
            "ndof": 4,
            "control_name": "nk",
            "control_value": 65,
            "wall_s": 3e-3,
            "peak_rss_bytes": 2.5 * 1024**2,
            "unique_evals": 65,
            "density_matrix_error": 1.4e-3,
            "filling_error": 1e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "uniform_grid",
            "ndof": 4,
            "control_name": "nk",
            "control_value": 17,
            "wall_s": 1.8e-3,
            "peak_rss_bytes": 1.8 * 1024**2,
            "unique_evals": 17,
            "density_matrix_error": 2.2e-2,
            "filling_error": 2e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "adaptive_simplex",
            "ndof": 8,
            "control_name": "density_matrix_tol",
            "control_value": 3e-3,
            "wall_s": 9e-3,
            "peak_rss_bytes": 5 * 1024**2,
            "unique_evals": 44,
            "density_matrix_error": 1.1e-3,
            "filling_error": 1e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "adaptive_simplex",
            "ndof": 8,
            "control_name": "density_matrix_tol",
            "control_value": 1e-2,
            "wall_s": 4e-3,
            "peak_rss_bytes": 7 * 1024**2,
            "unique_evals": 28,
            "density_matrix_error": 2.4e-2,
            "filling_error": 2e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "uniform_grid",
            "ndof": 8,
            "control_name": "nk",
            "control_value": 65,
            "wall_s": 6e-3,
            "peak_rss_bytes": 5 * 1024**2,
            "unique_evals": 65,
            "density_matrix_error": 7e-3,
            "filling_error": 1e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "uniform_grid",
            "ndof": 8,
            "control_name": "nk",
            "control_value": 17,
            "wall_s": 3e-3,
            "peak_rss_bytes": 3 * 1024**2,
            "unique_evals": 17,
            "density_matrix_error": 2.8e-2,
            "filling_error": 2e-6,
            "reference_nk": 1025,
            "reference_tol": 1e-6,
        },
    ]


def test_density_comparison_plots_are_written_and_replace_legacy_bundle(tmp_path: Path):
    legacy = tmp_path / "wall_vs_unique_evals.png"
    legacy.write_text("stale", encoding="utf-8")

    written = write_density_comparison_plots(
        _sample_records(),
        tmp_path,
        fixed_density_errors=(1e-2, 5e-3),
    )

    expected = {
        tmp_path / "wall_vs_error.png",
        tmp_path / "memory_vs_error.png",
        tmp_path / "wall_vs_ndof_at_fixed_error_1e-02.png",
        tmp_path / "wall_vs_ndof_at_fixed_error_5e-03.png",
    }
    assert expected.issubset(set(written))
    for path in expected:
        assert path.exists()
    assert not legacy.exists()


def test_pareto_frontier_drops_dominated_points():
    xs = np.array([1e-4, 2e-4, 5e-4, 1e-3], dtype=float)
    ys = np.array([4e-2, 5e-2, 2e-2, 3e-2], dtype=float)

    frontier_x, frontier_y = _pareto_frontier(xs, ys)

    assert frontier_x.tolist() == pytest.approx([1e-4, 5e-4])
    assert frontier_y.tolist() == pytest.approx([4e-2, 2e-2])


def test_cost_curves_keep_flat_rss_points():
    records = [
        {
            "scenario": "block_chain_zt_comparison",
            "method": "adaptive_simplex",
            "ndof": 8,
            "density_matrix_error": 1e-3,
            "peak_rss_bytes": 4096,
        },
        {
            "scenario": "block_chain_zt_comparison",
            "method": "adaptive_simplex",
            "ndof": 8,
            "density_matrix_error": 5e-3,
            "peak_rss_bytes": 4096,
        },
    ]

    curves = _cost_curves(records, y_key="peak_rss_bytes")

    assert len(curves) == 1
    _, xs, ys = curves[0]
    assert xs.tolist() == pytest.approx([1e-3, 5e-3])
    assert ys.tolist() == pytest.approx([4096, 4096])


def test_bytes_to_mib_uses_binary_units():
    assert _bytes_to_mib(1024**2) == pytest.approx(1.0)


def test_log_interpolation_returns_expected_value():
    xs = np.array([1e-3, 1e-2], dtype=float)
    ys = np.array([1.0, 0.1], dtype=float)

    interpolated = _interpolate_log_y_at_x(xs, ys, target_x=10 ** -2.5)

    assert interpolated == pytest.approx(10 ** -0.5)


def test_comparison_labels_include_method_and_ndof():
    record = {
        "scenario": "block_chain_zt_comparison",
        "method": "adaptive_simplex",
        "ndof": 8,
    }

    assert _comparison_label(record) == "AdaptiveSimplex (ndof=8)"


def test_dense_reference_backoff_relaxes_target(monkeypatch):
    calls = []

    def fake_converged_dense_reference(*args, **kwargs):
        calls.append(kwargs["target_tol"])
        if kwargs["target_tol"] < 3e-3:
            raise AssertionError("Dense reference did not self-converge by nk=4001")
        return DenseReference(
            mu=0.0,
            charge=1.0,
            rho={(0,): np.eye(1, dtype=complex)},
            nk=1025,
        )

    monkeypatch.setattr(
        density_scaling,
        "converged_dense_reference",
        fake_converged_dense_reference,
    )

    reference, meta = density_scaling._dense_reference_with_backoff(
        {(0,): np.eye(1, dtype=complex)},
        keys=[(0,)],
        kT=0.0,
        target_tol=1.5e-3,
        nk_start=251,
        nk_max=4001,
        mu=0.0,
    )

    assert reference.nk == 1025
    assert calls == pytest.approx([1.5e-3, 2.25e-3, 3e-3])
    assert meta["reference_nk"] == 1025
    assert meta["reference_tol"] == pytest.approx(3e-3)


def test_density_measurement_prefers_peak_rss(monkeypatch):
    expected_result = object()

    monkeypatch.setattr(
        density_scaling,
        "benchmark",
        lambda *args, **kwargs: BenchmarkResult(
            median_s=1.0,
            mean_s=1.0,
            stdev_s=0.0,
            last_result=expected_result,
            peak_traced_bytes=111,
        ),
    )
    monkeypatch.setattr(density_scaling, "_peak_density_rss_bytes", lambda **kwargs: 222)

    measurement, result = density_scaling._density_measurement(
        {(0,): np.eye(2, dtype=complex)},
        mu=0.0,
        kT=0.0,
        keys=[(0,)],
        integration=density_scaling.AdaptiveSimplex(density_matrix_tol=1e-3),
        repeat=1,
        warmup=0,
    )

    assert measurement.peak_traced_bytes == 222
    assert result is expected_result


def test_reduced_density_record_schema():
    measurement = BenchmarkResult(
        median_s=1.0,
        mean_s=1.0,
        stdev_s=0.0,
        last_result=None,
        peak_traced_bytes=1234,
    )
    density_result = type("DensityResult", (), {"info": type("Info", (), {"unique_evals": 12})()})()

    record = density_scaling._density_record(
        integration=density_scaling.AdaptiveSimplex(density_matrix_tol=1e-3),
        ndof=8,
        control_name="density_matrix_tol",
        control_value=1e-3,
        benchmark_result=measurement,
        density_result=density_result,
        density_matrix_error=2e-4,
        filling_error=1e-6,
        reference_nk=1025,
        reference_tol=1e-6,
    )

    assert set(record) == {
        "scenario",
        "method",
        "ndof",
        "control_name",
        "control_value",
        "wall_s",
        "peak_rss_bytes",
        "unique_evals",
        "density_matrix_error",
        "filling_error",
        "reference_nk",
        "reference_tol",
    }
    assert "peak_memory_bytes" not in record


def test_density_summary_is_lean_without_verbose(capsys):
    density_scaling.print_density_comparison_summary(_sample_records(), verbose=False)

    output = capsys.readouterr().out

    assert "best_peak_rss=" in output
    assert "log_slope" not in output
    assert "fixed_density_error" not in output
    assert "best@<=" not in output


def test_density_summary_verbose_prints_only_error_slopes(capsys):
    density_scaling.print_density_comparison_summary(_sample_records(), verbose=True)

    output = capsys.readouterr().out

    assert "log_slope_wall_vs_error=" in output
    assert "log_slope_rss_vs_error=" in output
    assert "log_slope_wall_vs_ndof=" not in output
    assert "fixed_density_error" not in output


def test_density_main_quick_writes_json_without_plots(tmp_path: Path, monkeypatch, capsys):
    sample_records = _sample_records()
    output_path = tmp_path / "density_scaling.json"

    monkeypatch.setattr(density_scaling, "_zero_temperature_records", lambda **kwargs: sample_records)
    monkeypatch.setattr(
        sys,
        "argv",
        ["density_scaling.py", "--profile", "quick", "--output", str(output_path)],
    )

    density_scaling.main()

    output = capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["records"] == sample_records
    assert "plot=" not in output
    assert "log_slope" not in output
    assert not list(tmp_path.glob("*.png"))


def test_density_main_manual_writes_plots_and_verbose_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    sample_records = _sample_records()
    output_path = tmp_path / "density_scaling.json"
    plot_dir = tmp_path / "plots"

    monkeypatch.setattr(density_scaling, "_zero_temperature_records", lambda **kwargs: sample_records)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "density_scaling.py",
            "--profile",
            "full",
            "--output",
            str(output_path),
            "--plot-dir",
            str(plot_dir),
        ],
    )

    density_scaling.main()

    output = capsys.readouterr().out

    assert "log_slope_wall_vs_error=" in output
    assert "log_slope_rss_vs_error=" in output
    assert "log_slope_wall_vs_ndof=" not in output
    assert "plot=" in output
    assert (plot_dir / "wall_vs_error.png").exists()
    assert (plot_dir / "memory_vs_error.png").exists()
    assert (plot_dir / "wall_vs_ndof_at_fixed_error_1e-02.png").exists()
