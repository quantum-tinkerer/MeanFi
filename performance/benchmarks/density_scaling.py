from __future__ import annotations

import argparse
from dataclasses import replace
from typing import NamedTuple

from meanfi import AdaptiveSimplex, UniformGrid, density_matrix_at_mu
from meanfi.tests.helpers import (
    benchmark,
    converged_dense_reference,
    max_density_error,
    peak_rss_bytes,
)
from performance._shared.common import canonical_method_name, unique_eval_count, write_records
from performance._shared.plots import (
    _cost_curves,
    _frontier_curves,
    _log_fit,
    write_density_comparison_plots,
)
from performance._shared.scenarios import block_chain_keys, block_chain_model


ADAPTIVE_SIMPLEX_TOLS = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
UNIFORM_GRID_NKS = (9, 17, 33, 65, 129, 257, 513, 1025)
COMPARISON_NDOF_VALUES = (4, 8, 16, 32, 48, 64, 80, 96)
FIXED_DENSITY_ERRORS = (1e-2, 5e-3, 2e-3, 1e-3)
QUICK_UNIFORM_GRID_NKS = (17, 33, 65, 129, 257)
QUICK_COMPARISON_NDOF_VALUES = (4, 8, 16, 32, 48)
QUICK_FIXED_DENSITY_ERRORS = (1e-2, 5e-3)

_METHOD_LABELS = {
    "adaptive_simplex": "AdaptiveSimplex",
    "uniform_grid": "UniformGrid",
}


class DensityScalingProfile(NamedTuple):
    ndof_values: tuple[int, ...]
    uniform_grid_nks: tuple[int, ...]
    fixed_density_errors: tuple[float, ...]


PROFILES = {
    "quick": DensityScalingProfile(
        ndof_values=QUICK_COMPARISON_NDOF_VALUES,
        uniform_grid_nks=QUICK_UNIFORM_GRID_NKS,
        fixed_density_errors=QUICK_FIXED_DENSITY_ERRORS,
    ),
    "full": DensityScalingProfile(
        ndof_values=COMPARISON_NDOF_VALUES,
        uniform_grid_nks=UNIFORM_GRID_NKS,
        fixed_density_errors=FIXED_DENSITY_ERRORS,
    ),
}


def _method_label(method: str) -> str:
    return _METHOD_LABELS.get(method, method)


def _bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0**2)


def _integration_expr(integration: AdaptiveSimplex | UniformGrid) -> str:
    if isinstance(integration, AdaptiveSimplex):
        return (
            "AdaptiveSimplex("
            f"density_matrix_tol={integration.density_matrix_tol!r}, "
            f"max_refinements={integration.max_refinements!r})"
        )
    if isinstance(integration, UniformGrid):
        return f"UniformGrid(nk={integration.nk!r})"
    raise TypeError(f"Unsupported integration method for RSS measurement: {type(integration)!r}")


def _peak_density_rss_bytes(
    *,
    ndof: int,
    mu: float,
    kT: float,
    keys,
    integration: AdaptiveSimplex | UniformGrid,
) -> int | None:
    body = f"""
from meanfi import AdaptiveSimplex, UniformGrid, density_matrix_at_mu
from performance._shared.scenarios import block_chain_model

tb = block_chain_model({ndof!r})
keys = {list(keys)!r}
integration = {_integration_expr(integration)}
density_matrix_at_mu(
    tb,
    mu={mu!r},
    kT={kT!r},
    keys=keys,
    integration=integration,
)
"""
    return peak_rss_bytes(body)


def _density_measurement(
    tb,
    *,
    mu: float,
    kT: float,
    keys,
    integration: AdaptiveSimplex | UniformGrid,
    repeat: int,
    warmup: int,
):
    result = benchmark(
        lambda: density_matrix_at_mu(
            tb,
            mu=mu,
            kT=kT,
            keys=keys,
            integration=integration,
        ),
        repeat=repeat,
        warmup=warmup,
        track_tracemalloc=True,
    )
    peak_rss = _peak_density_rss_bytes(
        ndof=next(iter(tb.values())).shape[0],
        mu=mu,
        kT=kT,
        keys=keys,
        integration=integration,
    )
    if peak_rss is not None:
        result = replace(result, peak_traced_bytes=peak_rss)
    return result, result.last_result


def _adaptive_simplex_tolerances(ndof: int) -> tuple[float, ...]:
    if ndof >= 80:
        return (3e-2, 1e-2, 3e-3, 1e-3)
    return ADAPTIVE_SIMPLEX_TOLS


def _dense_reference_with_backoff(
    tb,
    *,
    keys,
    kT: float,
    target_tol: float,
    nk_start: int,
    nk_max: int,
    mu: float | None = None,
    filling: float | None = None,
    factors: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0),
):
    attempts = tuple(dict.fromkeys(target_tol * factor for factor in factors))
    last_error = None
    for actual_tol in attempts:
        try:
            reference = converged_dense_reference(
                tb,
                mu=mu,
                filling=filling,
                kT=kT,
                keys=keys,
                target_tol=actual_tol,
                nk_start=nk_start,
                nk_max=nk_max,
            )
            return reference, {
                "reference_nk": reference.nk,
                "reference_tol": actual_tol,
            }
        except AssertionError as exc:
            if "Dense reference did not self-converge" not in str(exc):
                raise
            last_error = exc
    if last_error is not None:
        raise last_error
    raise AssertionError("Dense reference generation failed unexpectedly")


def _density_record(
    *,
    integration: AdaptiveSimplex | UniformGrid,
    ndof: int,
    control_name: str,
    control_value: float | int,
    benchmark_result,
    density_result,
    density_matrix_error: float,
    filling_error: float,
    reference_nk: int,
    reference_tol: float,
) -> dict[str, object]:
    return {
        "scenario": "block_chain_zt_comparison",
        "method": canonical_method_name(integration),
        "ndof": int(ndof),
        "control_name": control_name,
        "control_value": control_value,
        "wall_s": float(benchmark_result.median_s),
        "peak_rss_bytes": None
        if benchmark_result.peak_traced_bytes is None
        else int(benchmark_result.peak_traced_bytes),
        "unique_evals": unique_eval_count(density_result.info),
        "density_matrix_error": float(density_matrix_error),
        "filling_error": float(filling_error),
        "reference_nk": int(reference_nk),
        "reference_tol": float(reference_tol),
    }


def _zero_temperature_records(
    *,
    profile: DensityScalingProfile,
    repeat: int,
    warmup: int,
) -> list[dict[str, object]]:
    mu = 0.0
    keys = block_chain_keys()
    records: list[dict[str, object]] = []

    for ndof in profile.ndof_values:
        tb = block_chain_model(ndof)
        reference, reference_meta = _dense_reference_with_backoff(
            tb,
            mu=mu,
            kT=0.0,
            keys=keys,
            target_tol=5e-4,
            nk_start=251,
            nk_max=4001,
            factors=(1.0, 2.0, 4.0, 6.0, 8.0),
        )

        for density_matrix_tol in _adaptive_simplex_tolerances(ndof):
            integration = AdaptiveSimplex(
                density_matrix_tol=density_matrix_tol,
                max_refinements=None,
            )
            measurement, result = _density_measurement(
                tb,
                mu=mu,
                kT=0.0,
                keys=keys,
                integration=integration,
                repeat=repeat,
                warmup=warmup,
            )
            records.append(
                _density_record(
                    integration=integration,
                    ndof=ndof,
                    control_name="density_matrix_tol",
                    control_value=density_matrix_tol,
                    benchmark_result=measurement,
                    density_result=result,
                    density_matrix_error=max_density_error(result.density_matrix, reference.rho),
                    filling_error=abs(result.filling - reference.charge),
                    **reference_meta,
                )
            )

        for nk in profile.uniform_grid_nks:
            integration = UniformGrid(nk=nk)
            measurement, result = _density_measurement(
                tb,
                mu=mu,
                kT=0.0,
                keys=keys,
                integration=integration,
                repeat=repeat,
                warmup=warmup,
            )
            records.append(
                _density_record(
                    integration=integration,
                    ndof=ndof,
                    control_name="nk",
                    control_value=nk,
                    benchmark_result=measurement,
                    density_result=result,
                    density_matrix_error=max_density_error(result.density_matrix, reference.rho),
                    filling_error=abs(result.filling - reference.charge),
                    **reference_meta,
                )
            )

    return records


def print_density_comparison_summary(
    records: list[dict[str, object]],
    *,
    verbose: bool,
) -> None:
    wall_curves = {
        (str(record["method"]), int(record["ndof"])): (record, xs, ys)
        for record, xs, ys in _frontier_curves(records)
    }
    rss_curves = {
        (str(record["method"]), int(record["ndof"])): (record, xs, ys)
        for record, xs, ys in _cost_curves(records, y_key="peak_rss_bytes")
    }

    for key in sorted(wall_curves):
        record, wall_errors, wall_values = wall_curves[key]
        rss_curve = rss_curves.get(key)
        if rss_curve is None:
            continue
        _, rss_errors, rss_values = rss_curve
        parts = [
            _method_label(str(record["method"])),
            f"ndof={int(record['ndof'])}",
            f"points={wall_errors.size}",
            f"error_range=[{float(wall_errors.min()):.3e},{float(wall_errors.max()):.3e}]",
            f"best_wall={float(wall_values.min()):.6f}s",
            f"best_peak_rss={_bytes_to_mib(float(rss_values.min())):.2f}MiB",
        ]
        if verbose:
            wall_slope, _ = _log_fit(wall_errors, wall_values)
            rss_slope, _ = _log_fit(rss_errors, rss_values)
            if wall_slope is not None:
                parts.append(f"log_slope_wall_vs_error={wall_slope:.3f}")
            if rss_slope is not None:
                parts.append(f"log_slope_rss_vs_error={rss_slope:.3f}")
        print(" ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    records = _zero_temperature_records(
        profile=profile,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    write_records(records, args.output)
    print_density_comparison_summary(records, verbose=args.plot_dir is not None)
    if args.plot_dir:
        written = write_density_comparison_plots(
            records,
            args.plot_dir,
            fixed_density_errors=profile.fixed_density_errors,
        )
        for path in written:
            print(f"plot={path}")


if __name__ == "__main__":
    main()
