from __future__ import annotations

import argparse

import numpy as np

from meanfi import AdaptiveQuadrature, DirectDiagonalization, Model, RationalFOE
from meanfi.integrate.engines.bdg import solve_bdg_density_fixed_filling
from performance._shared.fixtures import benchmark
from performance._shared.common import density_record, print_summary, write_records


def _square_lattice_2d(t: float = 0.15):
    return {
        (0, 0): np.array([[0.1]], dtype=complex),
        (1, 0): np.array([[-t]], dtype=complex),
        (-1, 0): np.array([[-t]], dtype=complex),
        (0, 1): np.array([[-t]], dtype=complex),
        (0, -1): np.array([[-t]], dtype=complex),
    }


def _pairing(delta: float):
    return {(0, 0): np.array([[0.0, delta], [delta, 0.0]], dtype=complex)}


def _max_density_error(lhs, rhs) -> float:
    return max(float(np.max(np.abs(lhs[key] - rhs[key]))) for key in rhs)


def _problem():
    keys = [(0, 0), (1, 0)]
    meanfield = _pairing(0.25)
    model = Model(
        _square_lattice_2d(),
        {(0, 0): np.array([[1.0]], dtype=complex)},
        filling=0.6,
        kT=0.5,
        superconducting=True,
    )
    return model, meanfield, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    model, meanfield, keys = _problem()
    reference = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=5e-5,
            max_refinements=160,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=5e-5,
        mu_tol=5e-5,
        max_mu_iterations=80,
        mu_guess=0.0,
    )

    configurations = (
        ("direct_diagonalization", DirectDiagonalization()),
        (
            "rational_aaa",
            RationalFOE(initial_poles=4, max_poles=256, rational_scheme="aaa"),
        ),
        (
            "rational_ozaki",
            RationalFOE(initial_poles=4, max_poles=256, rational_scheme="ozaki"),
        ),
    )

    records = []
    for label, matrix_function in configurations:
        integration = AdaptiveQuadrature(
            density_matrix_tol=1e-4,
            max_refinements=120,
            matrix_function=matrix_function,
        )
        measurement = benchmark(
            lambda: solve_bdg_density_fixed_filling(
                model,
                meanfield,
                keys=keys,
                integration=integration,
                filling_tol=1e-4,
                mu_tol=1e-4,
                max_mu_iterations=80,
                mu_guess=0.0,
            ),
            repeat=args.repeat,
            warmup=args.warmup,
            track_tracemalloc=True,
        )
        result = measurement.last_result
        records.append(
            density_record(
                scenario="bdg_matrix_function_comparison",
                workflow="bdg_density_fixed_filling",
                integration=integration,
                kT=model.kT,
                ndof=model._ndof,
                benchmark_result=measurement,
                density_result=result,
                density_matrix_error=_max_density_error(
                    result.density_matrix,
                    reference.density_matrix,
                ),
                filling_error=abs(result.filling - model.filling),
                extra={
                    "matrix_function": label,
                    "reference_density_matrix_tol": 5e-5,
                },
            )
        )

    write_records(records, args.output)
    print_summary(records)


if __name__ == "__main__":
    main()
