from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import csr_array

from meanfi import PeriodicGrid, DirectDiagonalization, Model, RationalFOE
from meanfi.density.integrate.bdg import solve_bdg_density_fixed_filling
from meanfi.tb.bdg import assemble_bdg_tb
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
    return assemble_bdg_tb(
        {(0, 0): np.zeros((1, 1), complex)},
        {(1, 0): np.array([[delta]], complex), (-1, 0): np.array([[-delta]], complex)},
        ndof=1,
    )


def _problem():
    keys = [(0, 0), (1, 0)]
    meanfield = {key: csr_array(value) for key, value in _pairing(0.25).items()}
    model = Model(
        {key: csr_array(value) for key, value in _square_lattice_2d().items()},
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
    parser.add_argument(
        "--nk", type=int, default=256, help="Total prescribed grid points"
    )
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    model, meanfield, keys = _problem()
    reference = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=PeriodicGrid(
            nk=args.nk,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=5e-5,
        mu_tol=5e-5,
        max_charge_evaluations=80,
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
        integration = PeriodicGrid(
            nk=args.nk,
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
                max_charge_evaluations=80,
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
                density_matrix_error=float(
                    np.max(np.abs(result.density.values - reference.density.values))
                ),
                filling_error=abs(result.filling - model.filling),
                extra={
                    "matrix_function": label,
                    "reference_nk": args.nk,
                },
            )
        )

    write_records(records, args.output)
    print_summary(records)


if __name__ == "__main__":
    main()
