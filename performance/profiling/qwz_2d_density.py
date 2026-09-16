from __future__ import annotations
from dataclasses import replace
from meanfi import default_solver_tolerances

import argparse

from meanfi import UniformGrid, density_matrix
from performance._shared.fixtures import (
    benchmark,
    converged_dense_reference,
    max_density_error,
    qiwuzhang,
)
from performance._shared.common import density_record, print_summary, write_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    tb = qiwuzhang(m=0.5)
    keys = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    filling = 1.0
    kT = 0.1
    reference = converged_dense_reference(
        tb,
        filling=filling,
        kT=kT,
        keys=keys,
        target_tol=5e-4,
        nk_start=65,
        nk_max=513,
    )
    integration = UniformGrid()
    measurement = benchmark(
        lambda: density_matrix(
            tb,
            filling=filling,
            kT=kT,
            keys=keys,
            integration=integration,
            tol=replace(
                default_solver_tolerances(1e-3),
                density_matrix_integration=1e-6,
                charge_integration=1e-6,
                filling_residual=1e-6,
            ),
        ),
        repeat=args.repeat,
        warmup=args.warmup,
        track_tracemalloc=True,
    )
    result = measurement.last_result
    record = density_record(
        scenario="qiwuzhang_2d_density_profile",
        workflow="density_fixed_filling",
        integration=integration,
        kT=kT,
        ndof=2,
        benchmark_result=measurement,
        density_result=result,
        density_matrix_error=max_density_error(result.to_tb(), reference.rho),
        filling_error=abs(result.filling - filling),
        extra={"density_matrix_tol": 1e-6},
    )
    write_records([record], args.output)
    print_summary([record])


if __name__ == "__main__":
    main()
