from __future__ import annotations

import argparse

from meanfi import AdaptiveQuadrature, AndersonMixing, solver
from meanfi.tests.helpers import benchmark
from performance._shared.common import print_summary, scf_record, write_records
from performance._shared.scenarios import hubbard_chain_scf_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    model, guess = hubbard_chain_scf_problem(U=2.0, kT=0.1)
    integration = AdaptiveQuadrature(density_matrix_tol=1e-4)
    scf = AndersonMixing(M=0, max_iterations=40)
    measurement = benchmark(
        lambda: solver(
            model,
            guess,
            integration=integration,
            scf=scf,
            scf_tol=1e-4,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
        track_tracemalloc=True,
    )
    record = scf_record(
        scenario="hubbard_1d_scf_profile",
        integration=integration,
        scf_method=scf,
        kT=model.kT,
        ndof=model._ndof,
        benchmark_result=measurement,
        solver_result=measurement.last_result,
        extra={"density_matrix_tol": 1e-4},
    )
    write_records([record], args.output)
    print_summary([record])


if __name__ == "__main__":
    main()
