from __future__ import annotations

import argparse

from meanfi import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    AndersonMixing,
    UniformGrid,
    solver,
)
from meanfi.tests.helpers import benchmark
from performance._shared.common import print_summary, scf_record, write_records
from performance._shared.scenarios import hubbard_chain_scf_problem


def _scf_measurement(model, guess, *, integration, scf, repeat: int, warmup: int):
    result = benchmark(
        lambda: solver(
            model,
            guess,
            integration=integration,
            scf=scf,
            scf_tol=1e-4,
        ),
        repeat=repeat,
        warmup=warmup,
        track_tracemalloc=True,
    )
    return result, result.last_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    anderson = AndersonMixing(M=0, max_iterations=40)
    records: list[dict] = []

    ft_model, ft_guess = hubbard_chain_scf_problem(U=2.0, kT=0.1)
    zt_model, zt_guess = hubbard_chain_scf_problem(U=2.0, kT=0.0)
    cases = [
        (
            "hubbard_chain_ft_scf",
            ft_model,
            ft_guess,
            AdaptiveQuadrature(density_matrix_tol=1e-4),
            {
                "problem_family": "hubbard_chain",
                "held_constant": "U=2.0,filling=2.0,kT=0.1",
                "control_parameter": "density_matrix_tol",
                "control_value": 1e-4,
            },
        ),
        (
            "hubbard_chain_zt_scf",
            zt_model,
            zt_guess,
            AdaptiveSimplex(density_matrix_tol=1e-3, max_refinements=600),
            {
                "problem_family": "hubbard_chain",
                "held_constant": "U=2.0,filling=2.0,kT=0.0",
                "control_parameter": "density_matrix_tol",
                "control_value": 1e-3,
                "max_refinements": 600,
            },
        ),
        (
            "hubbard_chain_zt_scf",
            zt_model,
            zt_guess,
            UniformGrid(nk=65),
            {
                "problem_family": "hubbard_chain",
                "held_constant": "U=2.0,filling=2.0,kT=0.0",
                "control_parameter": "nk",
                "control_value": 65,
            },
        ),
    ]

    for scenario, model, guess, integration, extra in cases:
        measurement, result = _scf_measurement(
            model,
            guess,
            integration=integration,
            scf=anderson,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        records.append(
            scf_record(
                scenario=scenario,
                integration=integration,
                scf_method=anderson,
                kT=model.kT,
                ndof=model._ndof,
                benchmark_result=measurement,
                solver_result=result,
                extra=extra,
            )
        )

    write_records(records, args.output)
    print_summary(records)


if __name__ == "__main__":
    main()
