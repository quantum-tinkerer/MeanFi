from __future__ import annotations
from dataclasses import replace
from meanfi import default_solver_tolerances

import argparse

from meanfi import (
    UniformGrid,
    FermiSimplex,
    AndersonMixing,
    solver,
)
from performance._shared.fixtures import benchmark
from performance._shared.common import print_summary, scf_record, write_records
from performance._shared.scenarios import hubbard_chain_scf_problem


def _scf_measurement(
    model, guess, *, integration, density_target, scf, repeat: int, warmup: int
):
    result = benchmark(
        lambda: solver(
            model,
            guess,
            integration=integration,
            scf=scf,
            tol=replace(
                default_solver_tolerances(1e-3),
                scf_residual=1e-4,
                density_matrix_integration=density_target,
                charge_integration=density_target,
            ),
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

    anderson = AndersonMixing(history_size=0, max_iterations=40)
    records: list[dict] = []

    ft_model, ft_guess = hubbard_chain_scf_problem(U=2.0, kT=0.1)
    zt_model, zt_guess = hubbard_chain_scf_problem(U=2.0, kT=0.0)
    cases = [
        (
            "hubbard_chain_ft_scf",
            ft_model,
            ft_guess,
            UniformGrid(),
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
            FermiSimplex(max_refinements=600),
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
            density_target=extra["control_value"]
            if extra["control_parameter"] == "density_matrix_tol"
            else 2e-4,
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
