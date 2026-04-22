from __future__ import annotations

import argparse

from meanfi import AdaptiveQuadrature, AdaptiveSimplex, UniformGrid, density_matrix_at_mu
from meanfi.tests.helpers import (
    benchmark,
    converged_dense_reference,
    dimerized_chain,
    max_density_error,
)
from performance._shared.common import density_record, print_summary, write_records
from performance._shared.scenarios import block_chain_keys, block_chain_model


ADAPTIVE_QUADRATURE_TOLS = (3e-2, 1e-2, 3e-3, 1e-3)
ADAPTIVE_SIMPLEX_TOLS = (1e-2, 3e-3, 1e-3, 3e-4)
UNIFORM_GRID_NKS = (17, 33, 65, 129)


def _density_measurement(tb, *, mu: float, kT: float, keys, integration, repeat: int, warmup: int):
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
    return result, result.last_result


def _adaptive_quadrature_records(*, repeat: int, warmup: int) -> list[dict]:
    tb = block_chain_model(12)
    keys = block_chain_keys()
    mu = 0.0
    kT = 0.15
    reference = converged_dense_reference(
        tb,
        mu=mu,
        kT=kT,
        keys=keys,
        target_tol=3e-4,
        nk_start=129,
        nk_max=1025,
    )
    records: list[dict] = []

    for density_matrix_tol in ADAPTIVE_QUADRATURE_TOLS[:3]:
        integration = AdaptiveQuadrature(density_matrix_tol=density_matrix_tol)
        measurement, result = _density_measurement(
            tb,
            mu=mu,
            kT=kT,
            keys=keys,
            integration=integration,
            repeat=repeat,
            warmup=warmup,
        )
        records.append(
            density_record(
                scenario="block_chain_ft_unique_evals",
                workflow="density_at_mu",
                integration=integration,
                kT=kT,
                ndof=12,
                benchmark_result=measurement,
                density_result=result,
                density_matrix_error=max_density_error(result.density_matrix, reference.rho),
                filling_error=abs(result.filling - reference.charge),
                extra={
                    "density_matrix_tol": density_matrix_tol,
                    "sweep": "unique_evals",
                },
            )
        )

    target_error = 2e-3
    for ndof in (4, 8, 12):
        tb = block_chain_model(ndof)
        reference = converged_dense_reference(
            tb,
            mu=mu,
            kT=kT,
            keys=keys,
            target_tol=target_error / 5.0,
            nk_start=129,
            nk_max=1025,
        )
        chosen = None
        chosen_error = None
        for density_matrix_tol in ADAPTIVE_QUADRATURE_TOLS:
            candidate = density_matrix_at_mu(
                tb,
                mu=mu,
                kT=kT,
                keys=keys,
                integration=AdaptiveQuadrature(density_matrix_tol=density_matrix_tol),
            )
            actual_error = max_density_error(candidate.density_matrix, reference.rho)
            if actual_error <= target_error:
                chosen = density_matrix_tol
                chosen_error = actual_error
                break
        if chosen is None:
            chosen = ADAPTIVE_QUADRATURE_TOLS[-1]
            chosen_result = density_matrix_at_mu(
                tb,
                mu=mu,
                kT=kT,
                keys=keys,
                integration=AdaptiveQuadrature(density_matrix_tol=chosen),
            )
            chosen_error = max_density_error(chosen_result.density_matrix, reference.rho)

        integration = AdaptiveQuadrature(density_matrix_tol=chosen)
        measurement, result = _density_measurement(
            tb,
            mu=mu,
            kT=kT,
            keys=keys,
            integration=integration,
            repeat=repeat,
            warmup=warmup,
        )
        records.append(
            density_record(
                scenario="block_chain_ft_ndof_target",
                workflow="density_at_mu",
                integration=integration,
                kT=kT,
                ndof=ndof,
                benchmark_result=measurement,
                density_result=result,
                density_matrix_error=chosen_error,
                filling_error=abs(result.filling - reference.charge),
                extra={
                    "density_matrix_tol": chosen,
                    "target_density_error": target_error,
                    "sweep": "ndof",
                },
            )
        )

    return records


def _zero_temperature_records(*, repeat: int, warmup: int) -> list[dict]:
    tb = dimerized_chain()
    keys = [(0,), (1,), (-1,)]
    mu = 0.0
    reference = converged_dense_reference(
        tb,
        mu=mu,
        kT=0.0,
        keys=keys,
        target_tol=1e-6,
        nk_start=251,
        nk_max=4001,
    )
    records: list[dict] = []

    for density_matrix_tol in ADAPTIVE_SIMPLEX_TOLS[:3]:
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
            density_record(
                scenario="dimerized_chain_zt_unique_evals",
                workflow="density_at_mu",
                integration=integration,
                kT=0.0,
                ndof=2,
                benchmark_result=measurement,
                density_result=result,
                density_matrix_error=max_density_error(result.density_matrix, reference.rho),
                filling_error=abs(result.filling - reference.charge),
                extra={
                    "density_matrix_tol": density_matrix_tol,
                    "sweep": "unique_evals",
                },
            )
        )

    for nk in UNIFORM_GRID_NKS[:3]:
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
            density_record(
                scenario="dimerized_chain_zt_unique_evals",
                workflow="density_at_mu",
                integration=integration,
                kT=0.0,
                ndof=2,
                benchmark_result=measurement,
                density_result=result,
                density_matrix_error=max_density_error(result.density_matrix, reference.rho),
                filling_error=abs(result.filling - reference.charge),
                extra={
                    "nk": nk,
                    "sweep": "unique_evals",
                },
            )
        )

    target_error = 3e-3
    for ndof in (4, 8, 12):
        tb = block_chain_model(ndof)
        reference = converged_dense_reference(
            tb,
            mu=mu,
            kT=0.0,
            keys=block_chain_keys(),
            target_tol=max(target_error / 2.0, 2e-3),
            nk_start=251,
            nk_max=4001,
        )
        candidates = [
            ("adaptive_simplex", ADAPTIVE_SIMPLEX_TOLS, lambda value: AdaptiveSimplex(density_matrix_tol=value, max_refinements=None), "density_matrix_tol"),
            ("uniform_grid", UNIFORM_GRID_NKS, lambda value: UniformGrid(nk=value), "nk"),
        ]
        for _method_name, controls, factory, control_field in candidates:
            chosen_control = controls[-1]
            chosen_error = None
            for control in controls:
                candidate = density_matrix_at_mu(
                    tb,
                    mu=mu,
                    kT=0.0,
                    keys=block_chain_keys(),
                    integration=factory(control),
                )
                actual_error = max_density_error(candidate.density_matrix, reference.rho)
                if actual_error <= target_error:
                    chosen_control = control
                    chosen_error = actual_error
                    break
            if chosen_error is None:
                chosen_result = density_matrix_at_mu(
                    tb,
                    mu=mu,
                    kT=0.0,
                    keys=block_chain_keys(),
                    integration=factory(chosen_control),
                )
                chosen_error = max_density_error(chosen_result.density_matrix, reference.rho)

            integration = factory(chosen_control)
            measurement, result = _density_measurement(
                tb,
                mu=mu,
                kT=0.0,
                keys=block_chain_keys(),
                integration=integration,
                repeat=repeat,
                warmup=warmup,
            )
            records.append(
                density_record(
                    scenario="block_chain_zt_ndof_target",
                    workflow="density_at_mu",
                    integration=integration,
                    kT=0.0,
                    ndof=ndof,
                    benchmark_result=measurement,
                    density_result=result,
                    density_matrix_error=chosen_error,
                    filling_error=abs(result.filling - reference.charge),
                    extra={
                        control_field: chosen_control,
                        "target_density_error": target_error,
                        "sweep": "ndof",
                    },
                )
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    records = _adaptive_quadrature_records(repeat=args.repeat, warmup=args.warmup)
    records.extend(_zero_temperature_records(repeat=args.repeat, warmup=args.warmup))
    write_records(records, args.output)
    print_summary(records)


if __name__ == "__main__":
    main()
