import pytest

from meanfi import AdaptiveQuadrature, AndersonMixing, density_matrix, solver
from meanfi.tests.helpers import benchmark, spinful_chain
from performance._shared.common import density_record, scf_record
from performance._shared.scenarios import hubbard_chain_scf_problem


pytestmark = pytest.mark.performance


def test_density_record_exposes_required_schema_fields():
    measurement = benchmark(
        lambda: density_matrix(
            spinful_chain(),
            filling=0.7,
            kT=0.15,
            keys=[(0,), (1,), (-1,)],
            integration=AdaptiveQuadrature(density_matrix_tol=1e-3),
            filling_tol=1e-4,
        ),
        repeat=1,
        warmup=0,
        track_tracemalloc=True,
    )

    record = density_record(
        scenario="schema_density_smoke",
        workflow="density_fixed_filling",
        integration=AdaptiveQuadrature(density_matrix_tol=1e-3),
        kT=0.15,
        ndof=2,
        benchmark_result=measurement,
        density_result=measurement.last_result,
        density_matrix_error=None,
        filling_error=abs(measurement.last_result.filling - 0.7),
    )

    required = {
        "scenario",
        "method",
        "workflow",
        "kT",
        "ndof",
        "problem_family",
        "sweep_axis",
        "sweep_value",
        "held_constant",
        "control_parameter",
        "control_value",
        "reference_kind",
        "wall_s",
        "peak_memory_bytes",
        "unique_evals",
        "wall_per_unique_eval_s",
        "n_kernel_evals",
        "n_evaluator_evals",
        "root_iterations",
        "scf_iterations",
        "n_kpoints",
        "total_unique_evals",
        "density_matrix_error",
        "filling_error",
        "scf_residual",
    }

    assert required.issubset(record)
    assert record["unique_evals"] > 0
    assert record["wall_per_unique_eval_s"] is not None
    assert record["root_iterations"] is not None
    assert record["total_unique_evals"] is None
    assert record["problem_family"] is None


def test_scf_record_exposes_required_schema_fields():
    model, guess = hubbard_chain_scf_problem(U=2.0, kT=0.1)
    integration = AdaptiveQuadrature(density_matrix_tol=1e-3)
    scf = AndersonMixing(M=0, max_iterations=20)
    measurement = benchmark(
        lambda: solver(
            model,
            guess,
            integration=integration,
            scf=scf,
            scf_tol=1e-4,
        ),
        repeat=1,
        warmup=0,
        track_tracemalloc=True,
    )

    record = scf_record(
        scenario="schema_scf_smoke",
        integration=integration,
        scf_method=scf,
        kT=model.kT,
        ndof=model._ndof,
        benchmark_result=measurement,
        solver_result=measurement.last_result,
    )

    required = {
        "scenario",
        "method",
        "workflow",
        "kT",
        "ndof",
        "problem_family",
        "sweep_axis",
        "sweep_value",
        "held_constant",
        "control_parameter",
        "control_value",
        "reference_kind",
        "wall_s",
        "peak_memory_bytes",
        "unique_evals",
        "wall_per_unique_eval_s",
        "n_kernel_evals",
        "n_evaluator_evals",
        "root_iterations",
        "scf_iterations",
        "n_kpoints",
        "total_unique_evals",
        "density_matrix_error",
        "filling_error",
        "scf_residual",
        "scf_method",
    }

    assert required.issubset(record)
    assert record["workflow"] == "scf"
    assert record["total_unique_evals"] == record["unique_evals"]
    assert record["scf_iterations"] is not None
    assert record["scf_residual"] is not None
