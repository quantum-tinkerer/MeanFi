"""Zero-temperature integration on one mesh shared by charge and density."""

from dataclasses import replace

import numpy as np

from meanfi.density.filling import mu_bracket, solve_mu
from meanfi.density.kpoint.zero_dim import evaluate_zero_dim
from meanfi.density.problem import DensityProblem
from meanfi.errors import ErrorValues
from meanfi.results import DensityResult
from meanfi.tb.ops import to_dense
from meanfi.hamiltonian import BlochHamiltonian, hamiltonian_dimension
from .mesh import SimplexEvaluator, _zero_temperature_entropy
from .energy import integrate_energies


def solve_simplex(
    problem: DensityProblem,
    *,
    mu: float | None,
    filling: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
    compute_entropy: bool = False,
) -> DensityResult:
    """Finish the charge solve, then refine density at its chemical potential."""
    if hamiltonian_dimension(problem.hamiltonian) == 0:
        return evaluate_zero_dim(
            to_dense(problem.hamiltonian[()]),
            problem.density_coordinates,
            mu=mu,
            filling=filling,
            mu_guess=mu_guess,
            filling_tol=problem.tolerances.filling_residual,
            compute_entropy=compute_entropy,
        )
    evaluator = SimplexEvaluator(problem)
    adaptive = problem.integration.nk is None
    tolerances = problem.tolerances
    charge_evaluations = 0

    def frozen_charge(candidate):
        result = evaluator.charge(candidate, adaptive=False)
        # Root finding uses a frozen discretization; its integration error is
        # checked separately before accepting the result.
        return float(result.value), float(result.dcharge_dmu)

    charge = None
    if filling is not None:
        while True:
            remaining = (
                None
                if max_charge_evaluations is None
                else max_charge_evaluations - charge_evaluations
            )
            if remaining is not None and remaining <= 0:
                raise RuntimeError(
                    "Chemical-potential solve reached maximum charge-evaluation budget"
                )
            root = solve_mu(
                evaluate_charge=frozen_charge,
                initial_bracket=lambda: mu_bracket(
                    problem.hamiltonian,
                    0.0,
                    eigenvalues=evaluator.mesh.eigenvalues
                    if isinstance(problem.hamiltonian, BlochHamiltonian)
                    else None,
                ),
                filling=filling,
                mu_guess=mu_guess,
                filling_tol=tolerances.filling_residual,
                mu_tol=mu_tol,
                max_charge_evaluations=remaining,
                use_derivative=True,
            )
            charge_evaluations += root.charge_evaluations
            mu = mu_guess = root.mu
            if not adaptive:
                break
            charge = evaluator.charge(mu, adaptive=True)
            if abs(charge.value - filling) <= tolerances.filling_residual:
                break

    density = evaluator.density(mu)
    value = (
        density.trace()
        if filling is None
        else float(root.charge if charge is None else charge.value)
    )

    entropy = None
    if compute_entropy:
        if filling is None and not density.values.size:
            # No density/charge call populated the native cache. Entropy only
            # needs eigenvalues on this mesh, without refinement or eigenvectors.
            eigenvalues = np.array(
                [
                    np.linalg.eigvalsh(evaluator.mesh.evaluate(*point))
                    for point in evaluator.mesh.points
                ]
            )
            evaluator.work.evaluations += len(eigenvalues)
            evaluator.work.diagonalizations += len(eigenvalues)
            entropy = _zero_temperature_entropy(evaluator.mesh, mu, eigenvalues)
        else:
            entropy = _zero_temperature_entropy(evaluator.mesh, mu)

    needs_energy = filling is not None or (
        density.values.size and (compute_entropy or problem.compute_energy)
    )
    density_filling = (
        evaluator.density_trace(mu, density)
        if needs_energy and density.values.size
        else density.trace()
    )
    statistics = evaluator.statistics(charge_evaluations)

    result = DensityResult(
        entries=density,
        mu=float(mu),
        filling=value,
        errors=ErrorValues(
            density_matrix_integration=None
            if density.errors is None
            else float(np.max(density.errors, initial=0.0)),
            charge_integration=float(charge.stopping_error)
            if charge is not None
            else None,
            filling_residual=None if filling is None else abs(value - filling),
        ),
        statistics=statistics,
        density_filling=density_filling,
        entropy=entropy,
    )

    def with_energy():
        energy = integrate_energies(evaluator.mesh, mu=mu, filling=density_filling)
        return replace(
            result,
            band_energy=energy.band_energy,
            statistics=replace(
                statistics,
                n_kernel_evals=statistics.n_kernel_evals + energy.evaluations,
                n_energy_evaluations=energy.evaluations,
                n_energy_simplices=energy.simplices,
                n_diagonalizations=statistics.n_diagonalizations + energy.evaluations,
            ),
        )

    if needs_energy:
        if problem.defer_energy:
            return replace(result, _energy_evaluation=with_energy)
        return with_energy()
    return result
