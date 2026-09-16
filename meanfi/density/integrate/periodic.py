"""Periodic filling and integration refinement controlled by density accuracy."""

import numpy as np

from meanfi.density.filling import charge_diagonal, mu_bracket, solve_mu
from meanfi.density.problem import DensityProblem
from meanfi.density.kpoint.matrix_functions import DirectDiagonalization
from meanfi.errors import ErrorValues
from meanfi.results import _DensityEntries, DensityResult, IntegrationInfo
from meanfi.tb.validate import tb_dimension
from .periodic_grid import _Evaluator, _Grid, periodic_grid_resolution


def solve_periodic(
    problem: DensityProblem,
    *,
    mu: float | None,
    filling: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
    compute_entropy: bool = False,
) -> DensityResult:
    """Evaluate one prescribed grid, or refine until coarse/fine tests pass."""
    if filling is not None and problem.kT == 0:
        raise NotImplementedError(
            "UniformGrid fixed-filling calculations at kT=0 are not implemented: "
            "occupations can jump as mu crosses a discrete level, so the current "
            "root solver cannot reliably meet the filling target. "
            "Use FermiSimplex for normal zero-temperature models, a physically "
            "appropriate kT > 0, or density_matrix_at_mu with a supplied mu."
        )
    integration, tolerances = problem.integration, problem.tolerances
    hamiltonian, kT = problem.hamiltonian, problem.kT
    dimension = tb_dimension(hamiltonian)
    if kT == 0 and dimension and integration.nk is None:
        raise ValueError(
            "Zero-temperature UniformGrid at fixed mu requires explicit nk"
        )
    coordinates = problem.density_coordinates
    ndof = problem.electron_ndof
    q_diag = None if ndof is None else charge_diagonal(ndof)
    prescribed = integration.nk is not None
    filling_tol = tolerances.filling_residual
    evaluator = _Evaluator(
        hamiltonian,
        kT=kT,
        integration=integration,
        coordinates=coordinates,
        q_diag=q_diag,
        tolerances=tolerances,
        sparse_layout=problem.sparse_layout,
        fixed_filling=filling is not None,
        compute_entropy=compute_entropy,
    )
    starting_nk = integration.nk if prescribed else integration.initial_nk
    n = (
        periodic_grid_resolution(starting_nk, dimension)
        if starting_nk is not None
        else (4 if dimension else 1)
    )
    previous = None
    refinements = 0
    charge_evaluations = 0
    density_error = charge_error = energy_error = entropy_error = None

    while True:
        count = n**dimension
        if count > integration.max_points:
            raise RuntimeError(
                "UniformGrid total-grid-size limit reached: "
                f"the next mesh needs {count} points, max_points={integration.max_points}. "
                "Increase max_points or reduce nk / relax integration targets."
            )
        grid = _Grid(n, dimension)
        if filling is not None:
            evaluator.retain_spectra(grid, previous)
            remaining = (
                None
                if max_charge_evaluations is None
                else max_charge_evaluations - charge_evaluations
            )
            if remaining is not None and remaining <= 0:
                raise RuntimeError(
                    "UniformGrid chemical-potential solve reached max_charge_evaluations across refinement grids"
                )
            root = solve_mu(
                evaluate_charge=lambda candidate: evaluator.charge(grid, candidate),
                initial_bracket=lambda: mu_bracket(hamiltonian, kT),
                filling=filling,
                mu_guess=mu_guess,
                filling_tol=filling_tol,
                mu_tol=mu_tol,
                max_charge_evaluations=remaining,
                use_derivative=evaluator.normal
                and isinstance(evaluator.method, DirectDiagonalization)
                and kT > 0,
            )
            charge_evaluations += root.charge_evaluations
            resolved_mu = mu_guess = root.mu
        else:
            resolved_mu = float(mu)
        integral, parent = evaluator.density(
            grid, resolved_mu, compare_previous=previous is not None
        )
        if prescribed:
            break
        if dimension == 0:
            density_error = np.zeros(integral.values.size)
            charge_error = 0.0 if filling is not None else None
            energy_error = 0.0 if integral.band_energy is not None else None
            entropy_error = 0.0 if compute_entropy else None
            break
        if previous is not None:
            density_error, charge_error, energy_error, entropy_error = integral.errors(
                parent
            )
            if np.max(
                density_error, initial=0.0
            ) <= tolerances.density_matrix_integration and (
                filling is None or charge_error <= tolerances.charge_integration
            ):
                break
        if (
            integration.max_refinements is not None
            and refinements >= integration.max_refinements
        ):
            raise RuntimeError(
                "UniformGrid did not converge before max_refinements="
                f"{integration.max_refinements}; mesh={grid.shape}, "
                f"density_error={None if density_error is None else np.max(density_error, initial=0.0)}, "
                f"charge_error={charge_error}, band_energy_error={energy_error}, "
                f"entropy_error={entropy_error}. Increase the limit or relax integration targets."
            )
        previous = grid
        n *= 2
        refinements += 1
    entries = _DensityEntries(coordinates, integral.values, density_error)
    charge = root.charge if filling is not None else integral.charge
    if charge is None:
        charge = entries.trace(ndof)
    work = evaluator.work
    info = IntegrationInfo(
        requested_nk=integration.nk,
        n_kpoints=grid.count,
        grid_shape=grid.shape,
        n_kernel_evals=work.kernels,
        n_diagonalizations=work.diagonalizations,
        refinements=refinements,
        charge_evaluations=charge_evaluations,
        charge_integration_calls=work.charge_calls,
        density_integration_calls=work.density_calls,
        error_estimate_available=not prescribed,
        spectrum_bytes=work.spectrum_bytes,
    )
    return DensityResult(
        entries=entries,
        mu=resolved_mu,
        filling=charge,
        errors=ErrorValues(
            density_matrix_integration=None
            if density_error is None
            else float(np.max(density_error, initial=0.0)),
            charge_integration=charge_error if filling is not None else None,
            band_energy_integration=energy_error,
            entropy=None
            if entropy_error is None
            else entropy_error + (evaluator.entropy_approximation_error or 0.0),
            matrix_function_error=evaluator.matrix_function_error,
            filling_residual=None if filling is None else abs(charge - filling),
        ),
        statistics=info,
        band_energy=integral.band_energy,
        entropy=integral.entropy,
    )
