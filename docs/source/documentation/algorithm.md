---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
## Algorithm Overview

### Fixed-filling density update

The central operation in `meanfi` is the finite-temperature density evaluation at fixed filling.
For a trial density matrix `\rho`, {autolink}`~meanfi.mf.meanfield` constructs the mean-field Hamiltonian and
{autolink}`~meanfi.mf.density_matrix` computes the new density matrix by:

1. building a momentum-space Hamiltonian function with {autolink}`~meanfi.tb.transforms.tb_to_kfunc`,
2. evaluating and caching the eigensystems of `H(k)` on adaptive cubature nodes via `stateful_quadrature`,
3. solving the scalar filling equation `N(\mu)=\nu` with safeguarded Newton and bisection fallback,
4. reusing the adaptive cache to evaluate the final density-matrix integral.

The chemical-potential solve exposes a charge target `charge_tol`, while the final density integral uses
its own `density_atol` and `density_rtol`.

### Self-consistent loop

`meanfi` solves the mean-field problem as a fixed-point iteration in the reduced density-matrix parameters.
The parameterization map is given by {autolink}`~meanfi.params.rparams.tb_to_rparams` and its inverse
by {autolink}`~meanfi.params.rparams.rparams_to_tb`.

For a parameter vector `\theta`, the SCF loop:

1. reconstructs the trial reduced density matrix,
2. evaluates the fixed-filling density update,
3. forms the residual between the updated and trial reduced density matrices,
4. applies Anderson mixing by default to accelerate convergence.

The public {autolink}`~meanfi.solvers.solver` function exposes Anderson mixing as the default outer-loop method,
with a simple linear-mixing fallback for debugging or robustness checks.

### Supported regime

The main package supports finite-temperature calculations only.
Dense grids and repeated SciPy cubature are kept only as testing and benchmarking references; they are not
used by the production solver path.
