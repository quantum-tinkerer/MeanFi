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
# Algorithm overview

This page connects the theoretical objects introduced in the [Theory](./theory/index.md) section to the numerical steps used to compute them.
It is intentionally organized by physical task rather than by backend taxonomy.
For the detailed method catalogue, see the [Algorithm reference](./algorithms/index.md).

(algo-build)=
## Build the self-consistent quadratic Hamiltonian

Theory context: {ref}`Theory: mean-field approximation <theory-mean-field>` and {ref}`Theory: density matrix <theory-density-matrix>`.

The first numerical task is to turn the current density state into the quadratic Hamiltonian that will be solved next.
In `MeanFi`, that is the outer self-consistent map together with the reduced state representation used by the solver.

- [SCF loop](./algorithms/scf_loop.md): the fixed-point structure of the outer solve
- [Parametrization and symmetry reduction](./algorithms/parametrization.md): how the self-consistent state is compressed before iteration

(algo-filling)=
## Solve the filling constraint `N(\mu) = \nu`

Theory context: {ref}`Theory: self-consistency and filling <theory-filling>`.

In the default fixed-filling workflow, density evaluation is not complete until the chemical potential has been chosen so that the target filling is satisfied.
This solve sits inside every density update rather than outside the mean-field loop.

- [Fixed-filling solve](./algorithms/fixed_filling.md): bracketing, safeguarded root-finding, and the coupling to density evaluation

(algo-bz)=
## Integrate the density over the Brillouin zone

Theory context: {ref}`Theory: tight-binding and real-space notation <theory-tight-binding>` and {ref}`Theory: density matrix <theory-density-matrix>`.

For translationally invariant systems, the density is assembled from momentum-space information across the Brillouin zone.
The main algorithmic choice here is how the sampled region is traversed and refined.

- [Integration families](./algorithms/integration_families.md): the main integration strategies
- [Adaptive quadrature](./algorithms/method_notes/adaptive_quadrature.md)
- [Uniform grid](./algorithms/method_notes/uniform_grid.md)
- [Adaptive simplex](./algorithms/method_notes/adaptive_simplex.md)

(algo-single-k)=
## Evaluate the density at one sampled $k$

Theory context: {ref}`Theory: density matrix <theory-density-matrix>` and {ref}`Theory extensions: finite temperature <theory-finite-temperature>`.

Once a momentum point is selected, the remaining task is to evaluate the density contribution of the effective quadratic Hamiltonian at that point.
This is where direct diagonalization and matrix-function approximations enter.

- [Matrix-function backends](./algorithms/matrix_functions.md): the fixed-$k$ density-evaluation layer
- [Direct diagonalization](./algorithms/method_notes/direct_diagonalization.md)
- [Rational FOE](./algorithms/method_notes/rational_foe.md)

## Choose defaults and supported combinations

Theory context: the core [Theory](./theory/index.md) page defines the objects being solved, while the [Theory extensions](./theory/extensions.md) page defines the main variants of the problem.

Once the theoretical task is clear, the remaining question is which numerical combinations are available and which ones are chosen by default.

- [Defaults and capabilities](./algorithms/defaults_and_capabilities.md): supported combinations and current dispatch rules
- [Algorithm reference](./algorithms/index.md): the full method-oriented directory
