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
# Algorithms

`MeanFi` solves a self-consistent mean-field problem by combining several numerical layers:

1. an outer SCF update on a reduced real parameter vector,
2. an inner fixed-filling solve for the chemical potential,
3. a Brillouin-zone integration strategy,
4. a single-$k$ density-evaluation backend.

```{toctree}
:hidden:
:maxdepth: 1

scf_loop.md
parametrization.md
fixed_filling.md
integration_families.md
matrix_functions.md
defaults_and_capabilities.md
method_notes/adaptive_quadrature.md
method_notes/uniform_grid.md
method_notes/adaptive_simplex.md
method_notes/rational_foe.md
method_notes/direct_diagonalization.md
```

The pages are grouped by conceptual layer rather than by every possible method combination.

## Outer solve

- [SCF loop](./scf_loop.md): the fixed-point structure and outer update methods
- [Parametrization and symmetry reduction](./parametrization.md): how `MeanFi` compresses the SCF state using Hermitian/BdG symmetry and `h_int` support
- [Fixed-filling solve](./fixed_filling.md): how the chemical potential is found inside each density evaluation

## K-space integration

- [Integration families](./integration_families.md): `AdaptiveQuadrature`, `AdaptiveSimplex`, and `UniformGrid`
- [Adaptive quadrature](./method_notes/adaptive_quadrature.md)
- [Uniform grid](./method_notes/uniform_grid.md)
- [Adaptive simplex](./method_notes/adaptive_simplex.md)

## Matrix-function evaluation

- [Matrix-function backends](./matrix_functions.md): what happens at a fixed sampled $k$
- [Direct diagonalization](./method_notes/direct_diagonalization.md)
- [Rational FOE](./method_notes/rational_foe.md)

## Dispatch

- [Defaults and capabilities](./defaults_and_capabilities.md): current defaults, supported combinations, and backend selection rules
