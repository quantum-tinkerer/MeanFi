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
# `DirectDiagonalization`

`DirectDiagonalization` is the exact matrix-function backend.

At each sampled $k$ point, it diagonalizes the shifted Hamiltonian explicitly:

:::{math}
H(k) - \mu Q = U \Lambda U^\dagger.
:::

The density matrix is then reconstructed as

:::{math}
\rho(k) = U\, f(\Lambda)\, U^\dagger,
:::

with $f$ the Fermi-Dirac occupation function.

## What “exact” means here

Here “exact” means exact for the finite sampled matrix at that k-point, up to numerical diagonalization error.
It does not mean the full Brillouin-zone integral is exact, because that still depends on the chosen integration family.

## Derivatives

In derivative-aware fixed-filling paths, the same eigendecomposition also gives access to exact matrix-function derivatives through the Fréchet derivative of the occupation function.

## Cost and scaling

For an $n \times n$ dense matrix, the dominant cost is dense diagonalization, which scales roughly like

:::{math}
\mathcal{O}(n^3).
:::

That is why direct diagonalization is simple and robust, but becomes expensive for large sparse problems.

It is the default finite-temperature backend for dense calculations.
