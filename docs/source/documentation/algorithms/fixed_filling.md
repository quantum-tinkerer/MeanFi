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
# Fixed-filling solve

Every density update in `MeanFi` is done at fixed filling.
That means the code must solve

:::{math}
N(\mu) = \nu
:::

for the chemical potential before it can return the density matrix.

## Bracketing and root solve

The inner fixed-filling solver:

1. builds an initial bracket for $\mu$ from a spectral bound,
2. expands the bracket until the target filling is enclosed,
3. solves for $\mu$ with safeguarded Newton steps and midpoint fallback.

If derivative information is unavailable or unusable, the solve falls back to bracketed updates only.

## Why this is coupled to the integrator

Each trial value of $\mu$ requires a charge evaluation.
That charge evaluation is produced by the same Brillouin-zone backend used for the density matrix itself, so:

- adaptive quadrature reuses its quadrature tree and cached payloads,
- adaptive simplex reuses its refined simplicial structure,
- uniform grids reuse the same fixed nodes.

## Why BdG is heavier

In BdG calculations, the chemical potential enters as a shift by the charge operator $Q$, not by the identity.
Since $Q$ does not generally commute with the BdG Hamiltonian, changing $\mu$ changes the effective matrix in a way that requires repeated backend evaluation.

So the fixed-filling solve is more expensive in superconducting calculations because it repeatedly re-diagonalizes or re-evaluates the BdG matrix as $\mu$ changes.
