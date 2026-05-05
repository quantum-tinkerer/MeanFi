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
# Integration families

The Brillouin-zone integral and the single-$k$ density evaluation are two different choices in `MeanFi`.
This page focuses only on the **k-space integration family**.

## The three families

### `AdaptiveQuadrature`

This is the finite-temperature adaptive cubature path.
It chooses evaluation nodes in the Brillouin zone adaptively and is the default family for `kT > 0`.

### `UniformGrid`

This evaluates the problem on a fixed `nk` grid in each momentum direction.
It is simple, predictable, and useful for explicit coarse-grid workflows and for the zero-temperature BdG path.

### `AdaptiveSimplex`

This is the dedicated zero-temperature adaptive simplicial backend.
It refines a simplicial partition of the Brillouin zone and is the default normal-state family for `kT = 0`.

## What the family controls

The integration family determines:

- where in k-space the Hamiltonian is sampled,
- how errors are estimated, if at all,
- how work is reused as the fixed-filling solver tries new values of $\mu$.

It does **not** by itself determine how the density matrix is computed at one sampled $k$ point.
That is the job of the matrix-function backend.

## Main tradeoffs

| Family | Strength | Limitation |
| --- | --- | --- |
| `AdaptiveQuadrature` | Efficient finite-temperature adaptivity | Requires `kT > 0` |
| `UniformGrid` | Simple and explicit | No adaptive error control |
| `AdaptiveSimplex` | Native zero-temperature adaptive path | Normal-state only by default |
