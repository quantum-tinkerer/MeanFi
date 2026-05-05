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
# `RationalFOE`

`RationalFOE` is the finite-temperature matrix-function backend that avoids full diagonalization.

Instead of diagonalizing the sampled Hamiltonian exactly, it approximates the occupation matrix function

:::{math}
f(H) = \frac{1}{e^{H/kT}+1}
:::

by a rational approximation of the form

:::{math}
f(H) \approx c_0 I + \sum_{\ell=1}^{m} c_\ell (H - z_\ell I)^{-1}.
:::

So the calculation is reduced to solving shifted linear systems rather than computing the full eigendecomposition.

## Why it helps

For sparse matrices, solving a sequence of shifted sparse systems can be much cheaper than repeated dense diagonalization, especially when the Hamiltonian is large and the k-space integrator needs many evaluations.

## What it computes

At a fixed sampled $k$, the backend uses the rational approximation to evaluate:

- density blocks,
- and in supported paths, charge and derivative information for the fixed-filling solve.

## Cost and scaling

If the rational approximation uses $m$ poles, then the leading cost is roughly

:::{math}
\text{cost} \sim m \times \text{cost of one shifted linear solve}.
:::

That is why this approach is attractive primarily for sparse problems: sparse shifted solves can scale much better than dense diagonalization.

## Current practical notes

- sparse finite-temperature defaults choose `RationalFOE(rational_scheme="aaa")`,
- dense and sparse rational paths do not expose exactly the same capabilities,
- this is a single-node matrix-function backend, not a Brillouin-zone integration method by itself.
