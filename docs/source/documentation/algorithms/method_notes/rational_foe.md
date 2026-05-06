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

## Rational schemes

### `aaa`

`aaa` uses the AAA algorithm, an adaptive barycentric rational approximation scheme that selects poles and residues to fit the scalar occupation function efficiently on the relevant spectral interval; see [The AAA Algorithm for Rational Approximation](https://epubs.siam.org/doi/10.1137/16M1106122).

### `ozaki`

`ozaki` uses a pole expansion derived from continued-fraction ideas for the Fermi-Dirac function, giving a structured rational approximation with poles chosen analytically rather than adaptively; see [Continued Fraction Representation of the Fermi-Dirac Function for Large-Scale Electronic Structure Calculations](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.75.035123).

## Why it helps

For sparse matrices, solving a sequence of shifted sparse systems can be much cheaper than repeated dense diagonalization, especially when the Hamiltonian is large and the k-space integrator needs many evaluations.

## What it computes

At a fixed sampled $k$, the backend uses the rational approximation to evaluate:

- density blocks,
- and in supported paths, charge and derivative information for the fixed-filling solve.

## Cost versus error scaling

If the rational approximation uses $m$ poles, then the leading cost is roughly

:::{math}
\text{cost} \sim m \times C_{\mathrm{solve}}.
:::

For a highly sparse local Hamiltonian with bounded degree, the optimistic best case is that one shifted sparse solve is close to linear in the one-node matrix size $n$,

:::{math}
C_{\mathrm{solve}} \sim \mathcal{O}(n),
\qquad
C_{\mathrm{rat}} \sim \mathcal{O}(m n),
:::

where $m$ is the number of poles.
This is the regime in which rational FOE can be much cheaper than dense diagonalization,

:::{math}
C_{\mathrm{diag}} \sim \mathcal{O}(n^3).
:::

More generally, sparse direct rational FOE is better summarized as

:::{math}
C_{\mathrm{rat}} \sim \mathcal{O}(m n^{\alpha}),
\qquad
1 \le \alpha < 3,
:::

where $\alpha$ depends on the sparsity graph, dimension, fill-in during factorization, solver details, and whether the implementation needs a full inverse or only selected entries.
Typical optimistic sparse-direct heuristics are:

- 1D-like sparsity: `C_rat ~ O(m n)`
- 2D-like sparsity: `C_rat ~ O(m n^{3/2})`
- 3D-like sparsity: `C_rat ~ O(m n^2)`

The rational approximation error is then controlled separately by the pole scheme and pole count.
AAA often reaches a given scalar approximation error with a moderate adaptive pole count, while Ozaki uses a structured analytic pole set.

That is why this approach is attractive primarily for sparse problems: sparse shifted solves can scale much better than dense diagonalization.

## Current practical notes

- sparse finite-temperature defaults choose `RationalFOE(rational_scheme="aaa")`,
- dense and sparse rational paths do not expose exactly the same capabilities,
- this is a single-node matrix-function backend, not a Brillouin-zone integration method by itself.
