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
The reference pages linked below give the method-by-method details.

At a high level, the computation is the repeated evaluation of the sequence

:::{math}
\rho_n
\;\longrightarrow\;
\hat H_{\mathrm{MF}}[\rho_n]
\;\longrightarrow\;
\mu_n,
\qquad
\rho_n(k,\mu_n),
\qquad
\rho_{n+1}
=
\int_{\mathrm{BZ}} \rho_n(k,\mu_n)\, dk,
:::

until the self-consistency condition

:::{math}
\rho_{\mathrm{new}} = \rho
:::

is satisfied.

Equivalently, the algorithm is organized into four nested numerical tasks:

:::{math}
\rho_n
\;\xrightarrow{\text{mean-field update}}\;
\hat H_{\mathrm{MF}}[\rho_n]
\;\xrightarrow{\text{fixed filling}}\;
\mu_n
\;\xrightarrow{\text{single-$k$ evaluation}}\;
\rho_n(k,\mu_n)
\;\xrightarrow{\text{BZ integration}}\;
\rho_{n+1}
\qquad
\rho_{n+1} \xleftarrow{\text{SCF loop}} \rho_n
:::

(algo-build)=
## Mean-Field Update

The first numerical task is to turn the current density state into the quadratic Hamiltonian that will be solved next.
In `MeanFi`, that is the outer self-consistent map together with the reduced state representation used by the solver.

The object being built is

:::{math}
\hat H_{\mathrm{MF}}[\rho] = \hat H_0 + \hat V_{\mathrm{MF}}[\rho].
:::

**Reference pages**

- [SCF loop](./algorithms/scf_loop.md)
- [Parametrization and symmetry reduction](./algorithms/parametrization.md)

(algo-filling)=
## Fixed-Filling Solve

In the default fixed-filling workflow, density evaluation is not complete until the chemical potential has been chosen so that the target filling is satisfied.
This solve sits inside every density update rather than outside the mean-field loop.

The equation being solved is

:::{math}
N(\mu) = \nu.
:::

**Reference pages**

- [Fixed-filling solve](./algorithms/fixed_filling.md)

(algo-single-k)=
## Single-$k$ Density Evaluation

Once a momentum point is selected, the remaining task is to evaluate the density contribution of the effective quadratic Hamiltonian at that point.
This is where direct diagonalization and matrix-function approximations enter.

At a fixed sampled momentum, the object being computed is

:::{math}
\rho(k,\mu)
=
f\!\left(H_{\mathrm{MF}}(k)-\mu Q\right),
:::

with the normal-state case recovered by replacing $Q$ with the identity and, at zero temperature, replacing $f$ by the occupied-state projector.

**Reference pages**

- [Matrix-function backends](./algorithms/matrix_functions.md)
- [Direct diagonalization](./algorithms/method_notes/direct_diagonalization.md)
- [Rational FOE](./algorithms/method_notes/rational_foe.md)

(algo-bz)=
## Brillouin-Zone Integration

For translationally invariant systems, the density is assembled from momentum-space information across the Brillouin zone.
The main algorithmic choice here is how the sampled region is traversed and refined.

The object being evaluated is

:::{math}
\rho(\mu) = \int_{\mathrm{BZ}} \rho(k,\mu)\, dk
:::

or the corresponding discrete or adaptive approximation to that integral.

**Reference pages**

- [Integration families](./algorithms/integration_families.md)
- [Adaptive quadrature](./algorithms/method_notes/adaptive_quadrature.md)
- [Uniform grid](./algorithms/method_notes/uniform_grid.md)
- [Adaptive simplex](./algorithms/method_notes/adaptive_simplex.md)

## Choose defaults and supported combinations

Once the theoretical task is clear, the remaining question is which numerical combinations are available and which ones are chosen by default.

At this stage, the practical question is not a new theoretical equation but which numerical realization is used for the chain

:::{math}
\rho
\to
\hat H_{\mathrm{MF}}[\rho]
\to
\mu
\to
\rho(k,\mu)
\to
\rho(\mu).
:::

**Reference pages**

- [Defaults and capabilities](./algorithms/defaults_and_capabilities.md)
- [Algorithm reference](./algorithms/index.md)
