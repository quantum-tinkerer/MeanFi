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

This page connects the theoretical objects introduced in the [Theory](../theory/index.md) section to the numerical steps used to compute them.
It is intentionally organized by physical task rather than by backend taxonomy.
The reference pages linked below give the method-by-method details.

At a high level, the computation is the repeated evaluation of the sequence

:::{math}
p_n
\;\longrightarrow\;
\hat H_{\mathrm{MF}}[p_n]
\;\longrightarrow\;
\mu_n,
\qquad
\rho_n(k,\mu_n),
\qquad
p_\ast
=
\int_{\mathrm{BZ}} \rho_n(k,\mu_n)\, dk,
:::

until the self-consistency condition

:::{math}
p_{n+1} = p_n
:::

is satisfied.

Here $p_n$ denotes the current SCF state, $p_\ast$ the raw updated state returned by the density evaluation, and $p_{n+1}$ the next iterate after the SCF update step.

Equivalently, the algorithm is organized into five nested numerical tasks:

:::{math}
\begin{gathered}
p_n
\;\xrightarrow{\text{mean-field update}}\;
\hat H_{\mathrm{MF}}[p_n]
\;\xrightarrow{\text{fixed filling}}\;
\mu_n
\;\xrightarrow{\text{single-$k$ evaluation}}\;
\rho_n(k,\mu_n)
\;\xrightarrow{\text{BZ integration}}\;
p_\ast
\\[0.6em]
p_\ast
\;\xrightarrow{\text{SCF loop}}\;
p_{n+1}
\end{gathered}
:::

```{toctree}
:hidden:
:maxdepth: 1

scf_loop.md
parametrization.md
fixed_filling.md
integration_families.md
matrix_functions.md
defaults_and_capabilities.md
```

(algo-build)=
## Mean-Field Update

The first numerical task is to turn the current density state into the quadratic Hamiltonian that will be solved next.
In `MeanFi`, that is the outer self-consistent map together with the reduced state representation used by the solver.

The object being built is

:::{math}
\hat H_{\mathrm{MF}}[\rho] = \hat H_0 + \hat V_{\mathrm{MF}}[\rho].
:::

**Reference pages**

- [Parametrization and symmetry reduction](./parametrization.md)

(algo-filling)=
## Fixed-Filling Solve

In the default fixed-filling workflow, density evaluation is not complete until the chemical potential has been chosen so that the target filling is satisfied.
This solve sits inside every density update rather than outside the mean-field loop.

The equation being solved is

:::{math}
N(\mu) = \nu.
:::

**Reference pages**

- [Fixed-filling solve](./fixed_filling.md)

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

- [Matrix-function backends](./matrix_functions.md)

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

- [Integration families](./integration_families.md)

## SCF Loop

Once the raw update $p_\ast$ has been computed, the outer SCF method turns it into the next iterate $p_{n+1}$.
This is the step that closes the loop and determines how aggressively the self-consistent state is updated from one iteration to the next.

The object being updated is

:::{math}
p_{n+1} = \mathcal{S}_n(p_\ast, p_n, p_{n-1}, \dots, p_0),
:::

where $\mathcal{S}_n$ denotes the chosen SCF update rule, possibly with memory of earlier iterates.

**Reference pages**

- [SCF loop](./scf_loop.md)
- [Parametrization and symmetry reduction](./parametrization.md)

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

- [Defaults and capabilities](./defaults_and_capabilities.md)
