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
# SCF loop

The outer solve in `MeanFi` is a fixed-point problem for the self-consistent state.
That state may be represented directly as a density object or through a reduced real parametrization.

## Fixed-point map

Let $P$ denote the map from a density state to its reduced real parametrization,

:::{math}
\theta = P(\rho),
\qquad
\rho = P^{-1}(\theta).
:::

The construction of $P$ is described in [Parametrization and symmetry reduction](./parametrization.md).

One SCF iteration has the structure

:::{math}
\rho_n
\;\longrightarrow\;
\hat H_{\mathrm{MF}}[\rho_n]
\;\longrightarrow\;
\rho_{n+1},
:::

where the density update already includes the fixed-filling solve and the Brillouin-zone density evaluation.
If $D$ denotes that full density-evaluation map, then

:::{math}
\rho_{n+1} = D\!\left(\hat H_{\mathrm{MF}}[\rho_n]\right).
:::

In reduced coordinates, this defines the map

:::{math}
G(\theta)
=
P\!\left(
D\!\left(\hat H_{\mathrm{MF}}[P^{-1}(\theta)]\right)
\right).
:::

So the SCF problem is the fixed-point equation

:::{math}
\theta = G(\theta),
:::

or equivalently $\rho = D(\hat H_{\mathrm{MF}}[\rho])$ in the unreduced density representation.

With a history-dependent SCF update, the next iterate may depend on several previous states,

:::{math}
\theta_{n+1}
=
G_n(\theta_n, \theta_{n-1}, \dots, \theta_0),
:::

where the basic fixed-point map $G$ supplies the raw update and the chosen SCF scheme determines how that update is mixed with earlier iterates.

## SCF methods

The public `solver(...)` entry point accepts an explicit SCF method through `scf=...`.
Current built-in methods include:

- `AndersonMixing(...)`
- `LinearMixing(...)`

The default solver path uses Anderson mixing with conservative settings.

## Output

Once the fixed point converges, `MeanFi` converts the converged density result back into a mean-field correction and returns that as `result.mf`.
