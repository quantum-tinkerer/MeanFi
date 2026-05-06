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
Here $\nu$ is the target filling per unit cell, while the computed filling is obtained from a Brillouin-zone integral of the local-in-$k$ charge density,

:::{math}
N(\mu) = \int_{\mathrm{BZ}} n(\mu,k)\, dk,
:::

with

:::{math}
n(\mu,k) = \operatorname{tr}\!\bigl(W\,\rho(k,\mu)\bigr),
:::

where $W$ selects the local orbitals and weights that define the filling convention.
So the root solve is nested inside the same single-$k$ evaluation and Brillouin-zone integration machinery used for the density matrix itself.

## Bracketing and root solve

The inner fixed-filling solver:

1. builds an initial bracket for $\mu$ from a spectral bound,
2. expands the bracket until the target filling is enclosed,
3. solves for $\mu$ with safeguarded Newton steps when derivative information is available, and otherwise with safeguarded bracketed interpolation plus midpoint fallback.

If derivative information is unavailable or unusable, the solve falls back to bracketed updates only.
When the derivative can be computed, Newton uses

:::{math}
\mu_{j+1}
=
\mu_j - \frac{N(\mu_j)-\nu}{N'(\mu_j)},
\qquad
N'(\mu) = \int_{\mathrm{BZ}} \partial_\mu n(\mu,k)\, dk.
:::

So the same local-in-$k$ machinery that evaluates $n(\mu,k)$ may also supply the derivative information needed for faster root updates.
The derivative-aware path is safeguarded Newton, while the derivative-free path is a bracketed interpolation/bisection-type solver.

## Which paths use which root update?

The root solver depends on whether the underlying density-evaluation stack can provide $N'(\mu)$.

- `AdaptiveQuadrature` with `DirectDiagonalization`: safeguarded Newton
- `AdaptiveQuadrature` with derivative-aware rational evaluation: safeguarded Newton
- `AdaptiveQuadrature` with `RationalFOE(rational_scheme="aaa")`: bracketed interpolation/bisection
- `UniformGrid` at finite temperature with `DirectDiagonalization`: safeguarded Newton
- `UniformGrid` at finite temperature with derivative-aware rational evaluation: safeguarded Newton
- `UniformGrid` with `RationalFOE(rational_scheme="aaa")`: bracketed interpolation/bisection
- zero-temperature `AdaptiveSimplex`: bracketed interpolation/bisection
- zero-temperature `UniformGrid`: bracketed interpolation/bisection

So the practical distinction is not the integrator alone.
It is whether the full stack can evaluate both $N(\mu)$ and $N'(\mu)$, or only $N(\mu)$.

## Cost versus error scaling

If one charge evaluation has work $W_N$, then one root iteration costs roughly

:::{math}
\text{cost per root step} \sim C_N.
:::

For safeguarded Newton, once the iteration is in its local regime,

:::{math}
| \mu_{j+1} - \mu_\ast | \approx C |\mu_j - \mu_\ast|^2,
:::

so the error contracts quadratically in the ideal derivative-aware regime.

For pure bisection on a bracket $[\mu_-,\mu_+]$,

:::{math}
|\mu_j-\mu_\ast|
\le
\frac{\mu_+ - \mu_-}{2^{j+1}},
\qquad
j \sim \log_2\!\frac{\mu_+ - \mu_-}{\varepsilon_\mu}.
:::

So the total bisection cost scales like

:::{math}
\text{cost} \sim C_N \log\!\frac{1}{\varepsilon_\mu}.
:::

The derivative-free branch used in `MeanFi` is usually better than raw bisection when interpolation helps, but it retains the same logarithmic worst-case bracket-shrinking behavior.
So, at the level of asymptotic work versus root accuracy,

:::{math}
\text{Newton:}\quad \text{cost} \sim C_N \log\log\!\frac{1}{\varepsilon_\mu}
\qquad\text{(local ideal regime)},
:::

while

:::{math}
\text{bisection-type:}\quad \text{cost} \sim C_N \log\!\frac{1}{\varepsilon_\mu}.
:::

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
