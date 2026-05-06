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
# `AdaptiveQuadrature`

`AdaptiveQuadrature` is the adaptive finite-temperature Brillouin-zone integrator.
Its implementation is based on the
[stateful_quadrature package](https://github.com/Kostusas/stateful_quadrature).

It approximates Brillouin-zone integrals of the form

:::{math}
\rho(R) = \int_{\mathrm{BZ}} \frac{d^d k}{(2\pi)^d}\;
e^{i k \cdot R}\,\rho(k),
:::

but does not use a fixed tensor grid.
Instead it places cubature nodes adaptively and refines only where the estimated integration error is large.

## Main idea

At each sampled point $k_j$, the backend evaluates either:

- the charge integrand needed for the fixed-filling solve, or
- the density-matrix integrand needed for the final output.

The adaptive controller then builds an estimate

:::{math}
I \approx \sum_j w_j f(k_j)
:::

and refines the integration tree until the requested tolerance is met or the refinement budget is exhausted.

## What is reused

The same adaptive structure is reused across repeated chemical-potential evaluations during the fixed-filling solve.
This matters because solving $N(\mu)=\nu$ requires many charge evaluations before the final density integral is taken.

## Relation to matrix-function backends

`AdaptiveQuadrature` is only the k-space integrator.
At each sampled $k$ point it still delegates to a matrix-function backend:

- exact diagonalization,
- or rational FOE.

## Cost versus error scaling

If the adaptive tree ends up using $N_k$ effective sample points, then the total cost is roughly

:::{math}
\text{cost} \sim N_k \times C_k,
:::

multiplied again by however many charge evaluations are needed while solving for $\mu$.
If the cubature rule has effective algebraic order $q$ on a smooth $d$-dimensional integrand, then one expects a relation of the form

:::{math}
\varepsilon \sim N_k^{-q/d},
\qquad
\text{cost} \sim \varepsilon^{-d/q} C_k.
:::

So the method is attractive when:

- the integrand is smooth enough that adaptivity reduces the required number of k-points,
- and the per-node matrix-function evaluation is not too expensive.

## Practical knobs

- `density_matrix_tol`: integration target
- `max_refinements`: optional refinement cap
- `rule`: cubature rule choice
- `matrix_function`: backend used at each sampled node

It is the default integration family for `kT > 0`.
