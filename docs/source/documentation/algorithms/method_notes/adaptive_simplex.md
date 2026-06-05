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
# `AdaptiveSimplex`

`AdaptiveSimplex` is the dedicated zero-temperature adaptive integration backend for normal-state calculations.
Its physics-specific native implementation lives in the separate
[lineartetrahedron package](https://gitlab.kwant-project.org/qt/lineartetrahedron),
while the generic adaptive mesh engine lives in
[adaptivesimplex](https://gitlab.kwant-project.org/qt/adaptivesimplex).
MeanFi keeps the public integration API and dispatch logic.

At zero temperature, the occupation becomes discontinuous, so the finite-temperature quadrature machinery is no longer the natural default.
Instead, `AdaptiveSimplex` refines a simplicial partition of the Brillouin zone and estimates the integral from local simplex contributions.

## Main idea

The Brillouin zone is partitioned into simplices.
For each active simplex $\sigma$, the backend compares:

- a coarse contribution on that simplex,
- a preview contribution on refined descendants.

The difference between the two is used as an error indicator.
Refinement then focuses on the simplices with the largest contribution to the total estimated error.

## Fixed-filling workflow

At zero temperature the backend uses the same geometric infrastructure for both:

- the charge solve $N(\mu)=\nu$,
- and the final density integral at the converged $\mu$.

The density stage starts from the charge-converged refined mesh rather than rebuilding from scratch.

## Cost versus error scaling

The exact cost depends on how many simplices are activated by refinement.
Very roughly,

:::{math}
\text{cost} \sim N_{\sigma} \times C_{\sigma},
:::

where $N_{\sigma}$ is the number of active simplices visited by the adaptive controller.
For the linear tetrahedron type behavior that this method is designed around, a useful rule of thumb is

:::{math}
\text{cost} \sim \varepsilon^{-d/2} C_{\sigma}.
:::

This method is specialized but efficient when the zero-temperature integrand is difficult enough that a fixed grid would need many points.

## Practical knobs

- `density_matrix_tol`
- `max_refinements`
- `num_threads`

`num_threads` requests a per-integration native worker count from the `lineartetrahedron`
backend. Leave it as `None` to use the backend default. This is the preferred way to
control AdaptiveSimplex threading from notebooks or task schedulers such as Dask, where
setting OpenMP environment variables for each calculation is awkward.

It requires `kT = 0` and is the default normal-state integration family at zero temperature.
