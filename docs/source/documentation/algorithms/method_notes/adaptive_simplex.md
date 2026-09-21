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
[FermiSimplex package](https://gitlab.kwant-project.org/qt/lineartetrahedron),
while the generic adaptive mesh engine lives in
[adaptivesimplex](https://gitlab.kwant-project.org/qt/adaptivesimplex).
MeanFi keeps the public integration API and dispatch logic.

At zero temperature, the occupation becomes discontinuous, so the finite-temperature quadrature machinery is no longer the natural default.
Instead, `AdaptiveSimplex` refines a simplicial partition of the Brillouin zone and estimates the integral from local simplex contributions.

## Main idea

Charge integration retains adaptive mesh refinement and its existing sampled
error estimator. The density stage then keeps that mesh fixed and raises the
cubature degree on simplices with the largest density-error indicators.
At fixed chemical potential, MeanFi first resolves charge to `charge_tol`;
at fixed filling it reuses the mesh from the chemical-potential solve.

For a simplex of dimension $d$ and volume $V$, the first pair is

:::{math}
Q_1 = \frac{V}{d+1}\sum_i f(v_i), \qquad
Q_2 = \frac{V}{(d+1)(d+2)}\sum_i f(v_i)
    + \frac{V(d+1)}{d+2}f(c),
:::

where $c$ is the centroid. $Q_2$ integrates quadratics exactly. The first
indicator is the maximum absolute requested-component difference between
$Q_2$ and $Q_1$. Larger degrees use nested Grundmann–Moeller rules of degrees
3, 5, ..., 21, with analytic coefficients and reused interior samples.
Only the requested density components are stored at interior nodes.

The controller uses the same error policy as density h-refinement: the maximum
of the root-sum-square of local correction norms and the norm of the coherently
summed complex corrections, plus a separate floating-point floor. It promotes
the largest contributors until this estimate is below `density_matrix_tol`.
This is a sampled estimate, not a rigorous bound. Geometric splits and order
promotions are reported separately in the internal integration statistics as `refinements`
and `p_refinements`.

## Occupation approximation

At every density cubature point the projector of each band is weighted by
that band's linear-simplex occupied volume fraction. This applies to both
bulk and cut simplices and preserves the onsite trace of the charge result.
The existing occupied barycentric moments also correct the vertex-linear
projector/Fourier contribution inside cut simplices. This removes the leading
occupation correlation error without new diagonalizations. The cubature estimate
still excludes charge error and higher-order occupation correlation, so
p-refinement alone cannot remove all metallic density error. Tightening
`charge_tol` resolves the charge geometry but does not itself certify the
remaining density error. Unequally occupied degenerate bands can also limit
p-convergence.

## Cost versus error scaling

Linear bulk density integration can require work proportional to
$\varepsilon^{-d/2}$ even where occupations are fixed, because occupied
projectors vary. Raising the cubature order can reduce this bulk cost.
A fixed-degree rule alone does not guarantee arbitrary accuracy on a fixed
mesh: the initial vertices-plus-centroid approximation can still leave an
error floor. The order must increase, or the charge mesh must become finer.
Measured costs and errors should be used to assess scaling; neither the
initial error pair nor the cut approximation guarantees a universal exponent.

## Practical knobs

- `density_matrix_tol`: target for the density cubature estimate.
- `charge_tol`: target for the unchanged charge-mesh refinement.
- `density_max_degree`: 2 or an odd degree from 3 through 21, default 21.
  Set 2 to evaluate only the vertices/vertices-plus-centroid pair.
- `max_refinements`: caps charge splits and density order promotions separately.
- `num_threads`: defaults to 1; `None` leaves native thread limits unchanged.
  With OpenMP available and at least 32 orbitals, requesting more than one
  thread enables density promotion batches of up to 16 simplices. Smaller
  Hamiltonians and builds without OpenMP retain serial promotions.

A degree or work limit reached above tolerance raises a nonconvergence error.
The density controller does not silently fall back to simplex splitting.

It requires `kT = 0` and is the default normal-state integration family at zero temperature.
