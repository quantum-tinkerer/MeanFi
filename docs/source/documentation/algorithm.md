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
# Algorithm Overview

## Self-Consistent Mean-Field Loop

To calculate the mean-field interaction in {eq}`mf_infinite`, we require the finite-temperature density
matrix $\rho_{mn}(R)$ at the target filling $\nu$. However, {eq}`density` depends on both the chemical
potential $\mu$ and the mean-field interaction $\hat{V}_{\text{MF}}$ itself. Hence, these quantities must
still be determined self-consistently.

A single iteration of the self-consistency loop can be viewed as a function that, given an initial density
matrix $\rho_{mn}(R)_{\text{init}}$, computes a new density matrix $\rho_{mn}(R)_{\text{new}}$ while solving
for the chemical potential internally:

$$
\mathrm{MF}\bigl(\rho_{mn}(R)_{\text{init}}\bigr) \to \rho_{mn}(R)_{\text{new}}.
$$

This iteration is defined in the {autolink}`~meanfi.model.Model.density_matrix` method and proceeds as follows:

1. **Mean-field calculation** ({autolink}`~meanfi.mf.meanfield`): Compute the new mean-field potential
   $\hat{V}_{\text{new, MF}}(R)$ using {eq}`mf_infinite`.
2. **Density matrix update** ({autolink}`~meanfi.mf.density_matrix`):
   1. **Momentum-space Hamiltonian** ({autolink}`~meanfi.tb.transforms.tb_to_kfunc`): Transform the real-space
      tight-binding Hamiltonian into a function of $k$ that returns $\hat{H}(k)$.
   2. **Cached spectral data** (`stateful_quadrature`): Diagonalize $\hat{H}(k)$ on adaptive cubature nodes and
      cache the eigensystems. This is the expensive part of the calculation.
   3. **Chemical-potential solve**: Reuse the cached eigensystems to evaluate the total charge $N(\mu)$ and
      its derivative, then solve $N(\mu) = \nu$ with safeguarded Newton steps and bisection fallback.
   4. **Density integral**: Reuse the same adaptive quadrature tree to evaluate the final real-space density
      matrix $\rho_{mn}(R)_{\text{new}}$ at the converged $\mu$.

The related {autolink}`~meanfi.mf.density_matrix_at_mu` function performs the same spectral caching and density
integration steps when the chemical potential is supplied explicitly.

## Self-Consistency Criteria

To define the self-consistency condition clearly, we introduce a mapping function $f$ that transforms a Hermitian
tight-binding dictionary into a real vector of parameters:

$$
f : \rho_{mn}(R) \;\longmapsto\; f(\rho_{mn}(R)) \in \mathbb{R}^N.
$$

In the code, $f$ is represented by {autolink}`~meanfi.params.rparams.tb_to_rparams`, and its inverse is
{autolink}`~meanfi.params.rparams.rparams_to_tb`. Currently, $f$ takes only the upper-triangular elements of the
density matrix and separates them into real and imaginary parts, creating a real-valued vector.

### Fixed-Point Formulation

A solution to the mean-field problem requires two conditions:

1. **Fixed point**: $\rho_{mn}(R)$ must be a fixed point of the mean-field iteration:

   $$
   f\bigl(\mathrm{MF}(\rho_{mn}(R))\bigr) = f(\rho_{mn}(R)).
   $$

2. **Filling constraint**: The trace of $\rho_{mn}(0)$ equals the target filling $\nu$, i.e.,
   $\mathrm{Tr}[\rho_{mn}(0)] = \nu$.

In the current implementation, the filling constraint is enforced inside each density update by the internal
chemical-potential solve. This means the outer self-consistent loop works only with the reduced density-matrix
parameters and does not treat $\mu$ as an independent optimization variable.

## Current Solving Approach

### Reduced Parameter Vector

For clarity, let us denote

$$
\boldsymbol{\theta} = f(\rho_{mn}(R)) \in \mathbb{R}^N,
$$

the parameterized representation of the density matrix.

### Fixed-Point Objective

We define the fixed-filling update map

$$
\mathcal{G}(\boldsymbol{\theta}) =
f\Bigl(\mathrm{MF}\bigl(f^{-1}(\boldsymbol{\theta})\bigr)\Bigr),
$$

where the call to $\mathrm{MF}$ includes the internal solve for the chemical potential needed to satisfy the
filling target. The self-consistent solution therefore solves

$$
\mathcal{F}(\boldsymbol{\theta}) =
\mathcal{G}(\boldsymbol{\theta}) - \boldsymbol{\theta} = 0.
$$

This is the residual evaluated by {autolink}`~meanfi.solvers.solver`.

### Numerical Methods

The default solver applies internal linear mixing to the reduced density parameters,

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \alpha \, \mathcal{F}(\boldsymbol{\theta}_k),
$$

with a convergence test on the residual norm. For harder problems, the public
{autolink}`~meanfi.solvers.solver` interface also accepts an external optimizer callable through the
`optimizer=` hook, for example `scipy.optimize.anderson`.

Because the filling constraint is already handled inside each density update, such optimizers only see the
reduced-density fixed-point residual and do not need to manage the chemical potential separately.

### Chemical-Potential Stability

The chemical potential $\mu$ no longer appears as an independent variable in the outer solver. Instead, each
density update brackets $\mu$ from a bandwidth estimate, expands the bracket until the target filling is enclosed,
and then uses safeguarded Newton steps with bisection fallback to find the root.

This keeps the outer optimizer away from chemical-potential instabilities while still reusing the adaptive
quadrature cache across the repeated charge evaluations. The main package supports finite-temperature calculations
only; dense $k$-grid integrations and repeated SciPy cubature are kept only as testing and benchmarking references,
not as the production solver path.
