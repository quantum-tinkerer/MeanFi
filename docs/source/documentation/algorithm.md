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

To calculate the mean-field interaction in {eq}`mf_infinite`, we require the density
matrix $\rho_{mn}(R)$ at the target filling $\nu$. However, {eq}`density` depends on both the chemical
potential $\mu$ and the mean-field interaction $\hat{V}_{\text{MF}}$ itself. Hence, these quantities must
still be determined self-consistently.

A single iteration of the self-consistency loop can be viewed as a function that, given an initial density
matrix $\rho_{mn}(R)_{\text{init}}$, computes a new density matrix $\rho_{mn}(R)_{\text{new}}$ while solving
for the chemical potential internally:

$$
\mathrm{MF}\bigl(\rho_{mn}(R)_{\text{init}}\bigr) \to \rho_{mn}(R)_{\text{new}}.
$$

This iteration is exposed through the top-level density API and proceeds as follows:

1. **Mean-field calculation** ({autolink}`~meanfi.meanfield`): Compute the new mean-field potential
   $\hat{V}_{\text{new, MF}}(R)$ using {eq}`mf_infinite`.
2. **Density matrix update** ({autolink}`~meanfi.density_matrix`):
   1. **Momentum-space Hamiltonian** ({autolink}`~meanfi.tb.transforms.tb_to_kfunc`): Transform the real-space
      tight-binding Hamiltonian into a function of $k$ that returns $\hat{H}(k)$.
   2. **Temperature-dependent backend**:
      - For `kT > 0`, `stateful_quadrature` diagonalizes $\hat{H}(k)$ on adaptive cubature nodes, caches the
        eigensystems, solves $N(\mu)=\nu$, and then reuses the same quadrature tree for the final density integral.
      - For `kT = 0`, a separate adaptive simplicial backend first solves $N(\mu)=\nu$ on a hierarchically refined
        simplex mesh and then starts an adaptive density quadrature pass from that charge-converged mesh at frozen $\mu$.

The related {autolink}`~meanfi.density_matrix_at_mu` function performs the same spectral caching and density
integration steps when the chemical potential is supplied explicitly. In the zero-temperature backend, this means
starting directly from the simplicial density stage at fixed $\mu$.

## Self-Consistency Criteria

To define the self-consistency condition clearly, we introduce a mapping function $f$ that transforms a Hermitian
tight-binding dictionary into a real vector of parameters:

$$
f : \rho_{mn}(R) \;\longmapsto\; f(\rho_{mn}(R)) \in \mathbb{R}^N.
$$

In the code, $f$ is represented by {autolink}`~meanfi.state.normal.tb_to_rparams`, and its inverse is
{autolink}`~meanfi.state.normal.rparams_to_tb`. Currently, $f$ takes only the upper-triangular elements of the
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

This is the residual evaluated by {autolink}`~meanfi.solver`.

### Numerical Methods

The default solver applies internal linear mixing to the reduced density parameters,

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \alpha \, \mathcal{F}(\boldsymbol{\theta}_k),
$$

with a convergence test on the residual norm. For harder problems, the public
{autolink}`~meanfi.solver` interface switches to an explicit SCF method such as
{autolink}`~meanfi.scf.AndersonMixing`.

Because the filling constraint is already handled inside each density update, such SCF methods only see the
reduced-density fixed-point residual and do not need to manage the chemical potential separately.

### Chemical-Potential Stability

The chemical potential $\mu$ no longer appears as an independent variable in the outer solver. Instead, each
density update brackets $\mu$ from a bandwidth estimate, expands the bracket until the target filling is enclosed,
and then uses safeguarded Newton steps with bisection fallback to find the root.

This keeps the outer SCF method away from chemical-potential instabilities while still reusing the adaptive
quadrature or simplicial cache across the repeated charge evaluations. For `kT = 0`, the Brillouin zone is treated
as a torus mathematically. The geometry layer still keeps duplicated seam vertices so that the simplicial mesh never
crosses a periodic boundary, but the spectral layer wraps reduced coordinates before diagonalization so seam-related
geometry vertices collapse onto the same physical $k$ point in the eigensystem cache. To avoid seam-crossing
simplices, the root mesh partitions the fundamental cell into $2^d$ half-sized subcells before triangulating each
one. Dense $k$-grid integrations and repeated SciPy cubature are kept only as testing and benchmarking references,
not as the production solver path. Adaptive integration controls now live on explicit integration methods such as
`AdaptiveSimplex(max_refinements=...)`, `AdaptiveQuadrature(density_matrix_tol=...)`, and `UniformGrid(nk=...)`
rather than on `Model`. For `kT = 0`, adaptive density estimation compares a coarse owner-simplex vertex
cubature against a one-level preview vertex cubature and uses the preview value as the accepted contribution, but
the native controller updates those active-leaf preview sums incrementally rather than recomputing the whole active
frontier each round. All density and SCF stopping criteria are reported in componentwise max norms rather than L2
norms. Finally, `AdaptiveSimplex(max_refinements=0)` is a distinct root-mesh-only mode: it performs no preview, no
refinement, and therefore provides no density-error indicator.
