# Integration families

MeanFi has two integration families: `AdaptiveSimplex` uses FermiSimplex for
normal systems at zero temperature; `PeriodicGrid` samples an isotropic periodic
grid for normal and superconducting systems.

```{toctree}
:hidden:
:maxdepth: 1

method_notes/adaptive_simplex.md
method_notes/periodic_grid.md
```

Both use the same input-driven contract:

```python
meanfi.AdaptiveSimplex(nk=4096)
meanfi.PeriodicGrid(nk=4096)

meanfi.AdaptiveSimplex(density_matrix_tol=1e-5, charge_tol=1e-6)
meanfi.PeriodicGrid(density_matrix_tol=1e-5, charge_tol=1e-6)
```

An explicit `nk` requests a prescribed final mesh size. Omitting `nk` requests
accuracy control, using explicit targets or MeanFi's public tolerance policy.
Passing both `nk` and an integration target is an error. Safety limits such as
`max_points` and `max_refinements` do not select the mode. The top-level `tol`
continues to control filling roots and SCF convergence on a prescribed mesh.

`nk` is the requested **total number of mesh nodes**, with backend-dependent
rounding. It is never a per-axis resolution or a cumulative work count.
`result.statistics.requested_nk` and `result.statistics.n_kpoints` distinguish requested
and actual mesh sizes; cumulative diagonalizations are reported separately.

| Request | Periodic grid | Simplex mesh |
| --- | --- | --- |
| `nk=1000`, 2D | 32 × 32 = 1024 | 33 × 33 = 1089 |
| `nk=4096`, 2D | 64 × 64 = 4096 | 65 × 65 = 4225 |
| `nk=4096`, 3D | 16 × 16 × 16 = 4096 | 17 × 17 × 17 = 4913 |

A prescribed calculation evaluates that discretization without hidden refinement
or shifted validation. Integration errors are unavailable (`None`), and a
converged filling root does not establish Brillouin-zone integration accuracy.
In accuracy-controlled mode, failure to meet a target within a budget raises a
convergence error. Reported error estimates are empirical, not certificates.

Keep density error at the returned chemical potential, uncertainty transferred
from the chemical potential, charge-integration error, root residual and SCF
residual distinct. See [fixed filling](fixed_filling.md).

## Migration

`PeriodicGrid` replaces `UniformGrid`, `PeriodicQuadrature` and
`AdaptiveQuadrature`; the old names are removed. Convert an old
`UniformGrid(nk=n)` in dimension `d` to `PeriodicGrid(nk=n**d)` to preserve its
per-axis resolution. Replace adaptive quadrature with
`PeriodicGrid(density_matrix_tol=..., charge_tol=...)`. Remove old quadrature
rules, per-axis caps, derivative-accuracy and cache-policy options.

Prescribed finite-temperature sparse `RationalFOE` remains available with
`PeriodicGrid(nk=..., matrix_function=RationalFOE(...))`. Adaptive rational
integration is unsupported. Sparse calculations require an explicit supported
method rather than silently switching to a dense adaptive calculation.
