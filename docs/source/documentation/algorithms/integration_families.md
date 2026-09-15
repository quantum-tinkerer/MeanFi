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
meanfi.PeriodicGrid(
    density_matrix_tol=1e-5, charge_tol=1e-6,
    energy_tol=1e-7, entropy_tol=1e-7,
)
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

## Energy and entropy accuracy

`PeriodicGrid.energy_tol` is an absolute band-energy integration target in the
Hamiltonian's energy units per cell per physical orbital. `entropy_tol` is an
absolute entropy target in k_B per cell per physical orbital. These are
independent of the density-entry and charge targets. The default tolerance
policy sets each integration target to `tol/5`, the filling residual to `tol/10`,
and the SCF residual to `tol`. An explicit target overrides only its own budget.
For example, changing `density_matrix_tol` does not change the energy target.

Read estimates through `result.errors.band_energy_integration` and
`result.errors.entropy_integration`, alongside the density and charge errors.
All four integration estimates are `None` on prescribed meshes. FermiSimplex
currently supplies density and charge estimates but does not estimate band-energy
or entropy integration error; these fields remain `None` there, except for an
exact finite-system evaluation. `result.statistics` contains work and mesh
information, not physical quantities or their errors.

For a prescribed sparse calculation, scalar matrix-function targets still come
from `tol` or `tolerance_policy`; passing mesh integration targets with `nk` is
an error. A custom policy can independently replace
`ErrorTolerances.band_energy_integration` and `entropy_integration`. These control
the sampled matrix-function error; they do not estimate unsampled Brillouin-zone
integration error.

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
