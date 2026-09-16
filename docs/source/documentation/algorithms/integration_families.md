# Integration families

MeanFi has two integration families: `FermiSimplex` integrates normal systems at
zero temperature using the native FermiSimplex library; `UniformGrid` samples an
isotropic periodic grid for normal and superconducting systems.

```{toctree}
:hidden:
:maxdepth: 1

method_notes/fermi_simplex.md
method_notes/uniform_grid.md
```

All method settings are keyword-only. Both integrators use the same contract:

```python
meanfi.FermiSimplex(nk=4096)
meanfi.UniformGrid(nk=4096)

meanfi.FermiSimplex(density_matrix_tol=1e-5, charge_tol=1e-6)
meanfi.UniformGrid(density_matrix_tol=1e-5, charge_tol=1e-6)

meanfi.FermiSimplex(initial_nk=256)  # Refine from at least 256 vertices.
meanfi.UniformGrid(initial_nk=256)   # Refine from at least 256 points.
```

An explicit `nk` requests a prescribed final mesh size. Omitting `nk` requests
accuracy control, using explicit targets or MeanFi's public tolerance policy.
An explicit `density_matrix_tol` also supplies `charge_tol` when it is omitted.
An explicit `charge_tol` can be either tighter or looser. Without mesh overrides,
both targets come from the tolerance policy.
Passing both `nk` and an integration target is an error. Safety limits such as
`max_points` and `max_refinements` do not select the mode. The top-level `tol`
continues to control filling roots and SCF convergence on a prescribed mesh.

`nk` is the requested **total number of mesh nodes**, with backend-dependent
rounding. It is never a per-axis resolution or a cumulative work count.
`result.statistics.requested_nk` and `result.statistics.n_kpoints` distinguish requested
and actual mesh sizes; cumulative diagonalizations are reported separately.
Both return the same `IntegrationInfo` record. Optional details such as
`grid_shape` or `n_leaves` are `None` where inapplicable.

`initial_nk` uses the same units and rounding as `nk`, but selects the initial
mesh for adaptive refinement. It cannot be combined with `nk`. Without either,
UniformGrid starts at 4 points per axis; FermiSimplex starts at 3 vertices per
axis, including the periodic boundaries.

| Request | Periodic grid | Simplex mesh |
| --- | --- | --- |
| `nk=1000`, 2D | 32 × 32 = 1024 | 33 × 33 = 1089 |
| `nk=4096`, 2D | 64 × 64 = 4096 | 65 × 65 = 4225 |
| `nk=4096`, 3D | 16 × 16 × 16 = 4096 | 17 × 17 × 17 = 4913 |

A prescribed calculation evaluates that discretization without hidden refinement
or additional error-checking meshes. Integration errors are unavailable (`None`), and a
converged filling root does not establish Brillouin-zone integration accuracy.
In accuracy-controlled mode, failure to meet a target within a budget raises a
convergence error. Reported error estimates are empirical, not certificates.

Keep density error at the returned chemical potential, uncertainty transferred
from the chemical potential, charge-integration error, root residual and SCF
residual distinct. See [fixed filling](fixed_filling.md).

## Energy and entropy accuracy

Density accuracy controls integration; energy and entropy are evaluated on the
accepted mesh without separate targets. Their estimated errors are diagnostics,
not convergence conditions. Band-energy errors scale with the Hamiltonian's
energy units. The default tolerance policy sets density and charge integration
targets and `matrix_function_tol` to `tol/5`, the filling residual to `tol/10`,
and the SCF residual to `tol`. These are separate stage targets, not a certified
bound on their sum or on chemical-potential error.

Read estimates through `result.errors.band_energy_integration` and
`result.errors.entropy_integration`, alongside the density and charge errors.
All four integration estimates are `None` on prescribed meshes. FermiSimplex
currently supplies density and charge estimates but does not estimate band-energy
or entropy integration error; these fields remain `None` there, except for an
exact finite-system evaluation. `result.statistics` contains work and mesh
information, not physical quantities or their errors.

Matrix-function approximation has its own target and estimate on every result:
`tolerance_policy(tol).matrix_function_tol` and
`result.errors.matrix_function_error`. AAA reports the largest sampled scalar
Fermi-function error across the evaluated momenta, which estimates an upper
limit on every integrated density-entry error from that approximation. Positive
integration weights preserve the pointwise bound. Direct diagonalization and
FermiSimplex report `None` for this approximation estimate.

For AAA, charge traces may require a tighter scalar fit because they sum many
diagonal entries. Filling-search constraints apply only when finding filling.
Entropy reuses the accepted poles but has no accuracy target; its sampled error
remains a diagnostic. Neither estimate measures unsampled momentum variation.
Passing mesh integration targets with `nk` remains an error; the policy's
matrix-function target remains active on prescribed meshes.

`entry_errors` describes momentum integration only. UniformGrid reports each
entry's coarse/fine difference. FermiSimplex supplies its global worst-entry
estimate for each entry, so those estimates are conservative within its error
model. Both describe the integrated real-space density at the returned chemical
potential, rather than an error at a single momentum.

## Migration

The development names `AdaptiveSimplex` and `PeriodicGrid` are now
`FermiSimplex` and `UniformGrid`; their settings are unchanged and no aliases
remain. `PeriodicQuadrature` and `AdaptiveQuadrature` are removed.

The earlier per-axis `UniformGrid` implementation used a different `nk`
convention. For that API, replace `nk=n` with `nk=n**d` in dimension `d` to
preserve the per-axis resolution. Replace adaptive quadrature with
`UniformGrid(density_matrix_tol=..., charge_tol=...)`. Remove old quadrature
rules, per-axis caps, derivative-accuracy and cache-policy options.

Prescribed finite-temperature sparse `RationalFOE` remains available with
`UniformGrid(nk=..., matrix_function=RationalFOE(...))`. Adaptive rational
integration is unsupported. Sparse calculations require an explicit supported
method rather than silently switching to a dense adaptive calculation.
