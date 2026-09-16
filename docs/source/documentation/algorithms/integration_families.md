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

meanfi.FermiSimplex()  # Accuracy comes from tol on the calculation.
meanfi.UniformGrid()

meanfi.FermiSimplex(initial_nk=256)  # Refine from at least 256 vertices.
meanfi.UniformGrid(initial_nk=256)   # Refine from at least 256 points.
```

An explicit `nk` requests a prescribed final mesh size. Omitting `nk` requests
accuracy control, using explicit targets or MeanFi's public tolerance policy.
All numerical targets live in `ErrorTolerances`, passed as `tol` to the calculation.
A numeric `tol` uses the default policy. An omitted `charge_integration` in an
explicit record follows `density_matrix_integration`; charge can also be tighter
or looser. With `nk`, integration targets do not apply, while filling, matrix-function
and SCF targets still do. Resource limits such as `max_points` and `max_refinements`
do not select the mode.

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
`result.errors.entropy`, alongside the density and charge errors.
All four integration estimates are `None` on prescribed meshes. FermiSimplex
currently supplies density and charge estimates but does not estimate band-energy
or entropy integration error; these fields remain `None` there, except for an
exact finite-system evaluation. `result.statistics` contains work and mesh
information, not physical quantities or their errors.

Matrix-function approximation has its own target and estimate on every result:
`ErrorTolerances.matrix_function_tol` and
`result.errors.matrix_function_error`. AAA reports the largest sampled scalar
Fermi-function error across the evaluated momenta, which estimates an upper
limit on every integrated density-entry error from that approximation. Positive
integration weights preserve the pointwise bound. Direct diagonalization and
FermiSimplex report `None` for this approximation estimate.

For AAA, charge traces may require a tighter scalar fit because they sum many
diagonal entries. Filling-search constraints apply only when finding filling.
Entropy reuses the accepted poles but has no accuracy target; its sampled error
remains a diagnostic. Neither estimate measures unsampled momentum variation.
The matrix-function target remains active on prescribed meshes.

`entry_errors` describes momentum integration only. UniformGrid reports each
entry's coarse/fine difference. FermiSimplex supplies its global worst-entry
estimate for each entry, so those estimates are conservative within its error
model. Both describe the integrated real-space density at the returned chemical
potential, rather than an error at a single momentum.

## Migration

The development names `AdaptiveSimplex` and `PeriodicGrid` are now
`FermiSimplex` and `UniformGrid`; no aliases
remain. `PeriodicQuadrature` and `AdaptiveQuadrature` are removed.

The earlier per-axis `UniformGrid` implementation used a different `nk`
convention. For that API, replace `nk=n` with `nk=n**d` in dimension `d` to
preserve the per-axis resolution. Replace adaptive quadrature with
`UniformGrid()` and configure accuracy through `tol`. Former `density_matrix_tol`
and `charge_tol` settings are now `ErrorTolerances.density_matrix_integration` and
`charge_integration`. Former per-call `scf_tol`, `filling_tol` and `mu_tol` are now
`scf_residual`, `filling_residual` and `mu_tol` in that same record. Remove old
quadrature rules, per-axis caps, derivative-accuracy and cache-policy options.

Prescribed finite-temperature sparse `RationalFOE` remains available with
`UniformGrid(nk=..., matrix_function=RationalFOE(...))`. Adaptive rational
integration is unsupported. Sparse calculations require an explicit supported
method rather than silently switching to a dense adaptive calculation.

The shared `errors.entropy` is populated only when entropy is computed and its
total error can be estimated; otherwise it is `None`. Entropy does not control
refinement. Finite systems ignore mesh sizes with a warning and have zero
integration error.
