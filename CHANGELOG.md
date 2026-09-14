# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking changes

- EDIIS is the default for normal zero-temperature simplex SCF. Other workflows
  use Anderson with explicit initial scaling, a five-step history and Armijo
  search; reference subtraction, warm restarts and spatial symmetries no longer
  trigger the previous default-mixer divergence. SCF settings are keyword-only;
  `history_size` and `regularization` replace `M` and `w0`; use `scf_tol` for the
  residual target instead of the removed secondary stopping controls.
- Both density functions accept a `Model` and optional `mean_field`, including
  BdG models. Model inputs supply filling, temperature and required coordinates.
  One density evaluator now serves normal and BdG calculations.
- `DensityResult.to_tb(sparse=False)` replaces `to_matrix()` and the
  `density_matrix` compatibility property. `meanfield` accepts selected results.
  Model construction uses only `reference=DensityResult`; the dictionary alias
  is removed. `hamiltonian_from_density` replaces `hamiltonian_from_rho`, and
  `hamiltonian_from_meanfield` handles both normal and BdG models.
- Fourier grids use explicit `shape` tuples. Odd, even, rectangular and finite
  grids round-trip completely; even-axis Nyquist coefficients are shared between
  opposite displacements to retain Hermiticity. Sparse Fourier inputs work.
  `fermi_energy` uses `shape`; `fermi_dirac` uses `mu`.
- Model construction validates physical inputs and owns read-only dense/CSR
  copies. `PeriodicGrid.dtype` replaces `workspace_precision`. Explicit and
  implicit prescribed sparse RationalFOE now both default to AAA.
- Numerical density failures raise `ConvergenceError`. SCF exceptions extend it;
  `SolverFailure.result` is None when the initial density evaluation fails.
- Kwant sparse conversion assembles sparse blocks directly and inverse conversion
  visits nonzero site pairs. Finite builders and scalar multi-orbital terms work;
  `builder_to_tb` controls are keyword-only.
- Package roots export supported user types without internal helper re-exports.

- Coordinate constructors always return a layout, including empty selections;
  removed `allow_empty` and unused conversion helpers. Sparse model SCF
  reconstruction stays sparse through mean-field and BdG assembly.
- SCF spaces use one explicit parametrization. Read entry layouts through
  `space.active_coordinates.entries` and `space.required_coordinates.entries`;
  removed the redundant entry-access helpers and optional mapping fields.

- Density integration and SCF now share one `DensityEntries` payload in
  `DensityResult.entries`; the private slice/evaluation wrappers are removed.
  Manual result construction uses `DensityResult(entries=DensityEntries(...),
  ...)`. Read-only `coordinates` and `values` remain available, and `entry_errors`
  preserves the corresponding integration estimates through selection.
- `density_matrix_at_mu` accepts the same mutually exclusive `keys`,
  `coordinates`, or `interaction` selection modes as `density_matrix`.
- Removed the unused `BdGMatrixFunction` marker and
  `RationalFOE.dn_dmu_rtol` setting. Use `DirectDiagonalization` or `RationalFOE`
  directly; periodic filling roots need no derivative-accuracy setting.
- MUMPS is optional: install `meanfi[sparse]` for sparse RationalFOE. Dense
  periodic and FermiSimplex workflows no longer require `python-mumps`. The
  unused `accel` extra is removed.
- Development test environments now pin Python 3.11, 3.12 and 3.13 separately;
  optional MUMPS/Kwant coverage runs in `test-sparse`. CI also builds and checks
  installed wheels with and without the sparse extra. Unused Hatch-VCS and
  placeholder environments were removed.
- Density evaluation and SCF share one internal result path, with backend work
  diagnostics available through `DensityResult.statistics`. Redundant planning,
  result and compatibility wrappers were removed.
- Two integration families: `AdaptiveSimplex` and `PeriodicGrid`. Removed
  `UniformGrid`, `PeriodicQuadrature`, `AdaptiveQuadrature`, their adapters and
  the `stateful-quadrature` runtime dependency.
- `nk` requests the total final mesh size. Convert old per-axis
  `UniformGrid(nk=n)` to `PeriodicGrid(nk=n**dimension)`. Periodic grids round up
  to an isotropic tensor mesh; simplex meshes use native dyadic construction and
  count both periodically equivalent boundary nodes.
- Explicit `nk` selects prescribed-size operation. Omitting it selects accuracy
  control. Combining `nk` with integration targets is rejected. Safety budgets
  never change the mode, and top-level `tol` remains valid for roots and SCF.
- Dense finite-temperature calculations default to direct periodic integration
  with global refinement and mandatory shifted-grid validation. BdG at zero
  temperature requires `PeriodicGrid(nk=...)`.
- Adaptive rational integration is unsupported. Use an explicit prescribed
  `PeriodicGrid(nk=..., matrix_function=RationalFOE(...))` for sparse matrices at
  positive temperature.
  Automatic sparse selection raises migration guidance instead of densifying.

### Accuracy and resource contracts

- AAA uses a certified constant occupation for nearly degenerate spectra,
  avoiding unstable rational fits and meeting tight sparse accuracy targets.

- Selected simplex outputs are assembled directly from requested entries,
  avoiding a full density-matrix round trip.
- Sparse RationalFOE reuses bounded scalar-fit setup and uses Gershgorin spectral
  bounds, with no Hamiltonian eigensolves.
- Prescribed meshes report unavailable integration errors as `None`; a filling
  root residual is separate from integration accuracy and SCF convergence.
- Periodic integration streams eigenvectors in bounded batches, retains bounded
  normal spectra across root evaluations and nested refinement, and recomputes
  BdG spectra when chemical potential changes. Grid/storage/refinement limits
  fail clearly.
- FermiSimplex band energies remain available. Periodic density integration does
  not add a finite-temperature thermodynamic energy or free-energy method.
- Historical benchmark reports and compact evidence remain under `performance/`,
  outside the runtime package and release archives.

## [1.1.0] - 2024-05-31

### Fixed
- Ensure {autolink}`~meanfi.space.hermitian.tb_to_rparams` and {autolink}`~meanfi.space.hermitian.rparams_to_tb` use minimal parametrisation of the tight-binding dictionary.

### Added
- Density matrix cost for the mean-field solver.
- Functionality {autolink}`~meanfi.interop.kwant.tb_to_builder` to create `kwant` systems with the tight-binding dictionaries.
- Functionality {autolink}`~meanfi.interop.kwant.tb_to_kfunc` to create k-dependent function from tight-binding format.

### Changed
- Rewrote {autolink}`~meanfi.interop.kwant.builder_to_tb` to avoid potential bugs and remove the use of `copy`.

## [1.0.0] - 2024-05-09

- First release of _MeanFi_.
