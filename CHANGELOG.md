# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking changes

- The public integration methods are `FermiSimplex` and `UniformGrid`, replacing
  the development names `AdaptiveSimplex` and `PeriodicGrid` without aliases.
  Settings and numerical behavior are unchanged; `nk` still requests total points.

- Integration results keep physical quantities on the result and error estimates
  in `errors`; removed physical values, errors and unused energy counters from
  backend statistics. `band_energy_integration` and `entropy_integration` in
  `ErrorValues` are diagnostics only. Density accuracy controls integration;
  energy and entropy have no separate targets and do not trigger refinement.
  Sparse thermodynamics uses the density fit without extra energy-driven accuracy.
- Density preparation resolves settings once. Normal and BdG SCF share a single
  problem implementation; simplex fixed-mu and fixed-filling calculations share
  one refinement loop. Removed obsolete private dispatch functions and simplex
  compatibility switches. Sparse inverse layouts are immutable and shared across
  a calculation, while numeric factorizations remain local to each node.
- Added the contributor guidelines in `AGENTS.md`, the package design in
  `DESIGN.md`, a runnable quickstart, and isolated dependency-install CI coverage.
  Removed the unused `packaging` dependency; version is `2.0.0rc1`.

- Internal, free, and band energies and entropy are now per cell per physical
  orbital. Divide physical totals by N for an N-orbital model, including BdG
  models with 2N-dimensional Hamiltonians. Energy and entropy integration errors
  use the same normalization. Filling remains electrons per cell; the generic
  `expectation_value` remains an unnormalized trace.

- Final SCF results expose `internal_energy`, `free_energy`, and entropy
  in units of Boltzmann's constant. `free_energy = internal_energy - kT * entropy`.
  Iteration history and progress report internal energy only; verbose output
  prints entropy and free energy once at convergence.
  The old `total_energy` name is removed; use `internal_energy(model, density)`
  or `free_energy(model, density)` for observables. BdG pairing energy now uses
  the conjugate anomalous density, preserving phase invariance.
- EDIIS is the default for every supported SCF calculation. It minimizes a
  internal energy over the density history and never switches methods. A small
  quadratic history matrix replaces repeated model evaluations during
  coefficient optimization; entropy is excluded from the objective.
  Users can explicitly restart with another method after `NoConvergence`.
  An explicit density integration target supplies an omitted charge target;
  charge accuracy remains independently adjustable. SCF settings remain
  keyword-only; `history_size` and `regularization` replace `M` and `w0`; use
  `scf_tol` for the residual target instead of secondary stopping controls.
- RationalFOE now uses one AAA implementation for density and entropy with shared
  poles and factorizations. Ozaki and `rational_scheme` are removed: replace
  `RationalFOE(rational_scheme=...)` with `RationalFOE()`. Existing pole budgets
  remain available. The benchmark report records the density-only speed tradeoff
  and why Ozaki's poles were unsuitable for the shared-entropy calculation.
- BdG SCF interaction support now comes from the model instead of the keys in
  the initial guess. Removed dynamic key tracking and zero-block adapters that
  could discard nonlocal interactions.
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
  copies. `UniformGrid.dtype` replaces `workspace_precision`. Explicit and
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
- Two integration families: `FermiSimplex` and `UniformGrid`. Removed
  `PeriodicQuadrature`, `AdaptiveQuadrature`, their adapters and the
  `stateful-quadrature` runtime dependency. The earlier per-axis `UniformGrid`
  implementation is replaced by the shared periodic evaluator.
- `nk` requests the total final mesh size. For code using the earlier per-axis
  grid, replace `nk=n` with `nk=n**dimension`. Periodic grids round up
  to an isotropic tensor mesh; simplex meshes use native dyadic construction and
  count both periodically equivalent boundary nodes.
- Explicit `nk` selects prescribed-size operation. Omitting it selects accuracy
  control. Combining `nk` with integration targets is rejected. Safety budgets
  never change the mode, and top-level `tol` remains valid for roots and SCF.
- Dense finite-temperature calculations default to direct periodic integration
  with global refinement and mandatory shifted-grid validation. BdG at zero
  temperature requires `UniformGrid(nk=...)`.
- Adaptive rational integration is unsupported. Use an explicit prescribed
  `UniformGrid(nk=..., matrix_function=RationalFOE(...))` for sparse matrices at
  positive temperature.
  Automatic sparse selection raises migration guidance instead of densifying.

### Accuracy and resource contracts

- AAA fitting starts on a smaller grid and refines only when needed, while keeping
  the dense validation grid and final accuracy targets unchanged. QR reduces its
  weight solve to a small SVD. One validated fit is reused across nearby chemical
  potentials and k-points, with modest interval padding after a bounds miss and
  a retry on the actual spectrum when padding exceeds the pole budget. No new
  settings or dependencies are introduced.

- AAA can now refit residues from a nearly converged intermediate approximation.
  Final certification keeps the requested tolerance, fixing the 32-orbital
  thermodynamic regression at its original `1e-12` total-trace target.

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
- Density results retain occupied band energy and full-state entropy, including
  selected layouts. Dense evaluation reuses the eigensystem; sparse evaluation
  reuses selected inverse entries. Energy and entropy error estimates are
  diagnostics only. Zero-temperature flat half-filled bands retain residual
  entropy. FermiSimplex band-energy calculation uses its retained spectra.
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
