# Changelog

## [Unreleased]

This major revision simplifies the density, integration and SCF APIs without
preserving the previous interfaces.

- `Model` owns filling, temperature, interaction, symmetries and an optional
  reference density. Normal and superconducting models share the same density
  and reference-subtracted energy conventions.
- `density_matrix` solves for filling; `density_matrix_at_mu` evaluates a supplied
  chemical potential. Both return `DensityResult` with explicit selected entries,
  physical quantities, optional error estimates and work statistics.
- `FermiSimplex` handles normal zero-temperature integration. `UniformGrid` handles
  prescribed grids and coarse/fine refinement; sparse finite-temperature
  calculations use `RationalFOE` with AAA and optional MUMPS.
- `ErrorTolerances` contains the numerical targets. The default policy maps the
  scalar `tol` to independent SCF, integration, filling and matrix-function targets.
  Missing or inapplicable estimates are `None`; fixed-mu calculations do not run
  a separate charge integration to populate diagnostics.
- EDIIS is the default SCF method, uses internal energy and never switches methods
  automatically. Results retain the evaluated mean field, density and history,
  including the last valid state on nonconvergence.
- Energies and entropy are per cell per physical orbital. Entropy/free energy are
  opt-in through `compute_free_energy=True`, evaluated only after SCF terminates.
- Dense and sparse density layouts share coordinate selection. Fourier grids use
  explicit shapes; Kwant conversion supports sparse and finite systems.
- Python 3.11–3.13 are supported. MUMPS and Kwant are optional extras. FermiSimplex
  is currently pinned to a source revision; see the development guide for the
  remaining PyPI release prerequisite.
- Tutorials execute from scratch. Benchmarks use one script with live reference
  comparisons; generated data, historical benchmark archives and temporary
  release/API notes are no longer tracked.

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
