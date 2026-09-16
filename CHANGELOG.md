# Changelog

## [2.0.0] - Unreleased

MeanFi 2.0 redesigns the calculation API. Updating from 1.x requires code changes.

### Added

- Finite-temperature calculations, internal energy, and optional entropy/free energy.
- Superconducting Bogoliubov–de Gennes models with self-consistent pairing.
- Reference-subtracted interactions for normal and superconducting models, and
  spatial symmetry constraints including shifted orbital mappings.
- Adaptive zero-temperature integration with `FermiSimplex` and sparse
  finite-temperature density evaluation with AAA and optional MUMPS.
- Selected density entries, numerical error estimates and calculation statistics.
- EDIIS, linear mixing and Anderson mixing as explicit SCF methods.
- Executable tutorials, an API walkthrough, numerical reference tests and benchmarks.

### Changed

- `Model` owns validated inputs, temperature and filling. Use its `mean_field`,
  `hamiltonian_from_meanfield` and `random_meanfield` methods to construct calculations.
- `density_matrix` returns `DensityResult`; `density_matrix_at_mu` evaluates a
  supplied chemical potential. `solver` returns `SCFResult` with the evaluated
  correction, density, observables and iteration history.
- EDIIS is the default solver and uses internal energy. SCF convergence measures
  the largest density-entry residual. Failures retain the last valid result.
- A common `tol` and `ErrorTolerances` policy control SCF, integration, filling and
  matrix-function accuracy. Fixed grids report unavailable integration errors as `None`.
- Integration `nk` now means the total point count, not points per axis. Fourier
  grid helpers take an explicit shape. Zero-temperature fixed-filling calculations
  use `FermiSimplex`; `UniformGrid` supports fixed mu only at zero temperature.
- Reported energies and entropy are per cell per physical orbital; filling is per cell.
  Entropy/free energy are opt-in; SCF evaluates them only at termination.
- Kwant conversion moves to `meanfi.interop.kwant` and supports finite systems,
  sparse blocks and callable values. Kwant is now an optional dependency.
- Requires Python 3.11–3.13, NumPy 2.0 or newer and SciPy 1.13 or newer.

### Removed

- The 1.x tuple/dictionary result contracts and `optimizer`/`optimizer_kwargs` interface.
- Standalone `meanfield`, `guess_tb` and `fermi_energy` helpers. Use Model methods
  and the chemical potential returned by the density or SCF calculation.
- The old profiling script, replaced by the benchmark runner.

## [1.1.0] - 2024-05-31

### Fixed
- Ensure `meanfi.params.rparams.tb_to_rparams` and `meanfi.params.rparams.rparams_to_tb` use minimal parametrisation of the tight-binding dictionary.

### Added
- Density matrix cost for the mean-field solver.
- Functionality `meanfi.kwant_helper.utils.tb_to_builder` to create `kwant` systems with the tight-binding dictionaries.
- Functionality `meanfi.kwant_helper.utils.tb_to_kfunc` to create k-dependent function from tight-binding format.

### Changed
- Rewrote `meanfi.kwant_helper.utils.builder_to_tb` to avoid potential bugs and remove the use of `copy`.

## [1.0.0] - 2024-05-09

- First release of _MeanFi_.
