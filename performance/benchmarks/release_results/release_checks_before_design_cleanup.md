# Historical release-check report

Snapshot from `91fd1e7`, retained as historical evidence. Statements below
refer to earlier implementation stages; current instructions are in the
repository-root `RELEASE_CHECKS.md`.

# Simplification and release checks

MeanFi retains two integration families: native FermiSimplex for normal systems
at zero temperature, and periodic grids for normal/BdG calculations. Explicit
`nk` requests a total mesh size; omitting it enables supported accuracy control.

## Thermodynamics and simplification

Relative to `3443495`, runtime Python shrinks from **7,680 to 7,303 lines**
across the same **57 modules**, while adding complete thermodynamic reporting.

- `SCFResult` and its history report `internal_energy`, `free_energy`, and
  entropy in units of Boltzmann's constant. Helmholtz free energy is `U - kT*S`.
  `total_energy` is removed; the two public observable functions use the new names.
- Density results retain occupied band energy and full-state entropy even with
  selected entries. Dense calculations reuse their eigensystem; sparse calculations
  use one joint AAA fit and the same factorizations and selected inverse entries.
- EDIIS is the default for all supported SCF calculations. At finite temperature,
  it minimizes a history-based upper bound on free energy and uses the existing
  Anderson method to finish convergence, sharing one accepted-iteration budget.
- BdG pairing energy uses the conjugate anomalous density. Tests cover global
  pairing phase invariance and the Hamiltonian's energy derivative. BdG support
  now comes from the model, eliminating guess-dependent interaction truncation
  and the associated key tracking and zero-block adapters.
- Ozaki and `rational_scheme` are removed. One shorter AAA fitter handles density
  and entropy, including constant, asymmetric, and cold spectra. It checks the
  final rational expansion before sparse factorization and fails explicitly when
  the requested accuracy cannot be established.
- Periodic energy and entropy join nested and shifted mesh validation. BdG traces
  include every Nambu diagonal and the physical normalization. Zero-temperature
  flat half-filled modes retain their residual entropy.

See [migration notes](CHANGELOG.md), the
[executable API walkthrough](examples/api_walkthrough.py), and the
[package reference](docs/source/documentation/meanfi.md).

## Verification

| Environment | Required suite | Optional dependency skips |
| --- | ---: | ---: |
| Python 3.11.16, core | 457 passed | 14 |
| Python 3.12.13, core | 457 passed | 14 |
| Python 3.13.15, core | 457 passed | 14 |
| Python 3.12.13, MUMPS + Kwant | 475 passed | 0 |

All **34 slow numerical checks passed**. Required suites use warnings as errors,
coverage, and pytest-ruff. The final BdG support cleanup was additionally checked
against the superconducting physics reference after the full slow suite.

The walkthrough passes with and without optional extras. Strict Sphinx passes
with all five tutorials forced to execute. Installed core and sparse wheels pass
outside the checkout, including band-energy/entropy references. Each packaging
check builds a source distribution and then builds its wheel from that archive.

## Benchmarks

The [rational comparison](performance/benchmarks/release_results/rational_comparison.md)
records the AAA/Ozaki decision, raw errors, cold/warm timings, and factorization
counts. The old Ozaki charge stopping rule could hide density errors by trace
cancellation. Ozaki retains some density-only timing advantages, but its poles
cannot reliably meet the shared entropy targets. The replacement AAA fitter
eliminates the measured fitting timeouts and passes all 40 density/public-workflow
cases. Its thermal node sweep passes 34 of 35 cases; the absolute `1e-12` request
fails explicitly. Successful entropy/energy traces require no additional LU
factorizations or inverse queries after density evaluation.

[Dense regression data](performance/benchmarks/release_results/thermodynamics_dense.json)
compare `3443495` with this pass using seven repetitions, one CPU, and one
BLAS/OpenMP thread. Independent shifted-grid references pass in all five cases.

| Case | Before | After | Change |
| --- | ---: | ---: | ---: |
| Square metal | 49.70 ms | 50.41 ms | +1.4% |
| Gapped system | 7.36 ms | 6.85 ms | -7.0% |
| Cold BdG | 8.34 ms | 8.46 ms | +1.5% |
| Multichannel wire | 34.03 ms | 34.23 ms | +0.6% |
| Bounded 3D | 819.42 ms | 864.68 ms | +5.5% |

Meshes, refinement counts, and diagonalization counts are unchanged in these
cases. Small timing differences include measurement noise; these runs do not
establish universal performance guarantees. Historical evidence remains under
`performance/` and outside the runtime package and release archives.

## Supported scope and publication

No package was published. Archives remain at `0.0.dev0`; publication still needs
a release version and a target that accepts the pinned FermiSimplex Git dependency.
Adaptive RationalFOE remains unsupported. Zero-temperature BdG requires a prescribed
periodic mesh. Automatic finite-temperature sparse calls require an explicit
integration choice. Prescribed meshes do not claim continuous Brillouin-zone
accuracy; adaptive estimates and scalar approximation checks are empirical.
