# Simplification and release checks

MeanFi has two integration families: native FermiSimplex for normal zero-temperature
calculations, and periodic grids for normal/BdG density calculations. Explicit
`nk` requests a total mesh size; omitting it enables supported accuracy control.

## Current structure and API

This pass reduces runtime Python from **7,732 to 7,680 lines** and **58 to 57
modules**, relative to `185cf9e`, while completing the API review fixes.

- EDIIS is the default for normal zero-temperature simplex SCF. Other workflows
  use Anderson with explicit scaling, history and Armijo search. The reference,
  restart and symmetry examples that failed previously now converge by default.
- Public density functions live in `density/api.py`; package initialization only
  exports the API. A Model supplies filling, temperature and coordinate selection,
  including BdG. The duplicate BdG density adapter is removed.
- Model inputs are validated, owned and read-only. The two Hamiltonian helpers
  work for normal and BdG models. Reference subtraction uses `DensityResult`.
- Results convert with `to_tb(sparse=False)`; selected unknown entries stay
  unknown. `meanfield` accepts selected results. Removed legacy aliases,
  redundant package exports and secondary SCF stopping controls.
- Fourier grids use explicit `shape` tuples and preserve all odd/even/rectangular
  modes. Dense callables retain vectorized evaluation; sparse callables add only
  stored entries. Kwant conversions avoid dense unit-cell intermediates.
- `PeriodicGrid.dtype` names the actual complex precision. Explicit and implicit
  sparse RationalFOE default to AAA; nearly constant spectra use an endpoint-bounded
  constant occupation instead of an unstable rational fit.
- Density numerical failures use `ConvergenceError`. SCF exceptions carry the
  last valid result, or None when the initial density evaluation fails.

See [migration notes](CHANGELOG.md), the
[executable API walkthrough](examples/api_walkthrough.py), and the
[package reference](docs/source/documentation/meanfi.md).

## Verification

| Environment | Full required suite | Optional dependency skips |
| --- | ---: | ---: |
| Python 3.11.16, core | 420 passed | 7 |
| Python 3.12.13, core | 420 passed | 7 |
| Python 3.13.15, core | 420 passed | 7 |
| Python 3.12.13, MUMPS + Kwant | 431 passed | 0 |

All **40 slow numerical checks passed**, including tight AAA and Ozaki comparisons.
Required suites use warnings as errors, coverage and pytest-ruff. Final Fourier
vectorization and mixed-storage coverage were checked again with the complete
transform and public API contract test files after the full suite runs.

The walkthrough passes with both optional extras. Strict Sphinx passes with all
five tutorial notebooks executed. Ruff, formatting, pre-commit and whitespace
checks pass. Installed core and sparse wheels pass; the build also creates a
wheel from the source distribution. The wheel smoke check uses nondegenerate
coupled bands so it exercises MUMPS selected inverses, with thread limits set by
the check itself.

## Regression timings

[Raw results](performance/benchmarks/release_results/api_cleanup.json) and the
[benchmark report](performance/benchmarks/release_results/README.md) compare
`185cf9e` with this API cleanup. Five cases use one CPU, one BLAS/OpenMP thread,
one warmup and fifteen repetitions. Chemical potentials and work counts are
identical; all independent reference assertions pass. The largest timing increase
is 3.5%; the other cases are unchanged or faster in this run. These small cases
do not establish universal performance guarantees.

The separate dense Fourier check preserves callable speed (1.46 ms) and reduces
32-by-32 grid conversion from 10.59 ms to 2.90 ms. The report records its matrix
sizes, seed, point count and timings. Historical experiments remain unchanged and
outside release archives.

## Publication and supported scope

No package was published. Before publication, choose a release version and confirm
that the target accepts the pinned FermiSimplex Git dependency. Wheel and source
archives remain at `0.0.dev0`.

EDIIS remains restricted to normal zero-temperature simplex calculations; periodic
finite-temperature energy/free-energy estimates and adaptive RationalFOE remain
unsupported. Zero-temperature BdG requires a prescribed periodic mesh. Automatic
finite-temperature sparse calls require an explicit integration choice. Density
estimates apply at the returned chemical potential and do not certify its
uncertainty or continuous Brillouin-zone accuracy.
