# Simplification and release checks

MeanFi now has two integration families. Normal zero-temperature calculations
use `AdaptiveSimplex` (FermiSimplex); dense normal and superconducting calculations
at positive temperature default to `PeriodicGrid` with direct diagonalization.
Zero-temperature BdG requires explicit `PeriodicGrid(nk=...)`.

Both methods select prescribed-size operation from explicit `nk`, and accuracy
control otherwise. `nk` is a total mesh-size request. It cannot be combined with
explicit integration targets. Top-level `tol` still controls filling and SCF
convergence in prescribed calculations. See [migration notes](CHANGELOG.md) and
[integration contracts](docs/source/documentation/algorithms/integration_families.md).

The integration layer fell from 3,974 to 1,845 lines (54% smaller). The stateful quadrature
adapters, competing public methods, dependency, unused tolerance derivations and
external simplex build bootstrap were removed. Periodic density streams batches,
retains bounded normal eigenvalues for filling searches, and validates adaptive
results on a deterministically shifted grid. Requested nodes, actual nodes and
cumulative diagonalizations are separate fields in `result.statistics`.

## Verification

- Required pytest task, with warnings treated as errors, coverage and pytest-ruff:
  **361 passed**, 37 optional slow checks deselected (CPython 3.12.13).
- All **37 optional slow checks passed**, including sparse RationalFOE and
  interacting normal/BdG workflows. The graphene diagnostic uses an SCF target
  consistent with its integration tolerance; its three seeds and ordered-state
  assertion are preserved. Its former target failed even with the original adapter.
- Ruff check/format, all pre-commit hooks (including codespell), and whitespace
  checks passed for current code and documentation. Historical benchmark evidence
  is excluded from formatting hooks; all 23 archived checksums were verified.
- Strict Sphinx build passed, including execution of the five tutorials.
- Pixi lock consistency check passed.
- Wheel and source distribution built and inspected. The wheel excludes tests;
  both archives exclude historical experiments and obsolete quadrature modules.
- Installed-wheel smoke tests passed with `stateful_quadrature` unavailable:
  prescribed/adaptive periodic and simplex density, prescribed sparse RationalFOE,
  and normal/BdG SCF.
- Rebuilding the extracted source distribution with package-index access disabled
  produced a byte-identical wheel.

Local Python environments all use 3.12; this is not a claim of independent 3.11
or 3.13 test coverage. CI's declared test environments remain enabled.

## Representative regression timings

Medians of three sequential runs after warmup, with one BLAS/OpenMP thread.

| Case | Previous periodic method | PeriodicGrid |
| --- | ---: | ---: |
| Square metal | 0.0462 s | 0.0354 s |
| Gapped system | 0.0279 s | 0.0066 s |
| Cold BdG | 0.0065 s | 0.0083 s |
| Multichannel wire | 0.0289 s | 0.0340 s |
| Bounded 3D | 0.8881 s | 0.8120 s |

All five passed independent numerical references. No order-of-magnitude
regression was observed. See the [benchmark report](performance/benchmarks/release_results/README.md)
for tolerances, final mesh sizes, diagonalizations, reference errors and provenance.
The [historical study](performance/benchmarks/periodic_study/REPORT.md) and compact
evidence are retained for development, outside release archives.

## Scope and publication

Adaptive RationalFOE is unsupported; explicit prescribed sparse RationalFOE
remains supported. Automatic finite-temperature sparse calls give migration
guidance rather than silently allocating dense matrices. Periodic integration
does not supply finite-temperature energy/free-energy estimates. Density error
estimates apply at the returned chemical potential; they do not certify its
uncertainty or continuous Brillouin-zone accuracy.

Artifacts retain development version `0.0.dev0` and the existing exact FermiSimplex
Git dependency pin. Before publication, select a release version and confirm that
the distribution target accepts that dependency form. No package was published and no changes were pushed.
