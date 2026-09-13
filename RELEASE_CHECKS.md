# Simplification and release checks

MeanFi has two integration families: `AdaptiveSimplex` defaults to native
FermiSimplex for normal T=0 calculations; `PeriodicGrid` defaults to direct global
refinement for dense finite-temperature normal/BdG calculations. T=0 BdG requires
explicit `PeriodicGrid(nk=...)`. For both methods, explicit `nk` requests a total
mesh size; omitting it enables accuracy control. Top-level `tol` still controls
filling and SCF convergence on prescribed meshes.

## Code structure

The second simplification pass reduces runtime Python from **9,964 to 7,960 lines**
(20% smaller), and **65 to 59 modules**, relative to commit `595a8f2`. The density
subsystem alone falls from 5,310 to 3,632 lines. The integration layer is now 1,480
lines, compared with 3,974 before the original integration-family cleanup.

- One normalized `DensityProblem` goes directly through `evaluate_density` to
  simplex or periodic evaluation. Removed the plan object, forwarding modules,
  repeated key retargeting and duplicate raw statistics classes.
- SCF reuses normalized density inputs and the already-computed output state.
  Removed its duplicate result container and unused callback bookkeeping.
- Selected simplex values stay coordinate vectors. They no longer pass through
  full tight-binding matrices. Finite systems use one direct T=0 calculation.
- Removed the unused generic rational block solver, derivative controls, stale
  chemical-potential caches, eager MUMPS import and broad export shims.
- AAA/Ozaki retain bounded per-point state. Gershgorin bounds replace unnecessary
  spectral eigensolves. Sparse shape, symmetry and projection checks compare
  stored entries without dense allocation.
- MUMPS is an optional `meanfi[sparse]` dependency. Core installations support
  FermiSimplex and direct periodic workflows without it. Removed unused build
  extras and placeholder environments; CI checks actual Python versions and
  installed wheels.

See [migration notes](CHANGELOG.md) and the
[integration contracts](docs/source/documentation/algorithms/integration_families.md).

## Verification

| Environment | Required checks | Optional dependency skips |
| --- | ---: | ---: |
| Python 3.11.16, core | 350 passed | 7 |
| Python 3.12.13, core | 350 passed | 7 |
| Python 3.13.15, core | 350 passed | 7 |
| Python 3.12.13, MUMPS + Kwant | 359 passed | 0 |

All **37 optional slow physics checks passed** (36 plus the three-seed graphene
diagnostic). Required suites used warnings as errors, coverage and pytest-ruff.
Focused tests also cover optional dependency errors, selected simplex results
without matrix assembly, sparse constant tails, and validation of a 100,000-orbital
sparse system with dense conversion explicitly forbidden.

Ruff, formatting, pre-commit hooks, whitespace and lock consistency checks passed.
Strict Sphinx passed with all five tutorial notebooks executed. The wheel and
source distribution were built and inspected; installed-wheel checks passed both
without MUMPS/stateful-quadrature and with the explicit sparse extra. An offline
rebuild from the extracted source distribution produced a byte-identical wheel.

## Regression timings

Medians of three controlled sequential runs against the previously pushed branch.

| Case | Before | After |
| --- | ---: | ---: |
| Square metal | 0.0511 s | 0.0283 s |
| Gapped system | 0.0120 s | 0.0067 s |
| Cold BdG | 0.0159 s | 0.0083 s |
| Multichannel wire | 0.0640 s | 0.0332 s |
| Bounded 3D | 0.8846 s | 0.8824 s |
| Sparse chain, AAA | 7.7332 s | 3.2190 s |

The five dense cases retain identical chemical potentials, filling and independent
reference errors. Sparse changes remain within the requested tolerances; tighter
independent sparse checks passed. No measured regression appeared. These bounded
workloads do not establish universal speedups. See the
[benchmark report](performance/benchmarks/release_results/README.md) for protocols,
limits and compact evidence. Historical experiments remain outside release archives.

## Publication and supported scope

Artifacts retain `0.0.dev0` and the exact FermiSimplex Git dependency pin. Before
publication, choose a release version and confirm that the target accepts that
dependency form. No package was published.

Adaptive RationalFOE and periodic finite-temperature energy/free-energy estimates
remain unsupported. Automatic finite-temperature sparse calls give explicit
migration guidance. Density estimates apply at the returned chemical potential;
they do not certify its uncertainty or continuous Brillouin-zone accuracy.
