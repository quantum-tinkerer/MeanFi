# Simplification and release checks

MeanFi has two integration families: `AdaptiveSimplex` uses native FermiSimplex
for normal T=0 calculations; `PeriodicGrid` supports dense finite-temperature
normal/BdG calculations. Explicit `nk` prescribes a total mesh size; omitting it
enables accuracy control. T=0 BdG requires an explicit mesh. Top-level `tol`
continues to control filling and SCF convergence on prescribed meshes.

## Current structure

The third pass reduces runtime Python from **7,960 to 7,732 lines** and
**59 to 58 modules**, relative to `5a25a53`. Runtime is now
22.4% smaller than `595a8f2`, before the second pass.
The SCF coordinate/space layer is 1,352 lines, down from 1,527.

- `DensityProblem` goes directly to a simplex or periodic evaluator. Both return
  `DensityResult`, which shares immutable `DensityEntries` with SCF and public
  callers. Removed the private evaluation class and result conversion copies.
- Both public density functions accept the same keys, coordinates, or interaction
  selection. Selected results retain their individual error estimates; unknown
  entries cannot be converted to complete matrix blocks.
- `ActiveSCFSpace` holds one complete compact-orbit or dense-basis parametrization.
  Removed optional-mode fields, duplicate constructors, stored unused rows and
  forwarding helpers. General spatial symmetry retains its dense allocation guard.
- Coordinate extraction indexes sparse matrices directly; reconstruction preserves
  the model's sparse storage through Hartree/Fock terms and BdG assembly. Sparse
  observables and structural zero blocks avoid full dense allocation.
- Empty coordinate layouts are normal values. Removed `allow_empty`, unreachable
  None guards and obsolete conversion helpers. Complete-layout checks no longer
  construct quadratic Python sets. Internal state/integration types are concrete.

See [migration notes](CHANGELOG.md) and the
[integration contracts](docs/source/documentation/algorithms/integration_families.md).

## Verification

| Environment | Full required suite | Optional dependency skips |
| --- | ---: | ---: |
| Python 3.11.16, core | 398 passed | 7 |
| Python 3.12.13, core | 399 passed | 7 |
| Python 3.13.15, core | 398 passed | 7 |
| Python 3.12.13, MUMPS + Kwant | 408 passed | 0 |

The final mixed dense/sparse input regression was added after the 3.11/3.13 full
runs; all five sparse/mixed workflow tests passed again on those versions.
All **37 optional slow physics checks passed**. Required suites use warnings as
errors, coverage and pytest-ruff. Focused tests cover sparse/dense SCF agreement,
20,000-orbital sparse reconstruction without dense conversion, empty spaces,
compact/general symmetry mappings, shared immutable entries, selection errors and
consistent public selection modes.

Ruff, formatting, pre-commit and whitespace checks passed. Strict Sphinx passed
with all five tutorial notebooks executed. Core and optional-sparse installed
wheel checks passed; an offline rebuild from the extracted source distribution
produced a byte-identical wheel. Wheel and sdist retain `0.0.dev0`, the exact
FermiSimplex Git dependency pin, and optional `meanfi[sparse]` MUMPS installation.

## Regression timings

CPU 0, one BLAS/OpenMP thread, one warmup and fifteen timed runs per dense case.
The initial three-run comparison showed timing variation; the follow-up reversed
version order and found median changes within 1.2% in all five dense workloads.
Chemical potentials, filling, integration errors, work counts and independent
reference discrepancies are identical before and after the changes.

| Case | Before | After |
| --- | ---: | ---: |
| square_metal | 0.02768 s | 0.02736 s |
| gapped | 0.00662 s | 0.00657 s |
| cold_bdg | 0.00832 s | 0.00828 s |
| wire | 0.03396 s | 0.03427 s |
| bounded_3d | 0.83754 s | 0.82876 s |

A separate 2,000-orbital sparse SCF reconstruction/update workload, with three
repetitions in fresh processes, fell from **9.19 ms to 0.59 ms**. Peak process RSS,
including imports and setup, fell from **140.1 MiB to 79.0 MiB**. The complete
correction vectors have identical SHA-256 hashes. This measures reconstruction
and mean-field assembly, not an integrated SCF solve.

See the [benchmark report](performance/benchmarks/release_results/README.md) and
[raw evidence](performance/benchmarks/release_results/coordinates.json) for
protocols, initial timings and follow-up measurements. These bounded workloads
do not establish universal speedups. Historical experiments remain outside
release archives.

## Publication and supported scope

No package was published. Before publication, choose a release version and confirm
that the target accepts the pinned Git dependency. Adaptive RationalFOE and
periodic finite-temperature energy/free-energy estimates remain unsupported.
Automatic finite-temperature sparse calls require an explicit integration choice.
Density estimates apply at the returned chemical potential; they do not certify
its uncertainty or continuous Brillouin-zone accuracy.
