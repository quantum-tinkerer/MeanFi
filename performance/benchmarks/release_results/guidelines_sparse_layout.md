# Shared sparse layouts

The guidelines cleanup prepares one immutable `SparseRationalLayout` per periodic
calculation. It owns selected-inverse patterns and the mapping to requested
entries. Each Hamiltonian/chemical-potential node retains its own numeric factors.
Band-energy and entropy targets are independent; the occupation fit is tightened
for band energy, and entropy still shares its poles and inverse diagonals.

Measured on 2026-09-15 against archived commit `b15d729`, on CPU 6 of an Intel Xeon
Gold 5418Y with one BLAS thread, Python 3.12.13, NumPy 2.4.6 and SciPy 1.17.1.
The N=200 comparison uses three cold and three warm repetitions of nine cases:
three sparsity patterns, two node temperatures, and three complete filling
searches. The original 35-case thermodynamic sweep uses one cold and one warm
measurement. All runs use `max_poles=256`; dense-reference computation is excluded
from the timings. Baseline and changed implementations were measured sequentially.

Compact records retain individual timings, error tolerances, work counts,
environment details, and source hashes:

- [Before: N=200](guidelines_sparse_before.json)
- [After: N=200](guidelines_sparse_after.json)
- [After: original thermodynamic cases](guidelines_sparse_thermal.json)

The changed version is labelled `guidelines-working-tree`, not a commit. Its
recorded source hashes identify the measured uncommitted code. Both after files
share rational-source SHA-256
`18dfd931157f67b7024dcb2d3af21b3ef5326475a9fd6a6968648c84f6327357`.
Historical benchmark evidence remains unchanged.

Complete filling searches at kT=0.02 use four prescribed k-points. Medians of the
three cold runs are:

| N=200 pattern | Before (s) | After (s) | Speedup | New fits | Factorizations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chain | 0.872 | 0.753 | 1.16x | 6 | 556 |
| Rectangle | 1.163 | 0.946 | 1.23x | 7 | 556 |
| Random graph | 2.114 | 1.749 | 1.21x | 7 | 554 |

Fit and factorization counts are unchanged; selected-inverse calls remain
624, 624 and 622. The remainder after subtracting measured scalar setup,
factorization and selected-inverse work falls from 0.228 to 0.111 s for the chain,
0.316 to 0.133 s for the rectangle, and 0.481 to 0.164 s for the random graph.
This remainder includes layout preparation, assembly and bookkeeping; it is not a
separate measurement of layout construction alone. Timings vary with machine
conditions and sparsity; these measurements establish improvements in the three
complete searches, not a universal speedup for individual nodes or other sizes.

The retained three-repeat sweep measures the standalone chain node at kT=0.02
as 39 ms before and 70 ms after. A follow-up with eight repetitions shows abrupt
timing changes within the unchanged after process: its first four cold calls take
73–84 ms and its final four take 38–40 ms, comparable with the settled baseline.
The before/after JSON files include these diagnostic records separately under
`node_timing_followup`. Startup/timing transients limit conclusions from the
short standalone measurements; all diagnostic accuracy checks pass.

All 54 before and 54 after N=200 measurements pass their original checks against
independent dense eigensystems. Density-entry and total filling errors must be
at most 1e-8; energy and entropy errors must be at most 1e-8 per cell per physical
orbital. Entropy is in units of k_B. Maximum observed errors after cleanup are:

| Quantity | Maximum absolute error |
| --- | ---: |
| Density entry | 1.52e-12 |
| Filling | 2.88e-11 |
| Band energy per orbital | 3.46e-13 |
| Entropy per orbital | 3.59e-11 |

The references use the same prescribed k-grid: they check the matrix function
and filling solve, not Brillouin-zone convergence. All 70 measurements of the
35 original thermal cases also pass. That suite retains full-trace energy and
entropy targets, including the 32-orbital 1e-12 regression: its charge, density,
energy and entropy errors are 2.49e-14, 1.78e-15, 7.46e-14 and 6.16e-15. No final
accuracy requirement was relaxed, and thermodynamics adds no factorizations or
inverse queries. Scalar fit certification remains a sampled check.

Focused regressions cover complex inverse orientations, repeated density
coordinates, omitted hole diagonals in charge-only calculations, read-only
shared patterns, unequal energy/entropy targets, and cached entropy fits reused
by nodes that did not request thermodynamics. The latter must report the missing
entropy request instead of using a partial diagonal layout.

Reproduce with `performance/benchmarks/rational_comparison.py`, using
`--large-matrices --only n200 --repeat 3 --max-poles 256 --cpu 6` for the timing
comparison and `--thermodynamics --repeat 1 --max-poles 256 --cpu 6` for the thermal
sweep. Supply `--checkout` and `--source-revision b15d729` for an archived baseline;
every invocation also requires an `--output` path.
