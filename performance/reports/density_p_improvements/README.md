# Density p-cubature follow-up — 2026-09-22

This updates the initial [p-cubature experiment](../density_p_cubature/README.md).
The companion FermiSimplex revision is `af8b7f77870824af05f0d8ebc07899dd92a624da`. AdaptiveSimplex itself is
unchanged. MeanFi now pins this published companion revision.

## Implementation

1. **Linear cut correction.** Reuse the occupied barycentric moments M_bi and
   vertex spectra already computed for the charge simplex. Add
   `sum_bi (M_bi - f_b V/(d+1)) F_b(v_i)` to the fraction-weighted p-cubature.
   Here F includes the projector and real-space Fourier phase. Equivalently,
   integrate the vertex-linear interpolant over the occupied region exactly
   and p-integrate its nonlinear remainder with the occupation fraction.
   This removes the leading occupation correlation error, needs no additional
   diagonalizations, and preserves the onsite charge. Partial cells need an
   additional component contraction; bulk cells skip it. The correction is
   independent of degree, so it cancels from successive error differences.

2. **Existing error machinery.** Both h and p now share the same DensityGlobalError
   policy, moved into a common internal header. For complex correction vectors
   delta_sigma, the estimate is the maximum of
   `sqrt(sum ||delta_sigma||_infinity^2)`,
   `||sum delta_sigma||_infinity`, and a separately accumulated roundoff floor.
   The coherent term retains systematic errors; the statistical term prevents
   cancellations from hiding all local error. Selection reuses AdaptiveSimplex's
   RefinementQueue. No empirical rescaling factor was introduced.

3. **Selective parallel batches.** Batches of up to 16 selected cells use OpenMP
   only when at least 32 orbitals and more than one OpenMP thread are requested.
   Small matrices and the default one-thread setting retain serial promotion.
   Cells, work counters and exceptions are private to workers; queue and global
   error updates are serial and deterministic. Budgets are enforced before
   selecting each batch. One-cell batches bypass parallel scheduling.
   Only density_p.cpp is compiled with OpenMP, leaving charge compilation and
   threading unchanged. OpenMP is optional; disabling it builds the same serial
   implementation. MeanFi enables this via its existing `num_threads` setting.

## Matched before/after benchmark

Both binaries use the pinned AdaptiveSimplex sources, GCC 14.3 release builds,
and the same conda LAPACK provider on an Intel Xeon Gold 5418Y. Timings are the
median of seven measured runs after one warmup; BLAS/OpenMP are limited to one
thread. Baseline is the original p-adaptive backend, **not** h-refinement.
Every density solve starts from its charge-converged mesh. Requests contain
all 2-by-2 components at the onsite and positive nearest-axis lattice vectors.
Reference models and grid-doubling checks are as in the initial experiment.

At charge and density tolerance 1e-5:

| Model | Old / new spectra | Reduction | Old / new actual error | Old / new density time | Speedup |
|---|---:|---:|---:|---:|---:|
| bulk_2d | 1,138 / 958 | 16% | 5.59e-07 / 1.21e-06 | 3.70 / 1.64 ms | 2.26x |
| qwz_2d | 1,460 / 1,037 | 29% | 1.07e-06 / 4.62e-06 | 3.40 / 1.94 ms | 1.75x |
| bulk_3d | 17,172 / 9,846 | 43% | 3.7e-07 / 1.63e-06 | 30.32 / 17.66 ms | 1.72x |
| metal_2d | 2,561 / 1,637 | 36% | 1.14e-05 / 4.25e-06 | 4.90 / 4.32 ms | 1.13x |

The new estimator spends less effort on bulk accuracy: bulk actual errors grow,
but remain below the requested tolerance in these accepted runs. This is the
intended reduction of over-integration, not an equal-actual-error comparison.
The metallic result improves from the cut correction despite using fewer samples.

The moments-only ablation has exactly the same diagonalization counts and p
estimates as the old backend. Its metallic actual errors at tolerances 1e-3,
1e-4, 1e-5 are 4.55e-4, 4.75e-5, 4.28e-6, compared with 1.41e-3, 1.33e-4,
1.14e-5 without correction. The extra vertex contraction is small but not zero.

[Baseline data](baseline.json) · [Moments-only ablation](moments_only.json) ·
[Hybrid-estimator ablation](hybrid.json) · [Final data](final.json)

## Accuracy stress tests and remaining limits

Tested 36 cases: four models, three deterministic momentum shifts and complex
unitary orbital rotations, and tolerances 1e-3, 1e-4, 1e-5. References are analytic
for the separable metal and checked by grid doubling for the insulators.

- **32 converged**, all with actual error below both the requested tolerance and
  the reported estimate. Across all 36, the largest actual/estimate ratio is
  below 0.51. This is measured coverage, not a mathematical guarantee.
- **Four hit the degree-21 cap:** seed 2 of bulk_2d and qwz_2d, each at 1e-4 and
  1e-5. The original p backend also fails on exactly these four cases. They are
  explicitly reported as nonconverged; no mesh splitting fallback was added.
- Linear cut correction does not certify or eliminate nonlinear occupation
  correlation or charge-geometry error. An analytic affine-band test verifies
  the corrected cut-cell residual scales as O(h^3) in 1D, while remaining larger
  than the p estimate. A fixed charge mesh can therefore still have an error floor.

[Stress-test data](accuracy.json) · [Original-backend comparison](accuracy_baseline.json)

## Parallel experiment

Measured the final implementation with 1 versus 4 requested OpenMP threads,
using seven repetitions after warmup and the same already-resolved mesh:

| Orbitals | Serial time | Four-thread time | Speedup | New spectra in either run |
|---|---:|---:|---:|---:|
| 32 | 28.20 ms | 21.34 ms | 1.32x | 958 |
| 64 | 138.81 ms | 63.40 ms | 2.19x | 958 |

These are identical repeated blocks of the bulk 2D model, with all within-block
components requested. The 32-orbital threshold is a conservative choice based
on the tested sizes, not an optimized universal crossover. Small-matrix batching
was inconsistent and gave essentially no benefit for the metal, so it was removed
from the final small-matrix path. Larger-matrix measurements across runs ranged
from about 1.2x to 2.2x, so exact wall-time ratios should not be overinterpreted.

This environment's OpenBLAS shares an OpenMP runtime: requesting four OpenMP
threads also reports four BLAS threads. We recorded runtime settings and ran
additional controls holding the thread count at four while changing only the
prototype batch cap from 1 to 16. Those controls reduced the 32-orbital time
from 28.48 to 23.63 ms and the 64-orbital time from 140.64 to 103.09 ms.
Thus batching contributes a measurable gain beyond merely changing the thread
setting. The prototype's environment-variable batch override is not retained.

[Final thread data and runtime settings](threads.json) ·
[Prototype sweep](parallel_prototype.json) · [Same-thread-count controls](parallel_controls.json)

## Regression checks

- 130 FermiSimplex Python tests pass with OpenMP enabled and disabled.
- 163 MeanFi non-performance tests pass with warnings treated as errors;
  40 performance cases were deselected.
- Seven native test executables pass. These include polynomial moments through
  degree 21, nesting, the statistical/coherent error policy, and legacy charge
  and density integration checks.
- New analytic regressions cover affine projector components on cuts in 1D–3D,
  zero extra diagonalizations, residual cut-error order, parallel determinism,
  strict promotion budgets and Python callback exception propagation.
- Ruff, shell syntax and git whitespace checks pass.
- `pixi lock --check --dry-run` validates the updated immutable dependency pin.

## Reproduce

The normal dependency pin now refers to the published companion. The local
installer remains available for development. From this MeanFi checkout:

```bash
CXX="$PWD/.pixi/envs/latest/bin/x86_64-conda-linux-gnu-c++" tools/install_fermisimplex_local.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/latest/bin/python -m performance.benchmarks.density_p_cubature --methods p --models bulk_2d qwz_2d bulk_3d metal_2d --repeat 7
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/latest/bin/python -m performance.benchmarks.density_p_improvements accuracy --output performance/results/density_p_accuracy.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/latest/bin/python -m performance.benchmarks.density_p_improvements threads --output performance/results/density_p_threads.json
```

The stress/thread benchmark accepts `--backend DIRECTORY` for isolated before/after
packages; it explicitly bypasses the installed editable import hook. Historical
ablation binaries were built from baseline `813df5a`, the moments-only change,
and the shared-error-policy change. Numerical reference constructions and random
seeds are recorded in the benchmark source. No universal scaling exponent or
rigorous density-error certificate is claimed.
