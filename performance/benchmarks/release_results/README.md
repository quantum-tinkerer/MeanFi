# Release regression benchmark

Five bounded fixed-filling cases compare the recovered prior `PeriodicQuadrature`
implementation with `PeriodicGrid`. Each time is the median of three sequential
runs after one warmup, with BLAS/OpenMP limited to one thread. Imports, model
construction and independent reference evaluation are outside the timed region.
The matrix sizes are 16, 16, 16 (8 physical BdG orbitals), 16 and 8, respectively.
These small regressions test large implementation slowdowns; they do not replace
the historical 48-orbital performance study.

| Case | kT | Prior (s) | PeriodicGrid (s) | Ratio | Final points | Diagonalizations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| square_metal | 0.20 | 0.0462 | 0.0354 | 0.77 | 256 | 848 |
| gapped | 0.05 | 0.0279 | 0.0066 | 0.24 | 64 | 208 |
| cold_bdg | 0.01 | 0.0065 | 0.0083 | 1.28 | 16 | 160 |
| wire | 0.05 | 0.0289 | 0.0340 | 1.18 | 256 | 1020 |
| bounded_3d | 0.20 | 0.8881 | 0.8120 | 0.91 | 32768 | 102976 |

Density target: `1e-4` per requested entry. Charge-integration and filling-root
budgets: `2.5e-5` each. All five cases passed independent shifted, non-dyadic
reference comparisons at the returned chemical potential. Maximum density
discrepancy was below `1e-10`; physical filling error below `2.1e-5`. The two
reference resolutions agreed below `2e-14` in every requested density entry.
The harness independently assembles Fourier Hamiltonians and full covariance
matrices before selecting entries. This is an empirical check, not a rigorous
continuous-Brillouin-zone certificate. Full errors, reference changes, work and
environment metadata are in `baseline.json` and `periodic_grid.json`.

No order-of-magnitude slowdown was observed. The regression exposed an incorrect
normal-charge derivative normalization during development; the recorded release
run includes its correction. It also includes mandatory validation at irrational
per-axis shifts, which rejects nested-grid aliases. Ratios for these short calls
are sensitive to timing noise, and small constant-factor differences were not
tuned.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python performance/benchmarks/release_regression.py \
  --output build/release-regression.json --references
```

The historical baseline was run with `--checkout` pointing to recovered worktree
`a93c`, whose source is archived with the historical study. It uses its original
starting resolution of eight points per axis; the release starts at four. The
recorded counts include discovery, filling roots, density passes and validation.
The benchmark files are excluded from both wheel and source distribution.

## Further streamlining

The second pass compares commit `595a8f2` with the simplified code using the same
five cases and independent references. Runs were sequential, pinned to one CPU,
with one BLAS/OpenMP thread, three repetitions and one warmup per case. Python,
NumPy, timings and work counts are recorded in [streamlining.json](streamlining.json).

| Case | Before (s) | After (s) |
| --- | ---: | ---: |
| Square metal | 0.0511 | 0.0283 |
| Gapped system | 0.0120 | 0.0067 |
| Cold BdG | 0.0159 | 0.0083 |
| Multichannel wire | 0.0640 | 0.0332 |
| Bounded 3D | 0.8846 | 0.8824 |
| Sparse chain, fixed 128 points, AAA | 7.7332 | 3.2190 |

All five dense cases produced identical chemical potentials, filling and
reference discrepancies before and after the change. The reference assertions
pass; no measured slowdown appeared. Short-call timing ratios are sensitive to
CPU state and should not be interpreted as universal speedups.

The sparse case used three fresh processes per version on CPU 0, with one BLAS
thread. It used two identical nearest-neighbor chains (`h(0)=0`, `h(±1)=-I`),
`kT=0.15`, filling `0.7`, `filling_tol=0.01`, `mu_tol=1e-8`, and keys `0, ±1`.
`PeriodicGrid(nk=128)` selects AAA for this explicit sparse grid. Chemical
potential changed by `1.39e-5` and reported filling by `2.94e-6`; both solves met
the requested filling tolerance. Independent sparse reference tests passed at
tighter tolerances. This benchmark measures removing spectral setup work and
simplifying retained state; it does not establish integration convergence on a
prescribed mesh.

## Coordinate and result simplification

The third pass compares `5a25a53` with the shared-entry and sparse-coordinate
implementation. [coordinates.json](coordinates.json) records the original
three-run measurements with independent references and the reversed-order
fifteen-run follow-up. Both used CPU 0 and one BLAS/OpenMP thread. All five dense
median timings differ by less than 1.2%; physical values, work counts and
independent reference errors are identical. Initial short-call variation did not
persist in the follow-up.

Sparse reconstruction uses 2,000 orbitals with a nearest-neighbor onsite
interaction, 5,998 real SCF parameters and seeded random input. Each timed pass
reconstructs density, packs its parameters and assembles meanfield. The median
of three passes after one warmup fell from 9.19 ms to 0.59 ms; Linux peak process
RSS fell from 140.1 MiB to 79.0 MiB, including imports and setup. The resulting
correction vectors have identical SHA-256 hashes. This is a reconstruction
benchmark, not a full SCF or integration scaling claim.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0 python performance/benchmarks/scf_reconstruction.py \
  --output build/scf-reconstruction.json
```

Use `--checkout` to compare a previous checkout with the same harness. Benchmark
sources and evidence remain excluded from wheel and source distribution.

## API cleanup

The API pass compares `185cf9e` with the model-aware density implementation.
[api_cleanup.json](api_cleanup.json) records fifteen sequential repetitions after
warmup, pinned to CPU 0 with one BLAS/OpenMP thread. The five cases have identical
chemical potentials and work counts before and after; all independent reference
assertions pass. These bounded timings do not establish universal speedups.

| Case | Before (s) | After (s) |
| --- | ---: | ---: |
| square_metal | 0.04226 | 0.02792 |
| gapped | 0.00737 | 0.00679 |
| cold_bdg | 0.00843 | 0.00832 |
| wire | 0.03467 | 0.03588 |
| bounded_3d | 0.82832 | 0.82957 |

The largest increase was 3.5% in the wire case; the other cases were unchanged
or faster in this run. A separate dense Fourier-helper check used nine complex
16-by-16 blocks (keys `generate_tb_keys(1, 2)`, independent standard normal real
and imaginary parts, NumPy seed 81), 2,048 uniformly sampled points in
`[-pi, pi]^2`, and a 32-by-32 grid. Median callable time stayed at 1.46 ms;
FFT grid conversion decreased from 10.59 ms to 2.90 ms. Both used fifteen
repetitions after warmup. Dense callables retain their vectorized contraction;
sparse callables accumulate nonzero entries directly.

The sparse accuracy suite now checks both AAA and Ozaki at requested tight
accuracy, rather than depending on one scheme accidentally exceeding a loose
request. AAA also handles nearly degenerate spectra with a constant occupation
whose error is bounded by endpoint occupations over the full spectral interval.
This avoids an unnecessary ill-conditioned fit; the core regression verifies
cache reuse without eigensolves, and sparse references verify the density and
charge to `1e-8` with requests of `1e-9`.

## AAA fitting and reuse

[The AAA fitting report](aaa_fitting_and_reuse.md) compares the smaller fitting
grid, QR with a small SVD, and interval reuse against `56b8822`. It includes
N=100, 200, and 512 timings, scalar and matrix costs, accuracy checks, and the
tradeoff from occasional extra poles. The earlier
[normalization and fit-cost study](normalization_and_fit_cost.md) documents the
per-orbital quantities and the original tight-tolerance regression.

## Guidelines cleanup

[The shared-layout report](guidelines_sparse_layout.md) compares the guidelines
cleanup with `b15d729`. It records the N=200 fixed-filling timings, independent
dense-reference errors, the original 35 thermal cases, unchanged numerical work
counts, and the limits of the standalone-node timings. The cleanup also unifies
normal/BdG SCF and simplex solve setup and resolves density settings once.
The measured snapshot included independent energy and entropy targets; these
were removed in `bf412e6`. Their estimates are now diagnostics, and thermodynamics
uses the accepted density calculation. The recorded measurements describe the
earlier snapshot.
