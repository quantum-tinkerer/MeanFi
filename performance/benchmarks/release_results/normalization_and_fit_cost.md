# Thermodynamic normalization, AAA regression, and coefficient costs

Measured on 2026-09-14 using the implementation of this change above base commit
`20c0e88`. Machine, dependency versions, CPU affinity, source hashes, individual
measurements, and errors are retained in the linked JSON files. Historical
benchmark evidence is unchanged.

## Normalization

Public internal, free, and band energies and entropy are per cell per physical
orbital. N normal orbitals divide physical totals by N; a 2N-dimensional BdG
Hamiltonian still uses N after removing Nambu doubling. Their integration errors
use the same units. Filling remains electrons per cell. Generic observable
contractions remain unnormalized traces.

## The original AAA failure is fixed

[The original 35-case thermodynamic suite](rational_aaa_fixed_thermodynamics.json)
now passes **35/35**, with three cold and three warm measurements per case. All
210 measurements pass density, charge, energy, and entropy targets, and entropy
and energy require no additional matrix factorization or selected-inverse query.
The suite retains its original **total-trace** tolerances, so normalization did
not make this comparison easier.

The failing case was a 32-orbital complex chain at kT=0.02 and mu=0.13, requesting
absolute 1e-12 accuracy. Its intermediate barycentric approximation stalled just
above the derived scalar targets near machine precision. The old stopping gate
prevented the final pole/residue refit from being attempted. The gate now allows
that refit once intermediate errors are within a factor of ten of the target;
final acceptance still requires the original tolerance on a separate denser grid.
The accepted fit uses 21 conjugate pole pairs. Measured absolute errors were:

| Quantity | Error |
| --- | ---: |
| Charge | 1.030e-13 |
| Selected density entries | 4.330e-15 |
| Band energy, total trace | 1.066e-14 |
| Entropy, total trace | 4.990e-14 |

Regression tests check the final scalar functions on an independent 50,003-point
grid and the actual sparse matrix results against dense diagonalization. The
independent scalar-grid test allows four machine epsilons of evaluation roundoff
(about 9e-16): Python 3.11/3.13 environments reached 1.155e-14 versus the derived
occupation target of 1.092e-14. Runtime certification and the original matrix-level
1e-12 regression keep their exact requested tolerances. Sampled scalar
certification is not a rigorous uniform bound between samples.

## N=100, 200, and 512: does matrix work dominate?

[The larger-matrix study](rational_large_matrices.json) contains 18 single-node
cases and nine public fixed-filling workflows; all **27/27** pass every measured
accuracy check. There are three repetitions of each cold/warm measurement, one
CPU core and one BLAS thread, with a 256-pole cap. The Hamiltonians have chain,
rectangular nearest-neighbor, or random-graph sparsity. The random graph has
approximately eight off-diagonal neighbors per row before duplicates. Matrices
are scaled to a common Gershgorin radius of 2.5 to separate geometry cost from
scalar-approximation difficulty. Density/charge targets are 1e-8. Single-node
energy/entropy targets are 1e-8 **per orbital**; public workflows use the normal
`tol=1e-8` policy and `filling_tol=1e-8`.

The tables show medians at kT=0.02. The JSON also includes single-node cases at
kT=0.2. Dense reference calculations are excluded from all reported timings.
“Build” measures actual new AAA coefficient construction. “Matrix” combines LU
factorization and selected inversion. Total time also includes object setup,
spectral bounds, cache checks, sparse assembly, and result bookkeeping.

### One k-point at fixed chemical potential

A warm node reuses the same already-certified scalar fit, but constructs and
factors new matrices. This measures the benefit when the existing reuse policy
applies; it is not a prediction of the gain for an entire filling search.

| N | Geometry | Cold total (ms) | Build (ms) | Matrix (ms) | Reused-fit total (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Chain | 128.9 | 104.2 | 15.1 | 28.6 |
| 100 | 2D grid | 76.4 | 60.6 | 9.6 | 16.8 |
| 100 | Random graph | 77.7 | 56.1 | 14.8 | 23.9 |
| 200 | Chain | 84.6 | 60.4 | 18.4 | 26.2 |
| 200 | 2D grid | 88.3 | 59.5 | 21.7 | 30.6 |
| 200 | Random graph | 124.4 | 63.2 | 51.4 | 63.4 |
| 512 | Chain | 180.6 | 71.8 | 98.7 | 110.3 |
| 512 | 2D grid | 153.2 | 54.1 | 86.8 | 100.8 |
| 512 | Random graph | 425.1 | 70.1 | 332.3 | 358.6 |

### Public chemical-potential searches

Each call finds filling 0.43N on a four-point prescribed periodic grid and returns
selected density entries, band energy, and entropy. All calls start with the
ordinary per-calculation cache. Repeating a public call does not carry fits from
the previous call, so the script's cold/warm labels here denote repeated calls,
not cross-call cache reuse. These timings use the cold set of three repetitions.

| N | Geometry | Total (s) | Build (s) | Matrix (s) | Build / total | New fits |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 100 | Chain | 2.47 | 2.08 | 0.20 | 84% | 40 |
| 100 | 2D grid | 2.76 | 2.27 | 0.26 | 82% | 40 |
| 100 | Random graph | 2.75 | 2.05 | 0.39 | 75% | 40 |
| 200 | Chain | 2.95 | 2.25 | 0.44 | 76% | 40 |
| 200 | 2D grid | 2.80 | 1.98 | 0.52 | 71% | 40 |
| 200 | Random graph | 3.88 | 2.19 | 1.21 | 56% | 40 |
| 512 | Chain | 4.44 | 2.10 | 1.90 | 47% | 40 |
| 512 | 2D grid | 4.64 | 2.13 | 1.90 | 46% | 40 |
| 512 | Random graph | 9.18 | 2.03 | 6.17 | 22% | 36 |

Coefficient construction remains the main cost at N=100 and N=200 in these
workflows. At N=512 it takes about 22% for the random graph and 46–47% for the
chain/grid. Matrix size alone does not determine the crossover: sparse LU
fill-in, selected entries, temperature, and tolerance all matter.

The current cache keeps one fit and reuses it only when the next spectral
interval is contained in the cached interval and passes accuracy validation.
These searches performed 36–40 new fits across eight or nine charge evaluations
plus final density evaluation. Changed k-points and chemical potentials caused
intervals to move outside that single cached interval. The repeated work is
new coefficient construction, not expensive cache checks: bounds and checks
accounted for roughly 0.03 seconds per search, versus about 2 seconds of builds.

There is a measured reason to improve reuse for repeated N=100–200 searches.
The next bounded experiment should test fitting over a shared spectral interval
for a set of k-points and nearby chemical potentials, while keeping final scalar
accuracy checks. A wider interval may need more poles, increasing matrix work;
its net benefit must be measured. This change leaves the runtime caching policy
unchanged. For larger matrices with substantial fill-in, prioritize matrix work
according to profiles of the actual model.

## Reproduction

From the repository root, with MUMPS available:

```bash
.pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
    --thermodynamics --repeat 3 --max-poles 256 --cpu 4 \
    --output /tmp/aaa-fixed-thermodynamics.json
.pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
    --large-matrices --repeat 3 --max-poles 256 --timeout 60 \
    --output /tmp/large-matrices.json
```

Use `--per-cell` when comparing public APIs in older archived checkouts. Raw
single-node thermodynamic checks retain total-trace units unless the large-matrix
profile is selected.
