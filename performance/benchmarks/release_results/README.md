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
