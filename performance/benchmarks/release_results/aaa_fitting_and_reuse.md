# Faster AAA fitting and reuse

Measured on 2026-09-15 against commit `56b8822`, using the same Python environment,
CPU core, one BLAS thread, and three repetitions of each cold/warm case. The
baseline and new implementation were run sequentially. The raw records retain
source hashes, CPU and dependency details, individual timings, errors, and work
counts:

- [Before: 27 larger-matrix cases](rational_reuse_before.json)
- [After: the same 27 cases](rational_reuse_after.json)
- [After: the original 35 thermodynamic cases](rational_reuse_thermodynamics.json)

See [the previous study](normalization_and_fit_cost.md) for model construction,
normalization, and the original tight-tolerance regression. These benchmark
artifacts are excluded from the package distributions.

## Implementation

1. Start AAA on a smaller scalar grid, doubling its resolution only if the fit
   fails. Keep the full dense validation grid throughout. The final resolution
   and support budget remain available for difficult cases.
2. Compute `L = QR` and take the smallest right singular vector of the small
   triangular `R`. This preserves the least-squares objective without forming
   `L.T @ L` and squaring its condition number.
3. Keep one scalar fit shared within the calculation. After a nearby spectral
   interval extends beyond its bounds, add 20% of the new interval width at each
   end. Validate each reuse against its current accuracy requirements. If the
   wider fit exceeds the budget, retry the actual spectrum.

The first fit uses the actual spectrum, avoiding an interval-padding cost for
one-off nodes. Reuse checks temperature, requested functions, and pole budget;
charge-only fits are replaced when entropy is needed. There are no new public
settings, dependencies, or cache layers. Matrix factorizations remain local to
each Hamiltonian and chemical potential. Density, energy, and entropy continue
to share the same sparse work.

## Complete fixed-filling solves

Medians of three cold runs, in seconds, at kT=0.02. Each public call starts with
an empty scalar-fit cache and evaluates four k-points. All nine workflows meet
the existing density, filling, energy, and entropy checks against independent
dense references.

| Physical orbitals | Pattern | Before (s) | After (s) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 100 | Chain | 2.392 | 0.528 | 4.53x |
| 100 | Rectangle | 2.799 | 0.660 | 4.24x |
| 100 | Random graph | 2.810 | 0.887 | 3.17x |
| 200 | Chain | 2.992 | 0.864 | 3.46x |
| 200 | Rectangle | 2.894 | 1.061 | 2.73x |
| 200 | Random graph | 3.915 | 1.984 | 1.97x |
| 512 | Chain | 4.515 | 2.790 | 1.62x |
| 512 | Rectangle | 4.699 | 2.975 | 1.58x |
| 512 | Random graph | 9.110 | 7.800 | 1.17x |

New coefficient builds fall from 40 to 6–7 per solve (36 to 7 for the 512-orbital
random graph). For the 200-orbital chain, coefficient construction falls from
2.278 s to 0.053 s.
Including spectral bounds and validation of cache hits, total scalar setup is
2.309 s before and 0.165 s after.

## Cost of one new fit

For 200-orbital nodes at kT=0.02, these cold runs construct one new joint
occupation/entropy fit. The total includes sparse matrix work and setup.

| Pattern | Fit before (ms) | Fit after (ms) | Total before (ms) | Total after (ms) |
| --- | ---: | ---: | ---: | ---: |
| Chain | 62.3 | 15.2 | 88.0 | 40.4 |
| Rectangle | 62.4 | 21.0 | 94.7 | 49.3 |
| Random graph | 62.7 | 17.9 | 124.5 | 79.8 |

Broader intervals and the smaller fitting grid can choose slightly more poles.
The complete filling solves use 5.6%–12.6% more factorizations;
their reduced scalar setup cost still improves all nine total runtimes. For
example, the 512-orbital rectangle node at kT=0.02 accepts
16 pole pairs instead of 15.
With the scalar fit already cached, that node takes
107.8 ms versus 101.3 ms before.
These changes do not promise a speedup for every already-cached node. Sparse
fill-in, temperature, and tolerances determine the balance of costs.

## Validation

All 27 larger-matrix cases and all 35 original thermodynamic cases pass every
cold/warm measurement: 372 successful, accurate measurements of the new code.
The original 32-orbital regression retains its absolute 1e-12 total-trace
requirements. No final accuracy target or validation-grid resolution was relaxed.
Scalar validation remains sampled, rather than a rigorous bound between points.

New regression tests cover nearby normal and BdG intervals, temperature changes,
tighter tolerances, adding entropy, restricted pole budgets, expanded-fit
failure, and a sharp kT=1e-5 transition requiring finer fitting resolution.
Required suites pass with 471 tests on each of Python 3.11, 3.12, and 3.13, and
494 tests with the optional sparse backend. All 34 slower numerical tests pass.
The original 40-case density sweep also passes. Executed tutorial documentation,
installed core/sparse wheel checks, and both public API walkthroughs pass.
