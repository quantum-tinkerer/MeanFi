# Final entropy evaluation

Compared `eb0a56d` with the `codex/on-demand-thermodynamics` implementation on
2026-09-16. JSON reports include source fingerprints, versions and individual
samples. Run `performance/benchmarks/final_entropy.py` with `--checkout`, `--label`
and `--output` to repeat. Both checkouts use the same single-thread environment,
with affinity to one CPU; runs are sequential and discard one warmup.

The finite chain has nearest-neighbor hopping -1, a diagonal ramp from -0.4 to
0.4, nearest-neighbor interaction 0.12, filling 0.43 N and kT=0.2. Linear mixing
(alpha=0.7) converges in 10 iterations at tol=1e-6. Each reported time is the
median of three samples. This small benchmark measures overhead, not general
sparse scaling or performance for arbitrary sparsity patterns.

| Orbitals | Backend | Previous default (s) | Final-only default (s) | Entropy disabled (s) | Final pass (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 100 | Dense | 0.0440 | 0.0469 | 0.0443 | 0.0021 |
| 100 | AAA | 0.6582 | 0.6833 | 0.3780 | 0.0138 |
| 200 | Dense | 0.1022 | 0.1100 | 0.1054 | 0.0052 |
| 200 | AAA | 0.5625 | 0.5551 | 0.5474 | 0.0102 |

The earlier 100-orbital AAA samples varied from 0.413 to 0.705 seconds. Runs
at this size also changed materially between repeats of the complete benchmark;
neither a speedup nor the apparent disabled-entropy saving is reliable at N=100.
Differences of a few milliseconds overlap run-to-run variability. At 200 orbitals,
sparse runtime is effectively unchanged; dense evaluation pays roughly the
additional final diagonalization.
The repeatable structural improvement is 11 -> 1 entropy fits for AAA, or zero
with `compute_free_energy=False`. SCF iterations do no entropy work.

The returned density and internal energy agree between default and disabled
entropy, and with the previous implementation. Independent dense references give
maximum density errors of 6.84e-12 (N=100) and 7.05e-12 (N=200), well below the
1e-7 verification threshold. Dense density/entropy agrees to machine precision.

Final AAA entropy errors are 3.50e-4 and 3.48e-4 per orbital; the corresponding
sampled fit estimates are 0.00947 and 0.00952. The earlier entropy errors were
2.56e-5 and 2.43e-5. This change follows from rebuilding a fixed-mu density fit:
it uses the matrix-function target without the tighter charge-trace constraint
or interval-cache history of the previous filling search. Density-selected poles
do not promise density-level entropy accuracy. There is no new entropy target;
the final calculation reports its own diagnostic. At kT=0.2 the measured entropy
contributions to free-energy error are about 7e-5 per orbital. Prescribed periodic
meshes report no total entropy-error estimate, since integration error is unknown.

Regression tests additionally check success and nonconvergence, exact input
Hamiltonian reuse, skipped entropy, partial results on final entropy failure,
and serialization without a hidden model.

Validation of this change: core suites pass on Python 3.11, 3.12 and 3.13 (526
tests each); the sparse/Kwant suite passes 563 tests. All 34 extended numerical
cases pass, including the one updated finite-input test rerun after removing its
unused nk. Strict documentation with executed notebooks, the public walkthrough,
core and sparse installed wheels, a clean dependency installation and all
pre-commit checks pass. The protected periodic-study records are unchanged.
