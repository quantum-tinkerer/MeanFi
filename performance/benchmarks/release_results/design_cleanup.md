# Design cleanup comparison

Compared commit `91fd1e7` with the runtime on `codex/clarify-density-scf-design`
on 2026-09-15. The raw reports identify the exact runtime files by SHA-256:
[before](design_cleanup_before.json), [after](design_cleanup_after.json).

## Scope and conditions

These are narrow node benchmarks, not complete SCF timings or a claim about all
matrix geometries. They use sparse tridiagonal Hermitian matrices of size 100 and
200, `kT=0.1`, `mu=0.05`, every diagonal entry and one distant off-diagonal entry.
The density target is `1e-7`; the total-charge target is `1e-6`. Independent dense
eigensystems supply references outside the timed region.

Each case runs one warm-up and seven measured repetitions, each with a fresh
node and scalar-fit cache. The two revisions run sequentially, pinned to one
logical CPU with one BLAS thread. Python 3.12.13, NumPy 2.4.6, SciPy 1.17.1.

## Timings

Times below are medians in milliseconds, before → after. The first column of
timings covers the initial scalar fit (including entropy fitting when requested).
Evaluation covers charge, selected density and optional thermodynamics, including
cached-fit validation; it is not exclusively matrix factorization time.

| N | Thermodynamics | Initial fit, ms | Evaluation, ms | Poles |
|---|---|---|---|---|
| 100 | no | 7.47 → 6.44 | 9.24 → 8.39 | 6 → 6 |
| 100 | yes | 15.62 → 10.19 | 15.17 → 8.67 | 10 → 6 |
| 200 | no | 7.29 → 6.67 | 13.53 → 13.63 | 6 → 6 |
| 200 | yes | 15.95 → 10.02 | 23.76 → 13.13 | 10 → 6 |

For N=200 with thermodynamics, the sum of the two medians falls from 39.71 ms to
23.15 ms (42%). Without thermodynamics, N=200 evaluation time is effectively
unchanged. Entropy no longer drives AAA pole selection or acceptance, explaining
the reduction from ten poles to six in the thermodynamic cases.

## Accuracy and tradeoff

Every case passes the original density and charge targets. The largest errors
after cleanup are `9.82e-11` in selected density entries, `1.82e-9` in total charge,
and `1.99e-12` in energy per orbital. The suite separately retains strict density
and energy checks at tighter tolerances.

Entropy accuracy is deliberately no longer a fitting requirement. At N=200,
entropy error per orbital increases from `8.76e-12` to `1.70e-4`. The reported
sampled scalar entropy error is `1.24e-2`, which conservatively covers the observed
trace error here. Sampled errors are empirical estimates, not rigorous interval
certificates. Sparse free-energy comparisons must account for this diagnostic
multiplied by `kT`, as well as other numerical errors. EDIIS uses internal energy
only, so this change does not introduce entropy into its objective.

## Size and validation

Runtime Python code, excluding tests, falls from 6,573 lines in 57 modules to
6,119 lines in 55 modules: 454 fewer lines (6.9%). Class count falls from 57 to 54.
The main reductions remove duplicate interaction bookkeeping, SCF state/callback
layers, symmetry-basis construction and density configuration wrappers.

Validation of this runtime:

- Sparse/Kwant suite including slow regressions: 549 passed, 2 skipped, 96% coverage.
- Python 3.11, 3.12 and 3.13 core suites: each 484 passed, 29 skipped and 34 slow
  cases deselected; slow cases are covered by the full suite above.
- Strict Sphinx build with forced notebook execution, public API walkthrough,
  installed core/sparse wheels and clean dependency installation all pass.

See [release checks](../../../RELEASE_CHECKS.md) for the repeatable validation
commands. Protected historical periodic benchmarks remain unchanged.

## Reproduction

Prepare the baseline once:

```sh
mkdir -p /tmp/meanfi-before-91fd1e7
git archive 91fd1e7 | tar -x -C /tmp/meanfi-before-91fd1e7
```

Run the following sequentially from the updated checkout:

```sh
pixi run --locked -e test-sparse python performance/benchmarks/design_cleanup.py --checkout /tmp/meanfi-before-91fd1e7 --label 91fd1e7 --output /tmp/design_cleanup_before.json
pixi run --locked -e test-sparse python performance/benchmarks/design_cleanup.py --label codex/clarify-density-scf-design --output /tmp/design_cleanup_after.json
```
