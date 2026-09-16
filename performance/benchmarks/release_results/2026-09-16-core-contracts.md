# Core contracts review — 2026-09-16

Compared with `912d4e1` on `codex/clarify-core-contracts`. The after measurements
use the implementation in the commit containing this report.

## Implemented and reviewed

- `Model.mean_field(density)` covers normal, BdG, and reference-subtracted models.
- Returned SCF corrections reproduce their returned densities, including failures.
- All accuracy targets live in `ErrorTolerances`, accepted through `tol` alongside
  numeric shorthand. Mesh settings contain no accuracy overrides.
- Real interaction coefficients and valid corrections are enforced at public boundaries.
  Normal and BdG share omitted-zero-block semantics; energy contractions use the
  same canonical correction representation.
- Unsupported methods and nonfinite result entries fail early.
- Spatial transformations must have an invertible integer lattice map and a unitary
  combined Fourier symbol. A reproduced roundoff bug that made a global phase
  remove all normal-density variables is covered by an analytic regression.
- Accepted chemical-potential guesses avoid bracket work. Tall symmetry systems
  use economical SVD; wide systems retain all null directions.

The core package has 110 fewer lines overall than the parent revision, excluding
tests, examples, documentation and benchmarks. No dependencies were added.

## Numerical and timing evidence

Environment: Intel Xeon Gold 5418Y, Python 3.12.13, NumPy 2.4.6, SciPy 1.17.1,
one BLAS/OpenMP thread. Calls run sequentially; timing is the median of three
runs after one warm-up. Model construction and reference diagonalization are
outside the timed region. Entropy is disabled to isolate the density calculation.

The benchmark uses a half-filled chain at `kT=0.2` with optional nearest-neighbor
pairing, one prescribed momentum, selected density entries, and `tol=1e-6`.
The supplied starting chemical potential is already acceptable in these cases.
This specifically measures the shortcut, not a general SCF speedup.

| Physical orbitals | State | Backend | Before (ms) | After (ms) | Worst density-entry error vs dense |
| --- | --- | --- | ---: | ---: | ---: |
| 100 | Normal | Dense | 2.71 | 4.84 | 0.00e+00 |
| 100 | Normal | AAA | 20.79 | 22.45 | 4.24e-12 |
| 100 | BdG | Dense | 29.81 | 15.88 | 0.00e+00 |
| 100 | BdG | AAA | 48.70 | 19.57 | 2.04e-11 |
| 200 | Normal | Dense | 8.09 | 7.84 | 0.00e+00 |
| 200 | Normal | AAA | 27.59 | 18.08 | 4.24e-12 |
| 200 | BdG | Dense | 144.88 | 80.36 | 0.00e+00 |
| 200 | BdG | AAA | 107.22 | 44.00 | 3.21e-13 |

Charge evaluations drop from 3 to 1 in every case. Normal dense spectra were
already reused, so that case has little timing change. BdG and AAA remove two
matrix evaluations. All density errors remain below `2.1e-11`; the largest final
filling error is below `8e-10`.

`test_partial_result_correction_reproduces_density_and_energy` checks unconverged
normal/BdG, dense/sparse, and reference/nonreference states against independent
diagonalization with absolute density and energy error below `1e-12`.
`test_nullspace_preserves_tall_and_wide_null_directions` uses an analytically known
nullspace and requires projector and constraint-residual errors below `1e-13`.

Reproduce the timing study from the repository root (prepare a parent checkout
first when comparing revisions):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run -e test-sparse python performance/benchmarks/core_contracts.py --output /tmp/contracts-current.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run -e test-sparse python performance/benchmarks/core_contracts.py --checkout /path/to/parent-checkout --output /tmp/contracts-parent.json
```

## Validation

- Sparse/Kwant standard suite: 626 passed, 2 skipped.
- Core suite on each of Python 3.11, 3.12 and 3.13: 589 passed, 35 skipped.
- Extended numerical suite: 34 passed.
- Strict Sphinx build from scratch with forced execution of all tutorials: passed.
- Public API walkthrough including sparse and Kwant: passed.
- Source/wheel builds, installed dense/sparse wheels, and clean dependency install: passed.
- Repository-wide formatting, spelling, and static checks: passed.

## Remaining limits

- Periodic AAA requires a prescribed mesh; its scalar matrix-function estimate
  does not establish momentum-integration accuracy.
- Coarse/fine integration estimates can miss shared mesh aliases. `initial_nk`
  lets callers start with a resolution suitable for their Hamiltonian.
- General spatial symmetry reduction still constructs a dense constraint basis;
  the economical SVD reduces workspace but does not remove that scaling limit.
- User-supplied symmetries must be appropriate to the model; validation establishes
  a valid transformation, not that the Hamiltonian has the requested symmetry.
- Entropy has no independent target. It is final postprocessing, and its diagnostic
  can be unavailable. EDIIS continues to use internal energy exclusively.

The historical `performance/benchmarks/periodic_study/` is unchanged.
See `RELEASE_CHECKS.md` for the release commands and publication prerequisites.
