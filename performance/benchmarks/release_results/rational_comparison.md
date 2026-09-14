# Sparse rational density, energy and entropy

**Follow-up (2026-09-14):** The previously failing AAA case now passes at its
original tolerance, bringing the thermodynamic suite to 35/35. See the
[fix and larger-matrix cost study](normalization_and_fit_cost.md). The results
below describe the earlier implementation and are retained as historical evidence.

**Use AAA as the single thermodynamic backend.** In a matched 256-pole study,
AAA met density, charge, energy and entropy tolerances in 34 of 35 sparse-node
cases. An Ozaki implementation with corrected density stopping and entropy
fitted on the same poles succeeded in 4 of 35; the other 31 refused because
entropy could not be certified. The final five public fixed-filling workflows
also passed independent density, band-energy and entropy checks.

This is a coverage and maintenance decision, not a claim of universal AAA speed
superiority. Ozaki retains a small-calculation advantage: with the revised
Fermi-only fit and equal 128-pole caps, the hot 4-orbital node took 2.94 ms cold
for Ozaki versus 3.76 ms for AAA. It also wins the smallest hot joint-entropy
case below. Keeping a separate density-only option would preserve this narrow
advantage while complicating an API that now always supplies thermodynamics.

## What was tested

- 40 density cases: 35 individual sparse nodes and five public fixed-filling
  calculations. Nodes include normal and BdG matrices, complex hopping, gapped,
  symmetric, empty, full and exactly degenerate spectra.
- Chains have 4–512 electron orbitals; 2D squares have 64/256 sites; 3D cubes have
  64/216 sites. Temperatures range from 2 to 0.0002, with spectral width divided
  by temperature reaching 22,152.
- Most absolute density, charge and thermodynamic tolerances are `1e-8`.
  Additional requests cover `1e-4`, `1e-6`, `1e-10` and `1e-12`; 2D/3D cases
  use `1e-6`. Every success must satisfy its own requested tolerance.
- Independent NumPy Hermitian eigensystems and SciPy logistic/entropy functions
  provide the references. Fixed filling uses a separate Brent solve. Public
  energy and entropy are checked at the returned chemical potential; reference
  time is excluded. The same prescribed k mesh isolates matrix-function and
  charge-solve accuracy from Brillouin-zone integration error.
- Three cold and three warm runs per successful case, pinned to CPU 2 of an
  Intel Xeon Gold 5418Y. All BLAS/OpenMP pools use one thread. Cold clears scalar
  fits; warm creates a new matrix node sharing the fit. Both still factor their
  matrices. Public calls each have their own per-calculation AAA cache.
- Each measurement has a 15-second limit; initial failures are not repeated.
  The original density comparison used 128 poles; final density and both joint
  thermodynamic methods use 256. Raw records include timings, pole histograms,
  factorization counts, errors, source fingerprints and environment metadata.

## Accuracy and reuse

| Calculation | Cases satisfying requested tolerances | Other outcomes |
|---|---:|---|
| Original AAA density, 128-pole cap | 24 / 40 | 16 scalar-fitting timeouts |
| Original Ozaki density, 128-pole cap | 36 / 40 | 4 silently inaccurate densities |
| Scalar-certified Ozaki density, 128-pole cap | 39 / 40 | 1 explicit pole-budget failure |
| Final AAA density/public workflows, 256-pole cap | 40 / 40 | none |
| Ozaki shared-pole thermodynamics, 256-pole cap | 4 / 35 | 31 entropy-certification failures |
| Final AAA shared-pole thermodynamics, 256-pole cap | 34 / 35 | 1 explicit certification failure |

The remaining AAA failure requests absolute `1e-12` energy and entropy accuracy
for a 32-site cold chain. It reports `ConvergenceError` after exhausting the
scalar fitting budget. The same density-only request passes. This is an explicit
precision/budget limit; the benchmark does not treat it as a successful result.

Across all 204 successful final thermodynamic calls, extracting energy and
entropy required **zero additional LU factorizations and zero additional
selected-inverse queries**. The traces use the density calculation's retained
inverse entries. A joint fit can require more poles than a density-only fit;
reuse means factoring those shared poles once.

| Final thermodynamic quantity | Largest absolute error | Largest error / requested tolerance |
|---|---:|---:|
| Charge | `7.94e-8` | `0.00846` |
| Selected density entry | `6.27e-9` | `0.00195` |
| Energy trace | `2.41e-8` | `0.0165` |
| Entropy trace | `4.88e-8` | `0.0818` |

These maxima span different requested tolerances, including `1e-4`. All
successful results remain within their own budgets. Public band-energy and
entropy errors are at most `3.71e-11` and `3.48e-11`, respectively, for requested
`1e-8`. Node energies are the full matrix trace `Tr[H rho]`; BdG nodes deliberately
use the full Nambu trace, while public results include the physical half factors
and Hamiltonian constant.

## Timings and remaining cost

Median **cold / warm milliseconds**, with equal 256-pole caps. These are joint
density/energy/entropy calculations; a failed certification is not a fast result.

| Case | Final AAA | Ozaki with entropy fit | AAA factors per node |
|---|---:|---:|---:|
| `metal_n4_T2` | 16.2 / 6.7 | 6.1 / 5.7 | 3 |
| `metal_n32_T0.2` | 30.3 / 7.6 | not certified | 10 |
| `metal_n512_T0.02` | 213.2 / 138.5 | not certified | 19 |
| `symmetric_T0.0002` | 352.3 / 21.7 | not certified | 33 |
| `degenerate` | 2.5 / 1.6 | 4.3 / 3.6 | 0 |
| `empty` | 1.8 / 1.3 | 38.8 / 39.4 | 0 |
| `bdg_n64_T0.02` | 95.4 / 20.9 | not certified | 18 |
| `geometry_square_n256_T0.02` | 129.2 / 49.9 | not certified | 18 |
| `geometry_cube_n216_T0.02` | 107.2 / 44.6 | not certified | 17 |

Final public fixed-filling calculations, including energy and entropy:

| Case | Median cold seconds | Total factorizations |
|---|---:|---:|
| `filling_metal_n8_T0.2` | 1.806 | 574 |
| `filling_metal_n32_T0.02` | 4.056 | 800 |
| `filling_gapped_n16_T0.02` | 6.179 | 1,149 |
| `filling_bdg_n8_T0.2` | 0.970 | 1,008 |
| `filling_bdg_n16_T0.02` | 1.018 | 1,008 |

Scalar fitting/certification still accounts for roughly 84–92% of the normal
fixed-filling cases. A future optimization could maintain one conservatively
enlarged spectral enclosure per calculation to reuse fits as k and mu change.
That would target the measured remaining cost without retaining more sparse
matrix factorizations.

The original Ozaki stopping rule compared only successive total charges.
Symmetric spectra could make that difference vanish while density entries
remained wrong: at requested `1e-8`, errors reached `0.00261`, `0.227` and `0.404`
for the three colder symmetric chains, and `0.0347` for the gapped half-filled
chain. Scalar certification fixes this stopping defect, but its fixed poles still
failed the shared-entropy requirements above. The AAA replacement also eliminates
all original direct-node fitting timeouts and fixes the gapped public solve.

## Artifacts and reproduction

- [Baseline density data](rational_comparison_baseline.json): the original
  `3443495c60c7a0d1bd07423895a41f8edd846dda` runtime, both schemes, 40 cases.
- [Ozaki comparison data](rational_comparison_ozaki.json): corrected density
  stopping, matched small-node timings and shared-pole entropy results.
- [Final density and public API data](rational_comparison_final_density.json).
  `public_recheck` records the final caller snapshot used for the five workflows.
- [Final thermodynamic data](rational_comparison_final_thermodynamics.json).
- [Archived Ozaki thermodynamic backend](rational_comparison_ozaki_backend.patch):
  a minimal three-file patch retaining the experimental backend needed to
  reproduce the Ozaki entropy comparison after its runtime removal.

From the repository root, using the sparse environment with MUMPS:

```bash
.pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
    --output /tmp/final-density.json --max-poles 256 --repeat 3 --timeout 15
.pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
    --output /tmp/final-thermodynamics.json --max-poles 256 --repeat 3 \
    --timeout 15 --thermodynamics
mkdir -p /tmp/meanfi-rational-study
git archive 3443495 | tar -x -C /tmp/meanfi-rational-study
.pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
    --checkout /tmp/meanfi-rational-study --output /tmp/baseline.json \
    --source-revision 3443495 --max-poles 128 --repeat 3 --timeout 15
.pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
    --checkout /tmp/meanfi-rational-study --output /tmp/certified-ozaki.json \
    --max-poles 128 --repeat 3 --scheme ozaki --certify-ozaki
git -C /tmp/meanfi-rational-study apply --unidiff-zero \
    "$PWD/performance/benchmarks/release_results/rational_comparison_ozaki_backend.patch"
.pixi/envs/test-sparse/bin/python performance/benchmarks/rational_comparison.py \
    --checkout /tmp/meanfi-rational-study --output /tmp/ozaki-entropy.json \
    --max-poles 256 --repeat 3 --scheme ozaki --thermodynamics
```

The context-free archive patch requires `--unidiff-zero`; applying it to the
baseline and reproducing both a passing and a failing Ozaki entropy case was
verified. The driver detects which schemes a checkout provides. Its old backend
reference uses shifted energy traces; the final backend reference uses `Tr[H rho]`.
`rational_source_sha256` identifies the numerical backend independently of
unrelated API/test files in a development snapshot. For the study-only corrected
Ozaki stopping rule, scalar certification is included in total time but excluded
from `scalar_fit_seconds`, which instruments the production terms method.

This bounded study supports the simpler shared-entropy backend. It does not
establish behavior for every sparse graph, spectrum, temperature or tolerance,
and it does not measure SCF optimizer convergence.
