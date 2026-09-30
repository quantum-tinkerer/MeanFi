# Occupation enclosure: design and measured results

## Scope

Use one reduced matrix model for gap classification and charge error. The
experimental branch now has one implementation: no `charge_method`, `method`,
boolean switch or legacy charge backend. Historical builds supply comparisons.
The native surface classifier uses the same enclosure as charge integration.
The separate vertex-only `certify_simplex` remains an affine-matrix primitive
with an explicit remainder contract; adaptive calculations no longer use it.

Base: MeanFi `54eab84`, FermiSimplex `13aeda0`, AdaptiveSimplex `5ea4787`, paper
`b3735fe`. The previous experimental implementation is FermiSimplex `98c5009`.
Current native implementation: [9f06d36](https://gitlab.kwant-project.org/qt/lineartetrahedron/-/commit/9f06d36c74f6c9c88f3dcc77ef4b173155c3f5b4).
The pre-optimization version is `ae3ce87`; earlier tables explicitly describe
that revision. The larger-band follow-up below measures both revisions.
No new dependencies, long SCF runs, or changes to the deferred `nk` behavior.

## Larger-band follow-up and optimization

The 49% figure was the total time ratio for a 192-band **1D** tight-binding
model. Its probe locations did not change. With one active band and additional
safe bands, both implementations still use 15 vertices, 10 refinements and 24
simplex visits. All measured charges have error about `9.310467e-6` against the
analytic occupied length; the enclosure indicator is `9.775987e-6`.

### Measured cost through 1,024 bands

| Bands | Old estimator, seconds | Before optimization, seconds | After optimization, seconds | Before / old | After / old |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | .00144 | .00135 | .00136 | .94 | .94 |
| 36 | .00423 | .00552 | .00534 | 1.30 | 1.26 |
| 96 | .0314 | .0456 | .0437 | 1.45 | 1.39 |
| 192 | .168 | .260 | .252 | 1.55 | 1.50 |
| 384 | 1.168 | 1.793 | 1.637 | 1.54 | 1.40 |
| 768 | 10.045 | 13.881 | 13.375 | 1.38 | 1.33 |
| 1024 | 23.259 | 31.419 | 30.233 | 1.35 | 1.30 |

The earlier 49% result reproduced as **55% before optimization and 50% after**
in this run. Absolute timings vary on the shared host. The 192-band optimization
saves about 3% in pooled medians, but its timing distributions overlap; it is a
modest improvement. The measured reduction is 3–9% over 36–1024 bands, and below
noise at 12 bands. The substantial remaining overhead has not been eliminated.

The relative penalty **does not keep increasing** in this fixed-physics test.
It peaks around 192–384 bands and falls at 768–1024. Over 384–1024 bands, the
observed power of `N` is 3.05 for the old estimator, 2.92 before optimization,
and 2.97 after optimization. These finite-range slopes support cubic scaling;
they do not locate a universal asymptotic plateau or predict every Hamiltonian.

![Runtime and relative overhead through 1024 bands](experiments/occupation/large-bands.svg)

Through 192 bands, each median pools nine batches from three processes with
reversed build order. At larger sizes it uses three integrations per build,
with build order rotated between sizes. All use one pinned CPU and one BLAS
thread, fresh meshes, two warmups, and no concurrent builds or tests. Model
construction and imports are excluded; mesh construction is included. Error
bars show numerator 10th–90th percentiles divided by the baseline median, not
confidence intervals. An independent two-sample pilot agrees with the larger
ratios: 1.52, 1.38, 1.34 before optimization at 384, 768, 1024 bands.

### Where the time goes at 192 bands

Instrumentation of `ae3ce87`, averaged over nine integrations, gives:

| Phase | Milliseconds per integration | Share of measured phases |
| --- | ---: | ---: |
| Vertex eigensystems | 85.1 | 32% |
| Reconstruct vertex Hamiltonians | 34.5 | 13% |
| Rotate full Bernstein controls | 62.1 | 24% |
| Other interpolation assembly and matrix norms | 39.6 | 15% |
| Hamiltonian samples | 11.1 | 4% |
| Safe-sector bounds and other model work | 14.8 | 6% |
| Schur reduction and bounds | 16.4 | 6% |
| Reduced polynomial integration | 0.13 | 0.05% |

These phases do not overlap: the sector/model row is the residual of model
construction after interpolation, rotation and Schur work. The instrumented
total is about 264 ms per integration; the median runtime is
260 ms. The old estimator could evaluate projected tight-binding hoppings.
This algorithm first constructs and bounds full matrices to decide which
states are safe to eliminate. In 1D that entails two reconstruction products
and four rotation products per simplex, even when only one active band remains.
These six dense products account for most of the extra work. Counting
Hamiltonian calls or eigensystems alone therefore obscures the cost.

### General changes in `9f06d36`

- Store each symmetric Bernstein control once. The number of stored matrices
  drops from 4 to 3 in 1D, 9 to 6 in 2D, and 16 to 10 in 3D.
- Accumulate matrix norms and sector row bounds in column-major order. Both
  triangles still contribute, including any roundoff asymmetry.
- Store the diagonal safe anchor as a vector and shift copied safe blocks in
  place, removing dense zero storage and redundant copies.
- Use the first failed Cholesky pivot to find the largest definite prefix.
  Reverse the ordering for a positive suffix. Earlier successful controls remain
  definite on every smaller principal block. This removes the binary search
  and its potentially logarithmic number of dense factorizations.

These changes keep one production method. They add no probes, fitted thresholds,
Hamiltonian-specific cases or configuration options. A regression compares the
sector counts to exhaustive eigenvalue tests on 100 complex matrix families;
counts agree, and the largest margin excess is `2.34e-15`, below the documented
`1e-12` numerical comparison tolerance. Existing exact-volume, quartic-pocket
and cubic-Schur tests also pass.

### Checks beyond the one-active-band tight-binding case

| Case | Optimized / preceding enclosure time | Optimized / old estimator time |
| --- | ---: | ---: |
| 96-band callable | .97 | 1.11 |
| 192-band callable | .98 | 1.18 |
| 96-band repeated two-band blocks | .99 | .81 |
| 192-band repeated two-band blocks | .95 | .79 |
| 2D quartic annulus, scalar | .89 | — |
| Frozen 2D paper input, 36 bands, target .01 | .93 | — |

The paper example takes 2.372 → 2.205 seconds, retaining 1,750 refinements,
actual error `.002717470643` and indicator `.00999102436`. Both versions still
hit the 2,000-refinement cap at target `.001`. There are no new SCF runs.
All 24 analytic charge checks retain exactly the same charge values and
refinement counts after optimization, with every error covered. In particular,
the annulus error remains `6.663055e-5` and the disk error `9.262147e-5`.
Other small scalar cases vary by a few percent. A three-sample unbatched
12-band run was 9% slower, while the nine-batch comparison above showed only
0.3%; there is no resolved small-matrix improvement.

The repeated-block family's speed advantage over the old estimator is **at
matched requested tolerance**, not matched actual error. At 192 bands the old
estimator happens to achieve `4.57e-10` actual error, versus `6.32e-4` for both
enclosure revisions; the target is `9.6e-4`. Both indicators cover their errors.
The enclosure uses 18 vertices instead of 36. This family demonstrates that
mesh adaptation changes the runtime comparison, but does not show a speed
advantage at equal achieved error. The one-active-band scaling table has equal
actual errors and equal meshes throughout.

The cubic/quartic sweep remains unchanged: zero false claims in all 297 pocket
cases, all 549 interval charges enclosed, and all 270 true gaps resolved after
persistent refinement. Raw results are saved separately for this revision.

### What the scaling claim does and does not cover

For a fixed simplex dimension and fixed temporary depth, construction and
occupation integration now cost at most `O(N^3)` per simplex and use `O(N^2)`
temporary matrix storage, plus a fixed number of Hamiltonian evaluations.
For tight binding this assumes a fixed number of hopping matrices; an arbitrary
callable may itself have a different evaluation cost. There are a fixed number
of full control rotations, at most
one Cholesky factorization per control and sign, and at most one subsequent
margin eigensystem. Reduced matrices have dimension `q <= N`. Full vertex
eigensystems already have cubic cost in the historical method.

Thus the new method adds a cubic constant cost; it does not add a higher power
of band count. A binary sector search in the preceding implementation could
have added a logarithmic factor in difficult cases; that path is removed.
The number of persistent refinements is a separate issue: adding physical
bands, crossings or smaller gaps can change it. No band-count-only bound on
end-to-end adaptive runtime follows from these measurements.

## Why the 36-band case was 56% slower

The old estimator evaluates a projected tight-binding model. The enclosure
constructs full matrices, rotates their Bernstein controls, proves safe-sector
bounds, and only then reduces to the active band. Fewer eigensystems therefore
does not imply less work.

Phase instrumentation reproduced a **1.56×** slowdown on the original case:
7.59 ms for the historical estimator and 11.86 ms for the previous enclosure.
Both used 15 vertices, 10 refinements and identical reported charge. The
historical charge/error stage cost 3.77 ms; the new enclosure cost 7.70 ms.
Vertex eigensystems cost about 3.7 ms in both.

| Previous enclosure phase | Time per integration | Share of total |
| --- | ---: | ---: |
| Vertex eigensystems, shared with baseline | 3.73 ms | 31% |
| Evaluate probe Hamiltonians | 0.40 ms | 3% |
| Reconstruct vertex matrices | 0.61 ms | 5% |
| Rotate full control matrices | 1.63 ms | 14% |
| Safe-sector tests and margins | 1.79 ms | 15% |
| Schur reduction and its bounds | 1.18 ms | 10% |
| Other interpolation assembly and norms | 1.88 ms | 16% |
| Integrate the reduced polynomial | 0.18 ms | 1.5% |

These regions are exclusive; the remaining time is mesh, affine-charge and
binding overhead. Instrumented stage times are averages; overall timings are
medians. Extra Hamiltonian evaluations (15 → 87) explain only a small part of
the increase. Full matrix arithmetic and bounds dominate, although the active
space is usually only one dimensional. The old estimator's 63 eigensystems
include small reduced problems; the new method's 15 are full vertex problems.

The revised code uses the cached diagonal anchor instead of rotating it again,
reuses safe-sector margins, and avoids copying sector matrices when row bounds
suffice. This reduces the 36-band overhead to **about 31%**. Dense matrix
products still cost O(N³), and the full matrix scans cost O(N²).

## Earlier measurements through 192 bands (`ae3ce87`)

The following medians come from separate Release builds, alternating their
execution order, with one pinned CPU and one BLAS/OpenMP thread. Ratios below
one are faster than the historical estimator. The first family retains one
active band and adds distant safe bands. All runs retained the same 15 vertices
and 10 refinements, so these ratios measure computational overhead directly.

| Bands | Previous enclosure / old, tight binding | Revised / old, tight binding | Revised / old, callable |
| ---: | ---: | ---: | ---: |
| 4 | 0.79 | 0.71 | 0.66 |
| 12 | 1.11 | 0.94 | 0.77 |
| 36 | 1.60 | 1.31 | 1.02 |
| 64 | 1.70 | 1.43 | 1.09 |
| 96 | 1.74 | 1.45 | 1.15 |
| 128 | 1.83 | 1.43 | 1.12 |
| 192 | 1.88 | 1.49 | 1.23 |

For this family, relative overhead increases with matrix size and then stays
near 40–50% for the revised tight-binding calculation over the larger tested
sizes. This is not a universal monotonic law. A callable cannot use the old
projected-hopping shortcut, so the relative penalty is smaller.

A second family repeats coupled two-band blocks, increasing the number of
active bands. Its target is scaled by the number of copies to hold accuracy
per copy fixed:

| Bands | Revised / old time | Old → revised mesh vertices |
| ---: | ---: | ---: |
| 4 | 0.63 | 15 → 15 |
| 12 | 0.77 | 36 → 16 |
| 36 | 0.83 | 36 → 16 |
| 96 | 0.80 | 36 → 17 |

Here the revised method is faster because it needs fewer refinements. Matrix
size, active dimension, Hamiltonian representation and refinement count all
matter. These families are controlled examples, not a universal performance
prediction.

![Runtime ratios versus band count](experiments/occupation/scaling.svg)

Markers show pooled medians of nine timing batches across three separate
processes. Bars show the 10th–90th percentiles of numerator timings divided by
the baseline median; they are not confidence intervals. The shared host shows
some timing variation, particularly at 24 bands. Raw measurements are retained.

## What “charge error covered” means

For each test, compute `actual_error = abs(reported_charge - reference)`.
“Covered” means `actual_error <= reported_error_indicator` (with a stated
roundoff allowance). It does **not** mean the actual error decreased, and these
tests do not establish coverage for arbitrary Hamiltonians. The adaptive
indicator sums each simplex's greatest distance from its reported charge to
the two occupation-interval endpoints.

At the tight tested targets, errors are electrons per cell:

| Model | Target | Actual error: old → revised | Revised indicator | Indicator / actual error |
| --- | ---: | ---: | ---: | ---: |
| cosine | 1e-5 | 5.64e-6 → 5.64e-6 | 5.87e-6 | 1.04 |
| quadratic pocket | 1e-5 | 5.37e-6 → 5.37e-6 | 5.81e-6 | 1.08 |
| disk | 1e-4 | 8.62e-5 → 9.26e-5 | 9.97e-5 | 1.08 |
| quartic annulus | 1e-4 | 6.80e-5 → 6.66e-5 | 1.00e-4 | 1.50 |
| Dirac | 1e-4 | 8.82e-5 → 6.89e-5 | 9.99e-5 | 1.45 |
| mixed 12/36 bands | 1e-5 | 9.31e-6 → 9.31e-6 | 9.78e-6 | 1.05 |

For 36 bands the old indicator was 9.48e-6. Its replacement is only **3.1% larger**;
the measured error is unchanged. On identical initial meshes, the actual charge
is unchanged by construction. The indicators vary more: revised/old ratios
range from 0.33 for the pocket to 2.89 for Dirac; the mixed-band ratio is 2.21.
Thus the estimates can be substantially more conservative on coarse meshes,
while the final errors at matched targets remain comparable.

All 24 revised analytic-reference checks covered the error. The old estimator
slightly underestimated three initial-mesh errors; it covered every refined
case. At the disk target, full Hamiltonian evaluations fell from 19,587 to
9,011, and eigensystems from 4,844 to 251.

On the paper's frozen 36-orbital inputs:

| Input and target | Actual error: old → revised | Refined mesh work |
| --- | ---: | --- |
| 1D, 1e-3 | 5.92e-4 → 6.48e-4 | 37 → 36 refinements |
| 2D, 1e-2 | 4.04e-3 → 2.72e-3 | 1,259 → 1,750 refinements |

The revised 2D test takes about 2.49 s versus the previously measured 2.17 s;
its extra probes and larger remainder allowance have a real cost. These paper
measurements were not CPU-pinned, so small differences should not be treated as
precise ratios. Both methods exhaust the 2,000-refinement cap at target 1e-3.
The 1D reference is analytic. The saved 2D reference agrees under axis exchange
to 2.22e-16 and reports quadrature error 7.53e-12; its conservative trace
accuracy target, 3.6e-7, is small compared with these charge errors.

## Probe placement: generality and cost

The quadratic interpolant still uses vertices and edge midpoints. The additional
samples now complete the **degree-four barycentric lattice**: all points with
coordinates `alpha/4`, where the nonnegative integer coordinates sum to four.
This fixed, symmetric rule depends only on simplex dimension. It does not use
band count, an expected crossing location, fitted model parameters or a choice
of physical Hamiltonian.

| Dimension | Additional validation points | All nodes, before → now | New Hamiltonian evaluations per simplex, before → now |
| ---: | --- | ---: | ---: |
| 1 | Two quarter points on each edge | 5 → 5 | 3 → 3 |
| 2 | Edge quarter points; three permutations of `(1/2,1/4,1/4)` | 13 → 15 | 10 → 12 |
| 3 | Edge quarter points; three points on each face; tetrahedron center | 27 → 35 | 23 → 31 |

The evaluation counts exclude cached vertices and include edge midpoints.
The triangular face centroid was replaced by three face-interior points; edge
and tetrahedron-center probes are unchanged. There are no probe eigensystems,
and temporary polynomial subdivision adds no Hamiltonian evaluations.
Consequently the **192-band 1D slowdown cannot come from this probe change**.
The probe change adds 20% to new Hamiltonian calls per triangle and about 35%
per tetrahedron. These are sample-count increases, not total-runtime estimates.
Extra matrix assembly and norm scans cost `O(N^2)` for each added sample;
an expensive user-supplied Hamiltonian can dominate this cost.

Why degree four: values on the complete lattice determine any polynomial of
degree at most four. Its residual after quadratic interpolation vanishes at
vertices and midpoints. Exact rational bounds on the remaining cardinal
functions give the dimension factors 2, 4 and 8. This controls **every** matrix
polynomial through degree four, rather than only the cubic/quartic pocket
examples. General smooth functions still require a sampling assumption;
higher-degree features or localized bumps can escape a finite sample set.
The node counts equal the dimensions of the degree-four polynomial spaces:
5, 15 and 35. With fewer value samples, a nonzero quartic polynomial can vanish
at every sample. Thus these counts are minimal for a value-only rule that
controls every quartic residual; the lattice is one symmetric choice of such
nodes. For a smooth nonpolynomial Hamiltonian, this addresses its local Taylor
terms through degree four but does not bound the fifth and higher terms.
The larger factors in 2D/3D can also require more persistent refinement, so
their practical cost must be judged at the requested charge accuracy.

## Cubic and quartic false gap claims

The sweep uses exact polynomial roots and extrema in 1D, and explicit positive
and negative witnesses for interior triangle pockets. Counts compare the root
sign tests, not failures of the old complete charge integrator:

| Pocket family | Cases | False vertex-affine claims | False previous-enclosure claims | False revised claims |
| --- | ---: | ---: | ---: | ---: |
| 1D cubic | 135 | 135 | 0 | 0 |
| 1D quartic | 144 | 144 | 0 | 0 |
| 2D cubic interior bubble | 9 | 9 | 0 | 0 |
| 2D quartic interior bubble | 9 | 9 | 9 | 0 |

The quartic counterexample on `0 <= y <= x <= 1` is
`H(x,y) = .001 + y*(1-x)*(x-y)*(1-2*x+y)`.
Every old edge and face-center probe sees `.001`, but `H(.8,.3) = -.008`.
Increasing temporary polynomial depth cannot detect this missing information.
Three interior face probes replace the centroid and detect the pocket.

Probe placement alone is insufficient: adversarial quartic residuals can be
more than twice their largest sampled value in 2D/3D. The default factors are
now **2, 4 and 8** in dimensions one, two and three. An exact rational
Bernstein verification proves these bounds for any matrix polynomial of degree
at most four. The script proves cardinal-function bounds `14/9`, `4`, `71/9`,
respectively. Extra native regressions exercise extremizing residuals in 2D/3D.
For nonpolynomial Hamiltonians this remains a sampling assumption.

The 270 genuinely gapped cubic/quartic examples expose the conservative side:
only three were proved gapped on the initial simplex at temporary depth six.
All 270 were proved after **1–6 persistent mesh refinements (median 3)**;
their charge was exactly zero and the stopping indicator below 1e-12. Persistent
refinement lowers the remainder; subdividing the same polynomial does not.
All 549 interval charge checks enclosed their analytic occupied length.
An arbitrary compact bump between every probe still defeats sampling, as an
explicit regression demonstrates.

## Algorithm and assumptions

Interpolate `K=H-mu I` by a quadratic Bernstein matrix polynomial. Its nodes are
vertices and edge midpoints; validate at the remaining degree-four lattice
nodes. Use `eta=2**dimension*max_defect+roundoff`. Explicit uniform remainders
can replace the sampled allowance for inspection.

In a vertex eigenbasis, bound negative and positive safe sectors with margin
`Delta > 0`. Retain the other states as active. With active/safe blocks `A,B,D`,
let `B1` be the affine vertex coupling and `X=D0^-1 B1`, with `D0` the anchor's
safe block. Form

`P = A2 - B1† D0^-1 B1`,

and bound its Schur error by

`epsilon = eta + 2*b*x + d*x^2 + (b+d*x)^2/Delta`,

where `b >= ||B-B1||`, `d >= ||D-D0||`, `x >= ||X||`. Bernstein controls and
matrix norm bounds provide these quantities. The identity
`S=Y-F†D^-1F`, with `Y=A-B†X-X†B+X†DX`, `F=B-DX`, gives the allowance.
Under a smooth local family and a uniform safe gap, it is O(h³) with an O(h⁴)
solve-residual contribution. The reported affine-band charge remains generally
second order. Failed safe tests enlarge the active space.

Temporary bisection restricts this polynomial exactly, without new Hamiltonian
calls or a smaller `epsilon`. Shifted affine cuts in its center eigenbasis
bound integrated occupation; strict sign bounds tighten the endpoints. The
density-cut estimate includes spatial cut disagreement, so cancellation in
charge cannot hide a displaced occupied region.

Strict occupation and integrated charge are separate quantities. A structurally
constant tight-binding matrix has exact charge, including half-filled flat
bands, without implying a strict gap. General callable flat bands may remain
unresolved without a structural guarantee. Roundoff safeguards are not formal
interval arithmetic proofs.

## Reproduce and validation

Use FermiSimplex `98c5009` for the two historical paths and the companion branch
for the revised path. Both are Release builds with AdaptiveSimplex `5ea4787`.
The measurements used GCC 14.3, Python 3.13.15, NumPy 2.5.3, OpenBLAS and an
Intel Xeon Gold 5418Y. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

```sh
python performance/compare_occupation_builds.py \
  --baseline /path/to/historical/python --current /path/to/revised/python \
  --models /path/to/historical/benchmarks/occupation_enclosure.py \
  --output scaling.json
PYTHONPATH=/path/to/revised/python python performance/occupation_polynomial_gaps.py \
  --output gaps.json
python /path/to/revised/benchmarks/verify_remainder_factor.py
PYTHONPATH=/path/to/revised/python python /path/to/revised/benchmarks/occupation_enclosure.py \
  --output charge.json --repeats 3
PYTHONPATH=/path/to/revised/python python performance/occupation_enclosure.py \
  --paper /path/to/meanfi-paper --output paper.json --repeats 3
```

The larger-band follow-up uses three separate builds: the old estimator at
`98c5009`, the preceding enclosure at `ae3ce87`, and the optimized enclosure at
`9f06d36`. The original model definition is loaded from the historical checkout.

```sh
python performance/occupation_large_bands.py \
  --legacy /path/to/98c5009/python --before /path/to/ae3ce87/python \
  --after /path/to/9f06d36/python \
  --models /path/to/98c5009/benchmarks/occupation_enclosure.py \
  --bands 12 36 96 192 --rounds 3 --repeats 3 --output optimization-small.json
# Repeat with --bands 384 768 1024 --rounds 1 for optimization-large.json.
python performance/plot_occupation_large_bands.py \
  --input optimization-small.json optimization-large.json --output large-bands.svg
```

To reproduce the 192-band profile, use
[profile-ae3.patch](experiments/occupation/profile-ae3.patch) on `ae3ce87`, with
`git apply --unidiff-zero`, and the same profile header and environment below.
Select `--variant current --bands 192 --repeats 7`. The saved
[profile](experiments/occupation/profile-192.json) contains raw inclusive totals
and derived exclusive times; do not add nested regions together.

For the phase profile, apply [profile.patch](experiments/occupation/profile.patch)
with `git apply --unidiff-zero` to a separate checkout at `98c5009`, copy
[profile-header.h](experiments/occupation/profile-header.h) to
`cpp/include/fermisimplex/occupation_profile.h`, and rebuild. Run
`performance/occupation_scaling.py --variant previous --bands 36 --representation tb
--models /path/to/historical/benchmarks/occupation_enclosure.py` with
`OCCUPATION_PROFILE=/path/to/times.json`. Divide the accumulated phase times by
`2 + batch*repeats`; the two extra calls are warmups. Instrumentation is absent
from the shipped implementation.

Data: [scaling](experiments/occupation/scaling.json),
[phase profile](experiments/occupation/profile.json),
[charge comparison](experiments/occupation/error-comparison.json),
[ae3ce87 charge](experiments/occupation/charge-current.json),
[ae3ce87 paper](experiments/occupation/paper-current.json),
[previous gaps](experiments/occupation/gaps-previous.json),
[revised gaps](experiments/occupation/gaps-current.json),
[rational proof](experiments/occupation/remainder-proof.json).
Follow-up data: [small-band timings](experiments/occupation/optimization-small.json),
[large-band timings](experiments/occupation/optimization-large.json),
[independent pilot](experiments/occupation/large-bands-pilot.json),
[callable timings](experiments/occupation/optimization-callable.json),
[repeated-block timings](experiments/occupation/optimization-replicas.json),
[charge before optimization](experiments/occupation/charge-before-optimization.json),
[optimized charge](experiments/occupation/charge-optimized.json),
[paper before optimization](experiments/occupation/paper-before-optimization.json),
[optimized paper](experiments/occupation/paper-optimized.json),
[optimized gap checks](experiments/occupation/gaps-optimized.json).
Original [charge](experiments/occupation/charge.json) and
[paper](experiments/occupation/paper.json) measurements are retained.

Validation: **796 MeanFi checks passed** (43 skipped, 35 slow checks deselected),
**180 FermiSimplex Python tests passed**, and **all 10 native test groups passed**.
Tests include exact 1D/2D/3D volumes, cubic Schur convergence, complex matrices,
quartic interior failures, flat-band half occupation, cut cancellation and the
full MeanFi filling/Fourier-moment API against analytic references.
