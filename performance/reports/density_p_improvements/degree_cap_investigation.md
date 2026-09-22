# Degree-cap investigation for the four nonconverged accuracy cases

The four rows in `accuracy.json` are the seed-2 `bulk_2d` and `qwz_2d` models at target errors `1e-4` and `1e-5`. Both are two-band insulators with `mu = 0`. The `bulk_2d` model has `d_z = 3 + cos(k_x) + cos(k_y) >= 1`. The `qwz_2d` model has `d = (sin(k_x), sin(k_y), 1.3 + cos(k_x) + cos(k_y))`; its gap cannot close because at simultaneous zeros of both sine terms `d_z` is one of `3.3`, `1.3`, or `-0.7`. The seeded momentum shift and unitary rotation preserve those gaps. There are no occupied-band cuts, so cut moments cannot explain these failures.

An isolated FermiSimplex build changed only the Python and C++ validation ceiling from degree 21 to 41. The production branches were unchanged. For each configuration, charge integration and density integration ran on fresh meshes. The density reference was a 256×256 midpoint grid, checked against 128×128 (maximum difference `4.9e-15` for `bulk_2d`, `2.2e-15` for `qwz_2d`). All density runs used one BLAS thread. Times below are medians of 30 measured runs in the same process; they are indicative, not controlled hardware benchmarks. `root` is the initial dyadic mesh level, and `cap` is the permitted p-cubature degree.

| Model | Target | Root | Cap | Charge simplices | Degree reached | Density evaluations | Actual error | Estimated error | Reached target | Charge+density ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| bulk_2d | 1e-4 | 1 | 21 | 8 | 21 | 2,192 | 7.69e-5 | 2.08e-4 | no | 10.0 |
| bulk_2d | 1e-4 | 1 | 41 | 8 | 29 | 1,325 | 2.58e-5 | 9.14e-5 | yes | 3.78 |
| bulk_2d | 1e-4 | 2 | 21 | 32 | 9 | 635 | 4.60e-6 | 9.25e-5 | yes | 1.69 |
| bulk_2d | 1e-5 | 1 | 21 | 8 | 21 | 2,192 | 7.69e-5 | 2.08e-4 | no | 5.27 |
| bulk_2d | 1e-5 | 1 | 41 | 8 | 41 | 3,779 | 3.66e-6 | 9.76e-6 | yes | 12.0 |
| bulk_2d | 1e-5 | 2 | 21 | 32 | 13 | 935 | 6.81e-7 | 7.80e-6 | yes | 2.32 |
| qwz_2d | 1e-4 | 1 | 21 | 12 | 21 | 3,288 | 4.49e-5 | 1.47e-4 | no | 8.27 |
| qwz_2d | 1e-4 | 1 | 41 | 12 | 25 | 2,079 | 1.80e-5 | 9.60e-5 | yes | 5.84 |
| qwz_2d | 1e-4 | 2 | 21 | 32 | 9 | 764 | 8.55e-6 | 8.24e-5 | yes | 2.22 |
| qwz_2d | 1e-5 | 1 | 21 | 12 | 21 | 3,288 | 4.49e-5 | 1.47e-4 | no | 8.27 |
| qwz_2d | 1e-5 | 1 | 41 | 12 | 35 | 4,680 | 2.28e-6 | 9.05e-6 | yes | 13.5 |
| qwz_2d | 1e-5 | 2 | 21 | 32 | 13 | 1,220 | 1.11e-6 | 9.32e-6 | yes | 3.16 |

At `1e-4`, the original actual errors were already below target; the conservative stopping estimate alone caused the `target_reached=False` status. At `1e-5`, both actual errors exceeded target. Raising the cap fixed the stopping estimate and actual accuracy on the original charge mesh. The default rule reached degree 21 in every simplex, confirming that the cap was active.

The charge mesh contributes to the cap exhaustion: lowering the charge tolerance from `1e-4` to `1e-6` did not change its 8 or 12 simplices because charge is constant inside the gap. Starting at root level 2 let the existing degree-21 integrator converge with degree 13 or less and about one-quarter as many density evaluations as the raised-cap alternative at `1e-5`. Thus this is a resolution tradeoff between bulk mesh spacing and polynomial degree, not a cut-region error or a charge-integration error. The charge-only refinement policy does not supply the bulk resolution that density needs.

A blanket increase to degree 41 is unattractive. The 2D Grundmann–Möller rule has absolute-weight sum about `1.76e3` at degree 21 and `4.85e6` at degree 41, amplifying cancellation and roundoff. Its constant-moment sum deviates from one by roughly `1.4e-11` at degree 41 (using long-double accumulation), versus below `1e-15` at degree 21. This is acceptable for these `1e-5` tests but a poor default for tighter tolerances, more components, and higher dimensions. A density-aware mesh resolution strategy would be faster for these cases; it requires separate validation on metals and 3D models before changing defaults.
