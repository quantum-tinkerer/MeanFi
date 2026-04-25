# Main vs Quadrature Comparison Harness

`profiling/compare_main_vs_quadrature.py` compares the current finite-temperature quadrature branch against a k-grid reference loaded directly from a Git ref, without checking out another branch or modifying core package files.

## What it measures

Each case is evaluated in two stages.

1. Density-only comparison.
   The script first solves the reference branch, freezes the converged reference Hamiltonian, and then compares the two `density_matrix` implementations on that same Hamiltonian. This isolates density generation from the outer SCF loop.
2. Full SCF comparison.
   The script then solves both branches from the same initial guess and compares the resulting reduced density matrices, physical mean fields, gaps, order parameters, and chemical potentials.

The returned local correction is not compared directly because both branches fold `-mu I` into the local onsite term. The script reconstructs the physical mean field separately and compares `mu` as its own quantity.

## Search strategy

The script uses the staged search from the implementation plan rather than a full cartesian sweep.

1. Find the smallest reference `nk` that agrees with the next tighter `nk`.
2. At the tightest sampled quadrature tolerance, find the smallest sampled `kT` that agrees with the next larger sampled `kT`.
3. With that `kT`, relax `charge_tol == density_atol` until the observables move beyond the case thresholds.

The current branch retries quadrature runs at `max_subdivisions=50000` only when the initial run fails because adaptive quadrature did not converge at the default `max_subdivisions=10000`. The retry is recorded explicitly in the output.

## Cases

- `hubbard1d`
  Uses the 1D bipartite Hubbard calibration case with `U=8`, `filling=2`.
- `graphene`
  Uses the docs graphene point with `U=0.2`, `V=1.2`, `filling=2`.

Both cases use one shared seeded random mean-field guess per case so that the only intended algorithmic difference is the density generation path.

## Outputs

Run the script from the repo root with the `latest` pixi environment:

```bash
.pixi/envs/latest/bin/python profiling/compare_main_vs_quadrature.py --case all
```

CLI options:

- `--ref`
  Git ref used for the k-grid reference. Defaults to `main`.
- `--case`
  One of `hubbard1d`, `graphene`, or `all`.
- `--output-dir`
  Directory for `summary.md` and `summary.json`.

The script emits:

- a console summary,
- `summary.md`,
- `summary.json`.

`summary.md` contains:

- a full-SCF summary table,
- a density-only summary table,
- per-case sweep tables for `nk`, `kT`, and tolerance selection.

`summary.json` contains the same information in machine-readable form, including retry notes, selected parameters, and acceptance flags.

## Acceptance thresholds

The script treats a configuration as acceptable only when the density-only stage and the full-SCF stage both satisfy the case thresholds.

For `hubbard1d`:

- `|Delta gap| < 1e-5`
- `|Delta m_staggered| < 1e-4`
- `max rel diff(meanfield) < 5e-3`

For `graphene`:

- `|Delta gap| < 1e-3`
- `|Delta CDW| < 1e-3`
- `max rel diff(meanfield) < 1e-3`

The script also reports a case as unsuitable when the selected reference solution is effectively gapless, because then the finite-temperature comparison is no longer measuring the intended limit cleanly.
