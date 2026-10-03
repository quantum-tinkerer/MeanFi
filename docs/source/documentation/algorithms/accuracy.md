# Accuracy and diagnostics

All three calculation functions accept `tol`: `solver`, `density_matrix`, and
`density_matrix_at_mu`. A scalar passes through `default_solver_tolerances`
once. The policy depends only on `tol`, for every matrix size and calculation.
Numerical methods then use the resolved targets directly.

| Target | Default for `tol=t` | Meaning |
| --- | --- | --- |
| `scf_residual` | $t$ | Largest change in the active density at self-consistency |
| `density_matrix_integration` | $t/5$ | Requested quadrature error after integrating the density entries |
| `filling_residual` | $t/50$ | $|N(\mu)-N_{\mathrm{target}}|$ in the charge solve |
| `charge_integration` | $t/5$ | Estimated integration error of the charge calculation |
| `matrix_function_tol` | $t/40$ | AAA's density matrix-function approximation target |
| `mu_tol` | $10^{-10}$ | Newton-step threshold for switching to a bracketed solve, in energy units |

`mu_tol` does not replace the filling-residual condition. Reaching
floating-point resolution without meeting that condition is a convergence failure. There is no separate
energy or entropy tolerance.

To adjust one target, pass a complete record:

```python
from dataclasses import replace

targets = replace(meanfi.default_solver_tolerances(1e-3), charge_integration=1e-3)
result = meanfi.density_matrix(model, tol=targets)
```

An omitted `charge_integration` in a new `ErrorTolerances` record follows
`density_matrix_integration`. A custom `tolerance_policy` may define other
relationships through its single `tol` argument.
Root finding uses `filling_residual` directly. Its default is one tenth of
`charge_integration`, so the filling solve adds little error to the charge
integration budget. AAA uses `matrix_function_tol` directly, without hidden tightening by filling tolerance or matrix size.

## Independent simplex error targets

For adaptive zero-temperature FermiSimplex calculations, the charge step first
refines the Fermi-surface mesh to `charge_integration`. It also estimates density
error from the approximate occupied regions, `density_cut_estimate`. Smooth
projector quadrature then refines to its own requested target:

\[
  t_{\mathrm{quad}}=t_{\mathrm{density}}.
\]

The cut estimate never relaxes this target. With the default policy both charge
and density targets are `tol/5`. A larger cut estimate indicates remaining
occupation uncertainty; making quadrature more accurate cannot remove that
uncertainty. The two estimates remain visible separately.

When a cell has a fixed number of occupied bands, the cut estimate compares its
reported occupations directly with those known full and empty bands. It retains
any discrepancy from tolerance-induced half occupations. Fixed occupation does
not make the projectors constant, so smooth insulating densities still require
quadrature.

These rules apply at both fixed filling and fixed chemical potential.
Density-only bisection preserves the charge mesh's affine occupation cuts.
Prescribed `nk` calculations retain their density rule without adaptive error
estimates.

The reported `errors.density_matrix_integration` is the achieved quadrature
estimate and meets the requested target when adaptive integration succeeds.
`errors.density_cut_estimate` is reported separately. Their sum is not a
certified total-density error: the occupation model uses sampled remainders,
and the quadrature estimate also depends on the sampled density.

## What the reported errors mean

`result.errors` is an `ErrorValues` record. Unavailable or inapplicable estimates
are `None`, rather than quantities computed solely to fill this record.

- `density_matrix_integration` concerns entries **after momentum integration**.
  UniformGrid compares coarse and fine integrals; FermiSimplex supplies an
  adaptive estimate. These are indicators, not certified error bounds.
- `filling_residual` belongs to the chemical-potential solve. It is separate from
  the charge-integration estimate and from the trace of a subsequently refined
  or approximated density. Those numbers need not agree to this tolerance.
- `charge_integration` describes the adaptive charge calculation when one is
  performed, including fixed-chemical-potential FermiSimplex calculations
  that request density entries.
- `matrix_function_error` is AAA's sampled scalar approximation estimate; it is
  `None` for diagonalization and FermiSimplex.
- `scf_residual` is populated by SCF. Band-energy and entropy errors are optional
  diagnostics in their own units; they never control refinement.

A prescribed `nk` provides no integration-error estimate. For finite systems,
integration errors are zero where applicable. `entry_errors` holds density
errors at individual requested coordinates when available.

Integration and matrix-function errors describe different approximations. A
small root residual establishes convergence of the computed charge function;
it does not independently certify its approximation accuracy. Likewise, a
small integration estimate can miss structure absent from the sampled mesh.
Choose a representative starting mesh or compare with a denser calculation
when checking a new model.

## Experimental quadratic occupation enclosure

`FermiSimplex` uses a shared reduced matrix model
for occupation tests and charge-error intervals during the fixed-filling solve.
The interpolation remainder is sampled, so hidden features can still be missed. Its cubic local
matrix allowance does not imply cubic charge convergence or rigorous error
certification. The experiment and numerical comparisons are described in
`docs/occupation-enclosure.md` in the repository.
