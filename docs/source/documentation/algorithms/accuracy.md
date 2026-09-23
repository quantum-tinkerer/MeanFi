# Accuracy and diagnostics

All three calculation functions accept `tol`: `solver`, `density_matrix`, and
`density_matrix_at_mu`. A scalar passes through `default_solver_tolerances`
once. The policy depends only on `tol`, for every matrix size and calculation.
Numerical methods then use the resolved targets directly.

| Target | Default for `tol=t` | Meaning |
| --- | --- | --- |
| `scf_residual` | $t$ | Largest change in the active density at self-consistency |
| `density_matrix_integration` | $t/5$ | Requested error in an integrated density entry; adaptive FermiSimplex relaxes p/h work when its cut estimate dominates |
| `filling_residual` | $t/10$ | $|N(\mu)-N_{\mathrm{target}}|$ in the charge solve |
| `charge_integration` | $t/5$ | Estimated integration error of the charge calculation |
| `matrix_function_tol` | $t/40$ | AAA's density matrix-function approximation target |
| `mu_tol` | $10^{-10}$ | Smallest useful root-search step, in energy units |

`mu_tol` does not replace the filling-residual condition. Reaching a step limit
without meeting that condition is a convergence failure. There is no separate
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
Root finding uses `filling_residual` directly. AAA uses `matrix_function_tol`
directly, without hidden tightening by filling tolerance or matrix size.

## Cut-limited simplex density budget

For adaptive zero-temperature FermiSimplex calculations, the charge step first
refines the Fermi-surface mesh and estimates the density error associated with
its occupation cut, `density_cut_estimate`. Density cubature then uses

\[
  t_{\mathrm{p}}=\max\!\left(t_{\mathrm{density}},
                 \tfrac12\,\widehat e_{\mathrm{cut}}\right).
\]

The factor of one half keeps the estimated projector-quadrature error smaller
than the estimated cut error, while avoiding much more quadrature work once
the cut dominates. The charge estimate is evaluated at the chosen chemical
potential for both fixed-filling and fixed-chemical-potential calculations.
The original `density_matrix_integration` target remains independently
configurable; it is a tighter limit when the cut estimate is small. Prescribed
`nk` calculations do not estimate the cut and retain their existing density
rule. Density-only bisection keeps the charge mesh's affine occupation cut.

The reported `errors.density_matrix_integration` is the achieved p/h estimate,
not the effective target; it can exceed the requested density tolerance when
the cut-limited budget applies. `errors.density_cut_estimate` is reported
separately. Their sum is not a certified total-density error: the cut estimate
uses sampled curvature and may miss unresolved structure.

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
