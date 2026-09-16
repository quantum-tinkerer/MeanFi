# Fixed filling and fixed chemical potential

`density_matrix(..., filling=...)` finds the chemical potential satisfying
`N(mu) = filling`, then integrates the requested density entries.
`density_matrix_at_mu(..., mu=...)` uses the supplied value directly and omits
the filling-root search.

The filling solver first evaluates the supplied or previous chemical potential.
If its filling residual and charge error meet their targets, no bracket is needed.
Otherwise it brackets the root using spectral bounds and expands when necessary. It uses safeguarded Newton steps when a supported
backend supplies useful derivatives, and bracketed interpolation with midpoint
fallback otherwise. `ErrorTolerances.filling_residual` and `mu_tol` control this
solve through `tol`; `max_charge_evaluations` bounds work per root search. A small
chemical-potential step is not an accuracy estimate for the root. The filling
residual must still meet its target. Periodic integration adds no derivative-integration
accuracy controls.

Normal periodic spectra do not change with chemical potential. Retaining their
eigenvalues makes repeated charge evaluations inexpensive; density evaluation
streams eigenvectors. BdG uses `H(k) - mu Q`, where the charge operator generally
does not commute with the Hamiltonian. A changed `mu` therefore requires a new
spectral evaluation.

Accuracy-controlled periodic integration solves the filling on each new grid.
Coarse/fine comparisons are evaluated at that grid's same returned chemical
potential, without additional shifted grids. Simplex integration reuses its native spectral mesh
through charge and density refinement.

## Interpreting results

The public result separates:

- the filling root residual on the evaluated mesh;
- charge and density integration error estimates;
- the matrix-function approximation estimate, when available;
- the outer SCF residual.

`density_matrix_integration` estimates integration error at the returned chemical
potential. It does not include density uncertainty caused by uncertainty in that
chemical potential. Root residual and charge-integration error are reported
separately; no chemical-potential error bound is estimated.

A small root residual on a prescribed mesh does not establish continuous
Brillouin-zone accuracy. Prescribed integration errors are `None`, not fabricated
zeros. Accuracy-controlled calls fail when targets cannot be met within their
budgets. Error estimates and reference comparisons are empirical.

At zero temperature a point-sampled charge is discontinuous in chemical
potential, so a requested filling may not be realizable on a prescribed periodic
mesh. Use FermiSimplex for normal systems or choose a suitable manual mesh and
filling tolerance for zero-temperature BdG.
