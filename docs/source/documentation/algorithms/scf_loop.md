# SCF loop

The outer solve in `MeanFi` is a fixed-point problem for the self-consistent state.
That state may be represented directly as a density object or through a reduced real parametrization.

## Fixed-point map

Let $P$ denote the map from a density state to its reduced real parametrization,

:::{math}
\theta = P(\rho),
\qquad
\rho = P^{-1}(\theta).
:::

The construction of $P$ is described in [Parametrization and symmetry reduction](./parametrization.md).

One SCF iteration has the structure

:::{math}
\rho_n
\;\longrightarrow\;
\hat H_{\mathrm{MF}}[\rho_n]
\;\longrightarrow\;
\rho_{n+1},
:::

where the density update already includes the fixed-filling solve and the Brillouin-zone density evaluation.
If $D$ denotes that full density-evaluation map, then

:::{math}
\rho_{n+1} = D\!\left(\hat H_{\mathrm{MF}}[\rho_n]\right).
:::

In reduced coordinates, this defines the map

:::{math}
G(\theta)
=
P\!\left(
D\!\left(\hat H_{\mathrm{MF}}[P^{-1}(\theta)]\right)
\right).
:::

So the SCF problem is the fixed-point equation

:::{math}
\theta = G(\theta),
:::

or equivalently $\rho = D(\hat H_{\mathrm{MF}}[\rho])$ in the unreduced density representation.

With a history-dependent SCF update, the next iterate may depend on several previous states,

:::{math}
\theta_{n+1}
=
G_n(\theta_n, \theta_{n-1}, \dots, \theta_0),
:::

where the basic fixed-point map $G$ supplies the raw update and the chosen SCF scheme determines how that update is mixed with earlier iterates.

## SCF methods

The public `solver(...)` entry point accepts an explicit SCF method through `scf=...`.
Current built-in methods include:

- `EnergyDIIS(...)`
- `AndersonMixing(...)`
- `LinearMixing(...)`

`EnergyDIIS()` is the default for every supported normal and BdG calculation.
Its history contains evaluated density states, their one-body energies, and
entropies. For non-negative weights summing to one, it minimizes

:::{math}
\widetilde F(c)
=\sum_i c_i\left(E_{0,i}-kT S_i\right)
+E_{\mathrm{int}}\!\left(\sum_i c_i\rho_i\right),
\qquad c_i\geq0,\quad\sum_i c_i=1.
:::

The interaction term is the exact quadratic mean-field functional, including
normal reference subtraction or BdG pairing as appropriate. At zero temperature
this gives the energy of the mixed density. At finite temperature, entropy
concavity makes it an upper bound on that density's free energy. The weighted
history entropy is not the entropy of the mixed state.

This bound can stall near a finite-temperature solution. The solver switches
to bounded Anderson mixing for final convergence, stagnation, or a prolonged
EDIIS phase. Both phases share `max_iterations`. Physical free energy is
reported for every accepted evaluation; it need not decrease at every step.

SCF settings are keyword-only. Anderson exposes `alpha`, `history_size`,
`regularization`, `line_search` and `max_iterations`. Use `solver(..., scf_tol=...)`
for the absolute residual target. EDIIS exposes `history_size` and
`max_iterations`; linear mixing exposes `alpha` and `max_iterations`.

## Output

Once the fixed point converges, `MeanFi` returns two useful state views: the
layout-aware final `result.density` and the physical interaction correction
`result.mean_field`. It does not store a redundant effective Hamiltonian. The
chemical potential and filling remain available as `result.mu` and
`result.filling`, backed by the final density result. `result.internal_energy`
and `result.free_energy` report energies per unit cell; `result.entropy` is in
units of Boltzmann's constant. They satisfy
`free_energy = internal_energy - model.kT * entropy`. Errors, accepted-iteration
history, and the convergence flag are also direct fields of `SCFResult`.

The SCF density uses the same selected coordinates that drove the solve, so it
does not trigger a second full-matrix integration. It can be passed directly as
`Model(..., reference=result.density)` when the new model has a compatible
interaction layout.

`result.history` contains one `SCFIteration` per accepted residual evaluation. Each record contains its step, chemical potential, filling, internal energy, free energy, entropy, and unified `ErrorValues`. The SCF residual is the maximum absolute residual component. Passing `verbose=True` prints these same physical values while the solve runs.

`NoConvergence` and `SolverFailure` are exceptions rather than alternate result shapes. When at least one physical density evaluation succeeded, the exception carries the last valid state as `exception.result` with `converged=False`.

Density integration and filling failures raise `ConvergenceError`. SCF failures
are subclasses: `NoConvergence` means the iteration budget was exhausted;
`SolverFailure` means a numerical evaluation failed. A failure before the first
valid density has `result=None`. Invalid inputs retain `ValueError` or `TypeError`.
