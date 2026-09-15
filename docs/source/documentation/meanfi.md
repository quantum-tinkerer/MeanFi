# Package reference

{download}`Download the executable API walkthrough <../../../examples/api_walkthrough.py>`

## Interactive problem definition

```{eval-rst}
.. autoclass:: meanfi.model.Model
   :members: hamiltonian_from_density, hamiltonian_from_meanfield, random_meanfield
```

`Model(..., reference=reference)` enables full reference-state subtraction for
normal calculations. The effective Hamiltonian is built as
`h_0 + W[rho - rho_ref]`, where `W` is the complete density-density mean-field
correction, including both Hartree and exchange-like terms. This is not a
Hartree-only background subtraction. `reference` is a layout-aware
`DensityResult`: it may contain only the entries required by the interaction,
but every required coordinate is validated when the model is constructed.
Missing entries are never interpreted as zeros.

If the reference should be the non-interacting density at a chosen filling,
compute it explicitly and pass the result to the model:

```python
reference = meanfi.density_matrix(
    h_0,
    filling=reference_filling,
    kT=kT,
    interaction=h_int,
    tol=tol,
)
model = meanfi.Model(
    h_0,
    h_int,
    filling=filling,
    kT=kT,
    reference=reference,
)
```

`density_matrix(model)` uses the model's filling, temperature and required
coordinates. Both normal and superconducting models use this API; optional
`mean_field=correction` evaluates an interacting Hamiltonian. The same model
support is available in `density_matrix_at_mu(model, mu)`.

For a Hamiltonian dictionary, supply exactly one of `keys`, `coordinates` or
`interaction`. `keys` requests complete blocks, while the other two options
request selected entries. `result.to_tb()` returns a dictionary of complete
blocks; `result.to_tb(sparse=True)` returns CSR blocks.
Selected results expose their read-only `coordinates`, `values`, and optional
per-entry `entry_errors`; converting
one to complete matrix blocks raises instead of filling uncomputed entries with
zeros. `result.select(coordinates)` selects both values and entry errors while
preserving the chemical potential, filling, entropy, band energy, and integration statistics.

Integrators and SCF share the immutable `DensityEntries` payload in
`result.entries`. To construct a result from separately computed entries, use
`DensityResult(entries=DensityEntries(coordinates, values, entry_errors),
mu=mu, filling=filling, entropy=entropy, errors=ErrorValues(...))`. The optional entry errors
use the same coordinate order as the values; `None` means no estimate exists.

## Mean-field and density matrix

```{eval-rst}
.. autofunction:: meanfi.meanfield
```

```{eval-rst}
.. autofunction:: meanfi.density_matrix
```

```{eval-rst}
.. autofunction:: meanfi.density_matrix_at_mu
```

```{eval-rst}
.. autofunction:: meanfi.fermi_dirac
```

```{eval-rst}
.. autoclass:: meanfi.DensityEntries
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.DensityResult
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.ErrorTolerances
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.ErrorValues
   :show-inheritance:
```

`Model` validates finite filling and temperature, matching matrix sizes and
lattice dimensions, and Hermiticity. It owns read-only copies of dense or sparse
input matrices and symmetry data. Use a new model (or `dataclasses.replace`) to
change model parameters.

## Solvers

`EnergyDIIS()` is the default for normal and BdG models with every supported
integration backend. At finite temperature it uses a free-energy upper bound
formed from the history's energies and entropies. Entropy is not linear in a
mixed density, so this is a surrogate for the mixed state's free energy.
EDIIS runs only its own update and raises `NoConvergence` on iteration
exhaustion. Users can explicitly restart with another method using
`failure.result.mean_field`; see [user-controlled composition](algorithms/scf_loop.md).
Physical free energy need not fall on every iteration.

Sparse `RationalFOE()` uses AAA at positive temperature on a prescribed
`PeriodicGrid(nk=...)`. Density and entropy share poles and sparse factorizations.

SCF settings are keyword-only. To change the history or iteration budget, pass
`scf=meanfi.EnergyDIIS(history_size=6, max_iterations=100)`. Explicit
`AndersonMixing` and `LinearMixing` remain available.


```{eval-rst}
.. autofunction:: meanfi.solver
```

```{eval-rst}
.. autoclass:: meanfi.SCFResult
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.SCFIteration
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.ConvergenceError
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.SolverError
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.SolverFailure
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.NoConvergence
   :show-inheritance:
```

## Observables

```{eval-rst}
.. automodule:: meanfi.observables
   :members: expectation_value, internal_energy, free_energy
   :show-inheritance:
```

SCF results and history entries report `internal_energy`, `free_energy`, and
`entropy` per cell per physical orbital. Entropy is in units of Boltzmann's constant, so
`free_energy = internal_energy - model.kT * entropy`. This is Helmholtz free
energy at fixed electron filling. The chemical-potential term is not subtracted.
All thermodynamic totals are divided by N for N physical orbitals in the unit
cell. A BdG Hamiltonian has size 2N; its physical totals still divide by N after
removing Nambu doubling. Spin components count as separate orbitals.

| Quantity | Convention |
| --- | --- |
| `internal_energy`, `free_energy`, `band_energy` | Energy per cell per physical orbital |
| `entropy` | Entropy / k_B per cell per physical orbital |
| `errors.band_energy_integration`, `errors.entropy_integration` | Same normalized units as the corresponding quantity |
| `filling` | Electrons per cell, from 0 to N |
| `mu`, `kT` | Single-particle energy units |
| Density entries | Occupations and coherences, without normalization by N |
| `expectation_value(density, observable)` | Unnormalized observable trace per cell |

Multiplying an energy or entropy result by N recovers its physical total per
cell. The generic `expectation_value` remains a trace: for example, the identity
operator gives the electron count for a normal density. Divide that trace by N
when an observable per orbital is wanted.

Compare converged solutions at the same filling and temperature using their
free energies; SCF convergence alone does not establish a global minimum.

```python
solution = meanfi.solver(model, model.random_meanfield(rng=0, scale=0.1))
print(solution.internal_energy, solution.entropy, solution.free_energy)
```

`expectation_value` and `internal_energy` accept complete tight-binding
matrix dictionaries or `DensityResult` objects. Selected results must cover
every coordinate used by the observable, or the operation raises with the
missing coordinates. For a separate energy evaluation, request full blocks
covering the bare Hamiltonian and interaction:

```python
energy_keys = sorted(set(model.h_0) | set(model.scf_space.density_keys))
density = meanfi.density_matrix(
    model, mean_field=solution.mean_field, keys=energy_keys
)
print(meanfi.internal_energy(model, density))
print(meanfi.free_energy(model, density))
```

`free_energy` requires a `DensityResult`, because a dictionary of a few
real-space density blocks does not contain the full state's entropy.
`density.entropy` is computed during density evaluation and remains available
when selecting fewer entries. Normal reference subtraction affects the
interaction energy, not entropy. BdG entropy includes the factor of one half
that removes Nambu doubling.

`density.band_energy` is the expectation of the **input quadratic Hamiltonian**.
It includes the BdG normal-ordering constant, excludes chemical potential, and
uses the same normalization per physical orbital as the other energies.
It is not the interacting internal energy: `internal_energy` accounts for the
interaction's double counting. The SCF result computes both energies directly,
including when its density contains only the entries required by the interaction.

## Tight-binding dictionary utilities

```{eval-rst}
.. automodule:: meanfi.tb.ops
   :members: add_tb, scale_tb
   :show-inheritance:
```

```{eval-rst}
.. automodule:: meanfi.tb.transforms
   :members:
   :show-inheritance:
```

## Coordinates and symmetries

```{eval-rst}
.. automodule:: meanfi.space
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: meanfi.tb.utils
   :members:
   :show-inheritance:
```

## `kwant` interface

```{eval-rst}
.. automodule:: meanfi.interop.kwant
   :members:
   :show-inheritance:
```
