# Package reference

## Interactive problem definition

```{eval-rst}
.. autoclass:: meanfi.model.Model
   :members: hamiltonian_from_rho, hamiltonian_from_meanfield, bdg_hamiltonian_from_meanfield
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

`density_matrix(..., keys=keys)` requests complete blocks and preserves the
existing `.density_matrix` compatibility property.
`density_matrix(..., coordinates=coordinates)` requests an exact advanced
selection. `keys`, `coordinates`, and `interaction` are mutually exclusive.
Selected results expose their read-only `coordinates` and `values`; converting
one to complete matrix blocks raises instead of filling uncomputed entries with
zeros.

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

## Solvers

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
   :members: expectation_value, total_energy
   :show-inheritance:
```

`expectation_value` and `total_energy` accept either complete tight-binding
matrix dictionaries or layout-aware `DensityResult` objects. Selected results
must cover every coordinate used by the observable; otherwise the operation
raises with the missing coordinates. For an SCF solution, use
`solution.total_energy` directly. If a separate density must be evaluated for
an energy, request complete energy keys and pass the result itself, for example
`meanfi.total_energy(model, meanfi.density_matrix(..., keys=energy_keys))`.

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

## Developer internals

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
