# Package reference

## Interactive problem definition

```{eval-rst}
.. autoclass:: meanfi.model.Model
   :members: hamiltonian_from_rho, hamiltonian_from_meanfield, bdg_hamiltonian_from_meanfield
```

`Model(..., reference_density_matrix=rho_ref)` enables full reference-state
subtraction for normal calculations. The effective Hamiltonian is built as
`h_0 + W[rho - rho_ref]`, where `W` is the complete density-density mean-field
correction, including both Hartree and exchange-like terms. This is not a
Hartree-only background subtraction. Internally, the reference is copied into
a private, read-only coordinate state tied to the model's active SCF space; it
is never exposed as a zero-filled reduced density matrix.

If the reference should be the non-interacting density at a chosen filling,
compute it explicitly and pass the density matrix to the model:

```python
onsite = (0,) * len(next(iter(h_0)))
reference_keys = list(dict.fromkeys([*h_int, onsite]))
rho_ref = meanfi.density_matrix(
    h_0,
    filling=reference_filling,
    kT=kT,
    keys=reference_keys,
).density_matrix
model = meanfi.Model(
    h_0,
    h_int,
    filling=filling,
    kT=kT,
    reference_density_matrix=rho_ref,
)
```

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

`total_energy` expects a density matrix containing every key needed by both the
non-interacting Hamiltonian and the interaction. For normal calculations where
the solver only requested interaction keys, evaluate the density with energy
keys first, for example `keys=list(set(h_0) | set(h_int))`, then pass that
density matrix to `meanfi.total_energy(model, density_matrix)`.

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
