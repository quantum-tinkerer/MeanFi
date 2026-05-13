# Package reference

## Interactive problem definition

```{eval-rst}
.. autoclass:: meanfi.model.Model
   :members: hamiltonian_from_rho, hamiltonian_from_meanfield, bdg_hamiltonian_from_meanfield
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
.. autoclass:: meanfi.DensityMatrixResult
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.AdaptiveQuadratureInfo
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.AdaptiveSimplexInfo
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.UniformGridInfo
   :show-inheritance:
```

## Solvers

```{eval-rst}
.. autofunction:: meanfi.solver
```

```{eval-rst}
.. autoclass:: meanfi.SolverResult
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: meanfi.SCFInfo
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: meanfi.NoConvergence
   :show-inheritance:
```

## Observables

```{eval-rst}
.. automodule:: meanfi.observables
   :members: expectation_value
   :show-inheritance:
```

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
