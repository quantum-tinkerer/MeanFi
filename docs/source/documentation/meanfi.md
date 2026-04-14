# Package reference

## Interactive problem definition

```{eval-rst}
.. autoclass:: meanfi.model.Model
   :members: density_matrix, density_matrix_at_mu
```

## Mean-field and density matrix

```{eval-rst}
.. automodule:: meanfi.mf
   :members: meanfield, density_matrix, density_matrix_at_mu, fermi_dirac
   :show-inheritance:
```

## Solvers

```{eval-rst}
.. automodule:: meanfi.solvers
   :members: solver, SolverInfo
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
.. automodule:: meanfi.tb.tb
   :members: add_tb, scale_tb
   :show-inheritance:
```

```{eval-rst}
.. automodule:: meanfi.tb.transforms
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: meanfi.params.rparams
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
.. automodule:: meanfi.kwant_helper.utils
   :members:
   :show-inheritance:
```
