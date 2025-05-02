# Package reference

## Interactive problem definition

To define the interactive problem, we use the following class:

```{eval-rst}
.. autoclass:: meanfi.model.Model
   :members: mfield, density_matrix_iteration
```

## Mean-field and density matrix

```{eval-rst}
.. automodule:: meanfi.mf
   :members: meanfield, density_matrix, density_matrix_kgrid, fermi_level
   :show-inheritance:
```

## Observables

```{eval-rst}
.. automodule:: meanfi.observables
   :members: expectation_value
   :show-inheritance:
```

## Solvers

```{eval-rst}
.. automodule:: meanfi.solvers
   :members: solver, solver_mf, cost_mf, cost_density
   :show-inheritance:
```

## Tight-binding dictionary

### Manipulation

```{eval-rst}
.. automodule:: meanfi.tb.tb
   :members: add_tb, scale_tb
   :show-inheritance:
```

### Brillouin zone transformations

```{eval-rst}
.. automodule:: meanfi.tb.transforms
   :members:
   :show-inheritance:
```

### Parametrisation

```{eval-rst}
.. automodule:: meanfi.params.rparams
   :members:
   :show-inheritance:
```

### Utility functions

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
