# Package reference

## Interactive problem definition

To define the interactive problem, we use the following class:

```{eval-rst}
.. autoclass:: pymf.model.Model
   :members: mfield
```

## Mean-field and density matrix

```{eval-rst}
.. automodule:: pymf.mf
   :members: meanfield, construct_density_matrix, construct_density_matrix_kgrid, fermi_on_grid
   :show-inheritance:
```

## Observables

```{eval-rst}
.. automodule:: pymf.observables
   :members: expectation_value
   :show-inheritance:
```

## Solvers

```{eval-rst}
.. automodule:: pymf.solvers
   :members: solver, cost
   :show-inheritance:
```

## Tight-binding dictionary

### Manipulation

```{eval-rst}
.. automodule:: pymf.tb.tb
   :members: add_tb, scale_tb
   :show-inheritance:
```

### Brillouin zone transformations

```{eval-rst}
.. automodule:: pymf.tb.transforms
   :members:
   :show-inheritance:
```

### Parametrisation

```{eval-rst}
.. automodule:: pymf.params.rparams
   :members:
   :show-inheritance:
```

### Utility functions

```{eval-rst}
.. automodule:: pymf.tb.utils
   :members:
   :show-inheritance:
```

## `kwant` interface

```{eval-rst}
.. automodule:: pymf.kwant_helper.utils
   :members:
   :show-inheritance:
```
