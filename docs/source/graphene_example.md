---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Graphene and extended Hubbard

## The physics

This tutorial serves as a simple example of using the meanfield algorithm in two dimensions in combination with using Kwant. We will consider a simple tight-binding model of graphene with a Hubbard interaction. The graphene system is first created using Kwant. For the basics of creating graphene with Kwant we refer to [this](https://kwant-project.org/doc/1/tutorial/graphene) tutorial.

We begin with the basic imports

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pymf.model as model
import pymf.solvers as solvers
import pymf.tb as tb
import pymf.kwant_helper.kwant_examples as kwant_examples
import pymf.kwant_helper.utils as kwant_utils
```

##  Preparing the model

We first translate this model from a Kwant system to a tight-binding dictionary. In the tight-binding dictionary the keys denote the hoppings while the values are the hopping amplitudes.

```{code-cell} ipython3
# Create translationally-invariant `kwant.Builder`
graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
h_0 = kwant_utils.builder_to_tb(graphene_builder)
```

We also use Kwant to create the Hubbard interaction. The interaction terms are  described by:

$$ Hubbardd $$

Once we have both the non-interacting and the interacting part, we can assign the parameters for the Hubbard interaction and then combine both, together with a filling, into the model.

```{code-cell} ipython3
U=1
V=0.1
params = dict(U=U, V=V)
h_int = kwant_utils.builder_to_tb(int_builder, params)
model = model.Model(h_0, h_int, filling=2)
```

To start the meanfield calculation we also need a starting guess. We will use our random guess generator for this. It creates a random Hermitian hopping dictionary based on the hopping keys provided and the number of degrees of freedom specified. As we don't expect the mean-field solution to contain terms more than the hoppings from the interacting part, we can use the hopping keys from the interacting part. We will use the same number of degrees as freedom as both the non-interacting and interacting part, so that they match.

```{code-cell} ipython3
guess = tb.utils.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
mf_sol = solvers.solver(model, guess, nk=18)
full_sol = tb.tb.add_tb(h_0, mf_sol)
```

After we have defined the guess, we feed it together with the model into the meanfield solver. The meanfield solver will return a hopping dictionary with the meanfield approximation. We can then add this solution to the non-interacting part to get the full solution. In order to get the solution, we specified the number of k-points to be used in the calculation. This refers to the k-grid used in the Brillouin zone for the density matrix.

## Creating a phase diagram

We can now create a phase diagram of the gap of the interacting solution. We will use the same hopping dictionary for the non-interacting part as before. We will vary the onsite Hubbard interactio $U$ strength from $0$ to $2$ and the nearest neighbor interaction strength $V$ from $0$ to $1.5$.

```{code-cell} ipython3
