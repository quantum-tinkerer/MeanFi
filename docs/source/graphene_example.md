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
from codes imort model, solvers, kwant_examples, kwant_helper, tb
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
