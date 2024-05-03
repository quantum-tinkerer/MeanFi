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
import pymf.mf as mf
import pymf.tb as tb
import pymf.observables as observables
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
_model = model.Model(h_0, h_int, filling=2)
```

To start the meanfield calculation we also need a starting guess. We will use our random guess generator for this. It creates a random Hermitian hopping dictionary based on the hopping keys provided and the number of degrees of freedom specified. As we don't expect the mean-field solution to contain terms more than the hoppings from the interacting part, we can use the hopping keys from the interacting part. We will use the same number of degrees as freedom as both the non-interacting and interacting part, so that they match.

```{code-cell} ipython3
guess = tb.utils.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
mf_sol = solvers.solver(_model, guess, nk=18, optimizer_kwargs={'M':0})
full_sol = tb.tb.add_tb(h_0, mf_sol)
```

After we have defined the guess, we feed it together with the model into the meanfield solver. The meanfield solver will return a hopping dictionary with the meanfield approximation. We can then add this solution to the non-interacting part to get the full solution. In order to get the solution, we specified the number of k-points to be used in the calculation. This refers to the k-grid used in the Brillouin zone for the density matrix.

## Creating a phase diagram of the gap

We can now create a phase diagram of the gap of the interacting solution. In order to calculate the gap we first create a function which takes a hopping dictionary and a Fermi energy and returns the indirect gap. The gap is defined as the difference between the highest occupied and the lowest unoccupied energy level. We will use a dense k-grid to calculate the gap. In order to obtain the Hamiltonian on a dense k-grid, we use the `tb_to_khamvector` function from the `transforms` module.

```{code-cell} ipython3
def compute_gap(h, fermi_energy=0, nk=100):
    kham = tb.transforms.tb_to_khamvector(h, nk, ks=None)
    vals = np.linalg.eigvalsh(kham)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)
```

Now that we can calculate the gap, we create a phase diagram of the gap as a function of the Hubbard interaction strength $U$ and the nearest neighbor interaction strength $V$. We vary the onsite Hubbard interactio $U$ strength from $0$ to $2$ and the nearest neighbor interaction strength $V$ from $0$ to $1.5$.

```{code-cell} ipython3
def compute_phase_diagram(Us, Vs, int_builder, h_0):
    gap = []
    mf_sols = []
    for U in Us:
      gap_U = []
      mf_sols_U = []
      for V in Vs:
        params = dict(U=U, V=V)
        h_int = kwant_utils.builder_to_tb(int_builder, params)
        _model = model.Model(h_0, h_int, filling=2)
        converged=False
        while not converged:
          guess = tb.utils.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
          try:
            mf_sol = solvers.solver(_model, guess, nk=18, optimizer_kwargs={'M':0})
            converged=True
          except:
            converged=False
        gap_U.append(compute_gap(tb.tb.add_tb(h_0, mf_sol), fermi_energy=0, nk=300))
        mf_sols_U.append(mf_sol)
      guess = None
      gap.append(gap_U)
      mf_sols.append(mf_sols_U)
    return np.asarray(gap, dtype=float), np.asarray(mf_sols)
```
We chose to initialize a new guess for each $U$ value, but not for each $V$ value. Instead, for consecutive $V$ values we use the previous mean-field solution as a guess. We do this because the mean-field solution is expected to be smooth in the interaction strength and therefore by using an inspired guess we can speed up the calculation.

We can now compute the phase diagram and plot it.

```{code-cell} ipython3
Us = np.linspace(0, 4, 5)
Vs = np.linspace(0, 1.5, 5)
gap, mf_sols = compute_phase_diagram(Us, Vs, int_builder, h_0)
plt.imshow(gap.T, extent=(Us[0], Us[-1], Vs[0], Vs[-1]), origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('V')
plt.ylabel('U')
plt.title('Gap')
plt.show()
```

This phase diagram has gap openings at the same places as shown in the [literature](https://arxiv.org/abs/1204.4531).

## Order parameters

We might also want to calculate order parameters of the mean field solution. From literature we know to expect a charge density wave (CDW) and a spin density wave (SDW) in the mean field solution in different regions of the phase diagram. Here we show how to create both the CDW and SDW order parameters and evaluate them for the mean field solution in the phase diagram.

### Charge density wave

We first start with the CDW order parameter. In CDW the spins on different sublattices have opposite signs. This means that our order parameter can be constructed as:

```{code-cell} ipython3
sz = np.array([[1, 0], [0, -1]])
cdw_order_parameter = {}
cdw_order_parameter[(0,0)] = np.kron(sz, np.eye(2))
```

We choose a point in the phase diagram where we expect there to be a CDW phase and calculate the expectation value with the CDW order parameter. In order to do this we first construct the density matrix from the mean field solution.

```{code-cell} ipython3
params = dict(U=0, V=2)
h_int = kwant_utils.builder_to_tb(int_builder, params)
_model = model.Model(h_0, h_int, filling=2)
guess = tb.utils.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
mf_sol = solvers.solver(_model, guess, nk=18, optimizer_kwargs={'M':0})
full_sol = tb.tb.add_tb(h_0, mf_sol)

rho, _ = mf.construct_density_matrix(full_sol, filling=2, nk=40)
expectation_value = observables.expectation_value(rho, cdw_order_parameter)
print(expectation_value)
```

We can also perform the same calculation over the complete phase diagram where we calculated the gap earlier:

```{code-cell} ipython3
expectation_value_list = []
for mf_sol in mf_sols.flatten():
    rho, _ = mf.construct_density_matrix(tb.tb.add_tb(h_0, mf_sol), filling=2, nk=40)
    expectation_value = observables.expectation_value(rho, cdw_order_parameter)
    expectation_value_list.append(expectation_value)
```

```{code-cell} ipython3
expectation_value_list = np.asarray(expectation_value_list).reshape(mf_sols.shape)
plt.imshow(np.abs(expectation_value_list.T.real), extent=(Us[0], Us[-1], Vs[0], Vs[-1]), origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('V')
plt.ylabel('U')
plt.title('Charge Density Wave Order Parameter')
plt.show()
```
