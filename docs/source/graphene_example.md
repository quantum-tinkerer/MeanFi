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

# Graphene with interactions

## The physics

This tutorial serves as a simple example of using the mean-field algorithm in two dimensions in combination with using Kwant. We will consider a simple tight-binding model of graphene with a Hubbard interaction. The graphene system is first created using Kwant. For the basics of creating graphene with Kwant we refer to [this](https://kwant-project.org/doc/1/tutorial/graphene) tutorial.

We begin with the basic imports

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pymf
```

##  Preparing the model

We first translate this model from a Kwant system to a tight-binding dictionary. In the tight-binding dictionary the keys denote the hoppings while the values are the hopping amplitudes.

```{code-cell} ipython3
import pymf.kwant_helper.kwant_examples as kwant_examples
import pymf.kwant_helper.utils as kwant_utils

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
_model = pymf.Model(h_0, h_int, filling=2)
```

To start the mean-field calculation we also need a starting guess. We will use our random guess generator for this. It creates a random Hermitian hopping dictionary based on the hopping keys provided and the number of degrees of freedom specified. As we don't expect the mean-field solution to contain terms more than the hoppings from the interacting part, we can use the hopping keys from the interacting part. We will use the same number of degrees as freedom as both the non-interacting and interacting part, so that they match.

```{code-cell} ipython3
guess = pymf.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
mf_sol = pymf.solver(_model, guess, nk=18, optimizer_kwargs={'M':0})
full_sol = pymf.add_tb(h_0, mf_sol)
```

After we have defined the guess, we feed it together with the model into the mean-field solver. The mean-field solver will return a hopping dictionary with the mean-field approximation. We can then add this solution to the non-interacting part to get the full solution. In order to get the solution, we specified the number of k-points to be used in the calculation. This refers to the k-grid used in the Brillouin zone for the density matrix.

## Creating a phase diagram of the gap

We can now create a phase diagram of the gap of the interacting solution. In order to calculate the gap we first create a function which takes a hopping dictionary and a Fermi energy and returns the indirect gap. The gap is defined as the difference between the highest occupied and the lowest unoccupied energy level. We will use a dense k-grid to calculate the gap. In order to obtain the Hamiltonian on a dense k-grid, we use the `tb_to_kgrid` function from pymf.

```{code-cell} ipython3
def compute_gap(h, fermi_energy=0, nk=100):
    kham = pymf.tb_to_kgrid(h, nk)
    vals = np.linalg.eigvalsh(kham)

    emax = np.max(vals[vals <= fermi_energy])
    emin = np.min(vals[vals > fermi_energy])
    return np.abs(emin - emax)
```

Now that we can calculate the gap, we create a phase diagram of the gap as a function of the Hubbard interaction strength $U$ and the nearest neighbor interaction strength $V$. We vary the onsite Hubbard interactio $U$ strength from $0$ to $2$ and the nearest neighbor interaction strength $V$ from $0$ to $1.5$.

```{code-cell} ipython3
def gap_and_mf_sol(U, V, int_builder, h_0):
    params = dict(U=U, V=V)
    h_int = kwant_utils.builder_to_tb(int_builder, params)
    _model = pymf.Model(h_0, h_int, filling=2)
    guess = pymf.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
    mf_sol = pymf.solver(_model, guess, nk=18, optimizer_kwargs={'M':0})
    gap = compute_gap(pymf.add_tb(h_0, mf_sol), fermi_energy=0, nk=300)
    return gap, mf_sol
```

```{code-cell} ipython3
def compute_phase_diagram(Us, Vs, int_builder, h_0):
  gaps = []
  mf_sols = []
  for U in Us:
    for V in Vs:
      gap, mf_sol = gap_and_mf_sol(U, V, int_builder, h_0)
      gaps.append(gap)
      mf_sols.append(mf_sol)
  gaps = np.asarray(gaps, dtype=float).reshape((len(Us), len(Vs)))
  mf_sols = np.asarray(mf_sols).reshape((len(Us), len(Vs)))
  return gaps, mf_sols
```
We chose to initialize a new guess for each new $U$ and $V$ value. For certain mean-field problems, one might want to reuse the mean-field solution of a nearby parameter as the next guess in order to speed up computations. However, as the size of this system is still small, we can afford to initialize a new guess for each new $U$ and $V$ value.

We can now compute the phase diagram and then plot it

```{code-cell} ipython3
Us = np.linspace(0, 4, 10)
Vs = np.linspace(0, 1.5, 10)
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

We choose a point in the phase diagram where we expect there to be a CDW phase and calculate the expectation value with the CDW order parameter. In order to do this we first construct the density matrix from the mean field solution. We perform this calculation over the complete phase diagram where we calculated the gap earlier:

```{code-cell} ipython3
cdw_list = []
for mf_sol in mf_sols.flatten():
    rho, _ = pymf.construct_density_matrix(pymf.add_tb(h_0, mf_sol), filling=2, nk=40)
    expectation_value = pymf.expectation_value(rho, cdw_order_parameter)
    cdw_list.append(expectation_value)
```

```{code-cell} ipython3
cdw_list = np.asarray(cdw_list).reshape(mf_sols.shape)
plt.imshow(np.abs(cdw_list.T.real), extent=(Us[0], Us[-1], Vs[0], Vs[-1]), origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('V')
plt.ylabel('U')
plt.title('Charge Density Wave Order Parameter')
plt.show()
```

### Spin density wave

To check the other phase we expect in the graphene phase diagram, we construct a spin density wave order parameter. In our chosen graphene system the spin density wave has $SU(2)$ symmetry. This means that we need to sum over the pauli matrices when constructing this order parameter. We can construct the order parameter as:

```{code-cell} ipython3
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])

s_list = [sx, sy, sz]

order_parameter_list = []
for s in s_list:
    order_parameter = {}
    order_parameter[(0,0)] =  np.kron(sz, s)
    order_parameter_list.append(order_parameter)
```

Then, similar to what we did in the CDW phase, we calculate the expectation value of the order parameter with the density matrix of the mean field solution over the complete phase diagram. The main subtlety here is that we need to sum over expectation value of all SDW direction order parameters defined in order to get the total spin density wave order parameter.

```{code-cell} ipython3
sdw_list = []
for mf_sol in mf_sols.flatten():
    rho, _ = pymf.construct_density_matrix(pymf.add_tb(h_0, mf_sol), filling=2, nk=40)
    expectation_values = []
    for order_parameter in order_parameter_list:
        expectation_value = pymf.expectation_value(rho, order_parameter)
        expectation_values.append(expectation_value)

    sdw_list.append(np.sum(np.array(expectation_values)**2))
```

```{code-cell} ipython3
sdw_list = np.asarray(sdw_list).reshape(mf_sols.shape)
plt.imshow(np.abs(sdw_list.T.real), extent=(Us[0], Us[-1], Vs[0], Vs[-1]), origin='lower', aspect='auto')
plt.colorbar()
plt.xlabel('V')
plt.ylabel('U')
plt.title('Spin Density Wave Order Parameter')
plt.show()
```

## Full phase diagram

Finally, we can combine the gap, CDW and SDW phase diagrams into one plot. We naively do this by plotting the order parameter of CDW minus the order parameter of SDW. Furthermore, we normalize the gap such that it is between $0$ and $1$ and can thus be used for the transparency.

```{code-cell} ipython3
import matplotlib.ticker as mticker
normalized_gap = gap/np.max(gap)
plt.imshow(np.abs(cdw_list.T.real)-np.abs(sdw_list.T.real), extent=(Us[0], Us[-1], Vs[0], Vs[-1]), origin='lower', aspect='auto', cmap="coolwarm", alpha=normalized_gap.T)
plt.colorbar(ticks=[-1.75, 0, 1.75], format=mticker.FixedFormatter(['SDW', '0', 'CDW']), label='Order parameter', extend='both')
plt.xlabel('V')
plt.ylabel('U')
plt.show()
```
