---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 1D Hubbard model

This example solves a chain with hopping $t=1$ and onsite repulsion $U$ between
opposite spins. A two-site cell allows antiferromagnetic order. The basis is
$(A\uparrow,A\downarrow,B\uparrow,B\downarrow)$.

## Define a tight-binding model

A tight-binding dictionary maps cell displacements to matrices. `(0,)` contains
terms within one cell; `(1,)` and `(-1,)` connect neighboring cells. Hermitian
hoppings obey $H_{-R}=H_R^\dagger$.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import meanfi
import numpy as np

hopping = np.kron([[0, 1], [0, 0]], np.eye(2))
h_0 = {(0,): hopping + hopping.T, (1,): hopping, (-1,): hopping.T}
sigma_x = np.array([[0, 1], [1, 0]])
U = 2.0
h_int = {(0,): U * np.kron(np.eye(2), sigma_x)}
model = meanfi.Model(h_0, h_int, filling=2)
```

The interaction acts only between opposite spins on the same site.
`filling=2` specifies two electrons per four-orbital cell.

## Solve and inspect the result

```{code-cell} ipython3
result = meanfi.solver(model, model.random_meanfield(rng=0, scale=0.05))
print(f"Converged: {result.converged}")
print(f"SCF residual: {result.errors.scf_residual:.2e}")
print(f"Chemical potential: {result.mu:.6f}")
print(f"Internal energy per orbital: {result.internal_energy:.6f}")
```

The default solver uses EDIIS and `tol=1e-3`. A scalar `tol` sets the numerical
targets through the default tolerance policy. For example, pass `tol=1e-5` when
you need a tighter solution. Initial guesses select among possible mean-field
states; they are not a guarantee of the lowest energy.

`result.mean_field` is the correction to the bare Hamiltonian. The chemical
potential is returned separately, so subtract `result.mu` when plotting bands.
Entropy and free energy are optional: request them with `compute_free_energy=True`.

```{code-cell} ipython3
h = model.hamiltonian_from_meanfield(result.mean_field)
k = np.linspace(0, 2 * np.pi, 200, endpoint=False)
bare_bands = np.linalg.eigvalsh(meanfi.tb_to_kgrid(h_0, (len(k),)))
bands = np.linalg.eigvalsh(meanfi.tb_to_kgrid(h, (len(k),)))

fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
for ax, energies, title in zip(axes, [bare_bands, bands - result.mu], ["U = 0", "U = 2"]):
    ax.plot(k, energies, color="black")
    ax.set(title=title, xlabel="k", ylabel=r"$E-\mu$", xlim=(0, 2 * np.pi))
plt.show()
```

The interacting solution opens a gap at half filling.

## Vary the interaction

```{code-cell} ipython3
Us = np.linspace(0, 4, 30)
gaps = []
for U in Us:
    interaction = {(0,): U * np.kron(np.eye(2), sigma_x)}
    model = meanfi.Model(h_0, interaction, filling=2)
    solution = meanfi.solver(model, model.random_meanfield(rng=0, scale=0.05))
    h = model.hamiltonian_from_meanfield(solution.mean_field)
    bands = np.linalg.eigvalsh(meanfi.tb_to_kgrid(h, (400,)))
    gaps.append(max(0.0, bands[:, 2].min() - bands[:, 1].max()))

plt.plot(Us, gaps, color="black")
plt.xlabel("U / t")
plt.ylabel("Gap / t")
plt.show()
```

Small gaps may be unresolved at the default tolerance; this plot should not be
used to infer a critical interaction strength. All points are computed afresh.
