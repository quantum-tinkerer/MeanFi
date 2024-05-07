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

# 1d Hubbard

To simulate how the package can be used to simulate infinite systems, we show how to use it with a tight-binding model in 1 dimension.
We exemplify this by computing the ground state of an infinite spinful chain with onsite interactions.
First, the basic imports are done.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pymf
```

After this, we start by constructing the non-interacting Hamiltonian. As we expect the ground state to be an antiferromagnet, we build a two-atom cell. We name the two sublattices, $A$ and $B$. The Hamiltonian is then:
$$
H_0 = \sum_i c_{i, B}^{\dagger}c_{i, A} + c_{i, A}^{\dagger}c_{i+1, B} + h.c.
$$
We write down the spinful part by simply taking $H_0(k) \otimes \mathbb{1}$.

To translat ethis Hamiltonian into a tight-binding model, we specify the hopping vectors together with the hopping amplitudes. We ensure that the Hamiltonian is hermitian by letting the hopping amplitudes from $A$ to $B$ be the complex conjugate of the hopping amplitudes from $B$ to $A$.

```{code-cell} ipython3
hopp = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
h_0 = {(0,): hopp + hopp.T.conj(), (1,): hopp, (-1,): hopp.T.conj()}
```

We verify this tight-binding model by plotting the band structure and observing the two bands due to the Brillouin zone folding. In order to do this we transform the tight-binding model into a Hamiltonian on a k-point grid using the `tb_to_khamvector` and then diagonalize it.

```{code-cell} ipython3
# Set number of k-points
nk = 100
ks = np.linspace(0, 2*np.pi, nk, endpoint=False)
hamiltonians_0 = pymf.tb_to_khamvector(h_0, nk, ks=ks)

vals, vecs = np.linalg.eigh(hamiltonians_0)
plt.plot(ks, vals, c="k")
plt.xticks([0, np.pi, 2 * np.pi], ["$0$", "$\pi$", "$2\pi$"])
plt.xlim(0, 2 * np.pi)
plt.ylabel("$E - E_F$")
plt.xlabel("$k / a$")
plt.show()

```

After confirming that the non-interacting part is correct, we can set up the interacting Hamiltonian. We define the interaction similarly as a tight-binding dictionary. To keep the physics simple, we let the interaction be onsite only, which gives us the following interaction matrix:

$$
H_{int} =
\left(\begin{array}{cccc}
    U & U & 0 & 0\\
    U & U & 0 & 0\\
    0 & 0 & U & U\\
    0 & 0 & U & U
\end{array}\right)~.
$$

This is simply constructed by writing:

```{code-cell} ipython3
U=0.5
h_int = {(0,): U * np.kron(np.ones((2, 2)), np.eye(2)),}
```

In order to find a mean-field solution, we combine the non interacting Hamiltonian with the interaction Hamiltonian and the relevant filling into a `Model` object. We then generate a starting guess for the mean-field solution and solve the model using the `solver` function. It is important to note that the guess will influence the possible solutions which the `solver` can find in the mean-field procedure. The `generate_guess` function generates a random Hermitian tight-binding dictionary, with the keys provided as hopping vectors and the values of the size as specified.

```{code-cell} ipython3
filling = 2
full_model = pymf.Model(h_0, h_int, filling)
guess = pymf.generate_guess(frozenset(h_int), ndof=4)
mf_sol = pymf.solver(full_model, guess, nk=nk)
```

The `solver` function returns only the mean-field correction to the non-interacting Hamiltonian. To get the full Hamiltonian, we add the mean-field correction to the non-interacting Hamiltonian. To take a look at whether the result is correct, we first do the mean-field computation for a wider range of $U$ values and then plot the gap as a function of $U$.
