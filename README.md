# pymf

`pymf` is a Python package for numerical self-consistent mean-field calculations in tight-binding systems using Hartree-Fock approximation. Its interface is designed to allow users to control cost functions and numerical solvers. Additionally, it has an interface with [`Kwant`](https://kwant-project.org/) for simple construction of tight-binding models.

## Installation

```
pip install pymf
```

## Usage

Finding a mean-field groundstate Hamiltonian is a 4-step process. We exemplify with the one-dimensional Hubbard model with two atoms per unit cell.

1. Define the non-interacting and interacting part of the Hamiltonian separately as hopping dictionaries.
```python
# Hopping matrix
hopp = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
# Non-interacting Hamiltonian
h_0 = {(0,): hopp + hopp.T.conj(), (1,): hopp, (-1,): hopp.T.conj()}
# Interacting Hamiltonian
U=2
s_x = np.array([[0, 1], [1, 0]])
h_int = {(0,): U * np.kron(np.eye(2), s_x),}
```
2. Import `pymf` and combine the non-interacting and interacting Hamiltonians into a `Model` object.
```python
# Number of electrons per unit cell
filling = 2
# Define model
model = pymf.Model(h_0, h_int, filling)
```
3. Provide a starting guess and the number of k-points to use the `solver` function and find the mean-field Hamiltonian.
```python
# Generate a random guess
guess = pymf.generate_guess(frozenset(h_int), ndof=4)
# Compute groundstate Hamiltonian
mf_sol = pymf.solver(model, guess, nk=nk)
```
4. Add the mean-field correction to the non-interacting part to calculate the total Hamiltonian.
```python
h_mf = pymf.add_tb(h_0, mf_sol)
```

# Citing `pymf`

We provide `pymf` as a free software under BSD license. If you have used `pymf` for work that has lead to a scientific publication, please mention the fact that you used it explicitly in the text body. For example, you may add

> the numerical calculations were performed using the pymf code

to the description of your numerical calculations. In addition, we ask you to cite the Zenodo repository:

```
zenodo red here
```
