# `MeanFi`

## What is `MeanFi`?

`MeanFi` is a Python package that performs self-consistent Hartree-Fock calculations on tight-binding models.
It aims to find the groundstate of a Hamiltonian with density-density interactions

$$
\hat{H} = \hat{H_0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ij} v_{ij} \hat{n}_i \hat{n}_j,
$$

and computes the mean-field correction $\hat{V}_{\text{MF}}$ which approximates the interaction term:

$$
\hat{V} \approx \hat{V}\_{\text{MF}} =\sum_{ij} \tilde{v}\_{ij} c^{\dagger}\_{i} c_{j}.
$$

For more details, refer to the [theory overview](https://meanfi.readthedocs.io/en/latest/documentation/mf_notes.html) and [algorithm description](https://meanfi.readthedocs.io/en/latest/documentation/algorithm.html).

## How to use `MeanFi`?

The calculation of a mean-field Hamiltonian is a simple 3-step process:

1. **Define**

    To specify the interacting problem, use a `Model` object which collects:
    - Non-interacting Hamiltonian as a tight-binding dictionary.
    - Interaction Hamiltonian as a tight-binding dictionary.
    - Particle filling number in the unit cell.
2. **Guess**

    Construct a starting guess for the mean-field correction.

3. **Solve**

    Solve for the mean-field correction using the `solver` function and add it to the non-interacting part to obtain the total mean-field Hamiltonian.

```python
import meanfi

#Define
h_0 = {(0,) : onsite, (1,) : hopping, (-1,) : hopping.T.conj()}
h_int = {(0,) : onsite_interaction}
model = meanfi.Model(h_0, h_int, filling=2)

#Guess
guess = meanfi.guess_tb(guess_hopping_keys, ndof)

#Solve
mf_correction = meanfi.solver(model, guess)
h_mf = meanfi.add_tb(h_0, mf_correction)
```

For more details and examples on how to use the package, we refer to the [tutorials](docs/source/tutorial/hubbard_1d.md).

## Why `MeanFi`?

Here is why you should use `MeanFi`:

* Simple

    The workflow is straightforward.
    Interface with `Kwant` allows easy creation of complicated tight-binding systems and interactions.

* Extensible

    `MeanFi`'s code is structured to be easy to understand, modify and extend.

* Optimized numerical workflow

    Introduces minimal overhead to the calculation of the mean-field Hamiltonian.


## What `MeanFi` doesn't do (yet)

Here are some features that are not yet implemented but are planned for future releases:

- **Superconducting order parameters**. Mean-field Hamiltonians do not include pairing terms.
- **General interactions**. We allow only density-density interactions (e.g. Coulomb) which can be described by a rank two tensor.
- **Temperature effects**. Density matrix calculations are done at zero temperature.

## Installation

```
pip install meanfi
```

## Citing `MeanFi`

If you have used `MeanFi` for work that has led to a scientific publication, please cite us as:

```bibtex
@misc{meanfi,
  author = {Vilkelis, Kostas and Zijderveld,  R. Johanna and Akhmerov, Anton R. and Manesco, Antonio L.R.},
  doi = {10.5281/zenodo.11149850},
  month = {5},
  title = {MeanFi},
  year = {2024}
}
```
