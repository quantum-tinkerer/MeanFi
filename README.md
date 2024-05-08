# `pymf`

## What is `pymf`?

`pymf` is a Python package that performs self-consistent Hartree-Fock calculations on tight-binding models.
It aims to find the groundstate of a Hamiltonian with density-density interactions

$$
\hat{H} = \hat{H_0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ij} v_{ij} \hat{n}_i \hat{n}_j,
$$

and computes the mean-field correction $\hat{V}_{\text{MF}}$ which approximates the interaction term:

$$
\hat{V} \approx \hat{V}_{\text{MF}} = \sum_{ij} \tilde{v}_{ij} c^\dagger_{i} c_{j}.
$$

For more details, refer to the [theory overview](docs/source/documentation/mf_notes.md) and [algorithm description](docs/source/documentation/algorithm.md).

## How to use `pymf`?

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
import pymf

#Define
h_0 = {(0,) : onsite, (1,) : hopping, (-1,) : hopping.T.conj()}
h_int = {(0,) : onsite_interaction}
model = pymf.Model(h_0, h_int, filling=2)

#Guess
guess = pymf.generate_guess(guess_hopping_keys, ndof)

#Solve
mf_correction = pymf.solver(model, guess)
h_mf = pymf.add_tb(h_0, mf_correction)
```

For more details and examples on how to use the package, we refer to the [tutorials](docs/source/tutorial/hubbard_1d.md).

## Why `pymf`?

Here is why you should use `pymf`:

* Simple

    The workflow is straightforward.
    Interface with `Kwant` allows easy creation of complicated tight-binding systems and interactions.

* Extensible

    `pymf`'s code is structured to be easy to understand, modify and extend.

* Optimized numerical workflow

    Introduces minimal overhead to the calculation of the mean-field Hamiltonian.


## What `pymf` doesn't do (yet)

Here are some features that are not yet implemented but are planned for future releases:

- **Superconductive order parameters**. Mean-field Hamiltonians do not include pairing terms.
- **General interactions**. We allow only density-density interactions (e.g. Coulomb) which can be described by a second-order tensor.
- **Temperature effects**. Density matrix calculations are done at zero temperature.

## Installation

```
pip install pymf
```

## Citing `pymf`

If you have used `pymf` for work that has led to a scientific publication, please cite us as:

```bibtex
@misc{pymf,
  author = {Vilkelis, Kostas and Zijderveld,  R. Johanna and Akhmerov, Anton R. and Manesco, Antonio L.R.},
  doi = {10.5281/zenodo.11149850},
  month = {5},
  title = {pymf},
  year = {2024}
}
```
