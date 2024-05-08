# What is `pymf`?

`pymf` is a Python package that performs self-consistent mean-field calculations on tight-binding models.
It aims to solve the following interacting many-body Hamiltonians:
$$
\hat{H} = \hat{H_0} + \hat{V} = \sum_{ij} h_{ij} c^\dagger_{i} c_{j} + \frac{1}{2} \sum_{ij} v_{ij} \hat{n}_i \hat{n}_j,
$$
by finding the mean-field correction $\hat{V}_{\text{MF}}$ which approximates the interaction term:

$$
\hat{V} \approx \hat{V}_{\text{MF}} = \sum_{ij} \tilde{v}_{ij} c^\dagger_{i} c_{j}.
$$

For more details on the theory, we refer to the documentation.
# How to use `pymf`?

The calculation of a mean-field Hamiltonian is a simple 3-step process:

1. **Define**

    To specify the interacting problem, use a `Model` object which collects:
    - Non-interacting Hamiltonian as a tight-binding dictionary.
    - Interaction Hamiltonian as a tight-binding dictionary.
    - Particle filling number in the unit cell.
2. **Guess**

    Construct a starting guess for the mean-field correction as a tight-binding dictionary.

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

For more details and examples on how to use the package, we refer to the tutorials.

# Why `pymf`?

Here is why you should use `pymf`:

* Simple

    The workflow is simple and straightforward.
    Interface with `Kwant` allows easy creation of complicated tight-binding systems and interactions.

* Extensible

    `pymf`'s code is structured to be easy to understand, modify and extend.

* Sufficiently time-effective

    Introduces minimal overhead to the calculation of the mean-field Hamiltonian.


# What `pymf` doesn't do (yet)

Here are some features that are not yet implemented but are planned for future releases:

- **Superconductive order parameters**. Mean-field Hamiltonians do not include pairing terms.
- **General interactions**. We allow only density-density interactions (e.g. Coulomb) which can be described by a second-order tensor.
- **Temperature effects**. Density matrix calculations are done at zero temperature.

# Installation

```
pip install pymf
```

# Citing `pymf`

We provide `pymf` as a free software under BSD license. If you have used `pymf` for work that has lead to a scientific publication, please mention the fact that you used it explicitly in the text body. For example, you may add

> the numerical calculations were performed using the pymf code

to the description of your numerical calculations. In addition, we ask you to cite the Zenodo repository:

```
zenodo red here
```
