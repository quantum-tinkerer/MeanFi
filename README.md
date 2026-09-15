# MeanFi

MeanFi solves self-consistent tight-binding models with density-density
interactions, at zero or finite temperature, including superconducting BdG
models. Define the Hamiltonian and interaction, supply an initial mean field,
and solve:

```python
import numpy as np
import meanfi

# A spinful chain: nearest-neighbor hopping and on-site repulsion.
hopping = -np.eye(2)
h_0 = {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.T.conj()}
h_int = {(0,): np.array([[0., 1.], [1., 0.]])}
model = meanfi.Model(h_0, h_int, filling=1, kT=0.2)

guess = model.random_meanfield(rng=0, scale=0.1)
result = meanfi.solver(model, guess)
assert result.converged
h_mf = model.hamiltonian_from_meanfield(result.mean_field)
print(result.internal_energy, result.entropy, result.free_energy)
```

Dictionary keys are lattice displacements; each value is an orbital matrix.
`EnergyDIIS()` is the default solver and mixes densities using internal energy.
Final results report energies and entropy per
cell per physical orbital, with entropy in units of Boltzmann's constant:
`free_energy = internal_energy - kT * entropy`. Filling counts electrons per
cell. A BdG Hamiltonian of size 2N still has N physical orbitals.

See the [tutorials](https://meanfi.readthedocs.io/en/latest/tutorial/hubbard_1d.html)
and [API walkthrough](https://gitlab.kwant-project.org/qt/meanfi/-/blob/main/examples/api_walkthrough.py) for densities, observables,
reference subtraction, restarts, BdG, sparse calculations, and Kwant conversion.
The [theory](https://meanfi.readthedocs.io/en/latest/documentation/theory/index.html)
and [algorithms](https://meanfi.readthedocs.io/en/latest/documentation/algorithms/index.html)
explain the physics and numerical methods.

## Integration

Normal zero-temperature calculations default to `FermiSimplex()`, backed by
FermiSimplex. Dense finite-temperature calculations default to `UniformGrid()` with
direct diagonalization, automatic refinement, and shifted-grid validation.

```python
# Choose a total point count or request accuracy-controlled integration.
integration = meanfi.UniformGrid(nk=4096)
integration = meanfi.UniformGrid(density_matrix_tol=1e-5, charge_tol=1e-6)
```

An explicit `nk` fixes the mesh and cannot be combined with integration targets.
For example, `nk=4096` gives 64² periodic points in 2D. Prescribed meshes do not
estimate integration error. The solver's `tol` still controls filling and SCF
convergence. Zero-temperature BdG calculations require an explicit mesh.

Sparse finite-temperature calculations use
`UniformGrid(nk=..., matrix_function=RationalFOE())`. AAA shares poles and sparse
factorizations between density and entropy; MUMPS supplies selected inverse
entries. See the [integration guide and migration notes](https://meanfi.readthedocs.io/en/latest/documentation/algorithms/integration_families.html)
for supported combinations, mesh rounding, and changes from previous APIs.

## Installation

This development revision supports Python 3.11–3.13. Use [Pixi](https://pixi.sh/)
to install the pinned dependencies and native compiler:

```bash
git clone https://gitlab.kwant-project.org/qt/meanfi.git
cd meanfi
pixi install --locked
pixi run python -c "import meanfi; print(meanfi.__version__)"
```

FermiSimplex currently builds from a pinned Git revision; its required API is
newer than its PyPI release. It includes its own AdaptiveSimplex mesh engine.
Installing this checkout with `pip install .` also works when Git and a suitable
C++ compiler are available. See the [development guide](https://meanfi.readthedocs.io/en/latest/development.html)
for native requirements and the remaining PyPI release prerequisite.

Optional extras are `pip install ".[sparse]"` for MUMPS and
`pip install ".[kwant]"` for conversion helpers. The `test-sparse` Pixi environment
supplies both packages and their native libraries. The default dense and
FermiSimplex paths do not require MUMPS.

```bash
pixi run -e test-py312 tests
pixi run -e test-py312 check-install  # Fresh Python environment and dependencies.
pixi run -e docs docs-build
```

The [design document](https://gitlab.kwant-project.org/qt/meanfi/-/blob/main/DESIGN.md) describes the code's structure and contracts;
[AGENTS.md](https://gitlab.kwant-project.org/qt/meanfi/-/blob/main/AGENTS.md) records the coding guidelines.

## Citing MeanFi

```bibtex
@misc{meanfi,
  author = {Vilkelis, Kostas and Zijderveld, R. Johanna and Akhmerov, Anton R. and Manesco, Antonio L.R.},
  doi = {10.5281/zenodo.11149850},
  month = {5},
  title = {MeanFi},
  year = {2024}
}
```
