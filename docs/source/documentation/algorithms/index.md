# Algorithm overview

A `Model` supplies the bare Hamiltonian, interaction, filling, temperature and
optional reference density. Start from a mean-field guess:

```python
solution = meanfi.solver(model, model.random_meanfield(rng=0))
```

(one-solve)=
## MeanFi calculation loop

After evaluating the initial guess, SCF repeats this density update:

$$
\begin{array}{ccccc}
\rho_n & \xrightarrow{\text{1. Build Hamiltonian}} & H_n
       & \xrightarrow{\text{2. Find }\mu} & (H_n,\mu_n) \\
\uparrow\,\text{repeat} &&&& \downarrow\,\text{3. Density at }k \\
\rho_{n+1} & \xleftarrow{\text{5. SCF update}} & \rho_n^{\mathrm{out}}
       & \xleftarrow{\text{4. Integrate}} & \rho_n(k)
\end{array}
$$

Here $\rho_n$ is the trial density and $\rho_n^{\mathrm{out}}$ its calculated
update. Step 5 stops when their difference meets the SCF target; otherwise EDIIS
chooses the next trial density and the loop repeats.

## 1. Build the Hamiltonian

The interaction acts on the density relative to the optional reference:

$$H_n=h_0+W[\rho_n-\rho_{\mathrm{ref}}].$$

Only the interaction's required density entries are needed. MeanFi represents
these with independent real variables, respecting Hermiticity, pairing and
any imposed [spatial symmetries](parametrization.md).

## 2. Find the chemical potential

At fixed filling, solve $N(\mu)=\mathrm{filling}$ using `filling_residual`.
Charge probes compute the occupations or physical diagonal entries needed for
this search. FermiSimplex finishes charge refinement before its density stage;
density refinement does not restart the charge solve.

`density_matrix(model)` performs steps 2–4. `density_matrix_at_mu(model, mu)`
skips step 2 and uses the supplied chemical potential. Its filling comes from
already available occupations or density entries, or is `None`. The filling
residual and charge-integration estimate are [separate quantities](accuracy.md).

## 3. Evaluate density at each momentum

For a sampled momentum,

$$\rho_n(k)=f\!\left(H_n(k)-\mu_n Q\right),$$

where $Q=I$ for normal models and $Q=\operatorname{diag}(I,-I)$ for BdG models.
Dense diagonalization supplies occupied eigenstates; sparse AAA approximates
the Fermi matrix function and extracts selected inverse entries. Both compute
the requested density entries. See [matrix functions](matrix_functions.md).

## 4. Integrate over momentum

Real-space density entries are Fourier integrals over the Brillouin zone:

$$
(\rho_n^{\mathrm{out}})_R=\frac{1}{|\mathrm{BZ}|}\int_{\mathrm{BZ}}
e^{ik\cdot R}\rho_n(k)\,dk.
$$

| Calculation | Integration |
| --- | --- |
| Dense, normal, zero temperature | `FermiSimplex()` by default |
| Dense, positive temperature, normal or BdG | `UniformGrid()` by default |
| Periodic BdG at zero temperature | Explicit `UniformGrid(nk=...)` |
| Sparse at positive temperature | Explicit `UniformGrid(nk=...)`, using AAA |

`FermiSimplex` integrates interpolated spectra and density contributions on an
adaptive simplex mesh. Adaptive `UniformGrid` doubles each axis and compares
coarse and fine integrals; it requires positive temperature and dense
diagonalization. There is no shifted validation grid.

`nk` prescribes a **total point count** and disables refinement and integration
error estimation. For example, `UniformGrid(nk=4)` uses $2\times2$ points in two
dimensions; other requests round up to an isotropic $n^d$ grid. FermiSimplex
counts boundary vertices and its mesh construction can overshoot `nk`.

Without `nk`, `initial_nk` sets the starting mesh: defaults are $5^d$ simplex
vertices or $4^d$ grid points. These are starting sizes, not accuracy guarantees.
Finite systems have no momentum integral and ignore grid settings with a warning.

## 5. Check convergence and update the density

SCF stops when the largest active-density residual meets `tol` (default `1e-3`).
Otherwise, `EnergyDIIS()` chooses a convex combination of previous densities by
minimizing **internal energy**, then returns to step 1. It never switches methods.
`LinearMixing` and `AndersonMixing` are explicit alternatives through `scf=`.

Iteration exhaustion raises `NoConvergence`; `exception.result` contains the
last evaluated state. Restarting with another method is a user decision.

## Result and optional free energy

`solution.mean_field` is the input correction that produced `solution.density`.
Applying the interaction to that density gives the next correction; the two
agree to SCF accuracy at convergence.

Entropy is omitted by default. `compute_free_energy=True` evaluates it after
SCF terminates, also for a valid partial result. Then $F=U-kT\,S$, with energies
and entropy per cell per physical orbital. Entropy never participates in EDIIS
or the convergence tests.

```{toctree}
:maxdepth: 1

accuracy.md
matrix_functions.md
parametrization.md
```
