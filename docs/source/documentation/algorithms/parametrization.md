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
# Parametrization and symmetry reduction

The SCF loop is a fixed-point iteration on **real parameters**, not directly on a density-matrix dictionary.
This page explains the parametrization used by `model.scf_space` and how it connects to the SCF-loop step

:::{math}
\rho \longrightarrow h_{\mathrm{MF}}[\rho].
:::

The short version is:

:::{math}
\text{computed density entries}
\xrightarrow{\text{compress}}
y
\xrightarrow{\text{expand}}
\text{active density input}
\xrightarrow{h_{\mathrm{MF}}}
\text{mean-field correction}.
:::

Here `y` is the minimal real vector mixed by the SCF algorithm.

## Why a separate space exists

The full density matrix contains many entries that the mean-field update never uses.
The interaction dictionary `h_int` decides which real-space density entries can affect the mean-field Hamiltonian.
MeanFi calls these the **active entries**.

An active entry is a triple

:::{math}
(R,a,b),
:::

meaning the complex density value

:::{math}
z_i = \rho_{ab}(R).
:::

The ordered list of active entries is stored as density coordinates.
This list is only a layout: it can extract selected values from a tight-binding dictionary and put selected values back into tight-binding blocks.
It does not impose Hermiticity, particle-hole constraints, positivity, or SCF logic.

For normal-state models, the active entries are:

- exchange entries where `h_int(R)` has structural nonzeros;
- onsite diagonal density entries needed by the Hartree term.

For superconducting models, the active entries additionally include anomalous top-right BdG entries selected by the interaction structure.

## From complex entries to real variables

Suppose there are `n` active complex values:

:::{math}
z =
\begin{pmatrix}
z_1 \\
\vdots \\
z_n
\end{pmatrix}
\in \mathbb{C}^n.
:::

MeanFi stores them as a real vector

:::{math}
x =
\begin{pmatrix}
\operatorname{Re} z \\
\operatorname{Im} z
\end{pmatrix}
\in \mathbb{R}^{2n}.
:::

The SCF loop does not mix `x` directly.
After symmetry reduction, the allowed active densities are represented by

:::{math}
x = B y,
\qquad
B \in \mathbb{R}^{2n \times p},
\qquad
y \in \mathbb{R}^p.
:::

The vector `y` is the solver variable.
It is the object passed to linear mixing or Anderson mixing.

This is a linear SCF-variable space, not the full physical density-matrix manifold.
It does not enforce positivity, trace constraints beyond the fixed-filling density solve, or density-matrix representability.

## How symmetries produce `B`

All constraints are written as homogeneous real linear equations on `x`:

:::{math}
C x = 0.
:::

The columns of `B` form a basis for the nullspace:

:::{math}
\operatorname{im} B = \ker C.
:::

MeanFi applies three kinds of constraints.

### Hermiticity

Normal density entries obey

:::{math}
\rho_{ab}(R) = \rho_{ba}(-R)^*.
:::

If

:::{math}
i=(R,a,b),
\qquad
j=(-R,b,a),
:::

then Hermiticity gives

:::{math}
\operatorname{Re} z_i = \operatorname{Re} z_j,
\qquad
\operatorname{Im} z_i = -\operatorname{Im} z_j.
:::

For BdG calculations, this Hermiticity constraint is applied to the electron-electron block.

### Particle-hole structure in BdG

BdG anomalous entries live in the top-right block.
If the electron sector has `ndof` internal degrees of freedom, the anomalous value

:::{math}
F_{ab}(R)
:::

is stored as the coordinate

:::{math}
(R,a,\mathrm{ndof}+b).
:::

MeanFi imposes strict fermionic antisymmetry:

:::{math}
F_{ab}(R) = -F_{ba}(-R).
:::

For onsite scalar pairing this is self-paired, so there is no anomalous degree of freedom.

### Spatial and internal symmetries

A user symmetry acts on basis states as

:::{math}
g\lvert R,a\rangle
=
\sum_{s,c}
U_s[c,a]\,
\lvert A R+s,c\rangle.
:::

This representation covers ordinary point symmetries, orbital or spin rotations, antiunitary symmetries, and shifted spatial symmetries such as glides.

For normal density entries, the induced constraint is

:::{math}
\rho_{ab}(R)
=
\sum_{s,t,c,d}
U_s[c,a]^*
U_t[d,b]\,
\rho_{cd}(A R+t-s).
:::

For anomalous BdG entries, MeanFi uses pairing covariance:

:::{math}
F_{ab}(R)
=
\sum_{s,t,c,d}
U_s[c,a]
U_t[d,b]\,
F_{cd}(A R+t-s).
:::

Antiunitary symmetries conjugate the transformed density values before the real equations are assembled.

## Restricted active support

The active support is fixed by `h_int`.
MeanFi does not automatically enlarge it when a symmetry maps an active entry to a missing entry.

If a constraint maps an active entry outside the active support, that outside entry is treated as zero and a warning is emitted.
The constraint may therefore force the active entry, or a linear combination of active entries, to vanish.

This behavior is deliberate: the SCF space represents the variables that can affect the chosen mean-field map, not the full density matrix.

## Required density entries

The density solver also does not need to compute every active entry.
Once the basis `B` is known, MeanFi chooses enough active real-space entries to recover `y`.

Let `P` select the real and imaginary rows of the required complex entries.
MeanFi chooses entries so that

:::{math}
P B
:::

has full column rank.

The required complex entries are exposed by

```python
model.scf_space.required_realspace_entries()
```

Selected-density backends use exactly these entries.
For sparse selected-inverse calculations, this avoids computing Hermitian partners and other symmetry-related values that can be reconstructed from `B`.

## Compression and expansion

After a density solve, MeanFi has values for the required entries.
Those values are packed into `P x` and compressed to SCF parameters:

:::{math}
y = (P B)^+ P x.
:::

In the implementation, `model.scf_space` precomputes this compression map during model construction.
During SCF iterations, compression is therefore just a matrix-vector multiply.

The inverse direction expands SCF parameters back to active density input:

:::{math}
x = B y.
:::

The real vector `x = [\operatorname{Re} z, \operatorname{Im} z]` is converted back to complex values and returned as tight-binding blocks.

Conceptually:

```text
density result
  -> required active entries
  -> reduced real params y
  -> constrained active density input
  -> mean-field correction
```

## Where this appears in the SCF loop

The SCF loop uses this space at four points:

1. **Initial guess projection.**
   A user-provided mean-field guess is projected into the active constrained space.
   If projection removes components, MeanFi warns.

2. **Selected density evaluation.**
   When the backend supports selected entries, the density solve requests only `required_realspace_entries()`.

3. **Compression.**
   The computed density entries are compressed to the real vector `y`.
   This is the residual variable used by the fixed-point solver.

4. **Expansion and mean-field update.**
   The next `y` is expanded to active density input, then the normal or BdG mean-field map turns it into the next Hamiltonian correction.

Random guesses use the same map.
`model.random_meanfield(...)` samples the minimal real vector `y`, expands it through `model.scf_space`, and applies the model's mean-field map.
The returned object is a solver-ready mean-field correction, not raw density parameters.
