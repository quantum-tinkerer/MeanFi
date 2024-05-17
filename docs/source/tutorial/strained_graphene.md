---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Strained graphene

To showcase how meanfi can be used to study supercell calculations, we consider a strained graphene system. We use the physics and the system from https://arxiv.org/pdf/2104.00573

## Model creation:

We first create the atomistic model in kwant:

```{code-cell} ipython3
import kwant
import matplotlib.pyplot as plt
import meanfi
import numpy as np
from meanfi.kwant_helper import utils
from scipy.spatial.transform import Rotation
from kwant.linalg import lll
import tinyarray
import scipy


n=10
a = 0.142
t = 1
ndof = n**2 * 4
L_M = 18
vF_times_hbar = 3 / 2
sigma_0 = np.array([[1, 0], [0, 1]], dtype=float)

scaling_factor = n / a / L_M
# Create honeycomb lattice and define supercell translational symmetry
lat = kwant.lattice.honeycomb(a=a * scaling_factor, norbs=2)
sym_2d = kwant.TranslationalSymmetry(lat.vec((n, 0)), lat.vec((0, n)))
# Create kwant.Builder
bulk = kwant.Builder(sym_2d)

# Defining the buckling
# Constant to ensure the correct periodicity
const = 2 * np.pi / (n * a * scaling_factor) * 2 / np.sqrt(3)

# Basis of the triangular buckled lattice
b_1 = const * np.array([1, 0, 0])
b_2 = const * np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), 0])
b_3 = const * np.array([np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3), 0])

# Generate hoppig modulation
def dt(r, b, xi):
    arg = np.dot(b, r)
    return (vF_times_hbar * xi**2 / (L_M) ** 2) * np.sin(arg) / const

# Set hopping landscape
def hopping(site1, site2, t, xi):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    r1 = np.array([x1, y1, 0])
    r2 = np.array([x2, y2, 0])
    r_med = (r1 + r2) / 2
    rot = Rotation.from_euler("z", -30, degrees=True)
    r_med = rot.apply(r_med)
    dr = r1 - r2
    dr = rot.apply(dr)
    if np.abs(dr[1]) < 0.01:
        return -t * sigma_0 + dt(r_med, b_1, xi) * sigma_0
    elif np.sign(dr[0] * dr[1]) < 0:
        return -t * sigma_0 + dt(r_med, b_2, xi) * sigma_0
    else:
        return -t * sigma_0 + dt(r_med, b_3, xi) * sigma_0

def onsite(site):
    return 0 * sigma_0

# Define onsite and hopping energies
bulk[lat.shape(lambda pos: True, (0, 0))] = onsite
bulk[lat.neighbors()] = hopping
# Wrap system
wrapped_syst = kwant.wraparound.wraparound(bulk)
# Finalize wrapped system
wrapped_fsyst = wrapped_syst.finalized()
```

We first want to check that our model is created correctly. To do this we plot the band structure of the non-interacting Hamiltonian of the atomistic model which we just created. In order to do this we define a k-path in the Brillouin zone which goes from $\Gamma$ to $K$ to $K'$ and back to $\Gamma$. We then calculate the band structure along this path.


```{code-cell} ipython3
def relevant_kpath(wrapped_syst, nk=50):
    lat_ndim = 2
    # columns of B are lattice vectors
    B = np.array(wrapped_syst._wrapped_symmetry.periods).T
    # columns of A are reciprocal lattice vectors
    A = np.linalg.pinv(B).T

    # Get lattice points that neighbor the origin, in basis of lattice vectors
    reduced_vecs, transf = lll.lll(A.T)
    neighbors = tinyarray.dot(lll.voronoi(reduced_vecs), transf)
    # Add the origin to these points.
    klat_points = np.concatenate(([[0] * lat_ndim], neighbors))
    # Transform to cartesian coordinates and rescale.
    klat_points = 2 * np.pi * np.dot(klat_points, A.T)
    voronoi = scipy.spatial.Voronoi(klat_points)
    around_origin = voronoi.point_region[0]
    bz_vertices = voronoi.vertices[voronoi.regions[around_origin]]

    GammaK = np.linspace([0, 0], bz_vertices[0], nk, endpoint=False)
    KKprime = np.linspace(bz_vertices[0], bz_vertices[1], nk, endpoint=False)
    KprimeGamma = np.linspace(bz_vertices[1], [0, 0], nk, endpoint=True)
    k_points = np.concatenate((GammaK, KKprime, KprimeGamma))

    return k_points

def momentum_to_lattice(k, wrapped_syst):
    B = np.array(wrapped_syst._wrapped_symmetry.periods).T
    # columns of A are reciprocal lattice vectors
    A = np.linalg.pinv(B).T
    k, _ = scipy.linalg.lstsq(A, k)[:2]
    return k

eks = []
params = {"t": 1.0, "mu": 0.0, "delta_mu": 0.0, "xi": 6.0}
for k in relevant_kpath(wrapped_syst):
    k = momentum_to_lattice(k, wrapped_syst)
    ham_k = wrapped_fsyst.hamiltonian_submatrix(
        params={**params, **dict(k_x=k[0], k_y=k[1])}, sparse=False
    )
    energies = np.sort(np.linalg.eigvalsh(ham_k))
    eks.append(energies)
```

```{code-cell} ipython3
plt.plot(eks, c="k", lw=1)
plt.ylabel(r"$E-E_F\ [eV]$")
plt.ylim(-0.5, 0.5)
plt.xlim(0, 149)
plt.xticks(
    [0, 50, 75, 100, 150], [r"$\Gamma$", r"$K$", r"$M$", r"$K^{\prime}$", r"$\Gamma$"]
)
plt.axvline(x=50, c="k", ls="--")
plt.axvline(x=75, c="k", ls="--")
plt.axvline(x=100, c="k", ls="--")
plt.axhline(y=0, c="k", ls="--")
plt.tight_layout()
plt.show()
```

This is qualitatively in agreement with Figure 2 from https://arxiv.org/pdf/2104.00573. The differences are due to the fact that we have chosen to show a smaller supercell here in order to not have the calculations take too long.

Now that we have confirmed that the non-interacting model is as we expect we can turn to making the interacting part. We only add interactions to the onsite part and then once again use the kwant helper function `build_interacting_syst` to create the interacting system.

```{code-cell} ipython3
def func_hop(site1, site2):
    return 0 * np.ones((2, 2))


def func_onsite(site, U):
    return U * np.ones((2, 2))


int_builder = utils.build_interacting_syst(
    bulk,
    lat,
    func_onsite,
    func_hop,
    max_neighbor=0,
)
```