import kwant
import numpy as np
from . import pauli
from kwant.linalg import lll
import scipy
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Create a 10 x 10 supercell of strained graphene following Antonio L R Manesco and Jose L Lado 2021 2D Mater. 8 035057


def high_symmetry_line(bz_vertices, nk=50):
    GammaK = np.linspace([0, 0], bz_vertices[0], nk, endpoint=False)
    KKprime = np.linspace(bz_vertices[0], bz_vertices[1], nk, endpoint=False)
    KprimeGamma = np.linspace(bz_vertices[1], [0, 0], nk, endpoint=True)
    return np.concatenate((GammaK, KKprime, KprimeGamma))


def create_system(n=16, nk=15):
    # Hopping constant
    t = 1
    # hbar * v_F
    vF_times_hbar = 3 / 2

    # Create honeycomb lattice and define supercell translational symmetry
    lat = kwant.lattice.honeycomb(a=1, norbs=2)
    sym_2d = kwant.TranslationalSymmetry(lat.vec((n, 0)), lat.vec((0, n)))
    # Create kwant.Builder
    bulk = kwant.Builder(sym_2d)

    # Defining supercell hopping modulation
    # Extract lattice vectors
    B = np.array(bulk.symmetry.periods).T
    L_M = np.linalg.norm(B.T[0])
    # Compute reciprocal lattice vectors
    A = np.linalg.pinv(B).T

    bs = 2 * np.pi * np.array([A[:, 0], A[:, 1], -(A[:, 0] + A[:, 1])])

    # Generate hoppig modulation
    def dt(r, b):
        return np.sin(np.dot(b, r)) / np.linalg.norm(bs[0])

    # Set hopping landscape
    def hopping(site1, site2, xi):
        r1 = site1.pos
        r2 = site2.pos
        r_med = (r1 + r2) / 2
        dr = r1 - r2
        prefactor = vF_times_hbar * xi**2 / (L_M) ** 2 / np.sqrt(3)
        _b = bs[np.argmin(np.abs(np.cross(dr, bs)))]
        # print(np.cross(dr, _b))
        return (t + prefactor * dt(r_med, _b)) * pauli.s0

    # Define onsite and hopping energies
    bulk[lat.shape(lambda pos: True, (0, 0))] = 0 * pauli.s0
    bulk[lat.neighbors()] = hopping

    # Get lattice points that neighbor the origin, in basis of lattice vectors
    reduced_vecs, transf = lll.lll(A.T)
    neighbors = np.dot(lll.voronoi(reduced_vecs), transf)
    # Add the origin to these points.
    klat_points = np.concatenate(([[0] * len(B)], neighbors))
    # Transform to cartesian coordinates and rescale.
    # Will be used in 'outside_bz' function, later on.
    klat_points = 2 * np.pi * np.dot(klat_points, A.T)
    # Calculate the Voronoi cell vertices
    voronoi_cell = scipy.spatial.Voronoi(klat_points)
    around_origin = voronoi_cell.point_region[0]
    bz_vertices = voronoi_cell.vertices[voronoi_cell.regions[around_origin]]

    def momentum_to_lattice(k):
        k, _ = scipy.linalg.lstsq(A, k)[:2]
        return k

    k_path = np.array([momentum_to_lattice(k) for k in high_symmetry_line(bz_vertices)])

    return bulk, lat, k_path


def plot_bands(hk, k_path):
    eks = []
    for k in k_path:
        energies = np.sort(np.linalg.eigvalsh(hk(k)))
        eks.append(energies)

    nk = len(k_path)
    plt.plot(eks, c="k", lw=1)
    plt.ylabel(r"$E-E_F\ [eV]$")
    plt.ylim(-0.1, 0.1)
    plt.xlim(0, nk)
    plt.xticks(
        [0, nk // 3, nk // 2, int(2 * nk // 3), nk],
        [r"$\Gamma$", r"$K$", r"$M$", r"$K^{\prime}$", r"$\Gamma$"],
    )
    plt.axvline(x=nk // 3, c="k", ls="--")
    plt.axvline(x=nk // 2, c="k", ls="--")
    plt.axvline(x=2 * nk // 3, c="k", ls="--")
    plt.axhline(y=0, c="k", ls="--")
    plt.tight_layout()
    plt.show()
