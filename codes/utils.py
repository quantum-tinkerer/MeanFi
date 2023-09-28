#%%
import numpy as np
import kwant
#%%
def get_fermi_energy(vals, filling):
    norbs = vals.shape[-1]
    vals_flat = np.sort(vals.flatten())
    ne = len(vals_flat)
    ifermi = int(round(ne * filling / norbs))
    if ifermi >= ne:
        return vals_flat[-1]
    elif ifermi == 0:
        return vals_flat[0]
    else:
        fermi = (vals_flat[ifermi - 1] + vals_flat[ifermi]) / 2
        return fermi

def syst2hamiltonian(kxs, kys, syst, params={}):
    def h_k(kx, ky):
        return syst.hamiltonian_submatrix(params={**params, **dict(k_x=kx, k_y=ky)})
    return np.array(
        [[h_k(kx, ky) for kx in kxs] for ky in kys]
    )

def potential2hamiltonian(
    syst, lattice, func_onsite, func_hop, ks, params={}, max_neighbor=1
):
    V = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    V[syst.sites()] = func_onsite
    for neighbors in range(max_neighbor):
        V[lattice.neighbors(neighbors + 1)] = func_hop
    wrapped_V = kwant.wraparound.wraparound(V).finalized()
    return syst2hamiltonian(kxs=ks, kys=ks, syst=wrapped_V, params=params)

def generate_guess(nk, dim, hopping_vecs, ndof, scale=0.1):
    """
    nk : int
        number of k points
    dim : int
        dimension of the system
    hopping_vecs : np.array
                hopping vectors as obtained from extract_hopping_vectors
    norb_list : np.array
                number of orbitals in each atom
    scale : float
            scale of the guess. If scale=1 then the guess is random around 0.5
            Smaller values of the guess significantly slows down convergence but 
            does improve the result at phase instability points.

    Notes: 
    -----
    Assumes that the desired max nearest neighbour distance is included in the hopping_vecs information.
    Creates a square grid by definition, might still want to change that
    """
    dim = 2
    klenlist = [nk for i in range(dim)]
    guess = np.zeros((klenlist+[ndof, ndof]), dtype=complex)
    kgrid = np.asarray(np.meshgrid(*[np.linspace(-np.pi, np.pi, nk) for i in range(dim)])).reshape(dim, -1).T

    #always include onsite/internal hopping
    amplitude = np.random.rand(ndof, ndof) 
    phase = 2 * np.pi * np.random.rand(ndof, ndof)
    rand_hermitian = amplitude * np.exp(1j * phase)
    rand_hermitian += rand_hermitian.T.conj()
    rand_hermitian /= 2
    reshape_order = [1 for i in range(dim)] #could use a better name
    guess += rand_hermitian.reshape(reshape_order+[ndof, ndof]) #no k-dependence for onsite

    for hop in hopping_vecs:  
        k_dependence = np.exp(1j * np.dot(kgrid, hop)).reshape(klenlist+[1,1])
        amplitude = np.random.rand(ndof, ndof) 
        phase = 2 * np.pi * np.random.rand(ndof, ndof)
        rand_hermitian = amplitude * np.exp(1j * phase)
        rand_hermitian += rand_hermitian.T.conj()
        rand_hermitian /= 2
        guess += rand_hermitian.reshape(reshape_order+[ndof, ndof]) * k_dependence

    return guess*scale

def extract_hopping_vectors(builder):
    keep=None
    deltas=[]
    for hop, val in builder.hopping_value_pairs():
        a, b=hop
        b_dom = builder.symmetry.which(b)
        # Throw away part that is in the remaining translation direction, so we get
        # an element of 'sym' which is being wrapped
        b_dom = np.array([t for i, t in enumerate(b_dom) if i != keep])
        deltas.append(b_dom)
    return np.asarray(deltas)

def generate_scf_syst(max_neighbor, syst, lattice):  
    subs = np.array(lattice.sublattices)
    def scf_onsite(site, mat):
        idx = np.where(subs == site.family)[0][0]
        return mat[idx, 0] + 1j * mat[idx, 1]

    scf = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    scf[syst.sites()] = scf_onsite
    for neighbor in range(max_neighbor):
        def scf_hopping(site1, site2, mat):
            return (
                mat[len(lattice.sublattices) + neighbor, 0]
                + 1j * mat[len(lattice.sublattices) + neighbor, 1]
            )

        scf[lattice.neighbors(neighbor + 1)] = scf_hopping
    deltas = extract_hopping_vectors(scf)
    wrapped_scf = kwant.wraparound.wraparound(scf).finalized()
    return wrapped_scf, deltas

def hk2hop(hk, deltas, ks, dk):
    kxx, kyy = np.meshgrid(ks, ks)
    kxy = np.array([kxx, kyy])
    hopps = (
        np.sum(
            np.einsum(
                "ijk,jklm->ijklm",
                np.exp(1j * np.einsum("ij,jkl->ikl", deltas, kxy)),
                hk,
            ),
            axis=(1, 2),
        )
        * (dk) ** 2
        / (2 * np.pi) ** 2
    )
    return hopps


def hk2syst(deltas, hk, ks, dk, max_neighbor, norbs, lattice):
    hopps = hk2hop(hk, deltas, ks, dk)
    bulk_scf = kwant.Builder(kwant.TranslationalSymmetry(*lattice.prim_vecs))
    for i, delta in enumerate(deltas):
        for j, sublattice1 in enumerate(lattice.sublattices):
            for k, sublattice2 in enumerate(lattice.sublattices):
                if np.allclose(delta, [0, 0]):
                    bulk_scf[sublattice1.shape((lambda pos: True), (0, 0))] = hopps[
                        i, j * norbs : (j + 1) * norbs, j * norbs : (j + 1) * norbs
                    ]
                    if k != j:
                        hopping = (delta, sublattice1, sublattice2)
                        bulk_scf[kwant.builder.HoppingKind(*hopping)] = hopps[
                            i, j * norbs : (j + 1) * norbs, k * norbs : (k + 1) * norbs
                        ]
                else:
                    for k, sublattice2 in enumerate(lattice.sublattices):
                        hopping = (delta, sublattice1, sublattice2)
                        bulk_scf[kwant.builder.HoppingKind(*hopping)] = hopps[
                            i, j * norbs : (j + 1) * norbs, k * norbs : (k + 1) * norbs
                        ]
    wrapped_scf_syst = kwant.wraparound.wraparound(bulk_scf).finalized()
    return wrapped_scf_syst

def calc_gap(vals, E_F):
    emax = np.max(vals[vals <= E_F])
    emin = np.min(vals[vals > E_F])
    return np.abs(emin - emax)