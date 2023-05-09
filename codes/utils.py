import numpy as np
import kwant

def get_fermi_energy(vals, filling):
    norbs = vals.shape[-1]
    vals_flat = np.sort(vals.flatten())
    ne = len(vals_flat)
    ifermi = int(round(ne * filling / norbs))
    if ifermi >= ne:
        ifermi = ne - 1
    sorte = np.sort(vals_flat)  # sorted eigenvalues
    if ifermi == 0:
        return sorte[0]
    fermi = (sorte[ifermi - 1] + sorte[ifermi]) / 2.0  # fermi energy
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

def generate_guess(max_neighbor, norbs, lattice):
    n_sub = len(lattice.sublattices)
    guess = np.zeros((n_sub + max_neighbor, 2, norbs, norbs))
    for i in range(n_sub):
        # Real part
        guess[i, 0] = np.random.rand(norbs, norbs)
        guess[i, 0] += guess[i, 0].T
        # Imag part
        guess[i, 1] = np.random.rand(norbs, norbs)
        guess[i, 1] -= guess[i, 1].T
    for neighbor in range(max_neighbor):
        # Real part
        guess[n_sub + neighbor, 0] = np.random.rand(norbs, norbs)
        # Imag part
        guess[n_sub + neighbor, 1] = np.random.rand(norbs, norbs)
    return guess

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