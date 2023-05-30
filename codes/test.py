import kwant
import numpy as np


s0 = np.identity(2)
sz = np.diag([1, -1])

def graphene_onsite(U, N_ks):
    graphene = kwant.lattice.general(
        [[1, 0], [1 / 2, np.sqrt(3) / 2]], [[0, 0], [0, 1 / np.sqrt(3)]]
    )
    a, b = graphene.sublattices

    # create bulk system
    bulk_graphene = kwant.Builder(kwant.TranslationalSymmetry(*graphene.prim_vecs))
    # add sublattice potential
    m0 = 0
    bulk_graphene[a.shape((lambda pos: True), (0, 0))] = m0 * sz
    bulk_graphene[b.shape((lambda pos: True), (0, 0))] = -m0 * sz
    # add hoppings between sublattices
    bulk_graphene[graphene.neighbors(1)] = s0

    # use kwant wraparound to sample bulk k-space
    wrapped_syst = kwant.wraparound.wraparound(bulk_graphene).finalized()

    # return a hamiltonian for a given kx, ky
    @np.vectorize
    def hamiltonian_return(kx, ky, params={}):
        ham = wrapped_syst.hamiltonian_submatrix(params={**params, **dict(k_x=kx, k_y=ky)})
        return ham

    N_k_axis = np.linspace(0, 2 * np.pi, N_ks, endpoint=False)
    hamiltonians_0 = np.array(
        [[hamiltonian_return(kx, ky) for kx in N_k_axis] for ky in N_k_axis]
    )

    # we need block diagonal structure here since opposite spins interact on the same sublattice
    v_int = U * np.block(
        [[np.ones((2, 2)), np.zeros((2, 2))], [np.zeros((2, 2)), np.ones((2, 2))]]
    )
    # repeat the matrix on a k-grid
    V = np.array([[v_int for i in range(N_ks)] for j in range(N_ks)])
    
    return hamiltonians_0, V

def hubbard_2D(U, N_ks):
    square = kwant.lattice.square(a=1, norbs=2)
    # create bulk system
    bulk_hubbard = kwant.Builder(kwant.TranslationalSymmetry(*square.prim_vecs))
    bulk_hubbard[square.shape((lambda pos: True), (0, 0))] = 0 * np.eye(2)
    # add hoppings between lattice points
    bulk_hubbard[square.neighbors()] = -1

    # use kwant wraparound to sample bulk k-space
    wrapped_fsyst = kwant.wraparound.wraparound(bulk_hubbard).finalized()

    # return a hamiltonian for a given kx, ky
    @np.vectorize
    def hamiltonian_return(kx, ky, params={}):
        ham = wrapped_fsyst.hamiltonian_submatrix(params={**params, **dict(k_x=kx, k_y=ky)})
        return ham
    
    N_k_axis = np.linspace(0, 2 * np.pi, N_ks, endpoint=False)
    hamiltonians_0 = np.array(
        [[hamiltonian_return(kx, ky) for kx in N_k_axis] for ky in N_k_axis]
    )

    # onsite interactions
    v_int = U * np.ones((2,2))
    V = np.array([[v_int for i in range(N_ks)] for j in range(N_ks)])
    return hamiltonians_0, V
