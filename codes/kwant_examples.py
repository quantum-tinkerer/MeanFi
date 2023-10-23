import kwant
import numpy as np
from . import utils

s0 = np.identity(2)
sz = np.diag([1, -1])

def graphene_extended_hubbard():
    # Create graphene lattice
    graphene = kwant.lattice.general(
        [[1, 0], [1 / 2, np.sqrt(3) / 2]], [[0, 0], [0, 1 / np.sqrt(3)]],
        norbs=2
    )
    a, b = graphene.sublattices

    # Create bulk system
    bulk_graphene = kwant.Builder(kwant.TranslationalSymmetry(*graphene.prim_vecs))
    # Set onsite energy to zero
    bulk_graphene[a.shape((lambda pos: True), (0, 0))] = 0 * s0
    bulk_graphene[b.shape((lambda pos: True), (0, 0))] = 0 * s0
    # Add hoppings between sublattices
    bulk_graphene[graphene.neighbors(1)] = s0

    def onsite_int(site, U):
        return U * np.ones((2, 2))
    def nn_int(site1, site2, V):
        return V * np.ones((2, 2))

    syst_V = utils.build_interacting_syst(
        syst=bulk_graphene,
        lattice = graphene,
        func_onsite = onsite_int,
        func_hop = nn_int,
    )
    
    return bulk_graphene, syst_V

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


def hubbard_1D(U, N_ks):
    chain = kwant.lattice.chain(a=1, norbs=2)
    # create bulk system
    bulk_hubbard = kwant.Builder(kwant.TranslationalSymmetry(*chain.prim_vecs))
    bulk_hubbard[chain.shape((lambda pos: True), (0,))] = 0 * s0
    # add hoppings between lattice points
    bulk_hubbard[chain.neighbors()] = -1


    # return a hamiltonian for a given kx, ky
    @np.vectorize
    def hamiltonian_return(kx, params={}):
        ham = wrapped_fsyst.hamiltonian_submatrix(params={**params, **dict(k_x=kx)})
        return ham

    N_k_axis = np.linspace(0, 2 * np.pi, N_ks, endpoint=False)
    hamiltonians_0 = np.array(
        [hamiltonian_return(kx) for kx in N_k_axis]
        )

    return hamiltonians_0, V
