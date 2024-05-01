import kwant
import numpy as np

from codes.kwant_helper.utils import build_interacting_syst

s0 = np.identity(2)
sz = np.diag([1, -1])


def graphene_extended_hubbard():
    """ "
    Return
    ------
    bulk_graphene : kwant.builder.Builder
        The bulk graphene system.
    syst_V : kwant.builder.Builder
    """
    # Create graphene lattice
    graphene = kwant.lattice.general(
        [[1, 0], [1 / 2, np.sqrt(3) / 2]], [[0, 0], [0, 1 / np.sqrt(3)]], norbs=2
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

    syst_V = build_interacting_syst(
        builder=bulk_graphene,
        lattice=graphene,
        func_onsite=onsite_int,
        func_hop=nn_int,
    )

    return bulk_graphene, syst_V
