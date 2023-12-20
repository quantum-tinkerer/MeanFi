from . import utils
import numpy as np

class Model:
    """
    A period tight-binding model class.

    Attributes
    ----------
    tb_model : dict
        Non-interacting tight-binding model.
    int_model : dict
        Interacting tight-binding model.
    Vk : function
        Interaction potential V(k). Used if `int_model = None`.
    guess : dict
        Initial guess for self-consistent calculation.
    dim : int
        Number of translationally invariant real-space dimensions.
    ndof : int
        Number of internal degrees of freedom (orbitals).
    hamiltonians_0 : nd-array
        Non-interacting amiltonian evaluated on a k-point grid.
    H_int : nd-array
        Interacting amiltonian evaluated on a k-point grid.
    """

    def __init__(self, tb_model, int_model=None, Vk=None, guess=None):
        self.tb_model = tb_model
        self.dim = len([*tb_model.keys()][0])
        if self.dim > 0:
            self.hk = utils.model2hk(tb_model=tb_model)
        self.int_model = int_model
        if self.int_model is not None:
            self.int_model = int_model
            if self.dim > 0:
                self.Vk = utils.model2hk(tb_model=int_model)
        else:
            if self.dim > 0:
                self.Vk = Vk
        self.ndof = len([*tb_model.values()][0])
        self.guess = guess
        if self.dim == 0:
            self.hamiltonians_0 = tb_model[()]
            self.H_int = int_model[()]

    def random_guess(self, vectors):
        """
        Generate random guess.

        Parameters:
        -----------
        vectors : list of tuples
            Hopping vectors for the mean-field corrections.
        """
        if self.int_model is None:
            scale = 1
        else:
            scale = 1+np.max(np.abs([*self.int_model.values()]))
        self.guess = utils.generate_guess(
            vectors=vectors,
            ndof=self.ndof,
            scale=scale
        )

    def kgrid_evaluation(self, nk):
        """
        Evaluates hamiltonians on a k-grid.

        Parameters:
        -----------
        nk : int
            Number of k-points along each direction.
        """
        self.hamiltonians_0, self.ks = utils.kgrid_hamiltonian(
            nk=nk,
            hk=self.hk,
            dim=self.dim,
            return_ks=True
        )
        self.H_int = utils.kgrid_hamiltonian(nk=nk, hk=self.Vk, dim=self.dim)
        self.mf_k = utils.kgrid_hamiltonian(
            nk=nk,
            hk=utils.model2hk(self.guess),
            dim=self.dim,
        )
