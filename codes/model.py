from . import utils
import numpy as np

class Model:

    def __init__(self, tb_model, int_model=None, Vk=None, guess=None):
        self.tb_model = tb_model
        self.hk = utils.model2hk(tb_model=tb_model)
        if int_model is not None:
            self.int_model = int_model
            self.Vk = utils.model2hk(tb_model=int_model)
        else:
            self.Vk = Vk
        self.dim = len([*tb_model.keys()][0])
        self.ndof = len([*tb_model.values()][0])
        self.guess = guess
            

    def random_guess(self, vectors):
        self.guess = utils.generate_guess(
            vectors=vectors,
            ndof=self.ndof,
            scale=1+np.max(np.abs([*self.int_model.values()]))
        )

    def kgrid_evaluation(self, nk):
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

    def flatten_mf(self):
        flat = self.mf_k.flatten()
        return flat[:len(flat)//2] + 1j * flat[len(flat)//2:]

    def reshape_mf(self, mf_flat):
        mf_flat = np.concatenate((np.real(mf_flat), np.imag(mf_flat)))
        return mf_flat.reshape(*self.hamiltonians_0.shape)
