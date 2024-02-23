from . import utils, hf
import numpy as np


class BaseMfModel:
    """
    Base class for periodic hamiltonian with an interacting potential
    treated within the mean-field approximation.

    Attributes
    ----------
    H0_k : function
        Non-interacting hamiltonian part H0(k) evaluated on a k-point grid.
    V_k : function
        Interaction potential V(k) evaluated on a k-point grid.
    filling : float
        Filling factor of the system.

    Methods
    -------
    densityMatrix(mf_k)
        Returns the density matrix given the mean-field correction to the
        non-interacting hamiltonian mf_k.
    meanField(rho)
        Calculates the mean-field correction from a given density matrix.
    """

    def __init__(self, H0_k, V_k, filling):
        """
        Parameters
        ----------
        H0_k : function
            Non-interacting hamiltonian part H0(k) evaluated on a k-point grid.
        V_k : function
            Interaction potential V(k) evaluated on a k-point grid.
        filling : float
            Filling factor of the system.
        """
        self.H0_k = H0_k
        self.V_k = V_k
        self.filling = filling

    def densityMatrix(self, mf_k):
        """
        Parameters
        ----------
        mf_k : nd-array
            Mean-field correction to the non-interacting hamiltonian.

        Returns
        -------
        rho : nd-array
            Density matrix.
        """
        vals, vecs = np.linalg.eigh(self.H0_k + mf_k)
        vecs = np.linalg.qr(vecs)[0]
        E_F = utils.get_fermi_energy(vals, self.filling)
        return hf.density_matrix(vals=vals, vecs=vecs, E_F=E_F)

    def meanField(self, rho):
        """
        Parameters
        ----------
        rho : nd-array
            Density matrix.

        Returns
        -------
        mf_k : nd-array
            Mean-field correction to the non-interacting hamiltonian.
        """
        return hf.compute_mf(rho, self.V_k)


class MfModel(BaseMfModel):
    """
    BaseMfModel with the non-interacting hamiltonian and the interaction
    potential given as tight-binding models.
    The model is defined on a regular k-point grid.
    """

    def __init__(self, tb_model, int_model, filling, nk=100):
        """
        Parameters
        ----------
        tb_model : dict
            Non-interacting tight-binding model.
        filling : float
            Filling factor of the system.
        int_model : dict
            Interacting tight-binding model.
        """

        self.filling = filling
        dim = len([*tb_model.keys()][0])
        if dim > 0:
            self.H0_k = utils.tb2grid(tb_model, nk=nk)
            self.V_k = utils.tb2grid(int_model, nk=nk)
        if dim == 0:
            self.H0_k = tb_model[()]
            self.V_k = int_model[()]