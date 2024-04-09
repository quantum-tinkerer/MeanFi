import numpy as np
from .utils import quad_vecNDim
from scipy.fftpack import ifftn
import itertools as it


def tb2kfunc(tb_model):
    """
    Fourier transforms a real-space tight-binding model to a k-space function.

    Parameters
    ----------
    tb_model : dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    
    Returns
    -------
    function
        A function that takes a k-space vector and returns a complex np.array.
    """
    
    def bloch_ham(k):
        ham = 0
        for vector in tb_model.keys():
            ham += tb_model[vector] * np.exp(
                -1j * np.dot(k, np.array(vector, dtype=float))
            )
        return ham
    
    return bloch_ham

def ifftn2tb(ifftArray):
    """
    Converts an array from ifftn to a tight-binding model format.

    Parameters
    ----------
    ifftArray : ndarray
        An array obtained from ifftn.
    
    Returns
    -------
    dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    """

    size = ifftArray.shape[:-2]

    keys = [np.arange(-size[0]//2+1, size[0]//2) for i in range(len(size))]
    keys = it.product(*keys)
    return {tuple(k): ifftArray[tuple(k)] for k in keys}

def kfunc2tbFFT(kfunc, nSamples, ndim=1):
    """
    Applies FFT on a k-space function to obtain a real-space components.

    Parameters
    ----------
    kfunc : function
        A function that takes a k-space vector and returns a np.array.
    nSamples : int
        Number of samples to take in k-space.
    
    Returns
    -------
    
    dict
        A dictionary with real-space vectors as keys and complex np.arrays as values.
    """
    
    ks = np.linspace(-np.pi, np.pi, nSamples, endpoint=False) # for now nSamples need to be even such that 0 is in the middle
    ks = np.concatenate((ks[nSamples//2:], ks[:nSamples//2]), axis=0) # shift for ifft
    
    if ndim == 1:
        kfuncOnGrid = np.array([kfunc(k) for k in ks])
    if ndim == 2:
        kfuncOnGrid = np.array([[kfunc((kx, ky)) for ky in ks] for kx in ks])
    if ndim > 2:
        raise NotImplementedError("n > 2 not implemented")
    ifftnArray = ifftn(kfuncOnGrid, axes=np.arange(ndim))
    return ifftn2tb(ifftnArray)

def kfunc2tbQuad(kfunc, ndim=1):
    """
    Inverse Fourier transforms a k-space function to a real-space function using a 
    ndim quadrature integration.

    Parameters
    ----------
    kfunc : function
        A function that takes a k-space vector and returns a np.array.
    ndim : int
        Dimension of the k-space
    
    Returns
    -------
    function
        A function that takes a real-space integer vector and returns a np.array.
    """
    def tbfunc(vector): 
        def integrand(k):
            return kfunc(k) * np.exp(1j * np.dot(k, np.array(vector, dtype=float))) / (2*np.pi)
        return quad_vecNDim(integrand, -np.pi, np.pi, ndim=ndim)[0]
    return tbfunc