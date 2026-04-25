from itertools import product
import numpy as np

from meanfi._bdg import validate_bdg_tb
from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import tb_to_kgrid


def guess_tb(
    tb_keys: list[tuple[None] | tuple[int, ...]],
    ndof: int,
    scale: float = 1,
    *,
    superconducting: bool = False,
) -> _tb_type:
    """Generate hermitian guess tight-binding dictionary.

    Parameters
    ----------
    tb_keys :
       List of hopping vectors (tight-binding dictionary keys) the guess contains.
    ndof :
        Number internal degrees of freedom within the unit cell.
    scale :
        Scale of the random guess.
    superconducting :
        When true, generate an electron-first BdG correction with shape
        ``(2*ndof, 2*ndof)`` on each key.
    Returns
    -------
    :
        Hermitian guess tight-binding dictionary.
    """
    ndim = len(tb_keys[0]) if tb_keys else 0
    if superconducting:
        return _guess_bdg_tb(tb_keys, ndof=ndof, ndim=ndim, scale=scale)

    return _guess_electron_tb(tb_keys, ndof=ndof, scale=scale)


def _guess_electron_tb(
    tb_keys: list[tuple[None] | tuple[int, ...]], ndof: int, scale: float
) -> _tb_type:
    guess = {}
    for vector in tb_keys:
        if vector not in guess.keys():
            amplitude = scale * np.random.rand(ndof, ndof)
            phase = 2 * np.pi * np.random.rand(ndof, ndof)
            rand_hermitian = amplitude * np.exp(1j * phase)
            if np.linalg.norm(np.array(vector)) == 0:
                rand_hermitian += rand_hermitian.T.conj()
                rand_hermitian /= 2
                guess[vector] = rand_hermitian
            else:
                guess[vector] = rand_hermitian
                guess[tuple(-np.array(vector))] = rand_hermitian.T.conj()

    return guess


def _guess_bdg_tb(
    tb_keys: list[tuple[None] | tuple[int, ...]], *, ndof: int, ndim: int, scale: float
) -> _tb_type:
    support = set(tb_keys)
    support.update(tuple(-np.asarray(vector, dtype=int)) for vector in tb_keys)

    normal_block = _guess_electron_tb(sorted(support), ndof=ndof, scale=scale)
    anomalous_block = {
        vector: scale
        * np.random.rand(ndof, ndof)
        * np.exp(2j * np.pi * np.random.rand(ndof, ndof))
        for vector in support
    }

    zero = np.zeros((ndof, ndof), dtype=complex)
    guess = {}
    for vector in support:
        opposite = tuple(-np.asarray(vector, dtype=int))
        normal = normal_block.get(vector, zero)
        anomalous = anomalous_block.get(vector, zero)
        lower = anomalous_block.get(opposite, zero).conj().T
        hole = -normal_block.get(opposite, zero).T
        guess[vector] = np.block([[normal, anomalous], [lower, hole]])

    validate_bdg_tb(guess, ndof=ndof, ndim=ndim, name="BdG guess")
    return guess


def generate_tb_keys(cutoff: int, dim: int) -> list[tuple[None] | tuple[int, ...]]:
    """Generate tight-binding dictionary keys up to a cutoff.

    Parameters
    ----------
    cutoff :
        Maximum distance along each dimension to generate tight-bindign dictionary keys for.
    dim :
        Dimension of the tight-binding dictionary.

    Returns
    -------
    :
        List of generated tight-binding dictionary keys up to a cutoff.
    """
    return [*product(*([[*range(-cutoff, cutoff + 1)]] * dim))]


def fermi_energy(tb: _tb_type, filling: float, nk: int = 100):
    """
    Calculate the Fermi energy of a given tight-binding dictionary.

    Parameters
    ----------
    tb :
        Tight-binding dictionary.
    filling :
        Number of particles in a unit cell.
        Used to determine the Fermi level.
    nk :
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        Fermi energy.
    """
    kham = tb_to_kgrid(tb, nk)
    vals = np.linalg.eigvalsh(kham)
    flat = np.sort(vals.reshape(-1))
    n_kpoints = vals.shape[0] if vals.ndim == 2 else int(np.prod(vals.shape[:-1]))
    idx = int(np.clip(np.ceil(filling * n_kpoints) - 1, 0, flat.size - 1))
    return float(flat[idx])
