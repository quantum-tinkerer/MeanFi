import numpy as np

from scipy.optimize import minimize
from tb.tb import add_tb, _tb_type
from tb.transforms import tb_to_kgrid, kgrid_to_tb


def fermi_dirac(E: np.ndarray, kT: float, fermi: float) -> np.ndarray:
    """
    Calculate the value of the Fermi-Dirac distribution at energy `E` and temperature `T`.

    Parameters
    ----------
    E: np.ndarray(float)
        The energy at which to find the value of the distribution. Can also be an array of values.
    kT: float
        The temperature in Kelvin and Boltzmann constant.
    fermi: float
        The Fermi level.

    Returns
    -------
        The value of the Fermi-Dirac distribution.
    """
    if kT == 0:
        fd = E < fermi
        return fd
    else:
        fd = np.empty_like(E)
        exponent = (E - fermi) / kT
        sign_mask = E >= fermi

        pos_exp = np.exp(-exponent[sign_mask])
        neg_exp = np.exp(exponent[~sign_mask])

        fd[sign_mask] = pos_exp / (pos_exp + 1)
        fd[~sign_mask] = 1 / (neg_exp + 1)

        return fd


def fermi_on_kgrid(vals: np.ndarray, target_charge: float) -> float:
    """Compute the Fermi level on a grid of k-points.

    Parameters
    ----------
    `vals: np.ndarray` :
        Eigenvalues of a hamiltonian sampled on a k-point grid with shape (nk, nk, ..., ndof, ndof),
        where ndof is number of internal degrees of freedom.
    `target_charge: float` :
        Target charge of a unit cell.
        Used to determine the Fermi level.

    Returns
    -------
    :
        Fermi level
    """
    norbs = vals.shape[-1]
    vals_flat = np.sort(vals.flatten())
    neig = len(vals_flat)
    ifermi = int(round(neig * target_charge / norbs))
    if ifermi >= neig:
        return vals_flat[-1]
    elif ifermi == 0:
        return vals_flat[0]
    else:
        fermi = (vals_flat[ifermi - 1] + vals_flat[ifermi]) / 2
        return fermi


def density_matrix_charge(
    ham: _tb_type,
    charge_op: np.ndarray,
    kT: float,
    fermi: float,
    nk: int,
    ndim: int,
    einsum_path: list,
) -> float:
    """
    Calculate the charge of a Hamiltonian with a given `fermi` level offset.

    Parameters
    ----------
    ham: _tb_type
        Hamiltonian tight-binding dictionary for which to calculate the charge.
    charge_op: np.ndarray
        Charge operator of the system, should have the same ndof as `ham`.
    kT: float
        The temperature in Kelvin and Boltzmann constant.
    fermi: float
        The Fermi level.
    nk: int
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.
    ndim: int
        Number of dimensions in the system.
    einsum_path: list
        A `np.einsum_path` result providing the optimal order to perform the einsum calculation with.
        Can also be a `np.einsum(optimize=...)` argument.

    Returns
    -------
        The charge expectation for the Hamiltonian.
    """
    fermi_shift = {(0,) * ndim: -fermi * charge_op}
    ham = add_tb(ham, fermi_shift)
    kham = tb_to_kgrid(ham, nk)
    vals, vecs = np.linalg.eigh(kham)

    distribution = fermi_dirac(vals, kT, 0)
    charge_expectation = np.einsum(
        "...ji, jl, ...li, ...i",
        vecs.conj(),
        charge_op,
        vecs,
        distribution,
        optimize=einsum_path,
    ).sum() / (nk**ndim)

    return charge_expectation


def charge_difference(
    fermi: float,
    ham: _tb_type,
    charge_op: np.ndarray,
    target_charge: float,
    kT: float,
    nk: int,
    ndim: int,
    einsum_path: list,
) -> float:
    """
    Calculate the difference between the charge of a Hamiltonian and a chosen target charge.

    Parameters
    ----------
    fermi: float
        The Fermi level.
    ham: _tb_type
        Hamiltonian tight-binding dictionary for which to calculate the charge.
    charge_op: np.ndarray
        Charge operator of the system, should have the same ndof as `ham`.
    target_charge: float
        Target charge of a unit cell.
        Used to determine the Fermi level.
    kT: float
        The temperature in Kelvin and Boltzmann constant.
    nk: int
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.
    ndim: int
        Number of dimensions in the system.
    einsum_path: list
        A `np.einsum_path` result providing the optimal order to perform the einsum calculation with.
        Can also be a `np.einsum(optimize=...)` argument.

    Returns
    -------
        The absolute difference between calculated charge and target charge.
    """
    charge_expectation = density_matrix_charge(
        ham, charge_op, kT, fermi, nk, ndim, einsum_path
    )
    difference = charge_expectation - target_charge

    return np.abs(difference)


def construct_rho(
    vals: np.ndarray, vecs: np.ndarray, kT: float, fermi: float
) -> np.ndarray:
    """
    Constructs a density matrix from a set of eigenvalues and vectors at a chosen temperature and Fermi level.

    Parameters
    ----------
    vals: np.ndarray
        An array of eigenvalues of the Hamiltonian.
    vecs: np.ndarray
        An array of eigenvectors of the Hamiltonian.
    kT: float
        The temperature in Kelvin and Boltzmann constant.
    fermi: float
        The Fermi level.

    Returns
    -------
        The density matrix on a k-grid.
    """
    occ_distribution = np.sqrt(fermi_dirac(vals, kT, fermi))
    occ_distribution = occ_distribution[..., np.newaxis]
    occ_vecs = np.copy(vecs)  # Copy may not be required
    occ_vecs *= np.moveaxis(occ_distribution, -1, -2)
    _density_matrix = occ_vecs @ np.moveaxis(occ_vecs, -1, -2).conj()

    return _density_matrix


def density_matrix_kgrid(
    ham: _tb_type,
    charge_op: np.ndarray,
    target_charge: float,
    kT: float,
    nk: int,
    ndim: int,
) -> tuple[np.ndarray, float]:
    """Calculate density matrix on a k-space grid.

    Parameters
    ----------
    ham: _tb_type
        Hamiltonian tight-binding dictionary from which to construct the density matrix.
    charge_op: np.ndarray
        Charge operator of the system, should have the same ndof as `ham`.
    target_charge: float
        Target charge of a unit cell.
        Used to determine the Fermi level.
    kT: float
        The temperature in Kelvin and Boltzmann constant.
    nk: int
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.
    ndim: int
        Number of dimensions in the system.


    Returns
    -------
    :
        Density matrix on a k-space grid with shape (nk, nk, ..., ndof, ndof) and Fermi level.
    """
    fermi_0 = 0

    if (charge_op != np.eye(charge_op.shape[0])).all():
        Q_shape = charge_op.shape
        v_shape = np.empty((nk,) * ndim + Q_shape)
        F_shape = np.empty((nk,) * ndim + (Q_shape[0],))

        einsum_path = np.einsum_path(
            "...ji, jl, ...li, ...i",
            v_shape,
            charge_op,
            v_shape,
            F_shape,
            optimize="optimal",
        )[0]

        result = minimize(
            charge_difference,
            fermi_0,
            args=(ham, charge_op, kT, target_charge, nk, ndim, einsum_path),
            method="Nelder-Mead",
            options={"fatol": kT / 2, "xatol": kT / 2},
        )
        opt_fermi = float(result.x)
    elif kT > 0:
        # Run that other option.
        # Double check what this one is supposed to do.
        opt_fermi = 0
    else:  # Need to check if this is correct or if it should not run the 'normal' density matrix construction here.
        vals, vecs = np.linalg.eigh(tb_to_kgrid(ham, nk))
        opt_fermi = fermi_on_kgrid(vals, target_charge)

    Q_Ef = {(0,) * ndim: -opt_fermi * charge_op}
    ham = add_tb(ham, Q_Ef)
    kham = tb_to_kgrid(ham, nk)
    vals, vecs = np.linalg.eigh(kham)

    _density_matrix_kgrid = construct_rho(vals, vecs, kT, 0)

    return _density_matrix_kgrid, opt_fermi


def density_matrix(
    ham: _tb_type, charge_op: np.ndarray, target_charge: float, kT: float, nk: int
) -> tuple[_tb_type, float]:
    """Compute the real-space density matrix tight-binding dictionary.

    Parameters
    ----------
    ham: _tb_type
        Hamiltonian tight-binding dictionary from which to construct the density matrix.
    charge_op: np.ndarray
        Charge operator of the system, should have the same ndof as `ham`.
    target_charge: float
        Target charge of a unit cell.
        Used to determine the Fermi level.
    kT: float
        The temperature in Kelvin and Boltzmann constant.
    nk: int
        Number of k-points in a grid to sample the Brillouin zone along each dimension.
        If the system is 0-dimensional (finite), this parameter is ignored.

    Returns
    -------
    :
        Density matrix tight-binding dictionary and Fermi energy.
    """
    ndim = len(list(ham)[0])

    if ndim > 0:
        _density_matrix_kgrid, fermi = density_matrix_kgrid(
            ham, charge_op, target_charge, kT, nk, ndim
        )
        return kgrid_to_tb(_density_matrix_kgrid), fermi
    else:
        _density_matrix, fermi = density_matrix_kgrid(
            ham, charge_op, target_charge, kT, nk, ndim
        )
        return {(): _density_matrix}, fermi


def meanfield(density_matrix: _tb_type, h_int: _tb_type) -> _tb_type:
    """Compute the mean-field correction from the density matrix.

    Parameters
    ----------
    density_matrix :
        Density matrix tight-binding dictionary.
    h_int :
        Interaction hermitian Hamiltonian tight-binding dictionary.
    Returns
    -------
    :
        Mean-field correction tight-binding dictionary.

    Notes
    -----

    The interaction h_int must be of density-density type.
    For example, h_int[(1,)][i, j] = V means a repulsive interaction
    of strength V between two particles with internal degrees of freedom i and j
    separated by 1 lattice vector.
    """
    n = len(list(density_matrix)[0])
    local_key = tuple(np.zeros((n,), dtype=int))

    direct = {
        local_key: np.sum(
            np.array(
                [
                    np.diag(
                        np.einsum("pp,pn->n", density_matrix[local_key], h_int[vec])
                    )
                    for vec in frozenset(h_int)
                ]
            ),
            axis=0,
        )
    }

    exchange = {
        vec: -1 * h_int.get(vec, 0) * density_matrix[vec] for vec in frozenset(h_int)
    }
    return add_tb(direct, exchange)
