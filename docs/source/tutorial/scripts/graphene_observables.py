"""Order and band extrema; phase names are not inferred from initial guesses."""

import numpy as np
from scipy.optimize import minimize

from .rhombohedral_graphene import IDENTITY, LOCAL, PAULI


def local_blocks(parameters, density):
    """Recover only the local 4x4 blocks, using onsite Hermiticity explicitly."""
    available = dict(zip(density.coordinates.entries, density.values, strict=True))
    blocks = np.empty((2 * parameters.layers, 4, 4), complex)
    for site in range(2 * parameters.layers):
        for i in range(4):
            for j in range(4):
                row, col = 4 * site + i, 4 * site + j
                if (LOCAL, row, col) in available:
                    blocks[site, i, j] = available[LOCAL, row, col]
                else:
                    blocks[site, i, j] = available[LOCAL, col, row].conjugate()
    return blocks


def local_orders(parameters, density):
    blocks = local_blocks(parameters, density)
    spin = np.array(
        [
            [np.trace(block @ np.kron(IDENTITY, s)).real for s in PAULI]
            for block in blocks
        ]
    )
    valley = np.array(
        [
            [np.trace(block @ np.kron(t, IDENTITY)).real for t in PAULI]
            for block in blocks
        ]
    )
    return {
        "local_charge_average": np.trace(blocks, axis1=1, axis2=2).real.tolist(),
        "local_spin_average": spin.tolist(),
        "local_valley_average": valley.tolist(),
        "surface_laf_spin_average": ((spin[0] - spin[-1]) / 2).tolist(),
        "max_intervalley_coherence": float(np.max(np.abs(blocks[:, :2, 2:]))),
        "max_transverse_spin": float(np.max(np.linalg.norm(spin[:, :2], axis=1))),
        "surface_charge_difference_per_cell": float(
            parameters.measure * (np.trace(blocks[0]) - np.trace(blocks[-1])).real
        ),
    }


def band_edges(parameters, correction, *, radial_points=30, angular_points=60):
    """Sample the whole disk, then optimize extremal valence/conduction energies.

    This diagnoses an indirect gap without assuming its extrema lie on ky=0.
    Multiple starting points reduce, but do not prove absence of, missed extrema.
    """
    h0 = parameters.affine_hamiltonian()
    occupied = 4 * parameters.layers
    radius = np.linspace(0.0, parameters.cutoff, radial_points)
    angles = np.arange(angular_points) * 2 * np.pi / angular_points
    q = (radius[:, None] * np.exp(1j * angles)).ravel()
    coordinates = np.column_stack((q.real, q.imag))

    def bands(point):
        return np.linalg.eigvalsh(h0(*point) + correction)

    values = np.array(
        [bands(point)[occupied - 1 : occupied + 1] for point in coordinates]
    )
    bounds = [(-parameters.cutoff, parameters.cutoff)] * 2
    extrema = []
    for index, sign in ((0, -1), (1, 1)):

        def objective(point):
            return sign * bands(point)[occupied - 1 + index]

        candidates = np.argsort(sign * values[:, index])[:6]
        best = float(sign * values[candidates[0], index])
        point = coordinates[candidates[0]]
        for candidate in candidates:
            result = minimize(
                objective,
                coordinates[candidate],
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": lambda x: parameters.cutoff**2 - x @ x,
                },
                options={"ftol": 1e-12, "maxiter": 150},
            )
            if result.success and result.fun < best:
                best, point = float(result.fun), result.x
        extrema.append((sign * best, point.tolist()))
    return {
        "valence_max_eV": extrema[0][0],
        "conduction_min_eV": extrema[1][0],
        "indirect_gap_meV": 1000 * (extrema[1][0] - extrema[0][0]),
        "valence_extremum": extrema[0][1],
        "conduction_extremum": extrema[1][1],
        "sampled_direct_gap_meV": float(1000 * np.min(values[:, 1] - values[:, 0])),
        "gap_sampling": [radial_points, angular_points],
    }


def mass_rule(parameters, correction, *, mu, indirect_gap_meV, tolerance=1e-7):
    """Return supplement Eq. (34), conditional on its low-energy assumptions.

    This is not a Berry-curvature integration or an independent certification of
    the absence of additional band inversions. Refuse metallic states and spin
    or valley coherent states, to which the paper's four-flavor formula does not
    apply. Downfold at the chemical potential onto the outer nondimer sites.
    """
    size = len(correction)
    flavor = np.arange(size) % 4
    off_flavor = np.where(flavor[:, None] != flavor[None, :], correction, 0.0)
    if indirect_gap_meV <= 0:
        return {"conditional_chern": None, "reason": "metallic indirect gap"}
    if np.max(np.abs(off_flavor)) > tolerance:
        return {"conditional_chern": None, "reason": "spin or valley coherence"}
    h = parameters.hamiltonian(0.0, 0.0) + correction
    masses = []
    for f in range(4):
        indices = np.arange(f, size, 4)
        block = h[np.ix_(indices, indices)]
        low = np.array([0, len(indices) - 1])
        high = np.arange(1, len(indices) - 1)
        effective = block[np.ix_(low, low)] - block[
            np.ix_(low, high)
        ] @ np.linalg.solve(
            block[np.ix_(high, high)] - mu * np.eye(len(high)), block[np.ix_(high, low)]
        )
        masses.append(float((effective[0, 0] - effective[1, 1]).real / 2))
    if min(abs(m) for m in masses) <= tolerance:
        return {"conditional_chern": None, "reason": "unresolved surface mass"}
    return {
        "conditional_chern": float(
            parameters.layers / 2 * np.dot([1, 1, -1, -1], np.sign(masses))
        ),
        "surface_masses_meV": (1000 * np.array(masses)).tolist(),
        "assumption": "four conserved flavors; no additional inversions beyond the chiral surface bands",
    }
