"""Independent smooth-insulator quadrature, used only to audit FermiSimplex.

All self-consistency and reported production integrals use FermiSimplex. This
reference deliberately uses a different quadrature to measure its actual error.
It is invalid for a metal and refuses any sampled violation of the spectral gap.
"""

import numpy as np
from scipy.special import roots_legendre


def insulating_reference(parameters, correction, mu, *, radial_order, angular_order):
    nodes, weights = roots_legendre(radial_order)
    # r dr / (pi Lambda²) = ds dphi / (2 pi), s=(r/Lambda)².
    radius = parameters.cutoff * np.sqrt((nodes + 1) / 2)
    angle = 2 * np.pi * np.arange(angular_order) / angular_order
    q = (radius[:, None] * np.exp(1j * angle)).ravel()
    weights = np.repeat(weights / (2 * angular_order), angular_order)
    h0 = parameters.affine_hamiltonian()
    size = len(correction)
    occupied = size // 2
    density = np.zeros((size, size), complex)
    energy = 0.0
    margin = np.inf
    for start in range(0, len(q), 128):
        points = q[start : start + 128]
        h = np.array([h0(z.real, z.imag) + correction for z in points])
        energies, vectors = np.linalg.eigh(h)
        margin = min(
            margin,
            np.min(mu - energies[:, occupied - 1]),
            np.min(energies[:, occupied] - mu),
        )
        if margin <= 0:
            raise ValueError(
                "insulating reference requires a chemical potential in the global gap"
            )
        w = weights[start : start + len(points)]
        filled = vectors[:, :, :occupied]
        density += np.einsum("k,kib,kjb->ij", w, filled, filled.conj(), optimize=True)
        energy += w @ energies[:, :occupied].sum(axis=1)
    return {
        "density": density,
        "band_energy_average": float(energy),
        "sampled_mu_margin_eV": float(margin),
    }
