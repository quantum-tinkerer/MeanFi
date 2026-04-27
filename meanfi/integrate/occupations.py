from __future__ import annotations

import numpy as np


def fermi_dirac(energies: np.ndarray, kT: float, fermi: float) -> np.ndarray:
    """Evaluate the Fermi-Dirac distribution."""

    if kT < 0:
        raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")
    if kT == 0:
        energies = np.asarray(energies, dtype=float)
        occupation = np.where(energies < fermi, 1.0, 0.0)
        occupation = np.where(energies == fermi, 0.5, occupation)
        return occupation.astype(float, copy=False)

    occupation = np.empty_like(energies, dtype=float)
    exponent = (energies - fermi) / kT
    sign_mask = energies >= fermi

    pos_exp = np.exp(-exponent[sign_mask])
    neg_exp = np.exp(exponent[~sign_mask])

    occupation[sign_mask] = pos_exp / (pos_exp + 1.0)
    occupation[~sign_mask] = 1.0 / (neg_exp + 1.0)
    return occupation
