"""Fermi occupations at finite or zero temperature."""

import numpy as np
from scipy.special import expit


def fermi_dirac(energies: np.ndarray, kT: float, mu: float) -> np.ndarray:
    """Occupations with half filling exactly at mu when kT is zero."""
    if not np.isfinite(kT) or kT < 0 or not np.isfinite(mu):
        raise ValueError("kT must be finite and non-negative, and mu must be finite")
    energies = np.asarray(energies, dtype=float)
    if kT == 0:
        return np.where(energies == mu, 0.5, energies < mu).astype(float)
    return expit((mu - energies) / kT)
