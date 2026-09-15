"""Fermi occupations at finite or zero temperature."""

import numpy as np
from scipy.special import entr, expit


def fermi_dirac(energies: np.ndarray, kT: float, mu: float) -> np.ndarray:
    """Occupations with half filling exactly at mu when kT is zero."""
    if not np.isfinite(kT) or kT < 0 or not np.isfinite(mu):
        raise ValueError("kT must be finite and non-negative, and mu must be finite")
    energies = np.asarray(energies, dtype=float)
    if kT == 0:
        return np.where(energies == mu, 0.5, energies < mu).astype(float)
    with np.errstate(over="ignore"):
        return expit((mu - energies) / kT)


def occupation_entropy(occupation: np.ndarray) -> np.ndarray:
    """Dimensionless entropy of each mode, including exactly empty/full modes."""
    return entr(occupation) + entr(1.0 - occupation)
