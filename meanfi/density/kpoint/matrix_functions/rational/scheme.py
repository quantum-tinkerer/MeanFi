"""Fermi-function fits controlling matrix-entry accuracy; entropy is diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import linalg as scipy_linalg
from scipy.special import expit

from .common import SparseRationalTerms
from meanfi.density.kpoint.occupations import fermi_dirac


@dataclass(frozen=True)
class _AAAIntervalCacheEntry:
    lower: float
    upper: float
    kT: float
    terms: SparseRationalTerms


def thermal_targets(energies: np.ndarray, kT: float) -> np.ndarray:
    """Return occupation and entropy in units of k_B without log(0)."""
    with np.errstate(over="ignore"):
        scaled = energies / kT
    magnitude = np.abs(scaled)
    entropy = np.logaddexp(0.0, -magnitude) + np.multiply(
        magnitude,
        expit(-magnitude),
        out=np.zeros_like(magnitude),
        where=np.isfinite(magnitude),
    )
    return np.column_stack((expit(-scaled), entropy))


def _aaa_sample_grid(
    lower: float, upper: float, *, kT: float, count: int
) -> np.ndarray:
    """Resolve band edges, the Fermi transition, and exponentially small tails."""
    center, scale = 0.5 * (lower + upper), 0.5 * (upper - lower)
    lobatto = center + scale * np.cos(np.linspace(0.0, np.pi, count))
    thermal = np.linspace(-8.0 * kT, 8.0 * kT, 64)
    tails = np.geomspace(kT, max(abs(lower), abs(upper), kT), max(64, count // 8))
    grid = np.unique(np.concatenate((lobatto, thermal, tails, -tails, [lower, upper])))
    return grid[(grid >= lower) & (grid <= upper)]


def _aaa_poles(support: np.ndarray, weights: np.ndarray) -> np.ndarray:
    size = support.size
    arrowhead = np.zeros((size + 1, size + 1))
    arrowhead[0, 1:] = weights
    arrowhead[1:, 0] = 1.0
    arrowhead[1:, 1:] = np.diag(support)
    metric = np.diag(np.r_[0.0, np.ones(size)])
    poles = scipy_linalg.eigvals(arrowhead, metric)
    poles = poles[np.isfinite(poles)]
    # The pencil is real, so non-real poles arrive in conjugate pairs.
    return poles[poles.imag >= 0.0]


def _evaluate_canonical_rational(
    x: np.ndarray, *, constant: complex, shifts: np.ndarray, residues: np.ndarray
) -> np.ndarray:
    values = np.full(np.shape(x), constant, dtype=complex)
    for shift, residue in zip(shifts, residues, strict=True):
        values += residue / (x - shift)
        values += np.conjugate(residue) / (x - np.conjugate(shift))
    return values


def _fit_residues(x: np.ndarray, targets: np.ndarray, shifts: np.ndarray):
    resolvents = 1.0 / (x[:, None] - shifts)
    design = np.column_stack(
        (np.ones(x.size), 2.0 * resolvents.real, -2.0 * resolvents.imag)
    )
    scales = np.linalg.norm(design, axis=0)
    scales[scales == 0.0] = 1.0
    coefficients = np.linalg.lstsq(design / scales, targets, rcond=None)[0]
    coefficients /= scales[:, None]
    pole_count = shifts.size
    return (
        coefficients[0],
        coefficients[1 : pole_count + 1] + 1j * coefficients[pole_count + 1 :],
    )


def thermal_errors(
    terms: SparseRationalTerms, grid: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    coefficients = [(terms.constant, terms.residues)]
    if targets.shape[1] == 2:
        coefficients.append((terms.entropy_constant, terms.entropy_residues))
    values = np.column_stack(
        [
            _evaluate_canonical_rational(
                grid, constant=constant, shifts=terms.shifts, residues=residues
            )
            for constant, residues in coefficients
        ]
    )
    return np.max(np.abs(values - targets), axis=0)


def _aaa_terms_for_interval(
    pole_cap: int,
    *,
    lower: float,
    upper: float,
    kT: float,
    scalar_tolerance: float,
    initial_poles: int = 1,
) -> SparseRationalTerms:
    """Fit the Fermi function and check its error before matrix factorization.

    For Hermitian A, max_ij |[r(A)-f(A)]_ij| <= max_spectrum |r-f|.
    The sampled scalar check supports the density-entry target; it is not a
    rigorous interval bound. Entropy never changes these poles or acceptance.
    """
    max_poles = pole_cap
    validation = _aaa_sample_grid(lower, upper, kT=kT, count=max(2048, 32 * max_poles))
    validation_targets = fermi_dirac(validation, kT, 0.0)
    best_error = float("inf")
    sample_count = max(512, 8 * initial_poles)
    max_samples = max(512, 8 * max_poles)
    while True:
        x = _aaa_sample_grid(lower, upper, kT=kT, count=sample_count)
        targets = fermi_dirac(x, kT, 0.0)
        constant = float(np.mean(targets))
        error = float(np.max(np.abs(validation_targets - constant)))
        if error <= scalar_tolerance:
            return SparseRationalTerms(
                constant=complex(constant),
                shifts=np.empty(0, dtype=complex),
                residues=np.empty(0, dtype=complex),
                pole_count=0,
                error=error,
            )

        center, radius = 0.5 * (lower + upper), 0.5 * (upper - lower)
        scaled_x, unique = np.unique((x - center) / radius, return_index=True)
        x, targets = x[unique], targets[unique]
        support: list[int] = []
        mask = np.ones(x.size, dtype=bool)
        approximation = np.full_like(targets, constant)
        for _ in range(min(max_poles, sample_count // 8)):
            residual = np.abs(targets - approximation) / scalar_tolerance
            residual[~mask] = -np.inf
            pivot = int(np.argmax(residual))
            support.append(pivot)
            mask[pivot] = False
            support_x, support_y = scaled_x[support], targets[support]
            if len(support) == 1:
                approximation[:] = support_y[0]
                continue
            cauchy = 1.0 / (scaled_x[mask, None] - support_x)
            loewner = (targets[mask, None] - support_y) * cauchy
            # Q preserves norms: minimizing ||L w|| is equivalent to ||R w||.
            triangular = np.linalg.qr(loewner, mode="r")
            weights = np.linalg.svd(triangular, full_matrices=False)[2][-1]
            weighted_cauchy = cauchy * weights
            approximation[mask] = (weighted_cauchy @ support_y) / np.sum(
                weighted_cauchy, axis=1
            )
            approximation[support] = support_y
            # The residue refit can improve a nearly converged barycentric fit.
            # Only final partial-fraction errors decide acceptance below.
            if (
                len(support) < initial_poles
                or np.max(np.abs(approximation - targets)) > 10 * scalar_tolerance
            ):
                continue
            shifts = center + radius * _aaa_poles(support_x, weights)
            # Real poles inside the spectrum are spurious pole/zero pairs. Discard
            # them before refitting; certification checks the remaining expansion.
            shifts = shifts[
                (shifts.imag != 0.0) | (shifts.real < lower) | (shifts.real > upper)
            ]
            constants, residues = _fit_residues(x, targets[:, None], shifts)
            terms = SparseRationalTerms(
                constant=complex(constants[0]),
                shifts=shifts,
                residues=residues[:, 0],
                pole_count=len(support),
            )
            error = thermal_errors(terms, validation, validation_targets[:, None])[0]
            best_error = min(best_error, float(error / scalar_tolerance))
            if error <= scalar_tolerance:
                return replace(terms, error=float(error))
        if sample_count == max_samples:
            break
        sample_count = min(2 * sample_count, max_samples)
    raise ValueError(
        "AAA scalar certification failed within max_poles "
        f"(best error/tolerance={best_error:.3e})"
    )


def fit_entropy(
    terms: SparseRationalTerms, *, lower: float, upper: float, kT: float
) -> SparseRationalTerms:
    """Fit entropy on the accepted density poles, without a new accuracy target."""
    grid = _aaa_sample_grid(lower, upper, kT=kT, count=max(2048, 32 * terms.pole_count))
    constants, residues = _fit_residues(
        grid, thermal_targets(grid, kT)[:, 1:], terms.shifts
    )
    result = replace(
        terms, entropy_constant=complex(constants[0]), entropy_residues=residues[:, 0]
    )
    validation = _aaa_sample_grid(
        lower, upper, kT=kT, count=max(4096, 64 * terms.pole_count)
    )
    # Entropy residuals can peak sharply near the thermal transition even when
    # the wider spectral grid is dense. Resolve that region independently.
    thermal = np.linspace(-40 * kT, 40 * kT, 4097)
    validation = np.unique(
        np.r_[validation, thermal[(thermal >= lower) & (thermal <= upper)]]
    )
    error = thermal_errors(result, validation, thermal_targets(validation, kT))[1]
    return replace(result, entropy_error=float(error))
