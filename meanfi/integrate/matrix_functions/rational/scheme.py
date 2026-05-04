from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import linalg as scipy_linalg

from ..base import RationalFOE
from ...occupations import fermi_dirac
from .common import SparseRationalTerms

@dataclass
class _AAABuilderState:
    training_x: np.ndarray
    training_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    support_indices: list[int]
    support_mask: np.ndarray
    support_x: np.ndarray
    support_y: np.ndarray
    weights: np.ndarray
    approx_training: np.ndarray
    approx_validation: np.ndarray


@dataclass(frozen=True)
class _AAAIntervalCacheEntry:
    lower: float
    upper: float
    kT: float
    support_x: np.ndarray
    support_y: np.ndarray
    weights: np.ndarray
    terms: SparseRationalTerms


@lru_cache(maxsize=None)
def _ozaki_exact_poles_and_residues(pole_count: int) -> tuple[np.ndarray, np.ndarray]:
    if pole_count <= 0:
        raise ValueError("pole_count must be positive")

    matrix_size = 2 * int(pole_count)
    diagonal_weights = np.arange(1, 2 * matrix_size, 2, dtype=float)
    off_diagonal = -0.5 / np.sqrt(diagonal_weights[:-1] * diagonal_weights[1:])
    continued_fraction = np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(continued_fraction)
    negative = eigenvalues < 0.0
    if int(np.count_nonzero(negative)) != int(pole_count):
        raise ValueError("Ozaki pole construction returned an unexpected pole count")

    selected_eigenvalues = eigenvalues[negative]
    selected_vectors = eigenvectors[0, negative]
    poles = -1.0 / selected_eigenvalues
    residues = -0.25 * np.square(selected_vectors) * np.square(poles)
    return np.asarray(poles, dtype=float), np.asarray(residues, dtype=float)


def _lobatto_grid(lower: float, upper: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([0.5 * (lower + upper)], dtype=float)
    theta = np.linspace(0.0, np.pi, int(count), dtype=float)
    nodes = np.cos(theta)[::-1]
    center = 0.5 * (upper + lower)
    scale = 0.5 * (upper - lower)
    return center + scale * nodes


def _fermi_tail_bounds(kT: float, tolerance: float) -> tuple[float, float]:
    clipped = min(max(float(tolerance), 1e-15), 0.499999999999)
    bound = float(kT) * float(np.log((1.0 - clipped) / clipped))
    return -bound, bound


def _whole_interval_tail_constant(
    lower: float,
    upper: float,
    *,
    kT: float,
    tolerance: float,
) -> complex | None:
    lower_tail, upper_tail = _fermi_tail_bounds(kT, tolerance)
    if upper <= lower_tail:
        return complex(1.0)
    if lower >= upper_tail:
        return complex(0.0)
    return None


def _aaa_dense_zero_points(lower: float, upper: float, kT: float) -> np.ndarray:
    if not (lower <= 0.0 <= upper):
        return np.empty(0, dtype=float)
    half_window = max(8.0 * float(kT), 1e-12)
    lo = max(lower, -half_window)
    hi = min(upper, half_window)
    if hi <= lo:
        return np.asarray([0.0], dtype=float)
    return np.linspace(lo, hi, 64, dtype=float)


def _aaa_sample_grid(lower: float, upper: float, *, count: int, kT: float) -> np.ndarray:
    points = np.concatenate(
        [
            _lobatto_grid(lower, upper, count),
            np.asarray([lower, upper], dtype=float),
            _aaa_dense_zero_points(lower, upper, kT),
        ]
    )
    return np.unique(points)


def _barycentric_evaluate(
    x: np.ndarray,
    support_x: np.ndarray,
    support_y: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    support_x = np.asarray(support_x, dtype=float)
    support_y = np.asarray(support_y, dtype=complex)
    weights = np.asarray(weights, dtype=complex)
    if support_x.size == 0:
        raise ValueError("Barycentric evaluation requires at least one support point")
    if support_x.size == 1:
        return np.full(x.shape, support_y[0], dtype=complex)

    diff = x[:, np.newaxis] - support_x[np.newaxis, :]
    scale = max(
        1.0,
        float(np.max(np.abs(support_x), initial=0.0)),
        float(np.max(np.abs(x), initial=0.0)),
    )
    exact_hits = np.abs(diff) <= 32.0 * np.finfo(float).eps * scale
    safe_diff = np.where(exact_hits, 1.0, diff)
    scaled = weights[np.newaxis, :] / safe_diff
    values = (scaled @ support_y) / np.sum(scaled, axis=1)
    if np.any(exact_hits):
        hit_rows = np.any(exact_hits, axis=1)
        hit_cols = np.argmax(exact_hits[hit_rows], axis=1)
        values[hit_rows] = support_y[hit_cols]
    return values


def _fit_barycentric_weights(
    x: np.ndarray,
    y: np.ndarray,
    support_indices: list[int],
) -> _AAABuilderState:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=complex)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("AAA samples must be one-dimensional arrays of equal length")
    if x.size == 0:
        raise ValueError("AAA sample grid must be non-empty")
    support_mask = np.zeros(x.size, dtype=bool)
    support_mask[np.asarray(support_indices, dtype=int)] = True
    support_x = x[np.asarray(support_indices, dtype=int)]
    support_y = y[np.asarray(support_indices, dtype=int)]
    if len(support_indices) == 1:
        weights = np.ones(1, dtype=complex)
        approx = np.full(x.shape, support_y[0], dtype=complex)
        return _AAABuilderState(
            training_x=x,
            training_y=y,
            validation_x=np.empty(0, dtype=float),
            validation_y=np.empty(0, dtype=complex),
            support_indices=list(support_indices),
            support_mask=support_mask,
            support_x=support_x,
            support_y=support_y,
            weights=weights,
            approx_training=approx,
            approx_validation=np.empty(0, dtype=complex),
        )

    sample_x = x[~support_mask]
    sample_y = y[~support_mask]
    cauchy = 1.0 / (sample_x[:, np.newaxis] - support_x[np.newaxis, :])
    loewner = (sample_y[:, np.newaxis] - support_y[np.newaxis, :]) * cauchy
    _u, _s, vh = np.linalg.svd(loewner, full_matrices=False)
    weights = np.asarray(vh[-1], dtype=complex)
    approx = _barycentric_evaluate(x, support_x, support_y, weights)
    return _AAABuilderState(
        training_x=x,
        training_y=y,
        validation_x=np.empty(0, dtype=float),
        validation_y=np.empty(0, dtype=complex),
        support_indices=list(support_indices),
        support_mask=support_mask,
        support_x=support_x,
        support_y=support_y,
        weights=weights,
        approx_training=approx,
        approx_validation=np.empty(0, dtype=complex),
    )


def _initialize_aaa_builder(
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> _AAABuilderState:
    initial_index = int(np.argmax(np.abs(training_y - np.mean(training_y))))
    state = _fit_barycentric_weights(training_x, training_y, [initial_index])
    state.validation_x = np.asarray(validation_x, dtype=float)
    state.validation_y = np.asarray(validation_y, dtype=complex)
    state.approx_validation = np.full(validation_x.shape, state.support_y[0], dtype=complex)
    return state


def _advance_aaa_builder(state: _AAABuilderState) -> _AAABuilderState:
    residual = np.abs(state.training_y - state.approx_training)
    residual[state.support_mask] = -np.inf
    pivot = int(np.argmax(residual))
    if pivot in state.support_indices:
        return state
    next_state = _fit_barycentric_weights(
        state.training_x,
        state.training_y,
        [*state.support_indices, pivot],
    )
    next_state.validation_x = state.validation_x
    next_state.validation_y = state.validation_y
    next_state.approx_validation = _barycentric_evaluate(
        state.validation_x,
        next_state.support_x,
        next_state.support_y,
        next_state.weights,
    )
    return next_state


def _extract_aaa_poles_arrowhead(
    support_x: np.ndarray,
    support_y: np.ndarray,
    weights: np.ndarray,
) -> tuple[complex, np.ndarray, np.ndarray]:
    support_x = np.asarray(support_x, dtype=complex)
    support_y = np.asarray(support_y, dtype=complex)
    weights = np.asarray(weights, dtype=complex)
    if support_x.size <= 1:
        return complex(support_y[0]), np.empty(0, dtype=complex), np.empty(0, dtype=complex)

    size = support_x.size
    arrowhead = np.zeros((size + 1, size + 1), dtype=complex)
    metric = np.zeros((size + 1, size + 1), dtype=complex)
    arrowhead[0, 1:] = weights
    arrowhead[1:, 0] = 1.0
    arrowhead[1:, 1:] = np.diag(support_x)
    metric[1:, 1:] = np.eye(size, dtype=complex)
    poles = np.asarray(scipy_linalg.eigvals(arrowhead, metric), dtype=complex)
    finite_mask = np.isfinite(poles)
    poles = poles[finite_mask]
    if poles.size:
        support_scale = max(1.0, float(np.max(np.abs(support_x))))
        support_tol = 1e-9 * support_scale
        poles = poles[
            np.array(
                [
                    np.min(np.abs(pole - support_x), initial=np.inf) > support_tol
                    for pole in poles
                ],
                dtype=bool,
            )
        ]
    if poles.size == 0:
        constant = complex(np.sum(weights * support_y) / np.sum(weights))
        return constant, poles, np.empty(0, dtype=complex)

    numerator = np.array(
        [
            np.sum(weights * support_y / (pole - support_x))
            for pole in poles
        ],
        dtype=complex,
    )
    denominator_derivative = -np.array(
        [
            np.sum(weights / np.square(pole - support_x))
            for pole in poles
        ],
        dtype=complex,
    )
    residues = numerator / denominator_derivative
    constant = complex(np.sum(weights * support_y) / np.sum(weights))
    return constant, np.asarray(poles, dtype=complex), np.asarray(residues, dtype=complex)


def _canonicalize_conjugate_terms(
    constant: complex,
    poles: np.ndarray,
    residues: np.ndarray,
) -> tuple[complex, np.ndarray, np.ndarray]:
    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)
    if poles.size == 0:
        return complex(np.real_if_close(constant)), poles, residues

    scale = max(1.0, float(np.max(np.abs(poles))))
    pair_tol = 1e-7 * scale
    used = np.zeros(poles.size, dtype=bool)
    canonical_poles: list[complex] = []
    canonical_residues: list[complex] = []

    for index, pole in enumerate(poles):
        if used[index]:
            continue
        residue = residues[index]
        if abs(np.imag(pole)) <= pair_tol:
            canonical_poles.append(complex(np.real(pole)))
            canonical_residues.append(complex(np.real(residue)) / 2.0)
            used[index] = True
            continue

        if np.imag(pole) < 0.0:
            target = np.conjugate(pole)
            candidates = [
                candidate
                for candidate in range(poles.size)
                if not used[candidate]
                and abs(poles[candidate] - target) <= pair_tol
            ]
            if candidates:
                continue

        target = np.conjugate(pole)
        partner_index = None
        partner_error = float("inf")
        for candidate in range(poles.size):
            if candidate == index or used[candidate]:
                continue
            error = abs(poles[candidate] - target)
            if error <= pair_tol and error < partner_error:
                partner_index = candidate
                partner_error = error
        if partner_index is None:
            canonical_pole = pole if np.imag(pole) > 0.0 else np.conjugate(pole)
            canonical_residue = residue if np.imag(pole) > 0.0 else np.conjugate(residue)
            canonical_poles.append(complex(canonical_pole))
            canonical_residues.append(complex(canonical_residue))
            used[index] = True
            continue

        partner_pole = poles[partner_index]
        partner_residue = residues[partner_index]
        upper_pole = pole if np.imag(pole) > 0.0 else partner_pole
        upper_residue = residue if np.imag(pole) > 0.0 else partner_residue
        lower_pole = partner_pole if np.imag(pole) > 0.0 else pole
        lower_residue = partner_residue if np.imag(pole) > 0.0 else residue
        canonical_poles.append(
            complex(0.5 * (upper_pole + np.conjugate(lower_pole)))
        )
        canonical_residues.append(
            complex(0.5 * (upper_residue + np.conjugate(lower_residue)))
        )
        used[index] = True
        used[partner_index] = True

    return (
        complex(np.real_if_close(constant)),
        np.asarray(canonical_poles, dtype=complex),
        np.asarray(canonical_residues, dtype=complex),
    )


def _evaluate_canonical_rational(
    x: np.ndarray,
    *,
    constant: complex,
    shifts: np.ndarray,
    residues: np.ndarray,
    tail_lower_bound: float | None = None,
    tail_upper_bound: float | None = None,
) -> np.ndarray:
    del tail_lower_bound, tail_upper_bound
    x = np.asarray(x, dtype=float)
    values = np.full(x.shape, constant, dtype=np.complex128)
    for shift, residue in zip(shifts, residues, strict=True):
        values += residue / (x - shift)
        values += np.conjugate(residue) / (x - np.conjugate(shift))
    return values


def _cleanup_froissart_doublets(
    state: _AAABuilderState,
    *,
    lower: float,
    upper: float,
    scalar_tolerance: float,
    tail_lower_bound: float | None,
    tail_upper_bound: float | None,
) -> _AAABuilderState:
    if len(state.support_indices) <= 2:
        return state
    interval_scale = max(1.0, float(upper - lower))
    residue_tol = scalar_tolerance * interval_scale
    while len(state.support_indices) > 2:
        constant, poles, residues = _extract_aaa_poles_arrowhead(
            state.support_x,
            state.support_y,
            state.weights,
        )
        if poles.size == 0:
            return state
        tiny = np.flatnonzero(np.abs(residues) < residue_tol)
        if tiny.size == 0:
            return state

        removed_any = False
        for pole_index in tiny:
            nearest_support = int(np.argmin(np.abs(state.support_x - poles[pole_index])))
            reduced_support = [
                index
                for local_index, index in enumerate(state.support_indices)
                if local_index != nearest_support
            ]
            if len(reduced_support) < 2:
                continue
            candidate = _fit_barycentric_weights(
                state.training_x,
                state.training_y,
                reduced_support,
            )
            candidate.validation_x = state.validation_x
            candidate.validation_y = state.validation_y
            candidate.approx_validation = _barycentric_evaluate(
                candidate.validation_x,
                candidate.support_x,
                candidate.support_y,
                candidate.weights,
            )
            candidate_constant, candidate_poles, candidate_residues = _extract_aaa_poles_arrowhead(
                candidate.support_x,
                candidate.support_y,
                candidate.weights,
            )
            candidate_constant, candidate_shifts, candidate_residues = _canonicalize_conjugate_terms(
                candidate_constant,
                candidate_poles,
                candidate_residues,
            )
            pole_values = _evaluate_canonical_rational(
                candidate.validation_x,
                constant=candidate_constant,
                shifts=candidate_shifts,
                residues=candidate_residues,
                tail_lower_bound=tail_lower_bound,
                tail_upper_bound=tail_upper_bound,
            )
            scalar_error = float(
                np.max(np.abs(candidate.validation_y - pole_values), initial=0.0)
            )
            barycentric_gap = float(
                np.max(np.abs(pole_values - candidate.approx_validation), initial=0.0)
            )
            if scalar_error <= scalar_tolerance and barycentric_gap <= 0.1 * scalar_tolerance:
                state = candidate
                removed_any = True
                break
        if not removed_any:
            return state
    return state


def _aaa_terms_from_builder(
    state: _AAABuilderState,
    *,
    lower: float,
    upper: float,
    scalar_tolerance: float,
    tail_lower_bound: float | None,
    tail_upper_bound: float | None,
) -> tuple[SparseRationalTerms, float, float]:
    cleaned = _cleanup_froissart_doublets(
        state,
        lower=lower,
        upper=upper,
        scalar_tolerance=scalar_tolerance,
        tail_lower_bound=tail_lower_bound,
        tail_upper_bound=tail_upper_bound,
    )
    constant, poles, residues = _extract_aaa_poles_arrowhead(
        cleaned.support_x,
        cleaned.support_y,
        cleaned.weights,
    )
    constant, shifts, residues = _canonicalize_conjugate_terms(
        constant,
        poles,
        residues,
    )
    pole_values = _evaluate_canonical_rational(
        cleaned.validation_x,
        constant=constant,
        shifts=shifts,
        residues=residues,
        tail_lower_bound=tail_lower_bound,
        tail_upper_bound=tail_upper_bound,
    )
    scalar_error = float(
        np.max(np.abs(cleaned.validation_y - pole_values), initial=0.0)
    )
    barycentric_gap = float(
        np.max(np.abs(pole_values - cleaned.approx_validation), initial=0.0)
    )
    terms = SparseRationalTerms(
        constant=constant,
        shifts=np.asarray(shifts, dtype=np.complex128),
        residues=np.asarray(residues, dtype=np.complex128),
        pole_count=len(cleaned.support_indices),
        support_count=len(cleaned.support_indices),
        tail_lower_bound=tail_lower_bound,
        tail_upper_bound=tail_upper_bound,
    )
    return terms, scalar_error, barycentric_gap


def _aaa_terms_for_interval(
    pole_cap: int,
    *,
    lower: float,
    upper: float,
    kT: float,
    initial_poles: int = 1,
    scalar_tolerance: float,
) -> tuple[SparseRationalTerms, _AAABuilderState]:
    constant_tail = _whole_interval_tail_constant(
        lower,
        upper,
        kT=kT,
        tolerance=scalar_tolerance,
    )
    if constant_tail is not None:
        terms = SparseRationalTerms(
            constant=constant_tail,
            shifts=np.empty(0, dtype=np.complex128),
            residues=np.empty(0, dtype=np.complex128),
            pole_count=0,
            support_count=0,
        )
        dummy_grid = np.asarray([lower, upper], dtype=float)
        dummy_target = np.asarray(fermi_dirac(dummy_grid, kT, 0.0), dtype=complex)
        return terms, _AAABuilderState(
            training_x=dummy_grid,
            training_y=dummy_target,
            validation_x=dummy_grid,
            validation_y=dummy_target,
            support_indices=[],
            support_mask=np.zeros(dummy_grid.size, dtype=bool),
            support_x=np.empty(0, dtype=float),
            support_y=np.empty(0, dtype=complex),
            weights=np.empty(0, dtype=complex),
            approx_training=np.full(dummy_grid.shape, constant_tail, dtype=complex),
            approx_validation=np.full(dummy_grid.shape, constant_tail, dtype=complex),
        )

    tail_lower_bound, tail_upper_bound = _fermi_tail_bounds(kT, scalar_tolerance)
    training_grid = _aaa_sample_grid(
        lower,
        upper,
        count=max(256, 16 * int(pole_cap)),
        kT=kT,
    )
    target = fermi_dirac(training_grid, kT, 0.0)
    certification_grid = _aaa_sample_grid(
        lower,
        upper,
        count=max(1024, 64 * int(pole_cap)),
        kT=kT,
    )
    certification_target = np.asarray(fermi_dirac(certification_grid, kT, 0.0), dtype=complex)
    sample_target = np.asarray(target, dtype=complex)
    builder = _initialize_aaa_builder(
        training_grid,
        sample_target,
        certification_grid,
        certification_target,
    )
    best_error = float("inf")
    best_gap = float("inf")
    best_terms: SparseRationalTerms | None = None
    minimum_support = max(1, int(initial_poles))

    while True:
        if len(builder.support_indices) >= minimum_support:
            terms, scalar_error, barycentric_gap = _aaa_terms_from_builder(
                builder,
                lower=lower,
                upper=upper,
                scalar_tolerance=scalar_tolerance,
                tail_lower_bound=tail_lower_bound,
                tail_upper_bound=tail_upper_bound,
            )
            if (
                scalar_error < best_error
                or (np.isclose(scalar_error, best_error) and barycentric_gap < best_gap)
            ):
                best_error = scalar_error
                best_gap = barycentric_gap
                best_terms = terms
            if scalar_error <= scalar_tolerance and barycentric_gap <= 0.1 * scalar_tolerance:
                return terms, builder
        if len(builder.support_indices) >= int(pole_cap):
            break
        builder = _advance_aaa_builder(builder)

    raise ValueError(
        "AAA scalar certification failed within the requested pole budget"
        if best_terms is None
        else f"AAA scalar certification failed within the requested pole budget (best error={best_error:.3e})"
    )


def _scheme_terms(
    options: RationalFOE,
    pole_count: int,
    *,
    lower: float,
    upper: float,
    kT: float,
) -> tuple[complex, np.ndarray, np.ndarray]:
    if options.rational_scheme == "ozaki":
        poles, residues = _ozaki_exact_poles_and_residues(int(pole_count))
        return (
            complex(0.5),
            1j * np.asarray(poles, dtype=float) * float(kT),
            np.asarray(residues, dtype=float) * float(kT),
        )
    if options.rational_scheme == "aaa":
        raise ValueError(
            "rational_scheme='aaa' is currently supported only on the sparse MUMPS RationalFOE path"
        )
    raise ValueError(f"Unsupported RationalFOE scheme: {options.rational_scheme}")

