from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import linalg as scipy_linalg
from meanfi.core.matrix import as_sparse, is_sparse_like, sparse_linalg_module
from meanfi.integrate.density_support import DensityEntrySupport

from .base import RationalFOE, _BlockResult
from .common import (
    _derivative_convergence,
    scalar_derivative_converged,
    spectral_interval,
    workspace_matrix,
)
from ..occupations import fermi_dirac
from .common import shift_by_mu
from .mumps_backend import (
    MumpsSelectedEntryPattern,
    SelectedInverseFactorization,
    build_selected_entry_pattern,
    require_mumps,
)


def _support_requested_pairs(density_support: DensityEntrySupport) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for block_rows, block_cols in zip(
        density_support.row_indices,
        density_support.col_indices,
        strict=True,
    ):
        if block_rows.size == 0:
            continue
        rows.append(np.asarray(block_rows, dtype=int))
        cols.append(np.asarray(block_cols, dtype=int))
    if not rows:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    stacked_rows = np.concatenate(rows)
    stacked_cols = np.concatenate(cols)
    pairs = np.unique(np.stack([stacked_rows, stacked_cols], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


@dataclass(frozen=True)
class SparseChargePattern:
    pattern: MumpsSelectedEntryPattern
    diagonal_positions: np.ndarray
    charge_weights: np.ndarray

    def charge_from_inverse_entries(
        self,
        inverse_entries: dict[complex, np.ndarray],
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
    ) -> float:
        diagonal = np.full(self.diagonal_positions.size, complex(constant), dtype=np.complex128)
        for shift, residue in zip(shifts, residues, strict=True):
            pole_entries = inverse_entries[complex(shift)]
            diagonal += residue * pole_entries[self.diagonal_positions]
            diagonal += np.conjugate(residue) * np.conjugate(
                pole_entries[self.diagonal_positions]
            )
        return float(np.real(np.sum(self.charge_weights * diagonal)))


@dataclass(frozen=True)
class SparseDensityPattern:
    pattern: MumpsSelectedEntryPattern
    requested_rows: np.ndarray
    requested_cols: np.ndarray
    requested_col_positions: np.ndarray
    requested_positions: np.ndarray
    requested_reverse_positions: np.ndarray
    requested_is_diagonal: np.ndarray
    selected_columns: np.ndarray

    def density_columns_from_inverse_entries(
        self,
        inverse_entries: dict[complex, np.ndarray],
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
    ) -> np.ndarray:
        columns = np.zeros(
            (self.pattern.size, self.selected_columns.size),
            dtype=np.complex128,
        )
        if self.requested_positions.size == 0:
            return columns

        values = np.zeros(self.requested_positions.size, dtype=np.complex128)
        values[self.requested_is_diagonal] += complex(constant)
        for shift, residue in zip(shifts, residues, strict=True):
            pole_entries = inverse_entries[complex(shift)]
            values += residue * pole_entries[self.requested_positions]
            values += np.conjugate(residue) * np.conjugate(
                pole_entries[self.requested_reverse_positions]
            )
        columns[self.requested_rows, self.requested_col_positions] = values
        return columns


def build_sparse_charge_pattern(trace_weights_diag: np.ndarray) -> SparseChargePattern:
    weights = np.asarray(trace_weights_diag, dtype=float)
    diagonal = np.flatnonzero(np.abs(weights) > 0.0).astype(int, copy=False)
    pattern = build_selected_entry_pattern(size=weights.size, rows=diagonal, cols=diagonal)
    diagonal_positions = np.asarray(
        [pattern.lookup[(int(index), int(index))] for index in diagonal],
        dtype=int,
    )
    return SparseChargePattern(
        pattern=pattern,
        diagonal_positions=diagonal_positions,
        charge_weights=weights[diagonal],
    )


def build_sparse_density_pattern(
    *,
    size: int,
    density_support: DensityEntrySupport,
) -> SparseDensityPattern:
    requested_rows, requested_cols = _support_requested_pairs(density_support)
    reverse_rows = requested_cols
    reverse_cols = requested_rows
    pattern = build_selected_entry_pattern(
        size=size,
        rows=np.concatenate([requested_rows, reverse_rows]),
        cols=np.concatenate([requested_cols, reverse_cols]),
    )
    selected_lookup = np.full(size, -1, dtype=int)
    selected_lookup[np.asarray(density_support.selected_columns, dtype=int)] = np.arange(
        density_support.selected_columns.size,
        dtype=int,
    )
    requested_positions = np.asarray(
        [
            pattern.lookup[(int(row), int(col))]
            for row, col in zip(requested_rows, requested_cols, strict=True)
        ],
        dtype=int,
    )
    requested_reverse_positions = np.asarray(
        [
            pattern.lookup[(int(row), int(col))]
            for row, col in zip(reverse_rows, reverse_cols, strict=True)
        ],
        dtype=int,
    )
    requested_col_positions = (
        selected_lookup[requested_cols]
        if requested_cols.size
        else np.empty(0, dtype=int)
    )
    return SparseDensityPattern(
        pattern=pattern,
        requested_rows=np.asarray(requested_rows, dtype=int),
        requested_cols=np.asarray(requested_cols, dtype=int),
        requested_col_positions=np.asarray(requested_col_positions, dtype=int),
        requested_positions=requested_positions,
        requested_reverse_positions=requested_reverse_positions,
        requested_is_diagonal=np.asarray(requested_rows == requested_cols, dtype=bool),
        selected_columns=np.asarray(density_support.selected_columns, dtype=int),
    )


def _pattern_subset_mappings(
    source: MumpsSelectedEntryPattern,
    target: MumpsSelectedEntryPattern,
) -> tuple[np.ndarray, np.ndarray]:
    source_positions: list[int] = []
    target_positions: list[int] = []
    for pair, source_position in source.lookup.items():
        target_position = target.lookup.get(pair)
        if target_position is None:
            continue
        source_positions.append(int(source_position))
        target_positions.append(int(target_position))
    return (
        np.asarray(source_positions, dtype=int),
        np.asarray(target_positions, dtype=int),
    )


@dataclass(frozen=True)
class SparseRationalTerms:
    constant: complex
    shifts: np.ndarray
    residues: np.ndarray
    pole_count: int
    support_count: int | None = None
    tail_lower_bound: float | None = None
    tail_upper_bound: float | None = None


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


def _dense_shifted_matrix(matrix: np.ndarray, shift: complex) -> np.ndarray:
    shifted = np.array(matrix, copy=True)
    diagonal = shifted.diagonal().copy()
    diagonal -= complex(shift)
    np.fill_diagonal(shifted, diagonal)
    return shifted


def _sparse_shifted_matrix(matrix: Any, shift: complex):
    shifted = as_sparse(matrix).tocsc()
    shifted = shifted.copy()
    diagonal = np.asarray(shifted.diagonal(), dtype=complex)
    diagonal -= complex(shift)
    shifted.setdiag(diagonal)
    return shifted


def _sparse_shifted_lu(matrix: Any, shift: complex):
    return sparse_linalg_module().splu(_sparse_shifted_matrix(matrix, shift))


def _evaluate_rational_terms(
    matrix: Any,
    block: np.ndarray,
    *,
    constant: complex,
    shifts: np.ndarray,
    residues: np.ndarray,
    q_diag: np.ndarray,
    derivative: bool,
    workspace_dtype: np.dtype,
    lu_cache: dict[complex, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[complex, Any] | None]:
    matrix = workspace_matrix(matrix, workspace_dtype)
    block = np.asarray(block, dtype=workspace_dtype)
    density_result = np.asarray(constant * block, dtype=complex)
    derivative_block = np.zeros_like(block, dtype=complex) if derivative else None

    if is_sparse_like(matrix):
        active_cache = {} if lu_cache is None else dict(lu_cache)
        for shift, residue in zip(shifts, residues, strict=True):
            key = complex(shift)
            lu = active_cache.get(key)
            if lu is None:
                lu = _sparse_shifted_lu(matrix, key)
                active_cache[key] = lu
            rhs = np.asarray(block, dtype=workspace_dtype)
            y = np.asarray(lu.solve(rhs), dtype=complex)
            y_adj = np.asarray(
                lu.solve(np.asarray(block, dtype=workspace_dtype), trans="H"),
                dtype=complex,
            )
            density_result = density_result + residue * y + np.conjugate(residue) * y_adj

            if derivative:
                rhs = np.asarray(q_diag[:, np.newaxis] * y, dtype=workspace_dtype)
                rhs_adj = np.asarray(q_diag[:, np.newaxis] * y_adj, dtype=workspace_dtype)
                z = np.asarray(lu.solve(rhs), dtype=complex)
                z_adj = np.asarray(lu.solve(rhs_adj, trans="H"), dtype=complex)
                derivative_block = (
                    derivative_block
                    + residue * z
                    + np.conjugate(residue) * z_adj
                )
        return density_result, derivative_block, active_cache

    dense_matrix = np.asarray(matrix, dtype=workspace_dtype)
    for shift, residue in zip(shifts, residues, strict=True):
        shifted = _dense_shifted_matrix(dense_matrix, shift)
        shifted_adjoint = shifted.conj().T
        y = np.linalg.solve(shifted, block)
        y_adj = np.linalg.solve(shifted_adjoint, block)
        density_result = density_result + residue * y + np.conjugate(residue) * y_adj

        if derivative:
            rhs = q_diag[:, np.newaxis] * y
            rhs_adj = q_diag[:, np.newaxis] * y_adj
            z = np.linalg.solve(shifted, rhs)
            z_adj = np.linalg.solve(shifted_adjoint, rhs_adj)
            derivative_block = derivative_block + residue * z + np.conjugate(residue) * z_adj

    return density_result, derivative_block, lu_cache


def _evaluate_rational_poles(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    pole_count: int,
    derivative: bool,
    options: RationalFOE,
    workspace_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray | None]:
    lower, upper = spectral_interval(matrix)
    constant, shifts, residues = _scheme_terms(
        options,
        pole_count,
        lower=lower,
        upper=upper,
        kT=kT,
    )
    density_result, derivative_block, _ = _evaluate_rational_terms(
        matrix,
        block,
        constant=constant,
        shifts=shifts,
        residues=residues,
        q_diag=q_diag,
        derivative=derivative,
        workspace_dtype=workspace_dtype,
    )
    return density_result, derivative_block


def _rational_density_block(
    matrix: Any,
    block: np.ndarray,
    *,
    kT: float,
    q_diag: np.ndarray,
    derivative: bool,
    tolerance: float,
    options: RationalFOE,
    derivative_trace_monitor=None,
    derivative_context: str | None = None,
    workspace_dtype: np.dtype = np.dtype(complex),
) -> _BlockResult:
    accepted_block = None
    accepted_derivative = None
    accepted_error = float("inf")
    accepted_order = None
    derivative_converged = True

    half_poles = int(options.initial_poles)
    half_block, half_derivative = _evaluate_rational_poles(
        matrix,
        block,
        kT=kT,
        q_diag=q_diag,
        pole_count=half_poles,
        derivative=derivative,
        options=options,
        workspace_dtype=workspace_dtype,
    )

    while 2 * half_poles <= int(options.max_poles):
        pole_count = 2 * half_poles
        full_block, full_derivative = _evaluate_rational_poles(
            matrix,
            block,
            kT=kT,
            q_diag=q_diag,
            pole_count=pole_count,
            derivative=derivative,
            options=options,
            workspace_dtype=workspace_dtype,
        )
        accepted_error = float(np.max(np.abs(full_block - half_block)))

        if derivative:
            derivative_converged, _derivative_error = _derivative_convergence(
                full_derivative,
                half_derivative,
                derivative=derivative,
                dn_dmu_rtol=options.dn_dmu_rtol,
                derivative_trace_monitor=derivative_trace_monitor,
                derivative_context=derivative_context,
                matrix_function_name="Rational FOE",
            )

        accepted_block = full_block
        accepted_derivative = full_derivative
        accepted_order = pole_count
        if accepted_error <= tolerance and derivative_converged:
            break

        half_poles = pole_count
        half_block = full_block
        half_derivative = full_derivative

    if accepted_error > tolerance or (derivative and not derivative_converged):
        raise ValueError("Rational FOE did not converge within max_poles")

    return _BlockResult(
        block=accepted_block,
        derivative_block=accepted_derivative,
        error=accepted_error,
        order=accepted_order,
    )


class PreparedRationalNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        q_diag: np.ndarray,
        options: RationalFOE,
        charge_tolerance: float,
        workspace_dtype: np.dtype = np.dtype(complex),
        trace_weights_diag: np.ndarray | None = None,
    ) -> None:
        self.workspace_dtype = np.dtype(workspace_dtype)
        self.matrix = workspace_matrix(matrix, self.workspace_dtype)
        self.kT = float(kT)
        self.q_diag = np.asarray(q_diag, dtype=float)
        self.options = options
        self.charge_tolerance = float(charge_tolerance)
        self.size = int(getattr(matrix, "shape")[0])
        self._trace_weights = (
            np.ones(self.size, dtype=float)
            if trace_weights_diag is None
            else np.asarray(trace_weights_diag, dtype=float)
        )
        self._trace_columns = np.flatnonzero(np.abs(self._trace_weights) > 0.0).astype(
            int,
            copy=False,
        )
        if self._trace_columns.size == 0:
            raise ValueError("Exact weighted trace requires at least one nonzero trace weight")
        self._trace_basis = np.zeros(
            (self.size, self._trace_columns.size),
            dtype=self.workspace_dtype,
        )
        self._trace_basis[
            self._trace_columns,
            np.arange(self._trace_columns.size),
        ] = 1.0
        self._charge_cache: dict[float, tuple[float, float, int]] = {}
        self._last_mu: float | None = None
        self._last_lu_cache: dict[complex, Any] = {}

    def _trace_scalar(self, block: np.ndarray) -> float:
        values = block[self._trace_columns, np.arange(self._trace_columns.size)]
        return float(np.real(np.sum(self._trace_weights[self._trace_columns] * values)))

    def _evaluate_terms_for_basis(
        self,
        mu: float,
        basis: np.ndarray,
        *,
        pole_count: int,
        derivative: bool,
        lu_cache: dict[complex, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[complex, Any] | None]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        lower, upper = spectral_interval(shifted)
        constant, shifts, residues = _scheme_terms(
            self.options,
            pole_count,
            lower=lower,
            upper=upper,
            kT=self.kT,
        )
        return _evaluate_rational_terms(
            shifted,
            np.asarray(basis, dtype=self.workspace_dtype),
            constant=constant,
            shifts=shifts,
            residues=residues,
            q_diag=self.q_diag,
            derivative=derivative,
            workspace_dtype=self.workspace_dtype,
            lu_cache=lu_cache,
        )

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, derivative, _order = self._charge_cache[mu]
            return charge, derivative

        half_poles = int(self.options.initial_poles)
        half_block, half_derivative_block, lu_cache = self._evaluate_terms_for_basis(
            mu,
            self._trace_basis,
            pole_count=half_poles,
            derivative=True,
            lu_cache=None,
        )
        half_charge = self._trace_scalar(half_block)
        half_derivative = (
            0.0 if half_derivative_block is None else self._trace_scalar(half_derivative_block)
        )

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            full_block, full_derivative_block, lu_cache = self._evaluate_terms_for_basis(
                mu,
                self._trace_basis,
                pole_count=pole_count,
                derivative=True,
                lu_cache=lu_cache,
            )
            full_charge = self._trace_scalar(full_block)
            full_derivative = (
                0.0 if full_derivative_block is None else self._trace_scalar(full_derivative_block)
            )

            if (
                abs(full_charge - half_charge) <= self.charge_tolerance
                and scalar_derivative_converged(
                    full_derivative,
                    half_derivative,
                    rtol=self.options.dn_dmu_rtol,
                )
            ):
                self._charge_cache[mu] = (full_charge, full_derivative, pole_count)
                self._last_mu = float(mu)
                self._last_lu_cache = {} if lu_cache is None else dict(lu_cache)
                return full_charge, full_derivative

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_charge = full_charge
            half_derivative = full_derivative

    def density_columns_from_charge_order(self, mu: float, basis: np.ndarray) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError("Charge pole count is unavailable for the requested chemical potential")
        lu_cache = self._last_lu_cache if self._last_mu == float(mu) else None
        block, _derivative_block, lu_cache = self._evaluate_terms_for_basis(
            mu,
            basis,
            pole_count=charge_cache[2],
            derivative=False,
            lu_cache=lu_cache,
        )
        if lu_cache is not None:
            self._last_mu = float(mu)
            self._last_lu_cache = dict(lu_cache)
        return block

    def density_from_charge_order(self, mu: float) -> np.ndarray:
        block = self.density_columns_from_charge_order(
            mu,
            np.eye(self.size, dtype=self.workspace_dtype),
        )
        return 0.5 * (block + block.conj().T)


class PreparedMumpsRationalNode:
    def __init__(
        self,
        matrix: Any,
        *,
        kT: float,
        q_diag: np.ndarray,
        options: RationalFOE,
        charge_tolerance: float,
        density_support: DensityEntrySupport,
        density_tolerance: float,
        workspace_dtype: np.dtype = np.dtype(complex),
        trace_weights_diag: np.ndarray | None = None,
        shared_aaa_interval_cache: list[_AAAIntervalCacheEntry] | None = None,
    ) -> None:
        if options.rational_scheme not in {"ozaki", "aaa"}:
            raise ValueError("Sparse RationalFOE currently requires rational_scheme='ozaki' or 'aaa'")
        if not is_sparse_like(matrix):
            raise ValueError("Sparse MUMPS-backed RationalFOE requires sparse matrices")
        require_mumps()

        self.workspace_dtype = np.dtype(workspace_dtype)
        self.matrix = as_sparse(workspace_matrix(matrix, self.workspace_dtype)).tocsr()
        self.kT = float(kT)
        self.q_diag = np.asarray(q_diag, dtype=float)
        self.options = options
        self.charge_tolerance = float(charge_tolerance)
        self.density_tolerance = float(density_tolerance)
        self.size = int(getattr(matrix, "shape")[0])
        self._trace_weights = (
            np.ones(self.size, dtype=float)
            if trace_weights_diag is None
            else np.asarray(trace_weights_diag, dtype=float)
        )
        self._density_support = density_support
        self._charge_pattern = build_sparse_charge_pattern(self._trace_weights)
        self._density_pattern = build_sparse_density_pattern(
            size=self.size,
            density_support=density_support,
        )
        density_extra_positions = np.asarray(
            [
                position
                for position in range(self._density_pattern.pattern.nnz)
                if (
                    int(self._density_pattern.pattern.rows[position]),
                    int(self._density_pattern.pattern.cols[position]),
                )
                not in self._charge_pattern.pattern.lookup
            ],
            dtype=int,
        )
        self._density_extra_pattern = build_selected_entry_pattern(
            size=self.size,
            rows=self._density_pattern.pattern.rows[density_extra_positions],
            cols=self._density_pattern.pattern.cols[density_extra_positions],
        )
        self._charge_to_density_source_positions, self._charge_to_density_target_positions = (
            _pattern_subset_mappings(self._charge_pattern.pattern, self._density_pattern.pattern)
        )
        self._extra_to_density_source_positions, self._extra_to_density_target_positions = (
            _pattern_subset_mappings(self._density_extra_pattern, self._density_pattern.pattern)
        )
        self._charge_cache: dict[float, tuple[float, int, complex, np.ndarray, np.ndarray]] = {}
        self._sparse_terms_cache: dict[tuple[float, int, float], SparseRationalTerms] = {}
        self._aaa_interval_cache: list[_AAAIntervalCacheEntry] = (
            shared_aaa_interval_cache if shared_aaa_interval_cache is not None else []
        )
        self._last_mu: float | None = None
        self._last_pole_count: int | None = None
        self._last_constant: complex = complex(0.0)
        self._last_shifts = np.empty(0, dtype=np.complex128)
        self._last_residues = np.empty(0, dtype=np.complex128)
        self._last_factorizations: dict[complex, SelectedInverseFactorization] = {}
        self._last_charge_entries: dict[complex, np.ndarray] = {}

    def _density_scalar_tolerance(self) -> float:
        return float(self.density_tolerance)

    def _charge_scalar_tolerance(self) -> float:
        weight_sum = float(np.sum(np.abs(self._charge_pattern.charge_weights)))
        if weight_sum <= 0.0:
            return self._density_scalar_tolerance()
        # Charge is a weighted trace, so the scalar Fermi-operator error must
        # shrink with the total trace weight to keep the filling solve stable.
        return max(
            np.finfo(float).eps,
            min(self._density_scalar_tolerance(), float(self.charge_tolerance) / weight_sum),
        )

    def _aaa_cached_terms_for_interval(
        self,
        *,
        lower: float,
        upper: float,
        pole_cap: int,
        scalar_tolerance: float,
    ) -> SparseRationalTerms | None:
        for entry in self._aaa_interval_cache:
            if not np.isclose(entry.kT, self.kT):
                continue
            if lower < entry.lower or upper > entry.upper:
                continue
            if entry.terms.support_count is not None and entry.terms.support_count > pole_cap:
                continue
            certification_grid = _aaa_sample_grid(
                lower,
                upper,
                count=max(1024, 64 * int(pole_cap)),
                kT=self.kT,
            )
            exact = np.asarray(fermi_dirac(certification_grid, self.kT, 0.0), dtype=complex)
            barycentric = _barycentric_evaluate(
                certification_grid,
                entry.support_x,
                entry.support_y,
                entry.weights,
            )
            pole_values = _evaluate_canonical_rational(
                certification_grid,
                constant=entry.terms.constant,
                shifts=entry.terms.shifts,
                residues=entry.terms.residues,
                tail_lower_bound=entry.terms.tail_lower_bound,
                tail_upper_bound=entry.terms.tail_upper_bound,
            )
            scalar_error = float(np.max(np.abs(exact - pole_values), initial=0.0))
            barycentric_gap = float(np.max(np.abs(pole_values - barycentric), initial=0.0))
            if scalar_error <= scalar_tolerance and barycentric_gap <= 0.1 * scalar_tolerance:
                return entry.terms
        return None

    def _sparse_terms(
        self,
        mu: float,
        *,
        pole_count: int,
        scalar_tolerance: float | None = None,
    ) -> SparseRationalTerms:
        if scalar_tolerance is None:
            scalar_tolerance = self._density_scalar_tolerance()
        cache_key = (float(mu), int(pole_count), float(scalar_tolerance))
        cached = self._sparse_terms_cache.get(cache_key)
        if cached is not None:
            return cached

        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        lower, upper = spectral_interval(shifted)
        padding = 1e-12 * max(1.0, float(upper - lower))
        lower -= padding
        upper += padding
        if self.options.rational_scheme == "ozaki":
            constant, shifts, residues = _scheme_terms(
                self.options,
                pole_count,
                lower=lower,
                upper=upper,
                kT=self.kT,
            )
            terms = SparseRationalTerms(
                constant=constant,
                shifts=np.asarray(shifts, dtype=np.complex128),
                residues=np.asarray(residues, dtype=np.complex128),
                pole_count=int(pole_count),
            )
        else:
            cached_terms = self._aaa_cached_terms_for_interval(
                lower=lower,
                upper=upper,
                pole_cap=int(pole_count),
                scalar_tolerance=float(scalar_tolerance),
            )
            if cached_terms is not None:
                self._sparse_terms_cache[cache_key] = cached_terms
                return cached_terms
            terms, builder = _aaa_terms_for_interval(
                pole_cap=int(pole_count),
                lower=lower,
                upper=upper,
                kT=self.kT,
                initial_poles=int(self.options.initial_poles),
                scalar_tolerance=float(scalar_tolerance),
            )
            self._aaa_interval_cache.append(
                _AAAIntervalCacheEntry(
                    lower=lower,
                    upper=upper,
                    kT=self.kT,
                    support_x=np.asarray(builder.support_x, dtype=float),
                    support_y=np.asarray(builder.support_y, dtype=complex),
                    weights=np.asarray(builder.weights, dtype=complex),
                    terms=terms,
                )
            )
        self._sparse_terms_cache[cache_key] = terms
        return terms

    def _certified_sparse_terms(
        self,
        mu: float,
        *,
        scalar_tolerance: float | None = None,
    ) -> SparseRationalTerms:
        if scalar_tolerance is None:
            scalar_tolerance = self._density_scalar_tolerance()
        if self.options.rational_scheme == "aaa":
            try:
                return self._sparse_terms(
                    mu,
                    pole_count=int(self.options.max_poles),
                    scalar_tolerance=float(scalar_tolerance),
                )
            except ValueError as exc:
                if "AAA scalar certification failed" not in str(exc):
                    raise
                raise ValueError("Rational FOE did not converge within max_poles") from exc
        pole_count = int(self.options.initial_poles)
        while True:
            try:
                return self._sparse_terms(
                    mu,
                    pole_count=pole_count,
                    scalar_tolerance=float(scalar_tolerance),
                )
            except ValueError as exc:
                if pole_count == int(self.options.max_poles):
                    raise ValueError("Rational FOE did not converge within max_poles") from exc
                pole_count = min(int(self.options.max_poles), 2 * pole_count)

    def _evaluate_charge_for_pole_count(
        self,
        mu: float,
        *,
        pole_count: int,
        terms: SparseRationalTerms | None = None,
        factorization_cache: dict[complex, SelectedInverseFactorization] | None = None,
    ) -> tuple[
        float,
        complex,
        np.ndarray,
        np.ndarray,
        dict[complex, SelectedInverseFactorization],
        dict[complex, np.ndarray],
    ]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        if terms is None:
            terms = self._sparse_terms(mu, pole_count=pole_count)
        constant = terms.constant
        shifts = terms.shifts
        residues = terms.residues
        active_factorizations = {} if factorization_cache is None else dict(factorization_cache)
        charge_entries: dict[complex, np.ndarray] = {}
        for shift in shifts:
            key = complex(shift)
            factorization = active_factorizations.get(key)
            if factorization is None:
                factorization = SelectedInverseFactorization()
                active_factorizations[key] = factorization
            factorization.factor(_sparse_shifted_matrix(shifted, key))
            charge_entries[key] = factorization.selected_inverse(self._charge_pattern.pattern)
        charge = self._charge_pattern.charge_from_inverse_entries(
            charge_entries,
            constant=constant,
            shifts=shifts,
            residues=residues,
        )
        return charge, constant, shifts, residues, active_factorizations, charge_entries

    def charge_and_derivative(self, mu: float) -> tuple[float, float]:
        if mu in self._charge_cache:
            charge, _pole_count, _constant, _shifts, _residues = self._charge_cache[mu]
            return charge, float("nan")

        if self.options.rational_scheme == "aaa":
            terms = self._certified_sparse_terms(
                mu,
                scalar_tolerance=self._charge_scalar_tolerance(),
            )
            charge, _constant, _shifts, _residues, factorization_cache, charge_entries = (
                self._evaluate_charge_for_pole_count(
                    mu,
                    pole_count=terms.pole_count,
                    terms=terms,
                    factorization_cache=None,
                )
            )
            self._charge_cache[mu] = (
                charge,
                terms.pole_count,
                terms.constant,
                np.asarray(terms.shifts, dtype=np.complex128),
                np.asarray(terms.residues, dtype=np.complex128),
            )
            self._last_mu = float(mu)
            self._last_pole_count = terms.pole_count
            self._last_constant = terms.constant
            self._last_shifts = np.asarray(terms.shifts, dtype=np.complex128)
            self._last_residues = np.asarray(terms.residues, dtype=np.complex128)
            self._last_factorizations = dict(factorization_cache)
            self._last_charge_entries = dict(charge_entries)
            return charge, float("nan")

        half_poles = int(self.options.initial_poles)
        half_charge, _constant, _shifts, _residues, factorization_cache, _charge_entries = self._evaluate_charge_for_pole_count(
            mu,
            pole_count=half_poles,
            factorization_cache=None,
        )

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            full_charge, constant, shifts, residues, factorization_cache, charge_entries = self._evaluate_charge_for_pole_count(
                mu,
                pole_count=pole_count,
                factorization_cache=factorization_cache,
            )
            if abs(full_charge - half_charge) <= self.charge_tolerance:
                self._charge_cache[mu] = (
                    full_charge,
                    pole_count,
                    constant,
                    np.asarray(shifts, dtype=np.complex128),
                    np.asarray(residues, dtype=np.complex128),
                )
                self._last_mu = float(mu)
                self._last_pole_count = pole_count
                self._last_constant = constant
                self._last_shifts = np.asarray(shifts, dtype=np.complex128)
                self._last_residues = np.asarray(residues, dtype=np.complex128)
                self._last_factorizations = dict(factorization_cache)
                self._last_charge_entries = dict(charge_entries)
                return full_charge, float("nan")

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_charge = full_charge

    def _density_columns_from_inverse_entries(
        self,
        *,
        constant: complex,
        shifts: np.ndarray,
        residues: np.ndarray,
        inverse_entries: dict[complex, np.ndarray],
    ) -> np.ndarray:
        return self._density_pattern.density_columns_from_inverse_entries(
            inverse_entries,
            constant=constant,
            shifts=shifts,
            residues=residues,
        )

    def _request_density_inverse_entries(
        self,
        mu: float,
        *,
        pole_count: int,
        factorization_cache: dict[complex, SelectedInverseFactorization] | None = None,
        terms: SparseRationalTerms | None = None,
    ) -> tuple[
        complex,
        np.ndarray,
        np.ndarray,
        dict[complex, SelectedInverseFactorization],
        dict[complex, np.ndarray],
    ]:
        shifted = shift_by_mu(
            self.matrix,
            mu,
            self.q_diag,
            dtype=self.workspace_dtype,
        )
        resolved_terms = terms if terms is not None else self._sparse_terms(mu, pole_count=pole_count)
        constant = resolved_terms.constant
        shifts = resolved_terms.shifts
        residues = resolved_terms.residues
        factorizations = {} if factorization_cache is None else dict(factorization_cache)
        inverse_entries: dict[complex, np.ndarray] = {}
        for shift in shifts:
            key = complex(shift)
            factorization = factorizations.get(key)
            if factorization is None:
                factorization = SelectedInverseFactorization()
                factorizations[key] = factorization
            factorization.factor(_sparse_shifted_matrix(shifted, key))
            inverse_entries[key] = factorization.selected_inverse(self._density_pattern.pattern)
        return constant, shifts, residues, factorizations, inverse_entries

    def _request_extra_density_entries_from_cached_factorizations(
        self,
        *,
        shifts: np.ndarray,
    ) -> dict[complex, np.ndarray]:
        if self._density_extra_pattern.nnz == 0:
            return {complex(shift): np.empty(0, dtype=np.complex128) for shift in shifts}
        return {
            complex(shift): self._last_factorizations[complex(shift)].selected_inverse(
                self._density_extra_pattern
            )
            for shift in shifts
        }

    def _merge_charge_and_extra_entries(
        self,
        charge_entries: dict[complex, np.ndarray],
        extra_entries: dict[complex, np.ndarray],
        *,
        shifts: np.ndarray,
    ) -> dict[complex, np.ndarray]:
        merged: dict[complex, np.ndarray] = {}
        for shift in shifts:
            key = complex(shift)
            full_entries = np.zeros(self._density_pattern.pattern.nnz, dtype=np.complex128)
            if self._charge_to_density_source_positions.size:
                full_entries[self._charge_to_density_target_positions] = charge_entries[key][
                    self._charge_to_density_source_positions
                ]
            if self._extra_to_density_source_positions.size:
                full_entries[self._extra_to_density_target_positions] = extra_entries[key][
                    self._extra_to_density_source_positions
                ]
            merged[key] = full_entries
        return merged

    def density_columns(self, mu: float, *, tolerance: float) -> np.ndarray:
        if self.options.rational_scheme == "aaa":
            terms = self._certified_sparse_terms(mu)
            constant, shifts, residues, factorizations, inverse_entries = (
                self._request_density_inverse_entries(
                    mu,
                    pole_count=terms.pole_count,
                    factorization_cache=None,
                    terms=terms,
                )
            )
            self._last_mu = float(mu)
            self._last_pole_count = terms.pole_count
            self._last_constant = terms.constant
            self._last_shifts = np.asarray(terms.shifts, dtype=np.complex128)
            self._last_residues = np.asarray(terms.residues, dtype=np.complex128)
            self._last_factorizations = dict(factorizations)
            self._last_charge_entries = {}
            return self._density_columns_from_inverse_entries(
                constant=constant,
                shifts=shifts,
                residues=residues,
                inverse_entries=inverse_entries,
            )

        half_poles = int(self.options.initial_poles)
        half_constant, half_shifts, half_residues, factorization_cache, half_entries = (
            self._request_density_inverse_entries(
                mu,
                pole_count=half_poles,
                factorization_cache=None,
            )
        )
        half_columns = self._density_columns_from_inverse_entries(
            constant=half_constant,
            shifts=half_shifts,
            residues=half_residues,
            inverse_entries=half_entries,
        )

        while True:
            pole_count = min(int(self.options.max_poles), 2 * half_poles)
            full_constant, full_shifts, full_residues, factorization_cache, full_entries = (
                self._request_density_inverse_entries(
                    mu,
                    pole_count=pole_count,
                    factorization_cache=factorization_cache,
                )
            )
            full_columns = self._density_columns_from_inverse_entries(
                constant=full_constant,
                shifts=full_shifts,
                residues=full_residues,
                inverse_entries=full_entries,
            )
            if float(np.max(np.abs(full_columns - half_columns), initial=0.0)) <= tolerance:
                self._last_mu = float(mu)
                self._last_pole_count = pole_count
                self._last_constant = full_constant
                self._last_shifts = np.asarray(full_shifts, dtype=np.complex128)
                self._last_residues = np.asarray(full_residues, dtype=np.complex128)
                self._last_factorizations = dict(factorization_cache)
                self._last_charge_entries = {}
                return full_columns

            if pole_count == int(self.options.max_poles):
                raise ValueError("Rational FOE did not converge within max_poles")

            half_poles = pole_count
            half_columns = full_columns

    def density_columns_from_charge_order(self, mu: float, basis: np.ndarray | None = None) -> np.ndarray:
        charge_cache = self._charge_cache.get(mu)
        if charge_cache is None:
            raise ValueError("Charge pole count is unavailable for the requested chemical potential")
        pole_count = charge_cache[1]
        cached_constant = charge_cache[2]
        cached_shifts = np.asarray(charge_cache[3], dtype=np.complex128)
        cached_residues = np.asarray(charge_cache[4], dtype=np.complex128)
        if (
            self._last_mu == float(mu)
            and self._last_pole_count == pole_count
            and self._last_charge_entries
        ):
            extra_entries = self._request_extra_density_entries_from_cached_factorizations(
                shifts=self._last_shifts,
            )
            inverse_entries = self._merge_charge_and_extra_entries(
                self._last_charge_entries,
                extra_entries,
                shifts=self._last_shifts,
            )
            return self._density_columns_from_inverse_entries(
                constant=self._last_constant,
                shifts=self._last_shifts,
                residues=self._last_residues,
                inverse_entries=inverse_entries,
            )
        constant, shifts, residues, factorizations, inverse_entries = self._request_density_inverse_entries(
            mu,
            pole_count=pole_count,
            terms=SparseRationalTerms(
                constant=cached_constant,
                shifts=cached_shifts,
                residues=cached_residues,
                pole_count=pole_count,
            ),
        )
        self._last_mu = float(mu)
        self._last_pole_count = pole_count
        self._last_constant = constant
        self._last_shifts = np.asarray(shifts, dtype=np.complex128)
        self._last_residues = np.asarray(residues, dtype=np.complex128)
        self._last_factorizations = dict(factorizations)
        self._last_charge_entries = {}
        return self._density_columns_from_inverse_entries(
            constant=constant,
            shifts=shifts,
            residues=residues,
            inverse_entries=inverse_entries,
        )
