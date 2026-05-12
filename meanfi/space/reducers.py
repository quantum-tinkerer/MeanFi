from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from meanfi.space.coordinates import DensityEntry
from meanfi.space.symmetry import (
    HermiticityConstraint,
    ParticleHoleConstraint,
    SpatialSymmetry,
)


def complex_to_real(values: np.ndarray) -> np.ndarray:
    """Pack complex values as ``[real parts..., imaginary parts...]``."""

    array = np.asarray(values, dtype=complex).reshape(-1)
    return np.concatenate((array.real, array.imag))


def real_to_complex(values: np.ndarray) -> np.ndarray:
    """Unpack ``[real parts..., imaginary parts...]`` into complex values."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size % 2:
        raise ValueError("real_to_complex expects an even number of real values")
    midpoint = array.size // 2
    return array[:midpoint] + 1j * array[midpoint:]


def nullspace(equations: np.ndarray, variable_count: int) -> np.ndarray:
    if variable_count == 0:
        return np.zeros((0, 0), dtype=float)
    if equations.size == 0:
        return np.eye(variable_count, dtype=float)
    u, singular_values, vh = np.linalg.svd(equations, full_matrices=True)
    del u
    tolerance = np.finfo(float).eps * max(equations.shape) * (
        singular_values[0] if singular_values.size else 1.0
    )
    rank = int(np.sum(singular_values > tolerance))
    return np.asarray(vh[rank:].T, dtype=float)


def _zero_equations(index: int, value_count: int) -> list[np.ndarray]:
    rows = []
    for offset in (0, value_count):
        row = np.zeros(2 * value_count, dtype=float)
        row[offset + index] = 1.0
        rows.append(row)
    return rows


def _conjugate_pair_equations(left: int, right: int, value_count: int) -> list[np.ndarray]:
    real_row = np.zeros(2 * value_count, dtype=float)
    real_row[left] = 1.0
    real_row[right] -= 1.0
    imag_row = np.zeros(2 * value_count, dtype=float)
    imag_row[value_count + left] = 1.0
    imag_row[value_count + right] += 1.0
    return [real_row, imag_row]


def _negative_pair_equations(left: int, right: int, value_count: int) -> list[np.ndarray]:
    real_row = np.zeros(2 * value_count, dtype=float)
    real_row[left] = 1.0
    real_row[right] += 1.0
    imag_row = np.zeros(2 * value_count, dtype=float)
    imag_row[value_count + left] = 1.0
    imag_row[value_count + right] += 1.0
    return [real_row, imag_row]


def _linear_complex_equations(
    left: int,
    coefficients: dict[int, complex],
    value_count: int,
) -> list[np.ndarray]:
    real_row = np.zeros(2 * value_count, dtype=float)
    imag_row = np.zeros(2 * value_count, dtype=float)
    real_row[left] = 1.0
    imag_row[value_count + left] = 1.0
    for target, coefficient in coefficients.items():
        real = float(np.real(coefficient))
        imag = float(np.imag(coefficient))
        real_row[target] -= real
        real_row[value_count + target] += imag
        imag_row[target] -= imag
        imag_row[value_count + target] -= real
    return [real_row, imag_row]


def _antilinear_complex_equations(
    left: int,
    coefficients: dict[int, complex],
    value_count: int,
) -> list[np.ndarray]:
    """Rows for ``z_left = sum c_j * conj(z_j)``."""

    real_row = np.zeros(2 * value_count, dtype=float)
    imag_row = np.zeros(2 * value_count, dtype=float)
    real_row[left] = 1.0
    imag_row[value_count + left] = 1.0
    for target, coefficient in coefficients.items():
        real = float(np.real(coefficient))
        imag = float(np.imag(coefficient))
        real_row[target] -= real
        real_row[value_count + target] -= imag
        imag_row[target] -= imag
        imag_row[value_count + target] += real
    return [real_row, imag_row]


def _warn_missing_partner(entry: DensityEntry, partner: DensityEntry) -> None:
    warnings.warn(
        f"Active density entry {entry} maps outside the h_int active support to "
        f"{partner}; the outside entry is treated as zero.",
        UserWarning,
        stacklevel=4,
    )


@dataclass(frozen=True)
class OrbitReducer:
    """Fast reducer for entrywise conjugation/sign constraints."""

    entries: tuple[DensityEntry, ...]

    def basis(
        self,
        constraints: tuple[HermiticityConstraint | ParticleHoleConstraint, ...],
    ) -> np.ndarray:
        index = {entry: position for position, entry in enumerate(self.entries)}
        value_count = len(self.entries)
        equations: list[np.ndarray] = []
        for constraint in constraints:
            for position, entry in enumerate(self.entries):
                partner = constraint.partner(entry)
                if partner is None:
                    continue
                partner_position = index.get(partner)
                if partner_position is None:
                    _warn_missing_partner(entry, partner)
                    equations.extend(_zero_equations(position, value_count))
                elif isinstance(constraint, HermiticityConstraint):
                    equations.extend(
                        _conjugate_pair_equations(
                            position, partner_position, value_count
                        )
                    )
                else:
                    equations.extend(
                        _negative_pair_equations(position, partner_position, value_count)
                    )
        matrix = (
            np.vstack(equations)
            if equations
            else np.zeros((0, 2 * value_count), dtype=float)
        )
        return nullspace(matrix, 2 * value_count)


@dataclass(frozen=True)
class LinearConstraintReducer:
    """General reducer for spatial symmetries that mix density entries."""

    entries: tuple[DensityEntry, ...]
    ndof: int
    family: str

    def basis(self, starting_basis: np.ndarray, symmetries) -> np.ndarray:
        if not symmetries:
            return starting_basis
        equations = self._spatial_equations(tuple(symmetries))
        if not equations:
            return starting_basis
        matrix = np.vstack(equations) @ starting_basis
        reduced = nullspace(matrix, starting_basis.shape[1])
        return starting_basis @ reduced

    def _spatial_equations(
        self,
        symmetries: tuple[SpatialSymmetry, ...],
    ) -> list[np.ndarray]:
        index = {entry: position for position, entry in enumerate(self.entries)}
        equations: list[np.ndarray] = []
        for symmetry in symmetries:
            for position, entry in enumerate(self.entries):
                coefficients = self._transformed_entry_coefficients(
                    entry,
                    symmetry,
                    index,
                )
                equation_builder = (
                    _antilinear_complex_equations
                    if symmetry.antiunitary
                    else _linear_complex_equations
                )
                equations.extend(
                    equation_builder(position, coefficients, len(self.entries))
                )
        return equations

    def _transformed_entry_coefficients(
        self,
        entry: DensityEntry,
        symmetry: SpatialSymmetry,
        index: dict[DensityEntry, int],
    ) -> dict[int, complex]:
        key, row, col = entry
        anomalous = self.family == "bdg" and row < self.ndof <= col
        if self.family == "bdg" and not anomalous and not (row < self.ndof and col < self.ndof):
            return {}

        source_col = col - self.ndof if anomalous else col
        key_array = np.asarray(key, dtype=int)
        transformed_base = symmetry.lattice_matrix @ key_array
        coefficients: dict[int, complex] = {}
        for left_shift, left_unitary in symmetry.unitaries_by_shift.items():
            for right_shift, right_unitary in symmetry.unitaries_by_shift.items():
                target_key = tuple(
                    int(component)
                    for component in (
                        transformed_base
                        + np.asarray(right_shift, dtype=int)
                        - np.asarray(left_shift, dtype=int)
                    )
                )
                for target_row in range(self.ndof):
                    left_value = left_unitary[target_row, row]
                    if abs(left_value) <= 1e-14:
                        continue
                    for target_col in range(self.ndof):
                        right_value = right_unitary[target_col, source_col]
                        coefficient = (
                            left_value * right_value
                            if anomalous
                            else np.conjugate(left_value) * right_value
                        )
                        if abs(coefficient) <= 1e-14:
                            continue
                        target_entry = (
                            target_key,
                            int(target_row),
                            int(self.ndof + target_col if anomalous else target_col),
                        )
                        target_index = index.get(target_entry)
                        if target_index is None:
                            _warn_missing_partner(entry, target_entry)
                            continue
                        coefficients[target_index] = (
                            coefficients.get(target_index, 0.0) + coefficient
                        )
        return coefficients
