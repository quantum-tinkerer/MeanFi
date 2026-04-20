from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from itertools import permutations, product
import math

import numpy as np
try:
    from meanfi._zero_temp_native import (
        NativeChargeEvaluator as _NativeChargeEvaluator,
        NativeDensityEvaluator as _NativeDensityEvaluator,
        NativeFrontier as _NativeFrontier,
        NativeGeometry as _NativeGeometry,
        accumulate_density_terms as _native_accumulate_density_terms,
        density_tables_from_eigenvectors as _native_density_tables_from_eigenvectors,
        point_key_bytes as _native_point_key_bytes,
        prepare_charge_batch_metadata as _native_prepare_charge_batch_metadata,
        prepare_density_cells_metadata as _native_prepare_density_cells_metadata,
        unique_first_indices_int64 as _native_unique_first_indices_int64,
    )

    _NATIVE_ZERO_TEMP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when the native extension is unavailable
    _NativeChargeEvaluator = None
    _NativeDensityEvaluator = None
    _NativeFrontier = None
    _NativeGeometry = None
    _native_accumulate_density_terms = None
    _native_density_tables_from_eigenvectors = None
    _native_point_key_bytes = None
    _native_prepare_charge_batch_metadata = None
    _native_prepare_density_cells_metadata = None
    _native_unique_first_indices_int64 = None
    _NATIVE_ZERO_TEMP_AVAILABLE = False

from meanfi.tb.tb import _tb_type
from meanfi.tb.transforms import tb_to_kfunc, tb_to_native_spectral_cache


_GEOM_TOL = 1e-14
_BULK_THETA = 0.5
_ROOT_SUBCELLS_PER_AXIS = 2
_MU_ERROR_REFINE_LEVELS = 2
_PointKey = bytes


def _native_geometry_runtime_available() -> bool:
    return (
        _NATIVE_ZERO_TEMP_AVAILABLE
        and _NativeGeometry is not None
        and _NativeFrontier is not None
    )


def _native_charge_runtime_available(
    mesh: "_ZeroTempGeometryCache",
    spectral_cache: "_SpectralCache",
) -> bool:
    return (
        _NativeChargeEvaluator is not None
        and mesh._uses_native_runtime()
        and spectral_cache._native_cache is not None
    )


def _native_density_runtime_available(
    mesh: "_ZeroTempGeometryCache",
    spectral_cache: "_SpectralCache",
) -> bool:
    return (
        _NativeDensityEvaluator is not None
        and mesh._uses_native_runtime()
        and spectral_cache._native_cache is not None
    )


@dataclass
class _SimplexRecord:
    simplex_id: int
    vertex_ids: tuple[int, ...]
    parent_id: int | None = None
    children: tuple[int, ...] = ()
    active: bool = True
    level: int = 0
    split_edge: tuple[int, int] | None = None
    midpoint_vertex_id: int | None = None


@dataclass(frozen=True)
class _RefinementDescriptor:
    parent_id: int
    child_ids: tuple[int, ...]
    parent_vertex_ids: tuple[int, ...]
    child_vertex_ids: tuple[tuple[int, ...], ...]
    new_midpoint_vertex_id: int
    bisected_edge: tuple[int, int]


@dataclass
class _ZeroTempGeometryCache:
    ndim: int
    root_subcells_per_axis: int = _ROOT_SUBCELLS_PER_AXIS
    vertices: list[np.ndarray] = field(default_factory=list)
    simplices: list[int] = field(default_factory=list)
    _simplex_records: list[_SimplexRecord] = field(default_factory=list)
    _vertex_lookup: dict[_PointKey, int] = field(default_factory=dict)
    _simplex_points_cache: dict[int, np.ndarray] = field(default_factory=dict)
    _simplex_volume_cache: dict[int, float] = field(default_factory=dict)
    _refined_leaf_geometry_cache: dict[tuple[int, int], _RefinedLeafGeometry] = field(default_factory=dict)
    _descendant_leaf_cache: dict[tuple[int, int], tuple[int, ...]] = field(default_factory=dict)
    _last_refinement_descriptors: dict[int, _RefinementDescriptor] = field(default_factory=dict)
    _native_geometry: object | None = None
    _native_frontier: object | None = None
    _vertices_array_cache: np.ndarray | None = None

    @classmethod
    def root(cls, ndim: int) -> "_ZeroTempGeometryCache":
        if _native_geometry_runtime_available():
            native_geometry = _NativeGeometry.root(
                ndim,
                root_subcells_per_axis=_ROOT_SUBCELLS_PER_AXIS,
                tol=float(_GEOM_TOL),
            )
            native_frontier = _NativeFrontier.from_geometry(native_geometry)
            cache = cls(
                ndim=ndim,
                root_subcells_per_axis=_ROOT_SUBCELLS_PER_AXIS,
                _native_geometry=native_geometry,
                _native_frontier=native_frontier,
            )
            cache._sync_vertices_from_native()
            cache.simplices = [int(simplex_id) for simplex_id in np.asarray(native_frontier.active_simplex_ids(), dtype=int)]
            for simplex_id in cache.simplices:
                vertex_ids = tuple(int(vertex_id) for vertex_id in np.asarray(native_geometry.simplex_vertex_ids(simplex_id), dtype=int))
                cache._simplex_records.append(
                    _SimplexRecord(
                        simplex_id=int(simplex_id),
                        vertex_ids=vertex_ids,
                        level=0,
                        active=True,
                    )
                )
            return cache

        cache = cls(ndim=ndim, root_subcells_per_axis=_ROOT_SUBCELLS_PER_AXIS)
        step = 1.0 / cache.root_subcells_per_axis
        for offset in product(range(cache.root_subcells_per_axis), repeat=ndim):
            base = step * np.array(offset, dtype=float)
            for perm in permutations(range(ndim)):
                # Triangulate each half-sized subcell independently so no root simplex
                # spans opposite Brillouin-zone faces across the torus seam.
                point = np.array(base, copy=True)
                simplex = [cache.get_or_add_vertex(point)]
                for axis in perm:
                    point = point.copy()
                    point[axis] += step
                    simplex.append(cache.get_or_add_vertex(point))
                cache.simplices.append(cache.add_simplex(tuple(simplex)))
        return cache

    def _uses_native_runtime(self) -> bool:
        return _native_geometry_runtime_available() and self._native_geometry is not None and self._native_frontier is not None

    def _vertices_array(self) -> np.ndarray:
        if self._vertices_array_cache is None or self._vertices_array_cache.shape[0] != len(self.vertices):
            if not self.vertices:
                self._vertices_array_cache = np.empty((0, self.ndim), dtype=float)
            else:
                self._vertices_array_cache = np.stack(self.vertices, axis=0)
        return self._vertices_array_cache

    def _sync_vertices_from_native(self) -> None:
        if not self._uses_native_runtime():
            return
        vertices = np.asarray(self._native_geometry.vertices_array(), dtype=float)
        current = len(self.vertices)
        if vertices.shape[0] > current:
            self.vertices.extend(np.array(vertices[index], copy=True) for index in range(current, vertices.shape[0]))
        self._vertices_array_cache = np.array(vertices, copy=True)

    def _ensure_native_simplex_record(self, simplex_id: int) -> None:
        if not self._uses_native_runtime():
            return
        while len(self._simplex_records) <= int(simplex_id):
            next_simplex_id = len(self._simplex_records)
            vertex_ids = tuple(
                int(vertex_id)
                for vertex_id in np.asarray(self._native_geometry.simplex_vertex_ids(next_simplex_id), dtype=int)
            )
            self._simplex_records.append(
                _SimplexRecord(
                    simplex_id=int(next_simplex_id),
                    vertex_ids=vertex_ids,
                    active=bool(next_simplex_id in self.simplices),
                )
            )

    def active_simplex_ids_array(self) -> np.ndarray:
        if self._uses_native_runtime():
            return np.asarray(self._native_frontier.active_simplex_ids(), dtype=int)
        return np.asarray(self.simplices, dtype=int)

    def simplex_vertex_id_array(self, simplex_ids: np.ndarray | list[int]) -> np.ndarray:
        simplex_ids_arr = np.asarray(simplex_ids, dtype=int).reshape(-1)
        if simplex_ids_arr.size == 0:
            return np.empty((0, self.ndim + 1), dtype=int)
        if self._uses_native_runtime():
            active_ids = self.active_simplex_ids_array()
            if active_ids.shape == simplex_ids_arr.shape and np.array_equal(active_ids, simplex_ids_arr):
                return np.asarray(self._native_frontier.vertex_ids(), dtype=int)
        return np.asarray([self.simplex_vertex_ids(int(simplex_id)) for simplex_id in simplex_ids_arr], dtype=int)

    def simplex_volumes_array(self, simplex_ids: np.ndarray | list[int]) -> np.ndarray:
        simplex_ids_arr = np.asarray(simplex_ids, dtype=int).reshape(-1)
        if simplex_ids_arr.size == 0:
            return np.empty((0,), dtype=float)
        if self._uses_native_runtime():
            active_ids = self.active_simplex_ids_array()
            if active_ids.shape == simplex_ids_arr.shape and np.array_equal(active_ids, simplex_ids_arr):
                return np.asarray(self._native_frontier.volumes(), dtype=float)
        return np.asarray([self.simplex_volume(int(simplex_id)) for simplex_id in simplex_ids_arr], dtype=float)

    def vertex_points_array(self, vertex_ids: np.ndarray | list[int]) -> np.ndarray:
        vertex_ids_arr = np.asarray(vertex_ids, dtype=int)
        if vertex_ids_arr.size == 0:
            shape = vertex_ids_arr.shape + (self.ndim,)
            return np.empty(shape, dtype=float)
        return self._vertices_array()[vertex_ids_arr]

    def get_or_add_vertex(self, point: np.ndarray) -> int:
        key = _point_key(point)
        vertex_id = self._vertex_lookup.get(key)
        if vertex_id is not None:
            return vertex_id
        vertex_id = len(self.vertices)
        self.vertices.append(np.asarray(point, dtype=float).copy())
        self._vertex_lookup[key] = vertex_id
        self._vertices_array_cache = None
        return vertex_id

    def add_simplex(
        self,
        vertex_ids: tuple[int, ...],
        *,
        parent_id: int | None = None,
        level: int = 0,
        active: bool = True,
    ) -> int:
        simplex_id = len(self._simplex_records)
        self._simplex_records.append(
            _SimplexRecord(
                simplex_id=simplex_id,
                vertex_ids=vertex_ids,
                parent_id=parent_id,
                level=level,
                active=active,
            )
        )
        return simplex_id

    def simplex_vertex_ids(self, simplex_id: int) -> tuple[int, ...]:
        if self._uses_native_runtime() and simplex_id >= len(self._simplex_records):
            self._ensure_native_simplex_record(int(simplex_id))
        return self._simplex_records[simplex_id].vertex_ids

    def simplex_points(self, simplex_id: int) -> np.ndarray:
        cached = self._simplex_points_cache.get(simplex_id)
        if cached is not None:
            return cached

        if self._uses_native_runtime():
            points = np.asarray(self._native_geometry.simplex_points(simplex_id), dtype=float)
        else:
            points = np.stack(
                [self.vertices[vertex_id] for vertex_id in self.simplex_vertex_ids(simplex_id)],
                axis=0,
            )
        self._simplex_points_cache[simplex_id] = points
        return points

    def simplex_volume(self, simplex_id: int) -> float:
        cached = self._simplex_volume_cache.get(simplex_id)
        if cached is not None:
            return cached

        if self._uses_native_runtime():
            volume = float(self._native_geometry.simplex_volume(simplex_id))
        else:
            volume = _simplex_volume(self.simplex_points(simplex_id))
        self._simplex_volume_cache[simplex_id] = float(volume)
        return float(volume)

    def simplex_refined_leaf_geometry(self, simplex_id: int, refine_levels: int) -> _RefinedLeafGeometry:
        if refine_levels <= 0:
            points = self.simplex_points(simplex_id)
            return _RefinedLeafGeometry(
                unique_points=np.asarray(points, dtype=float).copy(),
                leaf_indices=np.arange(points.shape[0], dtype=int)[np.newaxis, :],
                leaf_volumes=np.array([self.simplex_volume(simplex_id)], dtype=float),
            )

        cache_key = (simplex_id, int(refine_levels))
        cached = self._refined_leaf_geometry_cache.get(cache_key)
        if cached is not None:
            return cached

        geometry = _refined_simplex_leaf_geometry(self.simplex_points(simplex_id), refine_levels=refine_levels)
        self._refined_leaf_geometry_cache[cache_key] = geometry
        return geometry

    def ensure_children(self, simplex_id: int) -> tuple[int, int]:
        record = self._simplex_records[int(simplex_id)]
        if record.children:
            return tuple(int(child) for child in record.children)

        if self._uses_native_runtime():
            children = tuple(int(child) for child in np.asarray(self._native_geometry.ensure_children(int(simplex_id)), dtype=int))
            self._sync_vertices_from_native()
            parent_vertices = tuple(int(vertex_id) for vertex_id in record.vertex_ids)
            midpoint_id: int | None = None
            split_vertices: list[int] = []
            parent_level = int(record.level)
            for child_id in children:
                child_vertex_ids = tuple(
                    int(vertex_id) for vertex_id in np.asarray(self._native_geometry.simplex_vertex_ids(int(child_id)), dtype=int)
                )
                differing = [
                    index
                    for index, (parent_vertex_id, child_vertex_id) in enumerate(zip(parent_vertices, child_vertex_ids, strict=False))
                    if parent_vertex_id != child_vertex_id
                ]
                if len(differing) != 1:
                    raise ValueError("Native child simplex does not differ from the parent on exactly one vertex")
                split_vertices.append(int(differing[0]))
                if midpoint_id is None:
                    midpoint_id = int(child_vertex_ids[differing[0]])
                elif midpoint_id != int(child_vertex_ids[differing[0]]):
                    raise ValueError("Native bisection children disagree on the midpoint vertex id")

                if child_id == len(self._simplex_records):
                    self._simplex_records.append(
                        _SimplexRecord(
                            simplex_id=int(child_id),
                            vertex_ids=child_vertex_ids,
                            parent_id=int(simplex_id),
                            level=parent_level + 1,
                            active=False,
                        )
                    )
                else:
                    child_record = self._simplex_records[int(child_id)]
                    child_record.vertex_ids = child_vertex_ids
                    child_record.parent_id = int(simplex_id)
                    child_record.level = parent_level + 1
                    child_record.active = False

            if midpoint_id is None or len(split_vertices) != 2:
                raise ValueError("Native geometry failed to provide consistent child split metadata")
            record.children = children
            record.midpoint_vertex_id = int(midpoint_id)
            record.split_edge = (int(split_vertices[0]), int(split_vertices[1]))
            return children

        child_vertices, midpoint_id, split_edge = self._bisect(simplex_id)
        children = tuple(
            self.add_simplex(
                vertex_ids,
                parent_id=record.simplex_id,
                level=record.level + 1,
                active=False,
            )
            for vertex_ids in child_vertices
        )
        record.children = children
        record.midpoint_vertex_id = midpoint_id
        record.split_edge = split_edge
        return children

    def descendant_leaves(self, simplex_id: int, levels: int) -> tuple[int, ...]:
        levels = int(levels)
        if levels <= 0:
            return (int(simplex_id),)

        cache_key = (int(simplex_id), levels)
        cached = self._descendant_leaf_cache.get(cache_key)
        if cached is not None:
            return cached

        children = self.ensure_children(simplex_id)
        if levels == 1:
            leaves = tuple(int(child) for child in children)
        else:
            leaves = tuple(
                child_leaf
                for child in children
                for child_leaf in self.descendant_leaves(child, levels - 1)
            )
        self._descendant_leaf_cache[cache_key] = leaves
        return leaves

    def refine(self, simplex_ids: list[int]) -> int:
        refinements, _ = self.refine_with_children(simplex_ids)
        return refinements

    def refine_with_children(
        self,
        simplex_ids: list[int],
    ) -> tuple[int, dict[int, _RefinementDescriptor]]:
        marked = set(simplex_ids)
        if not marked:
            return 0, {}

        if self._uses_native_runtime():
            (
                refinements,
                parent_ids,
                child_offsets,
                child_ids,
                parent_vertex_ids,
                child_vertex_ids,
                midpoint_ids,
                bisected_edges,
            ) = self._native_geometry.refine(np.asarray(sorted(marked), dtype=np.int64))
            refinements = int(refinements)
            descriptors: dict[int, _RefinementDescriptor] = {}
            if refinements:
                self._native_frontier.apply_refinement(parent_ids, child_offsets, child_ids)
            self._sync_vertices_from_native()
            child_ids_arr = np.asarray(child_ids, dtype=int)
            child_offsets_arr = np.asarray(child_offsets, dtype=int)
            parent_ids_arr = np.asarray(parent_ids, dtype=int)
            parent_vertex_ids_arr = np.asarray(parent_vertex_ids, dtype=int)
            child_vertex_ids_arr = np.asarray(child_vertex_ids, dtype=int)
            midpoint_ids_arr = np.asarray(midpoint_ids, dtype=int)
            bisected_edges_arr = np.asarray(bisected_edges, dtype=int)

            for local_index, simplex_id in enumerate(parent_ids_arr):
                parent_record = self._simplex_records[int(simplex_id)]
                parent_record.active = False
                start = int(child_offsets_arr[local_index])
                stop = int(child_offsets_arr[local_index + 1])
                child_ids_local = tuple(int(child_id) for child_id in child_ids_arr[start:stop])
                child_vertex_ids_local = tuple(
                    tuple(int(vertex_id) for vertex_id in child_vertex_ids_arr[start + offset])
                    for offset in range(stop - start)
                )
                parent_record.children = child_ids_local
                parent_record.midpoint_vertex_id = int(midpoint_ids_arr[local_index])
                parent_record.split_edge = tuple(int(edge) for edge in bisected_edges_arr[local_index])
                for child_id, vertex_ids in zip(child_ids_local, child_vertex_ids_local, strict=False):
                    self._ensure_native_simplex_record(int(child_id))
                    child_record = self._simplex_records[int(child_id)]
                    child_record.vertex_ids = vertex_ids
                    child_record.parent_id = int(simplex_id)
                    child_record.level = parent_record.level + 1
                    child_record.active = True
                descriptors[int(simplex_id)] = _RefinementDescriptor(
                    parent_id=int(simplex_id),
                    child_ids=child_ids_local,
                    parent_vertex_ids=tuple(int(vertex_id) for vertex_id in parent_vertex_ids_arr[local_index]),
                    child_vertex_ids=child_vertex_ids_local,
                    new_midpoint_vertex_id=int(midpoint_ids_arr[local_index]),
                    bisected_edge=tuple(int(edge) for edge in bisected_edges_arr[local_index]),
                )

            self.simplices = [int(simplex_id) for simplex_id in np.asarray(self._native_frontier.active_simplex_ids(), dtype=int)]
            self._last_refinement_descriptors = descriptors
            return refinements, descriptors

        refined: list[int] = []
        descriptors: dict[int, _RefinementDescriptor] = {}
        refinements = 0
        for simplex_id in self.simplices:
            if simplex_id not in marked:
                refined.append(simplex_id)
                continue

            refinements += 1
            parent_record = self._simplex_records[simplex_id]
            parent_record.active = False
            children = self.ensure_children(simplex_id)
            for child_id in children:
                self._simplex_records[child_id].active = True
            if parent_record.midpoint_vertex_id is None or parent_record.split_edge is None:
                raise ValueError("Refined simplex is missing split metadata")
            descriptors[simplex_id] = _RefinementDescriptor(
                parent_id=int(simplex_id),
                child_ids=tuple(int(child_id) for child_id in children),
                parent_vertex_ids=tuple(int(vertex_id) for vertex_id in parent_record.vertex_ids),
                child_vertex_ids=tuple(
                    tuple(int(vertex_id) for vertex_id in self._simplex_records[child_id].vertex_ids)
                    for child_id in children
                ),
                new_midpoint_vertex_id=int(parent_record.midpoint_vertex_id),
                bisected_edge=tuple(int(edge) for edge in parent_record.split_edge),
            )
            refined.extend(children)

        self.simplices = refined
        self._last_refinement_descriptors = descriptors
        return refinements, descriptors

    def _bisect(
        self,
        simplex_id: int,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], int, tuple[int, int]]:
        simplex = self.simplex_vertex_ids(simplex_id)
        points = self.simplex_points(simplex_id)
        edge_i, edge_j = _longest_edge(points)
        midpoint = 0.5 * (points[edge_i] + points[edge_j])
        midpoint_id = self.get_or_add_vertex(midpoint)

        child_a = list(simplex)
        child_b = list(simplex)
        child_a[edge_i] = midpoint_id
        child_b[edge_j] = midpoint_id
        return (tuple(child_a), tuple(child_b)), midpoint_id, (int(edge_i), int(edge_j))


@dataclass
class _ChargeSummary:
    charge: float
    derivative: float


@dataclass
class _ChargeIndicatorSummary:
    coarse_charge: float
    refined_charge: float
    indicator: float
    markers: list[tuple[int, float]]
    heap: list[tuple[float, int]]


@dataclass
class _DensitySummary:
    estimate: np.ndarray
    error_vector: np.ndarray
    error_scalar: float
    markers: list[tuple[int, float]]


@dataclass(frozen=True)
class _DensityCellValue:
    estimate: np.ndarray
    error_vector: np.ndarray
    error_scalar: float


@dataclass(frozen=True)
class _PreparedDensitySimplex:
    simplex_id: int
    volume: float
    vertex_ids: np.ndarray
    vertex_points: np.ndarray
    whole_bands: np.ndarray
    whole_weights: np.ndarray
    whole_centroid_tables: np.ndarray
    step_bands: np.ndarray
    step_centroid_tables: np.ndarray
    step_centroid_occ: np.ndarray
    step_vertex_occ: np.ndarray
    piece_volumes: np.ndarray
    piece_centroid_tables: np.ndarray
    piece_vertex_offsets: np.ndarray
    piece_vertex_tables: np.ndarray


def _bulk_mark_heap(error_heap: list[tuple[float, int]], total_error: float) -> list[int]:
    if not error_heap or total_error <= 0.0:
        return []
    target = _BULK_THETA * total_error
    accumulated = 0.0
    selected: list[int] = []
    local_heap = list(error_heap)
    heapq.heapify(local_heap)
    while local_heap and accumulated < target:
        neg_error, simplex_id = heapq.heappop(local_heap)
        selected.append(simplex_id)
        accumulated += -neg_error
    return selected


@dataclass
class _StageCounters:
    integration_calls: int = 0
    evaluator_evals: int = 0
    refinements: int = 0


@dataclass(frozen=True)
class _RefinedLeafGeometry:
    unique_points: np.ndarray
    leaf_indices: np.ndarray
    leaf_volumes: np.ndarray


@dataclass(frozen=True)
class _PreparedChargeBatch:
    points: np.ndarray
    vertex_energies: np.ndarray
    volumes: np.ndarray
    sorted_energies: np.ndarray
    simplex_weights: np.ndarray
    distinct_mask: np.ndarray
    band_min: np.ndarray
    band_max: np.ndarray
    flat_energy: np.ndarray
    flat_mask: np.ndarray
    cell_min: np.ndarray
    cell_max: np.ndarray
    dimension: int
    ndof: int
    owner_ids: np.ndarray
    owner_unique: np.ndarray
    owner_inverse: np.ndarray


@dataclass(frozen=True)
class _ChargeBatchEvaluation:
    summary: _ChargeSummary
    owner_ids: np.ndarray
    owner_charges: np.ndarray
    derivative_exact: bool


class _SpectralCache:
    def __init__(self, h: _tb_type) -> None:
        self.ndof = next(iter(h.values())).shape[0]
        self._python_hkfunc = None
        self._python_cache: dict[_PointKey, tuple[np.ndarray, np.ndarray]] = {}
        self._native_cache = None
        self._vertex_values: list[np.ndarray | None] = []
        self._vertex_vectors: list[np.ndarray | None] = []
        self._n_kernel_evals = 0

        if _NATIVE_ZERO_TEMP_AVAILABLE:
            try:
                self._native_cache = tb_to_native_spectral_cache(h, tol=float(_GEOM_TOL))
            except ImportError:  # pragma: no cover - exercised only when native bindings are unavailable
                self._python_hkfunc = tb_to_kfunc(h)
        else:
            self._python_hkfunc = tb_to_kfunc(h)

    def _ensure_vertex_capacity(self, vertex_id: int) -> None:
        missing = vertex_id + 1 - len(self._vertex_values)
        if missing <= 0:
            return
        self._vertex_values.extend([None] * missing)
        self._vertex_vectors.extend([None] * missing)

    def _set_vertex_cache(
        self,
        vertex_id: int,
        cached: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self._ensure_vertex_capacity(vertex_id)
        self._vertex_values[vertex_id] = cached[0]
        self._vertex_vectors[vertex_id] = cached[1]

    def _get_vertex_cached(self, vertex_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if vertex_id >= len(self._vertex_values):
            return None
        values = self._vertex_values[vertex_id]
        vectors = self._vertex_vectors[vertex_id]
        if values is None or vectors is None:
            return None
        return values, vectors

    def _get_vertex_values_cached(self, vertex_id: int) -> np.ndarray | None:
        if vertex_id >= len(self._vertex_values):
            return None
        return self._vertex_values[vertex_id]

    def _evaluate_point(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, vectors = self._evaluate_points(np.asarray(point, dtype=float)[np.newaxis, :])
        return values[0], vectors[0]

    def _to_k_points(self, points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=float)
        return 2.0 * np.pi * points_arr - np.pi

    def get(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._native_cache is not None:
            values, vectors = self._native_cache.get_many(self._to_k_points(np.asarray(point, dtype=float)[np.newaxis, :]))
            return np.asarray(values[0], dtype=float), np.asarray(vectors[0], dtype=complex)

        key = _point_key(point)
        cached = self._python_cache.get(key)
        if cached is not None:
            return cached

        cached = self._evaluate_point(point)
        self._python_cache[key] = cached
        return cached

    def get_values(self, point: np.ndarray) -> np.ndarray:
        if self._native_cache is not None:
            values = self._native_cache.get_many_values(self._to_k_points(np.asarray(point, dtype=float)[np.newaxis, :]))
            return np.asarray(values[0], dtype=float)

        key = _point_key(point)
        cached = self._python_cache.get(key)
        if cached is not None:
            return cached[0]

        cached = self._evaluate_point(point)
        self._python_cache[key] = cached
        return cached[0]

    def get_many(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_arr = np.asarray(points, dtype=float)
        if points_arr.ndim == 1:
            points_arr = points_arr[np.newaxis, :]
        n_points = points_arr.shape[0]
        eigenvalues = np.empty((n_points, self.ndof), dtype=float)
        eigenvectors = np.empty((n_points, self.ndof, self.ndof), dtype=complex)

        if self._native_cache is not None:
            values, vectors = self._native_cache.get_many(self._to_k_points(points_arr))
            eigenvalues[...] = np.asarray(values, dtype=float)
            eigenvectors[...] = np.asarray(vectors, dtype=complex)
            return eigenvalues, eigenvectors

        miss_indices: list[int] = []
        miss_keys: list[tuple[float, ...]] = []
        miss_points: list[np.ndarray] = []
        for index, point in enumerate(points_arr):
            key = _point_key(point)
            cached = self._python_cache.get(key)
            if cached is None:
                miss_indices.append(index)
                miss_keys.append(key)
                miss_points.append(point)
                continue
            eigenvalues[index] = cached[0]
            eigenvectors[index] = cached[1]

        if miss_indices:
            miss_values, miss_vectors = self._evaluate_points(np.stack(miss_points, axis=0))
            for local_index, index in enumerate(miss_indices):
                cached = (miss_values[local_index], miss_vectors[local_index])
                self._python_cache[miss_keys[local_index]] = cached
                eigenvalues[index] = cached[0]
                eigenvectors[index] = cached[1]
        return eigenvalues, eigenvectors

    def get_many_values(self, points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=float)
        if points_arr.ndim == 1:
            points_arr = points_arr[np.newaxis, :]
        n_points = points_arr.shape[0]
        eigenvalues = np.empty((n_points, self.ndof), dtype=float)

        if self._native_cache is not None:
            eigenvalues[...] = np.asarray(
                self._native_cache.get_many_values(self._to_k_points(points_arr)),
                dtype=float,
            )
            return eigenvalues

        miss_indices: list[int] = []
        miss_keys: list[tuple[float, ...]] = []
        miss_points: list[np.ndarray] = []
        for index, point in enumerate(points_arr):
            key = _point_key(point)
            cached = self._python_cache.get(key)
            if cached is None:
                miss_indices.append(index)
                miss_keys.append(key)
                miss_points.append(point)
                continue
            eigenvalues[index] = cached[0]

        if miss_indices:
            miss_values, miss_vectors = self._evaluate_points(np.stack(miss_points, axis=0))
            for local_index, index in enumerate(miss_indices):
                cached = (miss_values[local_index], miss_vectors[local_index])
                self._python_cache[miss_keys[local_index]] = cached
                eigenvalues[index] = cached[0]
        return eigenvalues

    def get_many_vertices(
        self,
        vertex_ids: tuple[int, ...],
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points_arr = np.asarray(points, dtype=float)
        n_points = points_arr.shape[0]
        eigenvalues = np.empty((n_points, self.ndof), dtype=float)
        eigenvectors = np.empty((n_points, self.ndof, self.ndof), dtype=complex)

        miss_indices: list[int] = []
        miss_vertex_ids: list[int] = []
        miss_points: list[np.ndarray] = []
        for index, (vertex_id, point) in enumerate(zip(vertex_ids, points_arr, strict=False)):
            cached = self._get_vertex_cached(vertex_id)
            if cached is not None:
                eigenvalues[index] = cached[0]
                eigenvectors[index] = cached[1]
                continue

            miss_indices.append(index)
            miss_vertex_ids.append(vertex_id)
            miss_points.append(point)

        if miss_indices:
            miss_values, miss_vectors = self._evaluate_points(np.stack(miss_points, axis=0))
            for local_index, index in enumerate(miss_indices):
                cached = (miss_values[local_index], miss_vectors[local_index])
                self._set_vertex_cache(miss_vertex_ids[local_index], cached)
                eigenvalues[index] = cached[0]
                eigenvectors[index] = cached[1]
        return eigenvalues, eigenvectors

    def get_many_vertex_values(
        self,
        vertex_ids: tuple[int, ...],
        points: np.ndarray,
    ) -> np.ndarray:
        points_arr = np.asarray(points, dtype=float)
        n_points = points_arr.shape[0]
        eigenvalues = np.empty((n_points, self.ndof), dtype=float)

        miss_indices: list[int] = []
        miss_vertex_ids: list[int] = []
        miss_points: list[np.ndarray] = []
        for index, (vertex_id, point) in enumerate(zip(vertex_ids, points_arr, strict=False)):
            cached_values = self._get_vertex_values_cached(vertex_id)
            if cached_values is not None:
                eigenvalues[index] = cached_values
                continue

            miss_indices.append(index)
            miss_vertex_ids.append(vertex_id)
            miss_points.append(point)

        if miss_indices:
            miss_values, miss_vectors = self._evaluate_points(np.stack(miss_points, axis=0))
            for local_index, index in enumerate(miss_indices):
                cached = (miss_values[local_index], miss_vectors[local_index])
                self._set_vertex_cache(miss_vertex_ids[local_index], cached)
                eigenvalues[index] = cached[0]
        return eigenvalues

    def _evaluate_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_arr = np.asarray(points, dtype=float)
        if points_arr.size == 0:
            return (
                np.empty((0, self.ndof), dtype=float),
                np.empty((0, self.ndof, self.ndof), dtype=complex),
            )

        if self._native_cache is not None:
            values, vectors = self._native_cache.evaluate_many(self._to_k_points(points_arr))
            return np.asarray(values, dtype=float), np.asarray(vectors, dtype=complex)

        h_k = self._python_hkfunc(self._to_k_points(points_arr))
        if h_k.ndim == 2:
            h_k = h_k[np.newaxis, :, :]
        eigenvalues, eigenvectors = np.linalg.eigh(h_k)
        self._n_kernel_evals += points_arr.shape[0]
        return eigenvalues.real.astype(float), eigenvectors

    def evaluate_many(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._evaluate_points(points)

    def evaluate_many_values(self, points: np.ndarray) -> np.ndarray:
        values, _ = self._evaluate_points(points)
        return values

    def get_vertex(self, vertex_id: int, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cached = self._get_vertex_cached(vertex_id)
        if cached is not None:
            return cached

        cached = self._evaluate_point(point)
        self._set_vertex_cache(vertex_id, cached)
        return cached

    def get_vertex_values(self, vertex_id: int, point: np.ndarray) -> np.ndarray:
        cached_values = self._get_vertex_values_cached(vertex_id)
        if cached_values is not None:
            return cached_values

        cached = self._evaluate_point(point)
        self._set_vertex_cache(vertex_id, cached)
        return cached[0]

    @property
    def n_cached_points(self) -> int:
        if self._native_cache is not None:
            return int(self._native_cache.size)
        return len(self._python_cache)

    @property
    def n_kernel_evals(self) -> int:
        if self._native_cache is not None:
            return int(self._native_cache.n_kernel_evals)
        return int(self._n_kernel_evals)


class _DensityTableStore:
    def __init__(self, spectral_cache: _SpectralCache, keys_arr: np.ndarray) -> None:
        self.spectral_cache = spectral_cache
        self.keys_arr = np.asarray(keys_arr, dtype=float)
        self._vertex_tables: np.ndarray | None = None
        self._vertex_ready = np.zeros(0, dtype=bool)
        self._ncomp = spectral_cache.ndof * spectral_cache.ndof * self.keys_arr.shape[0]

    def _ensure_capacity(self, vertex_ids: np.ndarray) -> None:
        if vertex_ids.size == 0:
            return
        required = int(np.max(vertex_ids)) + 1
        current = self._vertex_ready.shape[0]
        if current >= required:
            return
        ndof = self.spectral_cache.ndof
        new_tables = np.empty((required, ndof, self._ncomp), dtype=complex)
        if self._vertex_tables is not None:
            new_tables[:current] = self._vertex_tables
        self._vertex_tables = new_tables
        new_ready = np.zeros(required, dtype=bool)
        if current:
            new_ready[:current] = self._vertex_ready
        self._vertex_ready = new_ready

    def prefetch_vertex_tables(self, vertex_ids: np.ndarray, points: np.ndarray) -> None:
        vertex_ids_arr = np.asarray(vertex_ids, dtype=int).reshape(-1)
        if vertex_ids_arr.size == 0:
            return
        self._ensure_capacity(vertex_ids_arr)
        unique_ids, first_indices = _unique_first_indices_int64(vertex_ids_arr)
        missing_ids = unique_ids[~self._vertex_ready[unique_ids]]
        if missing_ids.size == 0:
            return

        missing_points = np.asarray(points, dtype=float).reshape(-1, points.shape[-1])[first_indices[~self._vertex_ready[unique_ids]]]
        _, missing_vectors = self.spectral_cache.get_many_vertices(
            tuple(int(vertex_id) for vertex_id in missing_ids),
            missing_points,
        )
        missing_tables = _density_tables_from_eigenvectors(missing_points, missing_vectors, self.keys_arr)
        self._vertex_tables[missing_ids] = missing_tables
        self._vertex_ready[missing_ids] = True

    def get_vertex_tables(
        self,
        vertex_ids: np.ndarray,
        points: np.ndarray,
        counters: _StageCounters,
        *,
        bands: np.ndarray | None = None,
    ) -> np.ndarray:
        vertex_ids_arr = np.asarray(vertex_ids, dtype=int).reshape(-1)
        self.prefetch_vertex_tables(vertex_ids_arr, points)
        stacked = self._vertex_tables[vertex_ids_arr]
        selected = stacked if bands is None else stacked[:, bands]
        counters.evaluator_evals += selected.shape[0] * selected.shape[1]
        return selected


class _VertexValueStore:
    def __init__(self, spectral_cache: _SpectralCache) -> None:
        self.spectral_cache = spectral_cache
        self._vertex_values: np.ndarray | None = None
        self._vertex_ready = np.zeros(0, dtype=bool)

    def _ensure_capacity(self, vertex_ids: np.ndarray) -> None:
        if vertex_ids.size == 0:
            return
        required = int(np.max(vertex_ids)) + 1
        current = self._vertex_ready.shape[0]
        if current >= required:
            return
        ndof = self.spectral_cache.ndof
        new_values = np.empty((required, ndof), dtype=float)
        if self._vertex_values is not None:
            new_values[:current] = self._vertex_values
        self._vertex_values = new_values
        new_ready = np.zeros(required, dtype=bool)
        if current:
            new_ready[:current] = self._vertex_ready
        self._vertex_ready = new_ready

    def prefetch_vertex_values(self, vertex_ids: np.ndarray, points: np.ndarray) -> None:
        vertex_ids_arr = np.asarray(vertex_ids, dtype=int).reshape(-1)
        if vertex_ids_arr.size == 0:
            return
        self._ensure_capacity(vertex_ids_arr)
        unique_ids, first_indices = _unique_first_indices_int64(vertex_ids_arr)
        missing_mask = ~self._vertex_ready[unique_ids]
        missing_ids = unique_ids[missing_mask]
        if missing_ids.size == 0:
            return

        flat_points = np.asarray(points, dtype=float).reshape(-1, points.shape[-1])
        missing_points = flat_points[first_indices[missing_mask]]
        values = self.spectral_cache.get_many_vertex_values(missing_ids, missing_points)
        self._vertex_values[missing_ids] = values
        self._vertex_ready[missing_ids] = True

    def get_vertex_values(self, vertex_ids: np.ndarray, points: np.ndarray) -> np.ndarray:
        vertex_ids_arr = np.asarray(vertex_ids, dtype=int).reshape(-1)
        self.prefetch_vertex_values(vertex_ids_arr, points)
        return self._vertex_values[vertex_ids_arr]


class _PythonChargePreparationStore:
    def __init__(self, mesh: _ZeroTempGeometryCache, spectral_cache: _SpectralCache) -> None:
        self.mesh = mesh
        self.spectral_cache = spectral_cache
        self._simplex_batches: dict[tuple[int, int], _PreparedChargeBatch] = {}
        self._mesh_batches: dict[int, _PreparedChargeBatch] = {}
        self._mesh_vertex_values: np.ndarray | None = None

    def clear_mesh_batches(self) -> None:
        self._mesh_batches.clear()

    def clear_all(self) -> None:
        self.clear_mesh_batches()
        self._simplex_batches.clear()

    def invalidate_refinement(self, descriptors: dict[int, _RefinementDescriptor]) -> None:
        if not descriptors:
            return
        self.clear_mesh_batches()
        removed_simplex_ids = {int(simplex_id) for simplex_id in descriptors}
        if not removed_simplex_ids:
            return
        self._simplex_batches = {
            key: batch
            for key, batch in self._simplex_batches.items()
            if key[0] not in removed_simplex_ids
        }

    def simplex_batch(self, simplex_id: int, levels: int) -> _PreparedChargeBatch:
        cache_key = (int(simplex_id), int(levels))
        cached = self._simplex_batches.get(cache_key)
        if cached is not None:
            return cached

        if levels <= 0:
            vertex_ids = np.asarray(self.mesh.simplex_vertex_ids(simplex_id), dtype=int)
            points = self.mesh.simplex_points(simplex_id)[np.newaxis, :, :]
            values = self._mesh_vertex_values_array()[vertex_ids][np.newaxis, :, :]
            volumes = np.array([self.mesh.simplex_volume(simplex_id)], dtype=float)
        else:
            leaf_ids = np.asarray(self.mesh.descendant_leaves(simplex_id, levels), dtype=int)
            batch = self._batch_from_leaf_ids(
                leaf_ids,
                np.full(leaf_ids.shape[0], simplex_id, dtype=int),
            )
            self._simplex_batches[cache_key] = batch
            return batch

        batch = _prepare_charge_batch_from_values(
            points,
            values,
            volumes=volumes,
            owner_ids=np.full(points.shape[0], simplex_id, dtype=int),
        )
        self._simplex_batches[cache_key] = batch
        return batch

    def mesh_batch(self, levels: int) -> _PreparedChargeBatch:
        cache_key = int(levels)
        cached = self._mesh_batches.get(cache_key)
        if cached is not None:
            return cached

        simplex_ids = self.mesh.active_simplex_ids_array()
        if simplex_ids.size == 0:
            batch = _empty_prepared_charge_batch()
            self._mesh_batches[cache_key] = batch
            return batch

        if levels <= 0:
            vertex_ids = self.mesh.simplex_vertex_id_array(simplex_ids)
            points = self.mesh.vertex_points_array(vertex_ids)
            volumes = self.mesh.simplex_volumes_array(simplex_ids)
            values = self._mesh_vertex_values_array()[vertex_ids]
            batch = _prepare_charge_batch_from_values(
                points,
                values,
                volumes=volumes,
                owner_ids=simplex_ids,
            )
            self._mesh_batches[cache_key] = batch
            return batch

        leaf_ids: list[int] = []
        owner_ids: list[int] = []
        for simplex_id in simplex_ids:
            descendants = self.mesh.descendant_leaves(simplex_id, levels)
            leaf_ids.extend(int(leaf_id) for leaf_id in descendants)
            owner_ids.extend([int(simplex_id)] * len(descendants))

        batch = self._batch_from_leaf_ids(
            np.asarray(leaf_ids, dtype=int),
            np.asarray(owner_ids, dtype=int),
        )
        self._mesh_batches[cache_key] = batch
        return batch

    def _batch_from_leaf_ids(
        self,
        leaf_ids: np.ndarray,
        owner_ids: np.ndarray,
    ) -> _PreparedChargeBatch:
        if leaf_ids.size == 0:
            return _empty_prepared_charge_batch()

        leaf_ids_arr = np.asarray(leaf_ids, dtype=int)
        owner_ids_arr = np.asarray(owner_ids, dtype=int)
        vertex_ids = self.mesh.simplex_vertex_id_array(leaf_ids_arr)
        points = self.mesh.vertex_points_array(vertex_ids)
        volumes = self.mesh.simplex_volumes_array(leaf_ids_arr)
        values = self._mesh_vertex_values_array()[vertex_ids]
        return _prepare_charge_batch_from_values(
            points,
            values,
            volumes=volumes,
            owner_ids=owner_ids_arr,
        )

    def _mesh_vertex_values_array(self) -> np.ndarray:
        n_vertices = len(self.mesh.vertices)
        if self._mesh_vertex_values is None:
            self._mesh_vertex_values = np.empty((n_vertices, self.spectral_cache.ndof), dtype=float)
            missing_ids = np.arange(n_vertices, dtype=int)
        elif self._mesh_vertex_values.shape[0] < n_vertices:
            previous_count = self._mesh_vertex_values.shape[0]
            new_values = np.empty((n_vertices, self.spectral_cache.ndof), dtype=float)
            new_values[:previous_count] = self._mesh_vertex_values
            self._mesh_vertex_values = new_values
            missing_ids = np.arange(previous_count, n_vertices, dtype=int)
        else:
            return self._mesh_vertex_values

        if missing_ids.size:
            points = self.mesh.vertex_points_array(missing_ids)
            values = self.spectral_cache.get_many_vertex_values(missing_ids, points)
            self._mesh_vertex_values[missing_ids] = values
        return self._mesh_vertex_values


class _PythonDensityPreparationStore:
    def __init__(
        self,
        mesh: _ZeroTempGeometryCache,
        spectral_cache: _SpectralCache,
        table_store: _DensityTableStore,
        value_store: _VertexValueStore,
        *,
        mu: float,
    ) -> None:
        self.mesh = mesh
        self.spectral_cache = spectral_cache
        self.table_store = table_store
        self.value_store = value_store
        self.mu = float(mu)
        self._prepared: dict[int, _PreparedDensitySimplex] = {}

    def prepare_initial_frontier(self, simplex_ids: list[int]) -> None:
        pending = [int(simplex_id) for simplex_id in simplex_ids if int(simplex_id) not in self._prepared]
        if not pending:
            return

        simplex_ids_arr = np.asarray(pending, dtype=int)
        vertex_id_array = self.mesh.simplex_vertex_id_array(simplex_ids_arr)
        simplex_points = self.mesh.vertex_points_array(vertex_id_array)
        n_simplex, n_vertices, ndim = simplex_points.shape
        ndof = self.spectral_cache.ndof

        flat_points = simplex_points.reshape(-1, ndim)
        flat_vertex_ids = vertex_id_array.reshape(-1)
        vertex_energies = self.value_store.get_vertex_values(flat_vertex_ids, flat_points).reshape(
            n_simplex,
            n_vertices,
            ndof,
        )
        self.table_store.prefetch_vertex_tables(flat_vertex_ids, flat_points)
        self._prepare_cells(
            simplex_ids_arr,
            simplex_points,
            vertex_id_array,
            vertex_energies,
        )

    def prepare_refined_children(self, descriptors: dict[int, _RefinementDescriptor]) -> None:
        pending_descriptors = [
            descriptor
            for descriptor in descriptors.values()
            if any(int(child_id) not in self._prepared for child_id in descriptor.child_ids)
        ]
        if not pending_descriptors:
            return

        midpoint_ids, midpoint_first = _unique_first_indices_int64(
            np.asarray([descriptor.new_midpoint_vertex_id for descriptor in pending_descriptors], dtype=int)
        )
        midpoint_points = np.stack([self.mesh.vertices[int(vertex_id)] for vertex_id in midpoint_ids], axis=0)
        self.value_store.prefetch_vertex_values(midpoint_ids, midpoint_points)
        self.table_store.prefetch_vertex_tables(midpoint_ids, midpoint_points)

        child_ids: list[int] = []
        child_vertex_ids: list[np.ndarray] = []
        child_points: list[np.ndarray] = []
        child_energies: list[np.ndarray] = []
        for descriptor in pending_descriptors:
            parent = self._prepared[descriptor.parent_id]
            parent_points = {
                int(vertex_id): parent.vertex_points[index]
                for index, vertex_id in enumerate(parent.vertex_ids)
            }
            midpoint_id = int(descriptor.new_midpoint_vertex_id)
            midpoint_point = self.mesh.vertices[midpoint_id]
            midpoint_energy = np.asarray(self.value_store._vertex_values[midpoint_id], dtype=float)

            for child_id, child_vertices in zip(descriptor.child_ids, descriptor.child_vertex_ids, strict=False):
                if int(child_id) in self._prepared:
                    continue
                points = np.stack(
                    [
                        midpoint_point if int(vertex_id) == midpoint_id else parent_points[int(vertex_id)]
                        for vertex_id in child_vertices
                    ],
                    axis=0,
                )
                energies = np.stack(
                    [
                        midpoint_energy
                        if int(vertex_id) == midpoint_id
                        else np.asarray(self.value_store._vertex_values[int(vertex_id)], dtype=float)
                        for vertex_id in child_vertices
                    ],
                    axis=0,
                )
                child_ids.append(int(child_id))
                child_vertex_ids.append(np.asarray(child_vertices, dtype=int))
                child_points.append(points)
                child_energies.append(energies)

        if not child_ids:
            return

        self._prepare_cells(
            np.asarray(child_ids, dtype=int),
            np.stack(child_points, axis=0),
            np.stack(child_vertex_ids, axis=0),
            np.stack(child_energies, axis=0),
        )

    def prepare_many(self, simplex_ids: list[int]) -> None:
        self.prepare_initial_frontier(simplex_ids)

    def _prepare_cells(
        self,
        simplex_ids: np.ndarray,
        simplex_points: np.ndarray,
        vertex_id_array: np.ndarray,
        vertex_energies: np.ndarray,
    ) -> None:
        n_simplex, n_vertices, ndim = simplex_points.shape
        ndof = self.spectral_cache.ndof

        centroid_points = np.mean(simplex_points, axis=1)
        centroid_energies, centroid_vectors = self.spectral_cache.evaluate_many(centroid_points)
        centroid_energies = np.asarray(centroid_energies, dtype=float)
        centroid_tables = _density_tables_from_eigenvectors(centroid_points, centroid_vectors, self.table_store.keys_arr)
        ncomp = centroid_tables.shape[2]

        vertex_occ = _step_occupation(vertex_energies, self.mu)
        centroid_occ = _step_occupation(centroid_energies, self.mu)

        if _native_prepare_density_cells_metadata is not None:
            (
                whole_offsets,
                whole_bands_flat,
                whole_weights_flat,
                step_offsets,
                step_bands_flat,
                piece_offsets,
                piece_bands_flat,
                piece_volumes_flat,
                piece_centroid_points_arr,
                piece_vertex_offsets_flat,
                piece_vertex_points_arr,
            ) = _native_prepare_density_cells_metadata(
                np.ascontiguousarray(simplex_points, dtype=np.float64),
                np.ascontiguousarray(vertex_energies, dtype=np.float64),
                np.ascontiguousarray(centroid_energies, dtype=np.float64),
                float(self.mu),
                float(_GEOM_TOL),
            )
            whole_offsets = np.asarray(whole_offsets, dtype=int)
            whole_bands_flat = np.asarray(whole_bands_flat, dtype=int)
            whole_weights_flat = np.asarray(whole_weights_flat, dtype=float)
            step_offsets = np.asarray(step_offsets, dtype=int)
            step_bands_flat = np.asarray(step_bands_flat, dtype=int)
            piece_offsets = np.asarray(piece_offsets, dtype=int)
            piece_bands_flat = np.asarray(piece_bands_flat, dtype=int)
            piece_volumes_flat = np.asarray(piece_volumes_flat, dtype=float)
            piece_centroid_points_arr = np.asarray(piece_centroid_points_arr, dtype=float)
            piece_vertex_offsets_flat = np.asarray(piece_vertex_offsets_flat, dtype=int)
            piece_vertex_points_arr = np.asarray(piece_vertex_points_arr, dtype=float)
        else:
            whole_bands_flat_list: list[int] = []
            whole_weights_flat_list: list[float] = []
            step_bands_flat_list: list[int] = []
            piece_bands_flat_list: list[int] = []
            piece_volumes_flat_list: list[float] = []
            piece_centroid_points: list[np.ndarray] = []
            piece_vertex_points: list[np.ndarray] = []
            whole_counts = np.zeros(n_simplex, dtype=int)
            step_counts = np.zeros(n_simplex, dtype=int)
            piece_counts = np.zeros(n_simplex, dtype=int)
            piece_vertex_offsets_list: list[int] = [0]

            vertex_matches_centroid = np.all(vertex_occ == centroid_occ[:, np.newaxis, :], axis=1)
            band_min = np.min(vertex_energies, axis=1)
            band_max = np.max(vertex_energies, axis=1)
            half_mask = np.all(np.abs(vertex_energies - self.mu) <= _GEOM_TOL, axis=1)
            empty_mask = (band_min > self.mu) & vertex_matches_centroid & ~half_mask
            full_mask = (band_max <= self.mu) & vertex_matches_centroid & ~half_mask
            whole_mask = half_mask | full_mask
            candidate_mask = ~(whole_mask | empty_mask)

            for local_index in range(n_simplex):
                whole_bands = np.flatnonzero(whole_mask[local_index]).astype(int)
                whole_weights = np.where(half_mask[local_index, whole_bands], 0.5, 1.0)
                whole_bands_flat_list.extend(int(band) for band in whole_bands)
                whole_weights_flat_list.extend(float(weight) for weight in whole_weights)
                whole_counts[local_index] = whole_bands.size

                candidate_bands = np.flatnonzero(candidate_mask[local_index]).astype(int)
                for band in candidate_bands:
                    pieces = _occupied_subsimplices(simplex_points[local_index], vertex_energies[local_index, :, band], self.mu)
                    if not pieces:
                        step_bands_flat_list.append(int(band))
                        step_counts[local_index] += 1
                        continue
                    for piece in pieces:
                        piece_points = np.asarray(piece, dtype=float)
                        piece_bands_flat_list.append(int(band))
                        piece_volumes_flat_list.append(_simplex_volume(piece_points))
                        piece_centroid_points.append(np.mean(piece_points, axis=0))
                        piece_vertex_points.extend(piece_points)
                        piece_vertex_offsets_list.append(piece_vertex_offsets_list[-1] + piece_points.shape[0])
                        piece_counts[local_index] += 1

            whole_offsets = np.zeros(n_simplex + 1, dtype=int)
            whole_offsets[1:] = np.cumsum(whole_counts)
            step_offsets = np.zeros(n_simplex + 1, dtype=int)
            step_offsets[1:] = np.cumsum(step_counts)
            piece_offsets = np.zeros(n_simplex + 1, dtype=int)
            piece_offsets[1:] = np.cumsum(piece_counts)
            whole_bands_flat = np.asarray(whole_bands_flat_list, dtype=int)
            whole_weights_flat = np.asarray(whole_weights_flat_list, dtype=float)
            step_bands_flat = np.asarray(step_bands_flat_list, dtype=int)
            piece_bands_flat = np.asarray(piece_bands_flat_list, dtype=int)
            piece_volumes_flat = np.asarray(piece_volumes_flat_list, dtype=float)
            piece_centroid_points_arr = (
                np.stack(piece_centroid_points, axis=0)
                if piece_centroid_points
                else np.empty((0, ndim), dtype=float)
            )
            piece_vertex_points_arr = (
                np.stack(piece_vertex_points, axis=0)
                if piece_vertex_points
                else np.empty((0, ndim), dtype=float)
            )
            piece_vertex_offsets_flat = np.asarray(piece_vertex_offsets_list, dtype=int)

        simplex_volumes = np.array([self.mesh.simplex_volume(int(simplex_id)) for simplex_id in simplex_ids], dtype=float)
        if piece_centroid_points_arr.size:
            _, piece_centroid_vectors = self.spectral_cache.evaluate_many(piece_centroid_points_arr)
            piece_centroid_tables = _density_tables_from_eigenvectors(
                piece_centroid_points_arr,
                piece_centroid_vectors,
                self.table_store.keys_arr,
            )
        else:
            piece_centroid_tables = np.empty((0, ndof, ncomp), dtype=complex)

        if piece_vertex_points_arr.size:
            _, piece_vertex_vectors = self.spectral_cache.evaluate_many(piece_vertex_points_arr)
            piece_vertex_tables = _density_tables_from_eigenvectors(
                piece_vertex_points_arr,
                piece_vertex_vectors,
                self.table_store.keys_arr,
            )
        else:
            piece_vertex_tables = np.empty((0, ndof, ncomp), dtype=complex)

        for local_index, simplex_id in enumerate(simplex_ids):
            whole_bands = whole_bands_flat[whole_offsets[local_index] : whole_offsets[local_index + 1]]
            whole_weights = whole_weights_flat[whole_offsets[local_index] : whole_offsets[local_index + 1]]
            step_bands = step_bands_flat[step_offsets[local_index] : step_offsets[local_index + 1]]
            piece_start = piece_offsets[local_index]
            piece_stop = piece_offsets[local_index + 1]
            piece_bands = piece_bands_flat[piece_start:piece_stop]

            if piece_start != piece_stop:
                global_vertex_start = int(piece_vertex_offsets_flat[piece_start])
                global_vertex_stop = int(piece_vertex_offsets_flat[piece_stop])
                local_piece_offsets = piece_vertex_offsets_flat[piece_start : piece_stop + 1] - global_vertex_start
                local_piece_counts = np.diff(local_piece_offsets)
                repeated_piece_bands = np.repeat(piece_bands, local_piece_counts)
                piece_centroid_tables_local = np.asarray(
                    piece_centroid_tables[np.arange(piece_start, piece_stop), piece_bands],
                    dtype=complex,
                )
                piece_vertex_tables_local = np.asarray(
                    piece_vertex_tables[
                        np.arange(global_vertex_start, global_vertex_stop, dtype=int),
                        repeated_piece_bands,
                        :,
                    ],
                    dtype=complex,
                )
                piece_volumes_local = piece_volumes_flat[piece_start:piece_stop]
                piece_vertex_offsets_local = np.asarray(local_piece_offsets, dtype=int)
            else:
                piece_centroid_tables_local = np.empty((0, ncomp), dtype=complex)
                piece_vertex_tables_local = np.empty((0, ncomp), dtype=complex)
                piece_volumes_local = np.empty((0,), dtype=float)
                piece_vertex_offsets_local = np.zeros(1, dtype=int)

            self._prepared[int(simplex_id)] = _PreparedDensitySimplex(
                simplex_id=int(simplex_id),
                volume=float(simplex_volumes[local_index]),
                vertex_ids=vertex_id_array[local_index].copy(),
                vertex_points=simplex_points[local_index].copy(),
                whole_bands=whole_bands.astype(int, copy=False),
                whole_weights=np.asarray(whole_weights, dtype=float),
                whole_centroid_tables=np.asarray(
                    centroid_tables[local_index, whole_bands] if whole_bands.size else np.empty((0, ncomp), dtype=complex),
                    dtype=complex,
                ),
                step_bands=step_bands.astype(int, copy=False),
                step_centroid_tables=np.asarray(
                    centroid_tables[local_index, step_bands] if step_bands.size else np.empty((0, ncomp), dtype=complex),
                    dtype=complex,
                ),
                step_centroid_occ=np.asarray(centroid_occ[local_index, step_bands], dtype=float),
                step_vertex_occ=np.asarray(vertex_occ[local_index, :, step_bands].T, dtype=float),
                piece_volumes=np.asarray(piece_volumes_local, dtype=float),
                piece_centroid_tables=piece_centroid_tables_local,
                piece_vertex_offsets=piece_vertex_offsets_local,
                piece_vertex_tables=piece_vertex_tables_local,
            )

    def get(self, simplex_id: int) -> _PreparedDensitySimplex:
        return self._prepared[int(simplex_id)]


# Backward-compatible aliases while the public zero-temperature API still
# exposes some internal symbols through tests and diagnostics.
_ChargePreparationStore = _PythonChargePreparationStore
_DensityPreparationStore = _PythonDensityPreparationStore

def density_matrix_zero_temp(
    h: _tb_type,
    *,
    filling: float,
    keys: list[tuple[int, ...]],
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
    mu_guess: float,
    mu_xtol: float,
    max_mu_iterations: int,
    max_subdivisions: int | None,
    geometry_cache: _ZeroTempGeometryCache | None = None,
):
    ndim = len(next(iter(h)))
    mesh = _ensure_geometry_cache(ndim, geometry_cache)
    spectral_cache = _SpectralCache(h)
    preparation_store = _ChargePreparationStore(mesh, spectral_cache)
    charge_counters = _StageCounters()
    density_counters = _StageCounters()

    remaining_refinements = max_subdivisions
    lower, upper = _mu_bracket_zero_temp(h)
    charge_evaluator = _ChargeEvaluator(preparation_store, mu_xtol, charge_counters)
    preview_charge_evaluator = _ChargeEvaluator(
        preparation_store,
        mu_xtol,
        charge_counters,
        refine_levels=_MU_ERROR_REFINE_LEVELS,
    )
    lower, upper = _expand_mu_bracket_zero_temp(
        preview_charge_evaluator.evaluate,
        filling=filling,
        lower=lower,
        upper=upper,
    )

    mu = float(np.clip(mu_guess, lower, upper))
    if not lower < mu < upper:
        mu = 0.5 * (lower + upper)

    charge = float("nan")
    derivative = float("nan")
    charge_error = float("nan")
    root_iterations = 0
    root_charge_tol = max(0.25 * charge_tol, np.finfo(float).eps)
    while True:
        # Solve on the finer preview model. The coarse active-frontier model is
        # retained only as the local reference for the refinement indicator.
        mu, charge, derivative, root_iterations = _solve_mu_zero_temp(
            preview_charge_evaluator.evaluate,
            filling=filling,
            mu_guess=mu,
            lower=lower,
            upper=upper,
            charge_tol=root_charge_tol,
            mu_xtol=mu_xtol,
            max_mu_iterations=max_mu_iterations,
        )
        indicator_summary = _charge_indicators(
            mesh,
            charge_evaluator,
            preview_charge_evaluator,
            mu,
        )
        charge = indicator_summary.refined_charge
        charge_error = indicator_summary.indicator
        if abs(charge - filling) <= charge_tol and charge_error <= charge_tol:
            break
        if remaining_refinements is not None and remaining_refinements <= 0:
            raise ValueError("Adaptive zero-temperature charge integration did not converge")

        marked = _bulk_mark_heap(indicator_summary.heap, charge_error)
        if not marked:
            break
        refinements, refinement_descriptors = mesh.refine_with_children(marked)
        charge_counters.refinements += refinements
        if remaining_refinements is not None:
            remaining_refinements -= refinements
            if remaining_refinements < 0:
                raise ValueError("Adaptive zero-temperature charge integration did not converge")
        preparation_store.invalidate_refinement(refinement_descriptors)
        charge_evaluator.clear_cache()
        preview_charge_evaluator.clear_cache()

    charge_kernel_evals = spectral_cache.n_kernel_evals
    density_summary = _adaptive_density(
        mesh,
        spectral_cache,
        keys=keys,
        mu=mu,
        density_atol=density_atol,
        density_rtol=density_rtol,
        remaining_refinements=remaining_refinements,
        counters=density_counters,
        start_from_current_mesh=True,
    )
    density_kernel_evals = spectral_cache.n_kernel_evals - charge_kernel_evals

    rho, error = _vector_to_density(
        density_summary.estimate,
        density_summary.error_vector,
        spectral_cache.ndof,
        keys,
    )
    info = _fixed_filling_info(
        mu=mu,
        charge=charge,
        charge_error=charge_error,
        derivative=derivative,
        root_iterations=root_iterations,
        charge_tol=charge_tol,
        density_atol=density_atol,
        density_rtol=density_rtol,
        charge_counters=charge_counters,
        density_counters=density_counters,
        charge_kernel_evals=charge_kernel_evals,
        density_kernel_evals=density_kernel_evals,
        spectral_cache=spectral_cache,
        mesh=mesh,
    )
    return rho, error, mu, info, mesh


def density_matrix_at_mu_zero_temp(
    h: _tb_type,
    *,
    mu: float,
    keys: list[tuple[int, ...]],
    density_atol: float,
    density_rtol: float,
    max_subdivisions: int | None,
    geometry_cache: _ZeroTempGeometryCache | None = None,
):
    ndim = len(next(iter(h)))
    mesh = _ensure_geometry_cache(ndim, geometry_cache)
    spectral_cache = _SpectralCache(h)
    counters = _StageCounters()

    density_summary = _adaptive_density(
        mesh,
        spectral_cache,
        keys=keys,
        mu=mu,
        density_atol=density_atol,
        density_rtol=density_rtol,
        remaining_refinements=max_subdivisions,
        counters=counters,
        start_from_current_mesh=False,
    )
    rho, error = _vector_to_density(
        density_summary.estimate,
        density_summary.error_vector,
        spectral_cache.ndof,
        keys,
    )
    info = _density_integration_info(
        counters=counters,
        spectral_cache=spectral_cache,
        mesh=mesh,
    )
    return rho, error, info, mesh


class _ChargeEvaluator:
    def __init__(
        self,
        preparation_store: _ChargePreparationStore,
        mu_xtol: float,
        counters: _StageCounters,
        refine_levels: int = 0,
    ) -> None:
        self.preparation_store = preparation_store
        self.mu_xtol = mu_xtol
        self.counters = counters
        self.refine_levels = int(refine_levels)
        self._charge_cache: dict[float, _ChargeBatchEvaluation] = {}
        self._simplex_charge_cache: dict[tuple[float, int, int], float] = {}
        mesh = preparation_store.mesh
        spectral_cache = preparation_store.spectral_cache
        self._native_evaluator = None
        if _native_charge_runtime_available(mesh, spectral_cache):
            self._native_evaluator = _NativeChargeEvaluator(
                mesh._native_geometry,
                spectral_cache._native_cache,
                refine_levels=self.refine_levels,
                tol=float(_GEOM_TOL),
            )

    def clear_cache(self) -> None:
        self._charge_cache.clear()
        self._simplex_charge_cache.clear()

    def _sync_native_preview_tree(self, levels: int) -> None:
        if self._native_evaluator is None or levels <= 0:
            return
        mesh = self.preparation_store.mesh
        for simplex_id in mesh.active_simplex_ids_array():
            mesh.descendant_leaves(int(simplex_id), int(levels))

    def evaluate(self, mu: float) -> _ChargeSummary:
        self.counters.integration_calls += 1
        return self._charge_evaluation(mu).summary

    def owner_charges(self, mu: float) -> tuple[np.ndarray, np.ndarray]:
        evaluation = self._charge_evaluation(mu)
        return evaluation.owner_ids, evaluation.owner_charges

    def simplex_charge(
        self,
        simplex_id: int,
        mu: float,
        *,
        refine_levels: int | None = None,
    ) -> float:
        levels = self.refine_levels if refine_levels is None else int(refine_levels)
        cache_key = (mu, simplex_id, levels)
        cached = self._simplex_charge_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._native_evaluator is not None:
            self._sync_native_preview_tree(levels)
            charge = float(self._native_evaluator.simplex_charge(int(simplex_id), float(mu), int(levels)))
            self._simplex_charge_cache[cache_key] = charge
            return charge

        batch = self.preparation_store.simplex_batch(simplex_id, levels)
        charge, _, _, _ = _charge_from_prepared_batch(batch, mu, self.counters)
        self._simplex_charge_cache[cache_key] = float(charge)
        return float(charge)

    def _charge_evaluation(self, mu: float) -> _ChargeBatchEvaluation:
        cached = self._charge_cache.get(mu)
        if cached is not None:
            return cached

        if self._native_evaluator is not None:
            self._sync_native_preview_tree(self.refine_levels)
            (
                charge,
                derivative,
                derivative_exact,
                owner_ids,
                owner_charges,
            ) = self._native_evaluator.evaluate(self.preparation_store.mesh._native_frontier, float(mu))
            self.counters.evaluator_evals += int(np.asarray(owner_ids).shape[0])
            evaluation = _ChargeBatchEvaluation(
                summary=_ChargeSummary(
                    charge=float(charge),
                    derivative=float(derivative) if bool(derivative_exact) else float("nan"),
                ),
                owner_ids=np.asarray(owner_ids, dtype=int),
                owner_charges=np.asarray(owner_charges, dtype=float),
                derivative_exact=bool(derivative_exact),
            )
            self._charge_cache[mu] = evaluation
            return evaluation

        batch = self.preparation_store.mesh_batch(self.refine_levels)
        charge, derivative, derivative_exact, owner_charges = _charge_from_prepared_batch(
            batch,
            mu,
            self.counters,
        )
        evaluation = _ChargeBatchEvaluation(
            summary=_ChargeSummary(
                charge=float(charge),
                derivative=float(derivative) if derivative_exact else float("nan"),
            ),
            owner_ids=batch.owner_unique,
            owner_charges=owner_charges,
            derivative_exact=derivative_exact,
        )
        self._charge_cache[mu] = evaluation
        return evaluation


def _adaptive_density(
    mesh: _ZeroTempGeometryCache,
    spectral_cache: _SpectralCache,
    *,
    keys: list[tuple[int, ...]],
    mu: float,
    density_atol: float,
    density_rtol: float,
    remaining_refinements: int | None,
    counters: _StageCounters,
    start_from_current_mesh: bool,
) -> _DensitySummary:
    if _native_density_runtime_available(mesh, spectral_cache):
        return _adaptive_density_native(
            mesh,
            spectral_cache,
            keys=keys,
            mu=mu,
            density_atol=density_atol,
            density_rtol=density_rtol,
            remaining_refinements=remaining_refinements,
            counters=counters,
            start_from_current_mesh=start_from_current_mesh,
        )
    return _adaptive_density_python(
        mesh,
        spectral_cache,
        keys=keys,
        mu=mu,
        density_atol=density_atol,
        density_rtol=density_rtol,
        remaining_refinements=remaining_refinements,
        counters=counters,
        start_from_current_mesh=start_from_current_mesh,
    )


def _adaptive_density_python(
    mesh: _ZeroTempGeometryCache,
    spectral_cache: _SpectralCache,
    *,
    keys: list[tuple[int, ...]],
    mu: float,
    density_atol: float,
    density_rtol: float,
    remaining_refinements: int | None,
    counters: _StageCounters,
    start_from_current_mesh: bool,
) -> _DensitySummary:

    if not start_from_current_mesh and not mesh.simplices:
        mesh = _ZeroTempGeometryCache.root(mesh.ndim)

    keys_arr = np.array(keys, dtype=float)
    table_store = _DensityTableStore(spectral_cache, keys_arr)
    value_store = _VertexValueStore(spectral_cache)
    preparation_store = _DensityPreparationStore(mesh, spectral_cache, table_store, value_store, mu=mu)
    ncomp = spectral_cache.ndof * spectral_cache.ndof * len(keys)
    total_estimate = np.zeros(ncomp, dtype=complex)
    total_error_vector = np.zeros(ncomp, dtype=float)
    total_error_scalar = 0.0
    cell_values: dict[int, _DensityCellValue] = {}
    error_versions: dict[int, int] = {}
    error_heap: list[tuple[float, int, int]] = []

    def evaluate_simplex(simplex_id: int) -> _DensityCellValue:
        prepared = preparation_store.get(simplex_id)
        estimate, error_vector, error_scalar = _simplex_density_python(prepared, table_store, counters)
        return _DensityCellValue(
            estimate=estimate,
            error_vector=error_vector,
            error_scalar=error_scalar,
        )

    def accumulate(value: _DensityCellValue, sign: float) -> None:
        nonlocal total_estimate, total_error_vector, total_error_scalar
        total_estimate += sign * value.estimate
        total_error_vector += sign * value.error_vector
        total_error_scalar += sign * value.error_scalar

    def store_simplex_value(simplex_id: int, value: _DensityCellValue) -> None:
        cell_values[simplex_id] = value
        version = error_versions.get(simplex_id, 0) + 1
        error_versions[simplex_id] = version
        if value.error_scalar > 0.0:
            heapq.heappush(error_heap, (-value.error_scalar, simplex_id, version))

    preparation_store.prepare_initial_frontier(mesh.simplices)
    for simplex_id in mesh.simplices:
        value = evaluate_simplex(simplex_id)
        store_simplex_value(simplex_id, value)
        accumulate(value, 1.0)

    while True:
        tolerance = density_atol + density_rtol * float(np.linalg.norm(total_estimate))
        summary = _DensitySummary(
            estimate=total_estimate,
            error_vector=total_error_vector,
            error_scalar=total_error_scalar,
            markers=[
                (simplex_id, value.error_scalar)
                for simplex_id, value in cell_values.items()
                if value.error_scalar > 0.0
            ],
        )
        if total_error_scalar <= tolerance:
            return summary
        if remaining_refinements is not None and remaining_refinements <= 0:
            raise ValueError("Adaptive zero-temperature density integration did not converge")

        target = _BULK_THETA * total_error_scalar
        marked: list[int] = []
        accumulated = 0.0
        while error_heap and accumulated < target:
            neg_error, simplex_id, version = heapq.heappop(error_heap)
            value = cell_values.get(simplex_id)
            if value is None or error_versions.get(simplex_id) != version:
                continue
            marked.append(simplex_id)
            accumulated += -neg_error
        if not marked:
            return summary
        refinements, refinement_descriptors = mesh.refine_with_children(marked)
        counters.refinements += refinements
        if remaining_refinements is not None:
            remaining_refinements -= refinements
            if remaining_refinements < 0:
                raise ValueError("Adaptive zero-temperature density integration did not converge")

        new_simplex_ids: list[int] = []
        for simplex_id in marked:
            value = cell_values.pop(simplex_id)
            accumulate(value, -1.0)
            error_versions[simplex_id] = error_versions.get(simplex_id, 0) + 1
            for child_id in refinement_descriptors[simplex_id].child_ids:
                new_simplex_ids.append(child_id)

        preparation_store.prepare_refined_children(refinement_descriptors)
        for child_id in new_simplex_ids:
            child_value = evaluate_simplex(child_id)
            store_simplex_value(child_id, child_value)
            accumulate(child_value, 1.0)


def _adaptive_density_native(
    mesh: _ZeroTempGeometryCache,
    spectral_cache: _SpectralCache,
    *,
    keys: list[tuple[int, ...]],
    mu: float,
    density_atol: float,
    density_rtol: float,
    remaining_refinements: int | None,
    counters: _StageCounters,
    start_from_current_mesh: bool,
) -> _DensitySummary:
    if not start_from_current_mesh and not mesh.simplices:
        mesh = _ZeroTempGeometryCache.root(mesh.ndim)

    keys_arr = np.asarray(keys, dtype=np.float64)
    native_evaluator = _NativeDensityEvaluator(
        mesh._native_geometry,
        spectral_cache._native_cache,
        np.ascontiguousarray(keys_arr, dtype=np.float64),
        tol=float(_GEOM_TOL),
    )
    ncomp = spectral_cache.ndof * spectral_cache.ndof * len(keys)
    total_estimate = np.zeros(ncomp, dtype=complex)
    total_error_vector = np.zeros(ncomp, dtype=float)
    total_error_scalar = 0.0
    cell_values: dict[int, _DensityCellValue] = {}
    error_versions: dict[int, int] = {}
    error_heap: list[tuple[float, int, int]] = []

    def evaluate_many(simplex_ids: list[int]) -> list[_DensityCellValue]:
        if not simplex_ids:
            return []
        estimates, error_vectors, error_scalars, evaluator_evals = native_evaluator.evaluate_many(
            np.asarray(simplex_ids, dtype=np.int64),
            float(mu),
        )
        counters.evaluator_evals += int(evaluator_evals)
        estimates_arr = np.asarray(estimates, dtype=complex)
        error_vectors_arr = np.asarray(error_vectors, dtype=float)
        error_scalars_arr = np.asarray(error_scalars, dtype=float)
        return [
            _DensityCellValue(
                estimate=estimates_arr[index].copy(),
                error_vector=error_vectors_arr[index].copy(),
                error_scalar=float(error_scalars_arr[index]),
            )
            for index in range(estimates_arr.shape[0])
        ]

    def accumulate(value: _DensityCellValue, sign: float) -> None:
        nonlocal total_estimate, total_error_vector, total_error_scalar
        total_estimate += sign * value.estimate
        total_error_vector += sign * value.error_vector
        total_error_scalar += sign * value.error_scalar

    def store_simplex_value(simplex_id: int, value: _DensityCellValue) -> None:
        cell_values[simplex_id] = value
        version = error_versions.get(simplex_id, 0) + 1
        error_versions[simplex_id] = version
        if value.error_scalar > 0.0:
            heapq.heappush(error_heap, (-value.error_scalar, simplex_id, version))

    initial_simplex_ids = [int(simplex_id) for simplex_id in mesh.simplices]
    for simplex_id, value in zip(initial_simplex_ids, evaluate_many(initial_simplex_ids), strict=False):
        store_simplex_value(simplex_id, value)
        accumulate(value, 1.0)

    while True:
        tolerance = density_atol + density_rtol * float(np.linalg.norm(total_estimate))
        summary = _DensitySummary(
            estimate=total_estimate,
            error_vector=total_error_vector,
            error_scalar=total_error_scalar,
            markers=[
                (simplex_id, value.error_scalar)
                for simplex_id, value in cell_values.items()
                if value.error_scalar > 0.0
            ],
        )
        if total_error_scalar <= tolerance:
            return summary
        if remaining_refinements is not None and remaining_refinements <= 0:
            raise ValueError("Adaptive zero-temperature density integration did not converge")

        target = _BULK_THETA * total_error_scalar
        marked: list[int] = []
        accumulated = 0.0
        while error_heap and accumulated < target:
            neg_error, simplex_id, version = heapq.heappop(error_heap)
            value = cell_values.get(simplex_id)
            if value is None or error_versions.get(simplex_id) != version:
                continue
            marked.append(simplex_id)
            accumulated += -neg_error
        if not marked:
            return summary

        refinements, refinement_descriptors = mesh.refine_with_children(marked)
        counters.refinements += refinements
        if remaining_refinements is not None:
            remaining_refinements -= refinements
            if remaining_refinements < 0:
                raise ValueError("Adaptive zero-temperature density integration did not converge")

        new_simplex_ids: list[int] = []
        for simplex_id in marked:
            value = cell_values.pop(simplex_id)
            accumulate(value, -1.0)
            error_versions[simplex_id] = error_versions.get(simplex_id, 0) + 1
            for child_id in refinement_descriptors[simplex_id].child_ids:
                new_simplex_ids.append(int(child_id))

        for child_id, child_value in zip(new_simplex_ids, evaluate_many(new_simplex_ids), strict=False):
            store_simplex_value(child_id, child_value)
            accumulate(child_value, 1.0)


def _simplex_charge(
    simplex_points: np.ndarray,
    spectral_cache: _SpectralCache,
    mu: float,
    counters: _StageCounters,
    *,
    vertex_ids: tuple[int, ...] | None = None,
    volume: float | None = None,
) -> float:
    prepared = _prepare_charge_batch(
        simplex_points,
        spectral_cache,
        vertex_ids=vertex_ids,
        volume=volume,
    )
    charge, _, _, _ = _charge_from_prepared_batch(prepared, mu, counters)
    return float(charge)


def _refined_simplex_charge(
    simplex_points: np.ndarray,
    spectral_cache: _SpectralCache,
    mu: float,
    counters: _StageCounters,
    *,
    refine_levels: int,
) -> float:
    batch = _prepare_refined_charge_batch(
        simplex_points,
        spectral_cache,
        refine_levels=refine_levels,
    )
    charge, _, _, _ = _charge_from_prepared_batch(batch, mu, counters)
    return float(charge)


def _prepare_charge_batch(
    simplex_points: np.ndarray,
    spectral_cache: _SpectralCache,
    *,
    vertex_ids: tuple[int, ...] | None = None,
    volume: float | None = None,
) -> _PreparedChargeBatch:
    simplex_volume = _simplex_volume(simplex_points) if volume is None else float(volume)
    if vertex_ids is None:
        vertex_energies = spectral_cache.get_many_values(simplex_points)[np.newaxis, :, :]
    else:
        vertex_energies = spectral_cache.get_many_vertex_values(vertex_ids, simplex_points)[np.newaxis, :, :]
    return _prepare_charge_batch_from_values(
        np.asarray(simplex_points, dtype=float)[np.newaxis, :, :],
        vertex_energies,
        volumes=np.array([float(simplex_volume)], dtype=float),
        owner_ids=np.array([0], dtype=int),
    )


def _prepare_refined_charge_batch(
    simplex_points: np.ndarray,
    spectral_cache: _SpectralCache,
    *,
    refine_levels: int,
) -> _PreparedChargeBatch:
    geometry = _refined_simplex_leaf_geometry(simplex_points, refine_levels=refine_levels)
    unique_values = spectral_cache.get_many_values(geometry.unique_points)
    return _prepare_charge_batch_from_values(
        geometry.unique_points[geometry.leaf_indices],
        unique_values[geometry.leaf_indices],
        volumes=geometry.leaf_volumes,
        owner_ids=np.zeros(geometry.leaf_volumes.shape[0], dtype=int),
    )


def _empty_prepared_charge_batch() -> _PreparedChargeBatch:
    return _PreparedChargeBatch(
        points=np.empty((0, 0, 0), dtype=float),
        vertex_energies=np.empty((0, 0, 0), dtype=float),
        volumes=np.empty(0, dtype=float),
        sorted_energies=np.empty((0, 0, 0), dtype=float),
        simplex_weights=np.empty((0, 0, 0), dtype=float),
        distinct_mask=np.empty((0, 0), dtype=bool),
        band_min=np.empty((0, 0), dtype=float),
        band_max=np.empty((0, 0), dtype=float),
        flat_energy=np.empty((0, 0), dtype=float),
        flat_mask=np.empty((0, 0), dtype=bool),
        cell_min=np.empty(0, dtype=float),
        cell_max=np.empty(0, dtype=float),
        dimension=0,
        ndof=0,
        owner_ids=np.empty(0, dtype=int),
        owner_unique=np.empty(0, dtype=int),
        owner_inverse=np.empty(0, dtype=int),
    )


def _owner_metadata(owner_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    owner_arr = np.asarray(owner_ids, dtype=int)
    if owner_arr.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    change = np.empty(owner_arr.shape[0], dtype=bool)
    change[0] = True
    change[1:] = owner_arr[1:] != owner_arr[:-1]
    owner_unique = owner_arr[change]
    owner_inverse = np.cumsum(change, dtype=int) - 1
    return owner_unique, owner_inverse


def _unique_first_indices_int64(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if values_arr.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if _native_unique_first_indices_int64 is not None:
        unique_values, first_indices = _native_unique_first_indices_int64(
            np.ascontiguousarray(values_arr, dtype=np.int64)
        )
        return np.asarray(unique_values, dtype=int), np.asarray(first_indices, dtype=int)

    seen: dict[int, int] = {}
    unique_values: list[int] = []
    first_indices: list[int] = []
    for index, value in enumerate(values_arr):
        ivalue = int(value)
        if ivalue in seen:
            continue
        seen[ivalue] = len(unique_values)
        unique_values.append(ivalue)
        first_indices.append(index)
    return np.asarray(unique_values, dtype=int), np.asarray(first_indices, dtype=int)


def _prepare_charge_batch_from_values(
    points: np.ndarray,
    vertex_energies: np.ndarray,
    *,
    volumes: np.ndarray,
    owner_ids: np.ndarray,
) -> _PreparedChargeBatch:
    points_arr = np.asarray(points, dtype=float)
    values = np.asarray(vertex_energies, dtype=float)
    volumes_arr = np.asarray(volumes, dtype=float)
    owners = np.asarray(owner_ids, dtype=int)
    if points_arr.size == 0:
        return _empty_prepared_charge_batch()
    if points_arr.ndim == 2:
        points_arr = points_arr[np.newaxis, :, :]
    if values.ndim == 2:
        values = values[np.newaxis, :, :]

    dimension = points_arr.shape[2]
    if _native_prepare_charge_batch_metadata is not None:
        (
            sorted_energies,
            simplex_weights,
            distinct_mask,
            band_min,
            band_max,
            flat_energy,
            flat_mask,
            cell_min,
            cell_max,
            owner_unique,
            owner_inverse,
        ) = _native_prepare_charge_batch_metadata(
            np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(volumes_arr, dtype=np.float64),
            np.ascontiguousarray(owners, dtype=np.int64),
            float(_GEOM_TOL),
        )
        sorted_energies = np.asarray(sorted_energies, dtype=float)
        simplex_weights = np.asarray(simplex_weights, dtype=float)
        distinct_mask = np.asarray(distinct_mask, dtype=bool)
        band_min = np.asarray(band_min, dtype=float)
        band_max = np.asarray(band_max, dtype=float)
        flat_energy = np.asarray(flat_energy, dtype=float)
        flat_mask = np.asarray(flat_mask, dtype=bool)
        cell_min = np.asarray(cell_min, dtype=float)
        cell_max = np.asarray(cell_max, dtype=float)
        owner_unique = np.asarray(owner_unique, dtype=int)
        owner_inverse = np.asarray(owner_inverse, dtype=int)
    else:
        sorted_energies = np.sort(values, axis=1).transpose(0, 2, 1)
        distinct_mask = ~np.any(np.diff(sorted_energies, axis=2) <= _GEOM_TOL, axis=2)
        simplex_weights = np.zeros_like(sorted_energies)
        if dimension > 3 and np.any(distinct_mask):
            diff = sorted_energies[:, :, :, np.newaxis] - sorted_energies[:, :, np.newaxis, :]
            diagonal = np.arange(diff.shape[2])
            diff[:, :, diagonal, diagonal] = 1.0
            simplex_weights[distinct_mask] = 1.0 / np.prod(diff[distinct_mask], axis=2)

        band_min = np.min(values, axis=1)
        band_max = np.max(values, axis=1)
        flat_mask = (band_max - band_min) <= _GEOM_TOL
        flat_energy = values[:, 0, :].copy()
        cell_min = np.min(band_min, axis=1)
        cell_max = np.max(band_max, axis=1)
        owner_unique, owner_inverse = _owner_metadata(owners)
    return _PreparedChargeBatch(
        points=points_arr,
        vertex_energies=values,
        volumes=volumes_arr,
        sorted_energies=sorted_energies,
        simplex_weights=simplex_weights,
        distinct_mask=distinct_mask,
        band_min=band_min,
        band_max=band_max,
        flat_energy=flat_energy,
        flat_mask=flat_mask,
        cell_min=cell_min,
        cell_max=cell_max,
        dimension=dimension,
        ndof=values.shape[2],
        owner_ids=owners,
        owner_unique=owner_unique,
        owner_inverse=owner_inverse,
    )


def _concatenate_charge_batches(batches: list[_PreparedChargeBatch]) -> _PreparedChargeBatch:
    non_empty = [batch for batch in batches if batch.volumes.size]
    if not non_empty:
        return _empty_prepared_charge_batch()
    if len(non_empty) == 1:
        return non_empty[0]

    owner_ids = np.concatenate([batch.owner_ids for batch in non_empty], axis=0)
    owner_unique, owner_inverse = _owner_metadata(owner_ids)

    return _PreparedChargeBatch(
        points=np.concatenate([batch.points for batch in non_empty], axis=0),
        vertex_energies=np.concatenate([batch.vertex_energies for batch in non_empty], axis=0),
        volumes=np.concatenate([batch.volumes for batch in non_empty], axis=0),
        sorted_energies=np.concatenate([batch.sorted_energies for batch in non_empty], axis=0),
        simplex_weights=np.concatenate([batch.simplex_weights for batch in non_empty], axis=0),
        distinct_mask=np.concatenate([batch.distinct_mask for batch in non_empty], axis=0),
        band_min=np.concatenate([batch.band_min for batch in non_empty], axis=0),
        band_max=np.concatenate([batch.band_max for batch in non_empty], axis=0),
        flat_energy=np.concatenate([batch.flat_energy for batch in non_empty], axis=0),
        flat_mask=np.concatenate([batch.flat_mask for batch in non_empty], axis=0),
        cell_min=np.concatenate([batch.cell_min for batch in non_empty], axis=0),
        cell_max=np.concatenate([batch.cell_max for batch in non_empty], axis=0),
        dimension=non_empty[0].dimension,
        ndof=non_empty[0].ndof,
        owner_ids=owner_ids,
        owner_unique=owner_unique,
        owner_inverse=owner_inverse,
    )


def _charge_from_prepared_batch(
    batch: _PreparedChargeBatch,
    mu: float,
    counters: _StageCounters,
) -> tuple[float, float, bool, np.ndarray]:
    if batch.volumes.size == 0:
        return 0.0, 0.0, True, np.empty(0, dtype=float)

    counters.evaluator_evals += batch.band_min.shape[0] * batch.band_min.shape[1]

    full_cell_mask = batch.cell_max <= mu
    empty_cell_mask = batch.cell_min > mu
    total_charge = float(np.sum(batch.volumes[full_cell_mask]) * batch.ndof)
    total_derivative = 0.0
    derivative_exact = True

    partial_indices = np.flatnonzero(~(full_cell_mask | empty_cell_mask))
    if partial_indices.size == 0:
        owner_charges = np.bincount(
            batch.owner_inverse,
            weights=np.where(full_cell_mask, batch.volumes * batch.ndof, 0.0),
            minlength=batch.owner_unique.size,
        )
        return total_charge, total_derivative, derivative_exact, owner_charges

    volumes = batch.volumes[partial_indices]
    band_min = batch.band_min[partial_indices]
    band_max = batch.band_max[partial_indices]
    flat_mask = batch.flat_mask[partial_indices]
    flat_energy = batch.flat_energy[partial_indices]

    full_mask = band_max <= mu
    empty_mask = band_min > mu
    flat_half_mask = flat_mask & ~(full_mask | empty_mask) & (np.abs(flat_energy - mu) <= _GEOM_TOL)

    cell_charge = volumes * (
        np.count_nonzero(full_mask, axis=1) + 0.5 * np.count_nonzero(flat_half_mask, axis=1)
    )
    cell_derivative = np.zeros(partial_indices.size, dtype=float)

    cut_mask = ~(full_mask | empty_mask | flat_half_mask)
    distinct_cut_mask = cut_mask & batch.distinct_mask[partial_indices]
    if np.any(distinct_cut_mask):
        fractions = np.zeros_like(band_min, dtype=float)
        derivative_fraction = np.zeros_like(band_min, dtype=float)
        cut_sorted_energies = batch.sorted_energies[partial_indices][distinct_cut_mask]
        cut_weights = batch.simplex_weights[partial_indices][distinct_cut_mask]
        cut_fraction_values, cut_derivative_values = _simplex_fraction_and_derivative_batch(
            cut_sorted_energies,
            mu,
            batch.dimension,
            weights=cut_weights,
        )
        fractions[distinct_cut_mask] = cut_fraction_values
        derivative_fraction[distinct_cut_mask] = cut_derivative_values
        cell_charge += volumes * np.sum(fractions, axis=1)
        cell_derivative += volumes * np.sum(derivative_fraction, axis=1)

    fallback_cut_mask = cut_mask & ~batch.distinct_mask[partial_indices]
    if np.any(fallback_cut_mask):
        derivative_exact = False
        for local_index, band in zip(*np.nonzero(fallback_cut_mask), strict=False):
            prepared_index = int(partial_indices[local_index])
            cell_charge[local_index] += _fallback_cut_band_charge(
                batch.points[prepared_index],
                batch.vertex_energies[prepared_index, :, int(band)],
                mu,
            )

    total_charge += float(np.sum(cell_charge))
    total_derivative += float(np.sum(cell_derivative))
    owner_charges = np.where(full_cell_mask, batch.volumes * batch.ndof, 0.0)
    owner_charges_partial = np.zeros(batch.volumes.shape[0], dtype=float)
    owner_charges_partial[partial_indices] = cell_charge
    owner_charges += owner_charges_partial
    aggregated_owner_charge = np.bincount(
        batch.owner_inverse,
        weights=owner_charges,
        minlength=batch.owner_unique.size,
    )
    return total_charge, total_derivative, derivative_exact, aggregated_owner_charge


def _fallback_cut_band_charge(simplex_points: np.ndarray, band_energies: np.ndarray, mu: float) -> float:
    return float(
        sum(
            _simplex_volume(piece)
            for piece in _occupied_subsimplices(simplex_points, band_energies, mu)
        )
    )


def _simplex_fraction_and_derivative_batch(
    sorted_energies: np.ndarray,
    mu: float,
    dimension: int,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(sorted_energies, dtype=float)
    if energies.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    if dimension == 1:
        e0 = energies[:, 0]
        e1 = energies[:, 1]
        denom = e1 - e0
        fraction = np.clip((mu - e0) / denom, 0.0, 1.0)
        derivative = np.where((mu > e0) & (mu < e1), 1.0 / denom, 0.0)
        return fraction, derivative

    if dimension == 2:
        e0 = energies[:, 0]
        e1 = energies[:, 1]
        e2 = energies[:, 2]
        fraction = np.empty(energies.shape[0], dtype=float)
        derivative = np.empty(energies.shape[0], dtype=float)

        lower = mu < e1
        upper = ~lower

        denom_lower = (e1 - e0) * (e2 - e0)
        x_lower = mu - e0
        fraction[lower] = np.clip((x_lower[lower] * x_lower[lower]) / denom_lower[lower], 0.0, 1.0)
        derivative[lower] = 2.0 * x_lower[lower] / denom_lower[lower]

        denom_upper = (e2 - e0) * (e2 - e1)
        x_upper = e2 - mu
        fraction[upper] = np.clip(1.0 - (x_upper[upper] * x_upper[upper]) / denom_upper[upper], 0.0, 1.0)
        derivative[upper] = 2.0 * x_upper[upper] / denom_upper[upper]
        return fraction, derivative

    if dimension == 3:
        e0 = energies[:, 0]
        e1 = energies[:, 1]
        e2 = energies[:, 2]
        e3 = energies[:, 3]
        fraction = np.empty(energies.shape[0], dtype=float)
        derivative = np.empty(energies.shape[0], dtype=float)

        region1 = mu < e1
        region2 = (mu >= e1) & (mu < e2)
        region3 = ~region1 & ~region2

        if np.any(region1):
            denom = (e1 - e0) * (e2 - e0) * (e3 - e0)
            x = mu - e0
            fraction[region1] = np.clip((x[region1] ** 3) / denom[region1], 0.0, 1.0)
            derivative[region1] = 3.0 * (x[region1] ** 2) / denom[region1]

        if np.any(region2):
            denom0 = (e1 - e0) * (e2 - e0) * (e3 - e0)
            denom1 = (e0 - e1) * (e2 - e1) * (e3 - e1)
            x0 = mu - e0
            x1 = mu - e1
            fraction[region2] = np.clip(
                (x0[region2] ** 3) / denom0[region2] + (x1[region2] ** 3) / denom1[region2],
                0.0,
                1.0,
            )
            derivative[region2] = 3.0 * (
                (x0[region2] ** 2) / denom0[region2] + (x1[region2] ** 2) / denom1[region2]
            )

        if np.any(region3):
            denom = (e3 - e0) * (e3 - e1) * (e3 - e2)
            x = e3 - mu
            fraction[region3] = np.clip(1.0 - (x[region3] ** 3) / denom[region3], 0.0, 1.0)
            derivative[region3] = 3.0 * (x[region3] ** 2) / denom[region3]
        return fraction, derivative

    if weights is None:
        raise ValueError("Generic simplex charge formula requires precomputed weights")

    delta = np.maximum(mu - energies, 0.0)
    fraction = np.sum(weights * delta**dimension, axis=1)
    if dimension == 1:
        derivative = np.sum(weights * (delta > _GEOM_TOL), axis=1)
    else:
        derivative = dimension * np.sum(
            weights * np.where(delta > _GEOM_TOL, delta ** (dimension - 1), 0.0),
            axis=1,
        )
    return np.clip(fraction, 0.0, 1.0), derivative


def _refined_simplex_leaf_geometry(
    simplex_points: np.ndarray,
    *,
    refine_levels: int,
) -> _RefinedLeafGeometry:
    if refine_levels <= 0:
        unique_points = np.asarray(simplex_points, dtype=float).copy()
        leaf_indices = np.arange(unique_points.shape[0], dtype=int)[np.newaxis, :]
        leaf_volumes = np.array([_simplex_volume(unique_points)], dtype=float)
        return _RefinedLeafGeometry(
            unique_points=unique_points,
            leaf_indices=leaf_indices,
            leaf_volumes=leaf_volumes,
        )

    unique_points: list[np.ndarray] = []
    lookup: dict[_PointKey, int] = {}
    leaf_indices: list[np.ndarray] = []
    leaf_volumes: list[float] = []

    def intern(point: np.ndarray) -> int:
        key = _point_key(point)
        index = lookup.get(key)
        if index is not None:
            return index
        index = len(unique_points)
        unique_points.append(np.asarray(point, dtype=float).copy())
        lookup[key] = index
        return index

    def visit(points: np.ndarray, levels: int) -> None:
        if levels <= 0:
            leaf_indices.append(np.array([intern(point) for point in points], dtype=int))
            leaf_volumes.append(_simplex_volume(points))
            return
        child_a, child_b = _bisect_simplex_points(points)
        visit(child_a, levels - 1)
        visit(child_b, levels - 1)

    visit(np.asarray(simplex_points, dtype=float), int(refine_levels))
    return _RefinedLeafGeometry(
        unique_points=np.stack(unique_points, axis=0),
        leaf_indices=np.stack(leaf_indices, axis=0),
        leaf_volumes=np.asarray(leaf_volumes, dtype=float),
    )


def _refined_simplex_leaves(
    simplex_points: np.ndarray,
    *,
    refine_levels: int,
) -> tuple[tuple[np.ndarray, float], ...]:
    if refine_levels <= 0:
        return ((simplex_points, _simplex_volume(simplex_points)),)

    child_a, child_b = _bisect_simplex_points(simplex_points)
    return _refined_simplex_leaves(child_a, refine_levels=refine_levels - 1) + _refined_simplex_leaves(
        child_b,
        refine_levels=refine_levels - 1,
    )


def _charge_indicators(
    mesh: _ZeroTempGeometryCache,
    coarse_evaluator: _ChargeEvaluator,
    refined_evaluator: _ChargeEvaluator,
    mu: float,
) -> _ChargeIndicatorSummary:
    coarse_owner_ids, coarse_charge = coarse_evaluator.owner_charges(mu)
    refined_owner_ids, refined_charge = refined_evaluator.owner_charges(mu)
    if not np.array_equal(coarse_owner_ids, refined_owner_ids):
        raise ValueError("Charge indicator evaluators disagree on simplex ownership")

    indicators = np.abs(refined_charge - coarse_charge)
    total_indicator = float(np.sum(indicators))
    markers = [
        (int(simplex_id), float(indicator))
        for simplex_id, indicator in zip(coarse_owner_ids, indicators, strict=False)
        if indicator > 0.0
    ]
    heap = [(-float(indicator), int(simplex_id)) for simplex_id, indicator in markers]

    heapq.heapify(heap)

    return _ChargeIndicatorSummary(
        coarse_charge=float(np.sum(coarse_charge)),
        refined_charge=float(np.sum(refined_charge)),
        indicator=total_indicator,
        markers=markers,
        heap=heap,
    )


def _simplex_density_python(
    prepared: _PreparedDensitySimplex,
    table_store: _DensityTableStore,
    counters: _StageCounters,
) -> tuple[np.ndarray, np.ndarray, float]:
    ncomp = table_store.spectral_cache.ndof * table_store.spectral_cache.ndof * len(table_store.keys_arr)
    estimate_high = np.zeros(ncomp, dtype=complex)
    estimate_low = np.zeros(ncomp, dtype=complex)
    whole_vertex_tables = np.empty((0, 0, ncomp), dtype=complex)
    step_vertex_tables = np.empty((0, 0, ncomp), dtype=complex)
    if prepared.whole_bands.size:
        whole_vertex_tables = table_store.get_vertex_tables(
            prepared.vertex_ids,
            prepared.vertex_points,
            counters,
            bands=prepared.whole_bands,
        )

    if prepared.step_bands.size:
        step_vertex_tables = table_store.get_vertex_tables(
            prepared.vertex_ids,
            prepared.vertex_points,
            counters,
            bands=prepared.step_bands,
        )

    if prepared.piece_volumes.size:
        counters.evaluator_evals += int(
            prepared.piece_volumes.size + prepared.piece_vertex_offsets[-1]
        )

    if _native_accumulate_density_terms is not None:
        low, high = _native_accumulate_density_terms(
            np.ascontiguousarray(prepared.whole_centroid_tables, dtype=np.complex128),
            np.ascontiguousarray(whole_vertex_tables, dtype=np.complex128),
            np.ascontiguousarray(prepared.whole_weights, dtype=np.float64),
            np.ascontiguousarray(prepared.step_centroid_tables, dtype=np.complex128),
            np.ascontiguousarray(step_vertex_tables, dtype=np.complex128),
            np.ascontiguousarray(prepared.step_centroid_occ, dtype=np.float64),
            np.ascontiguousarray(prepared.step_vertex_occ, dtype=np.float64),
            float(prepared.volume),
            np.ascontiguousarray(prepared.piece_volumes, dtype=np.float64),
            np.ascontiguousarray(prepared.piece_centroid_tables, dtype=np.complex128),
            np.ascontiguousarray(prepared.piece_vertex_offsets, dtype=np.int64),
            np.ascontiguousarray(prepared.piece_vertex_tables, dtype=np.complex128),
        )
        estimate_low += np.asarray(low, dtype=complex)
        estimate_high += np.asarray(high, dtype=complex)
    else:
        if prepared.whole_bands.size:
            high_tables = prepared.volume * np.mean(whole_vertex_tables, axis=0)
            low_tables = prepared.volume * prepared.whole_centroid_tables
            estimate_low += np.sum(prepared.whole_weights[:, np.newaxis] * low_tables, axis=0)
            estimate_high += np.sum(prepared.whole_weights[:, np.newaxis] * high_tables, axis=0)

        if prepared.step_bands.size:
            estimate_low += np.sum(
                prepared.volume * prepared.step_centroid_occ[:, np.newaxis] * prepared.step_centroid_tables,
                axis=0,
            )
            estimate_high += np.sum(
                prepared.volume * np.mean(prepared.step_vertex_occ[:, :, np.newaxis] * step_vertex_tables, axis=1),
                axis=0,
            )

        if prepared.piece_volumes.size:
            for piece_index, piece_volume in enumerate(prepared.piece_volumes):
                start = int(prepared.piece_vertex_offsets[piece_index])
                stop = int(prepared.piece_vertex_offsets[piece_index + 1])
                low = piece_volume * prepared.piece_centroid_tables[piece_index]
                high = piece_volume * np.mean(prepared.piece_vertex_tables[start:stop], axis=0)
                estimate_low += low
                estimate_high += high

    error = estimate_high - estimate_low
    error_vector = np.abs(error)
    error_scalar = float(np.linalg.norm(error))
    return estimate_high, error_vector, error_scalar


def _point_band_density_table(
    point: np.ndarray,
    eigenvectors: np.ndarray,
    keys_arr: np.ndarray,
    counters: _StageCounters,
    *,
    bands: np.ndarray | None = None,
) -> np.ndarray:
    return _points_band_density_tables(
        np.asarray(point, dtype=float)[np.newaxis, :],
        np.asarray(eigenvectors, dtype=complex)[np.newaxis, :, :],
        keys_arr,
        counters,
        bands=bands,
    )[0]


def _density_tables_from_eigenvectors(
    points: np.ndarray,
    eigenvectors: np.ndarray,
    keys_arr: np.ndarray,
    *,
    bands: np.ndarray | None = None,
) -> np.ndarray:
    points_arr = np.asarray(points, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=complex)
    if points_arr.ndim == 1:
        points_arr = points_arr[np.newaxis, :]
        vectors = vectors[np.newaxis, :, :]

    if bands is not None:
        selected = np.asarray(bands, dtype=int)
        vectors = vectors[:, :, selected]
    if _native_density_tables_from_eigenvectors is not None:
        return np.asarray(
            _native_density_tables_from_eigenvectors(
                np.ascontiguousarray(points_arr, dtype=np.float64),
                np.ascontiguousarray(vectors, dtype=np.complex128),
                np.ascontiguousarray(np.asarray(keys_arr, dtype=np.float64)),
            ),
            dtype=complex,
        )
    projectors = np.einsum("pib,pjb->pbij", vectors, vectors.conj(), optimize=True)
    k_points = 2.0 * np.pi * points_arr - np.pi
    phase = np.exp(1j * np.dot(k_points, keys_arr.T))
    values = projectors[..., np.newaxis] * phase[:, np.newaxis, np.newaxis, np.newaxis, :]
    return values.reshape(points_arr.shape[0], vectors.shape[2], -1)


def _points_band_density_tables(
    points: np.ndarray,
    eigenvectors: np.ndarray,
    keys_arr: np.ndarray,
    counters: _StageCounters,
    *,
    bands: np.ndarray | None = None,
) -> np.ndarray:
    points_arr = np.asarray(points, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=complex)
    if points_arr.ndim == 1:
        points_arr = points_arr[np.newaxis, :]
        vectors = vectors[np.newaxis, :, :]
    n_points = points_arr.shape[0]
    n_bands = vectors.shape[2] if bands is None else np.asarray(bands, dtype=int).shape[0]
    counters.evaluator_evals += n_points * n_bands
    return _density_tables_from_eigenvectors(points_arr, vectors, keys_arr, bands=bands)


def _band_density_value(
    point: np.ndarray,
    spectral_cache: _SpectralCache,
    band: int,
    keys_arr: np.ndarray,
    counters: _StageCounters,
    *,
    vertex_id: int | None = None,
) -> np.ndarray:
    counters.evaluator_evals += 1
    if vertex_id is None:
        eigenvalues, eigenvectors = spectral_cache.get(point)
    else:
        eigenvalues, eigenvectors = spectral_cache.get_vertex(vertex_id, point)
    del eigenvalues
    orbital = eigenvectors[:, band]
    projector = np.outer(orbital, orbital.conj())
    k_point = 2.0 * np.pi * np.asarray(point, dtype=float) - np.pi
    phase = np.exp(1j * np.dot(keys_arr, k_point))
    return (projector[..., np.newaxis] * phase[np.newaxis, np.newaxis, :]).reshape(-1)


def _occupied_subsimplices(simplex_points: np.ndarray, energies: np.ndarray, mu: float) -> list[np.ndarray]:
    points_arr = np.asarray(simplex_points, dtype=float)
    energy_arr = np.asarray(energies, dtype=float)
    ndim = simplex_points.shape[1]
    order = np.argsort(energy_arr, kind="mergesort")
    ordered_points = points_arr[order]
    ordered_energies = np.asarray(energy_arr[order], dtype=float)
    n_inside = int(np.searchsorted(ordered_energies, mu, side="right"))
    if n_inside <= 0:
        return []
    if n_inside >= ndim + 1:
        return [ordered_points]

    n_outside = ndim + 1 - n_inside
    grid: list[list[np.ndarray]] = [[ordered_points[i]] + [None] * n_outside for i in range(n_inside)]
    for i in range(n_inside):
        e_inside = ordered_energies[i]
        for offset in range(1, n_outside + 1):
            outside_index = n_inside + offset - 1
            e_outside = ordered_energies[outside_index]
            alpha = 0.0 if e_outside <= e_inside else (mu - e_inside) / (e_outside - e_inside)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            grid[i][offset] = (1.0 - alpha) * ordered_points[i] + alpha * ordered_points[outside_index]

    simplices: list[np.ndarray] = []

    def extend_path(i: int, j: int, path: list[np.ndarray]) -> None:
        if i == n_inside - 1 and j == n_outside:
            candidate = np.stack(path, axis=0)
            if _simplex_volume(candidate) > _GEOM_TOL:
                simplices.append(candidate)
            return
        if i < n_inside - 1:
            extend_path(i + 1, j, path + [grid[i + 1][j]])
        if j < n_outside:
            extend_path(i, j + 1, path + [grid[i][j + 1]])

    extend_path(0, 0, [grid[0][0]])
    return simplices


def _bulk_mark(markers: list[tuple[int, float]], total_error: float) -> list[int]:
    if not markers or total_error <= 0.0:
        return []
    ordered = sorted(markers, key=lambda item: item[1], reverse=True)
    target = _BULK_THETA * total_error
    selected: list[int] = []
    accumulated = 0.0
    for simplex_index, indicator in ordered:
        selected.append(simplex_index)
        accumulated += indicator
        if accumulated >= target:
            break
    return selected


def _solve_mu_zero_temp(
    evaluate_charge,
    *,
    filling: float,
    mu_guess: float,
    lower: float,
    upper: float,
    charge_tol: float,
    mu_xtol: float,
    max_mu_iterations: int,
) -> tuple[float, float, float, int]:
    lower_summary = evaluate_charge(lower)
    upper_summary = evaluate_charge(upper)
    lower_charge = lower_summary.charge
    upper_charge = upper_summary.charge
    mu = float(np.clip(mu_guess, lower, upper))
    if not lower < mu < upper:
        mu = 0.5 * (lower + upper)

    last_charge = float("nan")
    last_derivative = float("nan")
    for iteration in range(1, max_mu_iterations + 1):
        summary = evaluate_charge(mu)
        last_charge = summary.charge
        residual = last_charge - filling
        if abs(residual) <= charge_tol:
            return mu, last_charge, last_derivative, iteration

        if residual < 0.0:
            lower = mu
            lower_charge = last_charge
        else:
            upper = mu
            upper_charge = last_charge

        if upper - lower <= mu_xtol:
            return 0.5 * (lower + upper), last_charge, last_derivative, iteration

        slope = last_derivative
        if not np.isfinite(slope) or slope <= 0.0:
            slope = (upper_charge - lower_charge) / (upper - lower)
        last_derivative = float(slope)
        candidate = mu - residual / slope if slope > 0.0 else np.nan
        if not np.isfinite(candidate) or candidate <= lower or candidate >= upper:
            candidate = 0.5 * (lower + upper)
        mu = float(candidate)

    midpoint = 0.5 * (lower + upper)
    summary = evaluate_charge(midpoint)
    slope = last_derivative
    if not np.isfinite(slope) or slope <= 0.0:
        slope = (upper_charge - lower_charge) / (upper - lower)
    return float(midpoint), float(summary.charge), float(slope), max_mu_iterations


def _expand_mu_bracket_zero_temp(evaluate_charge, *, filling: float, lower: float, upper: float) -> tuple[float, float]:
    lower_charge = evaluate_charge(lower).charge
    upper_charge = evaluate_charge(upper).charge
    while lower_charge > filling or upper_charge < filling:
        lower *= 2.0
        upper *= 2.0
        lower_charge = evaluate_charge(lower).charge
        upper_charge = evaluate_charge(upper).charge
    return lower, upper


def _mu_bracket_zero_temp(h: _tb_type) -> tuple[float, float]:
    bound = sum(np.linalg.norm(matrix, ord=2) for matrix in h.values())
    return -float(bound + 1.0), float(bound + 1.0)


def _vector_to_density(
    estimate: np.ndarray,
    error: np.ndarray,
    ndof: int,
    keys: list[tuple[int, ...]],
):
    estimate = np.asarray(estimate).reshape(ndof, ndof, len(keys))
    error = np.asarray(error).reshape(ndof, ndof, len(keys))
    rho = {}
    rho_error = {}
    for index, key in enumerate(keys):
        rho[key] = estimate[..., index]
        rho_error[key] = error[..., index]
    return rho, rho_error


def _density_integration_info(*, counters: _StageCounters, spectral_cache: _SpectralCache, mesh: _ZeroTempGeometryCache):
    from meanfi.mf import DensityIntegrationInfo

    return DensityIntegrationInfo(
        n_kernel_evals=int(spectral_cache.n_kernel_evals),
        n_evaluator_evals=int(counters.evaluator_evals),
        n_cached_nodes=int(spectral_cache.n_cached_points),
        n_leaves=len(mesh.simplices),
        n_leaf_nodes=_count_leaf_vertices(mesh),
        subdivisions=int(counters.refinements),
    )


def _fixed_filling_info(
    *,
    mu: float,
    charge: float,
    charge_error: float,
    derivative: float,
    root_iterations: int,
    charge_tol: float,
    density_atol: float,
    density_rtol: float,
    charge_counters: _StageCounters,
    density_counters: _StageCounters,
    charge_kernel_evals: int,
    density_kernel_evals: int,
    spectral_cache: _SpectralCache,
    mesh: _ZeroTempGeometryCache,
):
    from meanfi.mf import FixedFillingInfo

    return FixedFillingInfo(
        mu=float(mu),
        charge=float(charge),
        charge_error=float(charge_error),
        dcharge_dmu=float(derivative),
        root_iterations=int(root_iterations),
        charge_integration_calls=int(charge_counters.integration_calls),
        density_integration_calls=1,
        charge_n_kernel_evals=int(charge_kernel_evals),
        density_n_kernel_evals=int(density_kernel_evals),
        n_kernel_evals=int(spectral_cache.n_kernel_evals),
        charge_n_evaluator_evals=int(charge_counters.evaluator_evals),
        density_n_evaluator_evals=int(density_counters.evaluator_evals),
        n_evaluator_evals=int(charge_counters.evaluator_evals + density_counters.evaluator_evals),
        n_cached_nodes=int(spectral_cache.n_cached_points),
        n_leaves=len(mesh.simplices),
        n_leaf_nodes=_count_leaf_vertices(mesh),
        subdivisions=int(charge_counters.refinements + density_counters.refinements),
        charge_integral_atol=float(charge_tol),
        density_atol=float(density_atol),
        density_rtol=float(density_rtol),
    )


def _ensure_geometry_cache(
    ndim: int,
    geometry_cache: _ZeroTempGeometryCache | None,
) -> _ZeroTempGeometryCache:
    if (
        geometry_cache is None
        or geometry_cache.ndim != ndim
        or geometry_cache.root_subcells_per_axis != _ROOT_SUBCELLS_PER_AXIS
    ):
        return _ZeroTempGeometryCache.root(ndim)
    return geometry_cache


def _count_leaf_vertices(mesh: _ZeroTempGeometryCache) -> int:
    used = {vertex_id for simplex_id in mesh.simplices for vertex_id in mesh.simplex_vertex_ids(simplex_id)}
    return len(used)


def _point_key(point: np.ndarray) -> _PointKey:
    if _native_point_key_bytes is not None:
        return bytes(_native_point_key_bytes(np.asarray(point, dtype=np.float64), float(_GEOM_TOL)))
    arr = np.asarray(point, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    if np.any(np.abs(arr) <= _GEOM_TOL):
        arr = np.array(arr, copy=True)
        arr[np.abs(arr) <= _GEOM_TOL] = 0.0
    else:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.tobytes()


def _step_occupation(energies: np.ndarray, mu: float) -> np.ndarray:
    energy_array = np.asarray(energies, dtype=float)
    occupation = np.where(energy_array < mu, 1.0, 0.0)
    occupation = np.where(np.abs(energy_array - mu) <= _GEOM_TOL, 0.5, occupation)
    return occupation.astype(float, copy=False)


def _longest_edge(simplex_points: np.ndarray) -> tuple[int, int]:
    best_pair = (0, 1)
    best_length = -1.0
    for i in range(simplex_points.shape[0]):
        for j in range(i + 1, simplex_points.shape[0]):
            length = float(np.dot(simplex_points[i] - simplex_points[j], simplex_points[i] - simplex_points[j]))
            if length > best_length:
                best_length = length
                best_pair = (i, j)
    return best_pair


def _bisect_simplex_points(simplex_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_i, edge_j = _longest_edge(simplex_points)
    midpoint = 0.5 * (simplex_points[edge_i] + simplex_points[edge_j])
    child_a = np.array(simplex_points, copy=True)
    child_b = np.array(simplex_points, copy=True)
    child_a[edge_i] = midpoint
    child_b[edge_j] = midpoint
    return child_a, child_b


def _simplex_volume(simplex_points: np.ndarray) -> float:
    ndim = simplex_points.shape[1]
    if ndim == 0:
        return 1.0
    edges = simplex_points[1:] - simplex_points[0]
    if ndim == 1:
        return float(abs(edges[0, 0]))
    if ndim == 2:
        determinant = edges[0, 0] * edges[1, 1] - edges[0, 1] * edges[1, 0]
        return float(abs(determinant) * 0.5)
    if ndim == 3:
        determinant = (
            edges[0, 0] * (edges[1, 1] * edges[2, 2] - edges[1, 2] * edges[2, 1])
            - edges[0, 1] * (edges[1, 0] * edges[2, 2] - edges[1, 2] * edges[2, 0])
            + edges[0, 2] * (edges[1, 0] * edges[2, 1] - edges[1, 1] * edges[2, 0])
        )
        return float(abs(determinant) / 6.0)
    determinant = np.linalg.det(edges)
    return float(abs(determinant) / math.factorial(ndim))
