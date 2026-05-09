"""General tight-binding dictionary API."""

from __future__ import annotations

from meanfi.tb.backend import tb_to_tight_binding_model, tb_to_vertex_cache
from meanfi.tb.bdg import (
    assemble_bdg_tb,
    electron_to_bdg_tb,
    particle_hole_conjugate,
    split_bdg_matrix,
    validate_bdg_tb,
    zero_bdg_array,
)
from meanfi.tb.ops import (
    _tb_type,
    add_tb,
    as_sparse,
    block_diag,
    compare_dicts,
    conjugate_transpose,
    elementwise_product,
    hermitian_spectral_bound,
    is_sparse_like,
    matrix_bound,
    matrix_shape,
    scale_tb,
    to_dense,
    transpose,
)
from meanfi.tb.storage import match_tb_storage, prefers_sparse_storage, tb_entries_changed
from meanfi.tb.transforms import ifftn_to_tb, kgrid_to_tb, tb_to_kfunc, tb_to_kgrid
from meanfi.tb.utils import fermi_energy, generate_tb_keys, guess_tb
from meanfi.tb.validate import (
    matrix_allclose,
    normalize_keys,
    require_zero_dim_local_key_only,
    tb_dimension,
    tb_orbital_count,
    validate_bdg_state,
    validate_hermiticity,
    validate_tb_dict,
    zero_key,
)

__all__ = [
    "_tb_type",
    "add_tb",
    "assemble_bdg_tb",
    "as_sparse",
    "block_diag",
    "compare_dicts",
    "conjugate_transpose",
    "elementwise_product",
    "electron_to_bdg_tb",
    "fermi_energy",
    "generate_tb_keys",
    "guess_tb",
    "hermitian_spectral_bound",
    "ifftn_to_tb",
    "is_sparse_like",
    "kgrid_to_tb",
    "matrix_allclose",
    "matrix_bound",
    "matrix_shape",
    "match_tb_storage",
    "normalize_keys",
    "particle_hole_conjugate",
    "prefers_sparse_storage",
    "require_zero_dim_local_key_only",
    "scale_tb",
    "split_bdg_matrix",
    "tb_dimension",
    "tb_orbital_count",
    "tb_to_kfunc",
    "tb_to_kgrid",
    "tb_to_tight_binding_model",
    "tb_to_vertex_cache",
    "tb_entries_changed",
    "validate_bdg_state",
    "to_dense",
    "transpose",
    "validate_bdg_tb",
    "validate_hermiticity",
    "validate_tb_dict",
    "zero_bdg_array",
    "zero_key",
]
