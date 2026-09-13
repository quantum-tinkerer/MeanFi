"""Shape, symmetry and projection checks must not densify sparse matrices."""

import numpy as np
import pytest
from scipy import sparse

from meanfi.tb.bdg import validate_bdg_tb
from meanfi.tb.storage import tb_entries_changed
from meanfi.tb.validate import matrix_allclose, validate_hermiticity


def test_large_sparse_validation_never_materializes_matrices(monkeypatch):
    size = 100_000
    electron = sparse.eye(size, format="csr", dtype=complex)
    bdg = sparse.diags(np.r_[np.ones(size), -np.ones(size)], format="csr")

    def forbid_dense(*args, **kwargs):
        raise AssertionError("validation attempted to densify a sparse matrix")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
    validate_hermiticity({(0,): electron})
    validate_bdg_tb({(0,): bdg}, ndof=size, ndim=1)
    assert not tb_entries_changed({(0,): electron}, {(0,): electron.copy()})
    assert tb_entries_changed({(0,): electron}, {})


@pytest.mark.parametrize("sparse_left", [False, True])
@pytest.mark.parametrize("sparse_right", [False, True])
def test_matrix_comparison_uses_absolute_entry_tolerance(sparse_left, sparse_right):
    left = np.array([[1.0, 0.0], [0.3, -1.0]])
    right = left.copy()
    right[0, 1] += 1e-7
    lhs = sparse.csr_matrix(left) if sparse_left else left
    rhs = sparse.csr_matrix(right) if sparse_right else right
    assert matrix_allclose(lhs, rhs, atol=1e-6)
    assert not matrix_allclose(lhs, rhs, atol=1e-8)


def test_projection_comparison_detects_removed_nonzero_keys():
    zero = np.zeros((2, 2))
    assert tb_entries_changed({(1,): np.eye(2)}, {})
    assert tb_entries_changed({}, {(1,): np.eye(2)})
    assert not tb_entries_changed({(1,): zero}, {})
    assert not tb_entries_changed({}, {(1,): zero})
