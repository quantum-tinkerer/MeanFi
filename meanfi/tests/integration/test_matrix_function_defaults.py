import pytest
import scipy.sparse as sp

from meanfi import AdaptiveSimplex, DirectDiagonalization, PeriodicGrid, RationalFOE
from meanfi.density.integrate.defaults import select_default_integration
from meanfi.density.integrate.periodic import resolve_periodic_matrix_function
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration


def test_rational_foe_defaults_to_ozaki():
    assert RationalFOE().rational_scheme == "ozaki"


def test_prescribed_sparse_periodic_defaults_to_aaa():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    resolved = resolve_periodic_matrix_function(
        None, sparse_tb, kT=0.15, prescribed=True
    )
    assert isinstance(resolved, RationalFOE)
    assert resolved.rational_scheme == "aaa"


def test_dense_periodic_defaults_to_direct():
    for prescribed in (False, True):
        assert isinstance(
            resolve_periodic_matrix_function(
                None, spinful_chain(), kT=0.15, prescribed=prescribed
            ),
            DirectDiagonalization,
        )


def test_dense_finite_temperature_defaults_to_periodic():
    resolved = select_default_integration(spinful_chain(), kT=0.15)
    assert isinstance(resolved, PeriodicGrid)
    assert resolved.nk is None
    assert isinstance(resolved.matrix_function, DirectDiagonalization)


def test_sparse_automatic_integration_requires_explicit_choice():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    with pytest.raises(ValueError, match="Automatic finite-temperature sparse"):
        select_default_integration(sparse_tb, kT=0.15)
    with pytest.raises(ValueError, match="Automatic sparse"):
        resolve_periodic_matrix_function(None, sparse_tb, kT=0.15, prescribed=False)
    assert isinstance(
        resolve_periodic_matrix_function(
            DirectDiagonalization(), sparse_tb, kT=0.15, prescribed=False
        ),
        DirectDiagonalization,
    )


def test_adaptive_rational_is_explicitly_unsupported():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    with pytest.raises(ValueError, match="adaptive RationalFOE is unsupported"):
        resolve_periodic_matrix_function(
            RationalFOE(), sparse_tb, kT=0.15, prescribed=False
        )


def test_zero_temperature_defaults():
    assert isinstance(
        select_default_integration(spinful_chain(), kT=0), AdaptiveSimplex
    )
    with pytest.raises(NotImplementedError, match="PeriodicGrid"):
        select_default_integration(spinful_chain(), kT=0, superconducting=True)
