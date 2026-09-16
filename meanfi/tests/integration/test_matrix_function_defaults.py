import pytest
import scipy.sparse as sp

from meanfi import FermiSimplex, DirectDiagonalization, UniformGrid, RationalFOE
from meanfi.density.problem import resolve_integration
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration


def test_prescribed_sparse_periodic_defaults_to_aaa():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    resolved = resolve_integration(
        sparse_tb,
        kT=0.15,
        integration=UniformGrid(nk=1 if True else None, matrix_function=None),
    ).matrix_function
    assert isinstance(resolved, RationalFOE)


def test_dense_periodic_defaults_to_direct():
    for prescribed in (False, True):
        assert isinstance(
            resolve_integration(
                spinful_chain(),
                kT=0.15,
                integration=UniformGrid(
                    nk=1 if prescribed else None, matrix_function=None
                ),
            ).matrix_function,
            DirectDiagonalization,
        )


def test_dense_finite_temperature_defaults_to_periodic():
    resolved = resolve_integration(spinful_chain(), kT=0.15)
    assert isinstance(resolved, UniformGrid)
    assert resolved.nk is None
    assert isinstance(resolved.matrix_function, DirectDiagonalization)


def test_sparse_automatic_integration_requires_explicit_choice():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    with pytest.raises(ValueError, match="Automatic sparse"):
        resolve_integration(sparse_tb, kT=0.15)
    with pytest.raises(ValueError, match="Automatic sparse"):
        resolve_integration(
            sparse_tb,
            kT=0.15,
            integration=UniformGrid(nk=1 if False else None, matrix_function=None),
        ).matrix_function
    assert isinstance(
        resolve_integration(
            sparse_tb,
            kT=0.15,
            integration=UniformGrid(
                nk=1 if False else None, matrix_function=DirectDiagonalization()
            ),
        ).matrix_function,
        DirectDiagonalization,
    )


def test_adaptive_rational_is_explicitly_unsupported():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    with pytest.raises(ValueError, match="adaptive RationalFOE is unsupported"):
        resolve_integration(
            sparse_tb,
            kT=0.15,
            integration=UniformGrid(
                nk=1 if False else None, matrix_function=RationalFOE()
            ),
        ).matrix_function


def test_zero_temperature_defaults():
    assert isinstance(resolve_integration(spinful_chain(), kT=0), FermiSimplex)
    with pytest.raises(ValueError, match="UniformGrid"):
        resolve_integration(spinful_chain(), kT=0, superconducting=True)
