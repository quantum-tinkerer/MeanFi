import pytest


DENSITY_TOLERANCE_LADDER = (1e-2, 3e-3, 1e-3)
SCALAR_TOLERANCE_LADDER = (1e-3, 3e-4, 1e-4)


@pytest.fixture(scope="session")
def density_tolerance_ladder():
    return DENSITY_TOLERANCE_LADDER


@pytest.fixture(scope="session")
def scalar_tolerance_ladder():
    return SCALAR_TOLERANCE_LADDER


@pytest.fixture(scope="session")
def require_mumps():
    pytest.importorskip("mumps", reason="requires the meanfi[sparse] extra")
