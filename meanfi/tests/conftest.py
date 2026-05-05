import pytest

from meanfi.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE


DENSITY_TOLERANCE_LADDER = (1e-2, 3e-3, 1e-3)
SCALAR_TOLERANCE_LADDER = (1e-3, 3e-4, 1e-4)


@pytest.fixture(scope="session")
def density_tolerance_ladder():
    return DENSITY_TOLERANCE_LADDER


@pytest.fixture(scope="session")
def scalar_tolerance_ladder():
    return SCALAR_TOLERANCE_LADDER


@pytest.fixture(scope="session")
def zero_temp_ext_available():
    return _ZERO_TEMP_EXT_AVAILABLE
