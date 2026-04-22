import pytest

from meanfi.zero_temp import _NATIVE_ZERO_TEMP_AVAILABLE


DENSITY_TOLERANCE_LADDER = (1e-2, 3e-3, 1e-3)
SCALAR_TOLERANCE_LADDER = (1e-3, 3e-4, 1e-4)


@pytest.fixture(scope="session")
def density_tolerance_ladder():
    return DENSITY_TOLERANCE_LADDER


@pytest.fixture(scope="session")
def scalar_tolerance_ladder():
    return SCALAR_TOLERANCE_LADDER


@pytest.fixture(scope="session")
def perf_smoke_benchmark_config():
    return {"repeat": 5, "warmup": 2}


@pytest.fixture(scope="session")
def perf_slow_benchmark_config():
    return {"repeat": 3, "warmup": 1}


@pytest.fixture(scope="session")
def native_zero_temp_available():
    return _NATIVE_ZERO_TEMP_AVAILABLE
