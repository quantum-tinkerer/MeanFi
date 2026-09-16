from itertools import product
import numpy as np

from meanfi.tb.ops import _tb_type
from meanfi.tb.transforms import tb_to_kgrid


def generate_tb_keys(cutoff: int, dim: int) -> list[tuple[int, ...]]:
    """Generate integer displacement keys within ``[-cutoff, cutoff]`` per axis."""
    return list(product(range(-cutoff, cutoff + 1), repeat=dim))


def fermi_energy(tb: _tb_type, filling: float, *, shape: tuple[int, ...] | None = None):
    """Estimate the zero-temperature Fermi level on a sampled Fourier grid.

    ``shape`` gives points per axis (default: 100 each). This is an order
    statistic of sampled energies, without an integration error estimate.
    Use ``density_matrix`` for a filling solve with controlled tolerances.
    """
    if not np.isfinite(filling) or not 0 <= filling <= next(iter(tb.values())).shape[0]:
        raise ValueError(
            "filling must be finite and between zero and the orbital count"
        )
    if shape is None:
        shape = (100,) * len(next(iter(tb)))
    vals = np.linalg.eigvalsh(tb_to_kgrid(tb, shape))
    flat = np.sort(vals.reshape(-1))
    n_kpoints = int(np.prod(vals.shape[:-1]))
    idx = int(np.clip(np.ceil(filling * n_kpoints) - 1, 0, flat.size - 1))
    return float(flat[idx])
