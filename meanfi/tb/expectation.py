"""Observable contractions for complete or selected real-space densities."""

import numpy as np
from meanfi.results import DensityResult
from meanfi.space.coordinates import opposite_key
from meanfi.tb.ops import _tb_type, elementwise_product, is_sparse_like


def expectation_value(
    density_matrix: _tb_type | DensityResult,
    observable: _tb_type,
) -> complex:
    """Compute the expectation value of an observable with respect to a density matrix.

    Parameters
    ----------
    density_matrix :
        Layout-aware density result or complete density-matrix tight-binding
        dictionary.
    observable :
        Observable tight-binding dictionary.

    Returns
    -------
    :
        Unnormalized trace per cell. Unlike the thermodynamic energy helpers,
        this general observable contraction is not divided by orbital count.
    """
    if isinstance(density_matrix, DensityResult):
        available = dict(
            zip(
                density_matrix.coordinates.entries,
                density_matrix.values,
                strict=True,
            )
        )
        total = 0.0j
        missing = []
        for key, block in observable.items():
            if is_sparse_like(block):
                coordinate_matrix = block.tocoo(copy=True)
                coordinate_matrix.sum_duplicates()
                coordinate_matrix.eliminate_zeros()
                rows, cols, weights = (
                    coordinate_matrix.row,
                    coordinate_matrix.col,
                    coordinate_matrix.data,
                )
            else:
                block = np.asarray(block)
                rows, cols = np.nonzero(block)
                weights = block[rows, cols]
            density_key = opposite_key(tuple(key))
            for row, col, weight in zip(rows, cols, weights, strict=True):
                entry = (density_key, int(col), int(row))
                value = available.get(entry)
                if value is None:
                    missing.append(entry)
                else:
                    total += weight * value
        if missing:
            unique_missing = tuple(dict.fromkeys(missing))
            preview = ", ".join(map(str, unique_missing[:3]))
            suffix = "" if len(unique_missing) <= 3 else ", ..."
            raise ValueError(
                "density is missing "
                f"{len(unique_missing)} coordinate(s) required by the observable: "
                f"{preview}{suffix}"
            )
        return complex(total)

    total = 0.0j
    missing_keys = []
    for key, block in observable.items():
        density_key = opposite_key(tuple(key))
        if density_key not in density_matrix:
            if (block != 0).sum():
                missing_keys.append(density_key)
            continue
        total += elementwise_product(block.T, density_matrix[density_key]).sum()
    if missing_keys:
        raise ValueError(
            "density_matrix is missing keys required by the observable: "
            f"{sorted(set(missing_keys))}"
        )
    return complex(total)
