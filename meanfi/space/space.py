from __future__ import annotations

from meanfi.model import Model
from meanfi.space.bdg import BdGMeanFieldDensitySpace
from meanfi.space.normal import NormalMeanFieldDensitySpace


class MeanFieldDensitySpace:
    """Factory for the active density variables in a mean-field problem."""

    @staticmethod
    def normal(model: Model) -> NormalMeanFieldDensitySpace:
        return NormalMeanFieldDensitySpace.from_model(model)

    @staticmethod
    def bdg(model: Model) -> BdGMeanFieldDensitySpace:
        return BdGMeanFieldDensitySpace.from_model(model)


__all__ = [
    "BdGMeanFieldDensitySpace",
    "MeanFieldDensitySpace",
    "NormalMeanFieldDensitySpace",
]
