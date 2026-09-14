"""Private coordinate-bound active-density state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from meanfi.space.space import ActiveSCFSpace


@dataclass(frozen=True)
class ActiveDensityState:
    """Independent real density coordinates tied to one active space."""

    space: ActiveSCFSpace
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True).reshape(-1)
        if np.any(~np.isfinite(values)):
            raise ValueError("active density state values must be finite")
        if values.size != self.space.num_params:
            raise ValueError(
                "active density state values do not match their active space"
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def relative_to(
        self,
        reference: ActiveDensityState | None,
    ) -> ActiveDensityState:
        """Return this state relative to a compatible optional reference."""

        if reference is None:
            return self
        require_same_space(reference, self.space)
        return ActiveDensityState(self.space, self.values - reference.values)


def require_same_space(state: ActiveDensityState, space: ActiveSCFSpace) -> None:
    if state.space is not space:
        raise ValueError("density state belongs to a different active space")
