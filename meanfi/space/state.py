"""Private coordinate-bound active-density state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActiveDensityState:
    """Independent real density coordinates tied to one active space."""

    space: object
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True).reshape(-1)
        if np.any(~np.isfinite(values)):
            raise ValueError("active density state values must be finite")
        expected = getattr(self.space, "num_params", values.size)
        if values.size != int(expected):
            raise ValueError("active density state values do not match their active space")
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


def require_same_space(state: ActiveDensityState, space: object) -> None:
    if state.space is not space:
        raise ValueError("density state belongs to a different active space")
