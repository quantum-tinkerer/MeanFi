from __future__ import annotations

import numpy as np

from meanfi import Model
from meanfi.tests.helpers import (
    antiferromagnetic_guess,
    bipartite_hubbard_1d,
    local_dense_model,
)


def block_chain_model(ndof: int, *, coupling_scale: float = 0.2):
    onsite = np.asarray(local_dense_model(ndof)[(0, 0)], dtype=complex)
    hopping = -float(coupling_scale) * np.eye(ndof, dtype=complex)
    return {
        (0,): onsite,
        (1,): hopping,
        (-1,): hopping.conj().T,
    }


def block_chain_keys() -> list[tuple[int, ...]]:
    return [(0,), (1,), (-1,)]


def hubbard_chain_scf_problem(*, U: float, kT: float, delta: float = 0.2):
    h_0, h_int = bipartite_hubbard_1d(U)
    model = Model(h_0, h_int, filling=2.0, kT=kT)
    guess = antiferromagnetic_guess(delta, 1)
    return model, guess
