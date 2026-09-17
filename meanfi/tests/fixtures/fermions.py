"""Small exact Fock-space references for interaction tests."""

import numpy as np


def annihilators(size):
    operators = np.zeros((size, 2**size, 2**size), complex)
    for orbital in range(size):
        for state in range(2**size):
            if state & (1 << orbital):
                parity = (state & ((1 << orbital) - 1)).bit_count()
                operators[orbital, state ^ (1 << orbital), state] = (-1) ** parity
    return operators
