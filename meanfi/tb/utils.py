from itertools import product


def generate_tb_keys(cutoff: int, dim: int) -> list[tuple[int, ...]]:
    """Generate integer displacement keys within ``[-cutoff, cutoff]`` per axis."""
    return list(product(range(-cutoff, cutoff + 1), repeat=dim))
