"""Full sublattice, valley and spin model from arXiv:2401.13413v2."""

from dataclasses import dataclass

import numpy as np

import meanfi as mf

LOCAL = (0, 0)


IDENTITY = np.eye(2)
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
VALLEYS = (np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))


def disk_coordinates(k1, k2, cutoff):
    """Concentric equal-area map from the computational BZ to a momentum disk."""
    a, b = k1 / np.pi - 1, k2 / np.pi - 1
    if a == 0 and b == 0:
        return 0.0, 0.0
    if abs(a) > abs(b):
        radius, angle = cutoff * a, np.pi / 4 * b / a
    else:
        radius, angle = cutoff * b, np.pi / 2 - np.pi / 4 * a / b
    return radius * np.cos(angle), radius * np.sin(angle)


def graphene_interaction(*, sites, u, v, hund, measure):
    """Embed each sublattice's four-flavor operators in the full orbital space."""
    plus, minus = [np.kron(p, IDENTITY) for p in VALLEYS]
    terms = []
    for site in range(sites):
        projector = np.diag(np.arange(sites) == site)

        def embed(operator):
            return np.kron(projector, operator)

        number = embed(np.eye(4))
        terms.append(mf.BilinearTerm(measure * u / 2, number, number))
        terms.append(mf.BilinearTerm(measure * v, embed(plus), embed(minus)))
        for spin in PAULI:
            terms.append(
                mf.BilinearTerm(
                    -measure * hund,
                    embed(np.kron(VALLEYS[0], spin)),
                    embed(np.kron(VALLEYS[1], spin)),
                )
            )
    return mf.BilinearInteraction(terms)


@dataclass(frozen=True)
class Graphene:
    layers: int = 5
    delta: float = -0.012
    soc: float = 0.001
    hund: float = 10.0
    u: float = 40.0
    v: float = -8.0
    gamma0: float = 3.1
    gamma1: float = 0.38
    gamma2: float = -0.015
    gamma3: float = -0.29
    gamma4: float = -0.141
    offset: float = 0.0105
    cutoff: float = 0.16
    offset_sites: str = "dimer"

    def __post_init__(self):
        if not isinstance(self.layers, int) or self.layers < 2:
            raise ValueError("layers must be an integer >= 2")
        if self.offset_sites not in ("dimer", "outer"):
            raise ValueError("offset_sites must be dimer or outer")

    @property
    def measure(self):
        """A_u times the disk measure, with dimensionless q = a_0 k."""
        return (np.sqrt(3) / 2) * self.cutoff**2 / (4 * np.pi)

    def valley_hamiltonian(self, qx, qy, valley):
        n = 2 * self.layers
        pi = valley * qx + 1j * qy
        v0, v3, v4 = np.sqrt(3) / 2 * np.array([self.gamma0, self.gamma3, self.gamma4])
        offsets = np.full(n, self.offset, dtype=float)
        offsets[[0, -1]] = 0.0
        if self.offset_sites == "outer":
            offsets = self.offset - offsets
        offsets += np.repeat(np.linspace(self.delta, -self.delta, self.layers), 2)
        h = np.diag(offsets).astype(complex)
        adjacent = np.array(
            [[v4 * pi.conjugate(), v3 * pi], [self.gamma1, v4 * pi.conjugate()]]
        )
        next_nearest = np.array([[0.0, self.gamma2 / 2], [0.0, 0.0]])
        for layer in range(self.layers):
            start = 2 * layer
            h[start, start + 1] = v0 * pi.conjugate()
            if layer + 1 < self.layers:
                h[start : start + 2, start + 2 : start + 4] = adjacent
            if layer + 2 < self.layers:
                h[start : start + 2, start + 4 : start + 6] = next_nearest
        return h + np.triu(h, 1).conj().T

    def hamiltonian(self, qx, qy):
        h = sum(
            np.kron(self.valley_hamiltonian(qx, qy, valley), np.kron(p, IDENTITY))
            for valley, p in zip((1, -1), VALLEYS)
        )
        bottom = np.diag([0.0] * (2 * self.layers - 2) + [1.0, 1.0])
        return h + self.soc * np.kron(bottom, np.kron(PAULI[2], PAULI[2]))

    def affine_hamiltonian(self):
        """Precompute the exactly affine Cartesian Hamiltonian for native callbacks."""
        constant = self.hamiltonian(0.0, 0.0)
        x = self.hamiltonian(1.0, 0.0) - constant
        y = self.hamiltonian(0.0, 1.0) - constant
        basis = np.stack([constant, x, y]).reshape(3, -1)

        def evaluate(qx, qy):
            # One BLAS contraction avoids several full-matrix temporaries.
            weights = np.array([1.0, qx, qy], dtype=complex)
            return (weights @ basis).reshape(constant.shape)

        return evaluate

    def bloch_hamiltonian(self):
        cartesian = self.affine_hamiltonian()

        def h(k1, k2):
            return cartesian(*disk_coordinates(k1, k2, self.cutoff))

        return mf.BlochHamiltonian(h)

    def model(self):
        interaction = graphene_interaction(
            sites=2 * self.layers,
            u=self.u,
            v=self.v,
            hund=self.hund,
            measure=self.measure,
        )
        return mf.Model(
            self.bloch_hamiltonian(),
            interaction,
            filling=4 * self.layers,
        )

    def seed(self, name, *, amplitude=0.025, random_seed=81):
        surface = np.zeros(2 * self.layers)
        surface[0], surface[-1] = 1.0, -1.0
        if name == "laf_x":
            order = np.kron(IDENTITY, PAULI[0])
        elif name == "laf_z":
            order = np.kron(IDENTITY, PAULI[2])
        elif name == "laf_valley_x":
            order = np.kron(PAULI[2], PAULI[0])
        elif name == "lp":
            order = -np.eye(4) if self.delta < 0 else np.eye(4)
        elif name == "qsh":
            order = np.kron(PAULI[2], PAULI[2])
        elif name == "ivc":
            order = np.kron(PAULI[0], PAULI[2])
        elif name.startswith("qah"):
            signs = -np.ones(4)
            signs[int(name[3:])] = 1.0
            order = np.diag(signs)
        elif name == "random":
            rng = np.random.default_rng(random_seed)
            order = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            order = (order + order.conj().T) / 2
            order /= np.linalg.norm(order, 2)
        else:
            raise ValueError(f"unknown seed {name}")
        return {LOCAL: amplitude * np.kron(np.diag(surface), order)}
