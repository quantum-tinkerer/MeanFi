import matplotlib.pyplot as plt
import numpy as np

import meanfi


def hubbard_chain(U: float = 2.0):
    hop = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {(0,): hop + hop.T.conj(), (1,): hop, (-1,): hop.T.conj()}
    h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
    return h_0, h_int


def plot_bands(tb, *, nk: int = 150):
    hk = meanfi.tb_to_kfunc(tb)
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    bands = np.linalg.eigvalsh(hk(ks[:, None]))

    plt.figure(figsize=(6, 4))
    plt.plot(ks, bands, color="black")
    plt.xlabel(r"$k$")
    plt.ylabel("Energy")
    plt.axhline(0.0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    h_0, h_int = hubbard_chain(U=2.0)
    model = meanfi.Model(h_0, h_int, filling=2.0, kT=0.0)
    guess = meanfi.guess_tb(frozenset(h_int), ndof=4)

    solution = meanfi.solver(
        model,
        guess,
        integration=meanfi.AdaptiveSimplex(density_matrix_tol=1e-6),
        scf=meanfi.LinearMixing(max_iterations=80),
        scf_tol=1e-6,
    )

    h_mf = meanfi.add_tb(h_0, solution.mf)
    print(f"method: {solution.info.method}")
    print(f"iterations: {solution.info.iterations}")
    print(f"residual norm: {solution.info.residual_norm:.3e}")
    print(f"mu: {solution.density_matrix_result.mu:.6f}")

    plot_bands(h_mf)
