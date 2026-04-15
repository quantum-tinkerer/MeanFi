import numpy as np
from scipy.optimize import anderson

from meanfi import Model, add_tb, guess_tb, solver, tb_to_kfunc


def _dense_gap(tb, nk=4000):
    hkfunc = tb_to_kfunc(tb)
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    eigenvalues = np.linalg.eigvalsh(hkfunc(ks[:, None])).reshape(-1)
    below = eigenvalues[eigenvalues <= 0]
    above = eigenvalues[eigenvalues > 0]
    return float(np.min(above) - np.max(below))


def _gap_relation_hubbard(Us, tol=2e-1):
    hopp = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
    h_0 = {(0,): hopp + hopp.T.conj(), (1,): hopp, (-1,): hopp.T.conj()}
    gaps = []
    for U in Us:
        h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
        np.random.seed(int(100 * U))
        guess = guess_tb(frozenset(h_int), len(next(iter(h_0.values()))))
        model = Model(
            h_0,
            h_int,
            filling=2,
            kT=0.05,
            charge_tol=1e-6,
            density_atol=1e-6,
            scf_tol=1e-5,
        )
        mf_sol = solver(
            model,
            guess,
            optimizer=anderson,
            optimizer_kwargs={
                "M": 0,
                "line_search": "wolfe",
                "maxiter": 80,
                "f_tol": model.scf_tol,
            },
            max_scf_steps=80,
        )
        gaps.append(_dense_gap(add_tb(h_0, mf_sol)))

    fit_gap = np.polyfit(Us, np.array(gaps), 1)[0]
    assert np.abs(fit_gap - 1.0) < tol


def test_gap_hubbard():
    _gap_relation_hubbard(np.linspace(8, 10, 5, endpoint=True))
