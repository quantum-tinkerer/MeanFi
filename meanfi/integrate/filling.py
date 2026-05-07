from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from meanfi.tb.ops import _tb_type, matrix_bound


ChargeEvaluation = Callable[[float], tuple[float, float, float | None]]
_CHARGE_ERROR_ACCEPTANCE_FRACTION = 0.5
_CHARGE_INTEGRAL_ATOL_FRACTION = 0.25
_MAX_BRACKET_EXPANSIONS = 64
_BRENT_MAXITER = 10_000


def _conservative_spectral_bound(hamiltonian: _tb_type) -> float:
    """Return a conservative spectral bound for the tight-binding Hamiltonian."""

    return float(sum(matrix_bound(matrix) for matrix in hamiltonian.values()))


def mu_bracket(hamiltonian: _tb_type, kT: float) -> tuple[float, float]:
    """Return a conservative chemical-potential bracket."""

    if not hamiltonian:
        raise ValueError("Hamiltonian cannot be empty")
    if not np.isfinite(kT) or kT < 0.0:
        raise ValueError("kT must be a nonnegative finite number")
    bound = _conservative_spectral_bound(hamiltonian)
    padding = max(1.0, 10.0 * kT)
    return -float(bound + padding), float(bound + padding)


def charge_integral_tolerance(filling_tol: float) -> tuple[float, float]:
    """Translate filling tolerance to charge-integral tolerances.

    The fixed-filling solver only accepts a charge sample when both
    `abs(charge - filling) <= filling_tol` and
    `charge_error <= filling_tol / 2`. We therefore budget at most one quarter of
    the total filling tolerance to the charge integration error so the root solver
    still has headroom to resolve the physical residual.
    """

    filling_tol_value = float(filling_tol)
    if not np.isfinite(filling_tol_value) or filling_tol_value <= 0.0:
        raise ValueError("filling_tol must be a positive finite number")
    return filling_tol_value * _CHARGE_INTEGRAL_ATOL_FRACTION, 0.0


@dataclass(frozen=True)
class FixedFillingSolve:
    mu: float
    charge: float
    charge_error: float
    residual: float
    derivative: float | None
    charge_evaluations: int


@dataclass(frozen=True)
class _ChargeSample:
    mu: float
    charge: float
    charge_error: float
    residual: float
    derivative: float | None


def _evaluate_charge_sample(
    evaluate_charge: ChargeEvaluation,
    *,
    filling: float,
    mu: float,
) -> _ChargeSample:
    charge, charge_error, derivative = evaluate_charge(float(mu))
    charge_value = float(charge)
    charge_error_value = float(charge_error)
    derivative_value = None if derivative is None else float(derivative)
    if not np.isfinite(charge_value):
        raise ValueError(f"Charge evaluation returned non-finite charge at mu={mu}")
    if not np.isfinite(charge_error_value) or charge_error_value < 0.0:
        raise ValueError(
            f"Charge evaluation returned invalid charge error at mu={mu}: {charge_error}"
        )
    if derivative_value is not None and not np.isfinite(derivative_value):
        raise ValueError(f"Charge evaluation returned non-finite derivative at mu={mu}")
    return _ChargeSample(
        mu=float(mu),
        charge=charge_value,
        charge_error=charge_error_value,
        residual=charge_value - float(filling),
        derivative=derivative_value,
    )


@dataclass
class _ChargeBracket:
    lower: _ChargeSample
    upper: _ChargeSample

    def update(self, sample: _ChargeSample) -> None:
        if sample.residual < 0.0 and sample.mu > self.lower.mu:
            self.lower = sample
        elif sample.residual > 0.0 and sample.mu < self.upper.mu:
            self.upper = sample

    @property
    def pair(self) -> tuple[float, float]:
        return self.lower.mu, self.upper.mu


class _MaxRootIterations(RuntimeError):
    pass


class _AcceptedSample(RuntimeError):
    # scipy.brentq exposes only a residual-based stopping rule, so we use a
    # private exception to stop once the physical charge acceptance criterion is met.
    def __init__(self, sample: _ChargeSample) -> None:
        self.sample = sample


class _ChargeRootSolver:
    def __init__(
        self,
        evaluate_charge: ChargeEvaluation,
        *,
        filling: float,
        filling_tol: float,
        mu_xtol: float,
        max_charge_evaluations: int | None,
        use_derivative: bool,
    ) -> None:
        self.evaluate_charge = evaluate_charge
        self.filling = float(filling)
        self.filling_tol = float(filling_tol)
        self.mu_xtol = float(mu_xtol)
        self.max_charge_evaluations = max_charge_evaluations
        self.use_derivative = bool(use_derivative)
        self.charge_evaluations = 0
        self.cache: dict[float, _ChargeSample] = {}
        self.last: _ChargeSample | None = None
        self.best: _ChargeSample | None = None

    def solve_with_expansion(
        self,
        *,
        lower: float,
        upper: float,
        mu_guess: float,
    ) -> FixedFillingSolve:
        bracket = self._expand_bracket(lower=lower, upper=upper)
        return self.solve_in_bracket(
            lower=bracket.lower.mu,
            upper=bracket.upper.mu,
            mu_guess=mu_guess,
            bracket=bracket,
        )

    def solve_in_bracket(
        self,
        *,
        lower: float,
        upper: float,
        mu_guess: float,
        bracket: _ChargeBracket | None = None,
    ) -> FixedFillingSolve:
        if bracket is None:
            bracket = _ChargeBracket(self.sample(lower), self.sample(upper))
        if bracket.lower.residual > 0.0 or bracket.upper.residual < 0.0:
            raise ValueError(
                "Chemical-potential bracket does not enclose the requested filling"
            )
        for sample in (bracket.lower, bracket.upper):
            if self.accepted(sample):
                return self._result(sample)

        mu = float(np.clip(mu_guess, bracket.lower.mu, bracket.upper.mu))
        if bracket.lower.mu < mu < bracket.upper.mu:
            sample = self.sample(mu)
            bracket.update(sample)
            if self.accepted(sample):
                return self._result(sample)

        lower_mu, upper_mu = bracket.pair
        mu0 = self.last.mu if self.last is not None else 0.5 * (lower_mu + upper_mu)
        try:
            final = (
                self._solve_with_newton(bracket, mu0=mu0)
                if self.use_derivative
                else self._solve_with_brent(bracket)
            )
        except _MaxRootIterations as exc:
            self._fail(
                "maximum charge-evaluation budget reached before satisfying the filling tolerance",
                self.best if self.best is not None else self.last,
            )
            raise AssertionError("unreachable") from exc

        if not self.accepted(final):
            self._fail(
                "root search ended before satisfying the filling tolerance",
                self.best if self.best is not None else final,
            )
        return self._result(final)

    def accepted(self, sample: _ChargeSample) -> bool:
        return (
            abs(sample.residual) <= self.filling_tol
            and sample.charge_error
            <= self.filling_tol * _CHARGE_ERROR_ACCEPTANCE_FRACTION
        )

    def sample(self, mu: float, *, enforce_limit: bool = True) -> _ChargeSample:
        mu_value = float(mu)
        if mu_value in self.cache:
            sample = self.cache[mu_value]
            self.last = sample
            return sample
        if enforce_limit and self.max_charge_evaluations is not None:
            if self.charge_evaluations >= self.max_charge_evaluations:
                raise _MaxRootIterations
        sample = _evaluate_charge_sample(
            self.evaluate_charge,
            filling=self.filling,
            mu=mu_value,
        )
        self.charge_evaluations += 1
        self.cache[mu_value] = sample
        self.last = sample
        if self.best is None or abs(sample.residual) < abs(self.best.residual):
            self.best = sample
        return sample

    def _expand_bracket(
        self,
        *,
        lower: float,
        upper: float,
        max_expansions: int = _MAX_BRACKET_EXPANSIONS,
    ) -> _ChargeBracket:
        lower_value = float(lower)
        upper_value = float(upper)
        if not lower_value < upper_value:
            raise ValueError(
                "Expected lower < upper for the chemical-potential bracket"
            )
        if max_expansions <= 0:
            raise ValueError("Bracket expansion limit must be positive")

        lower_sample = self.sample(lower_value)
        upper_sample = self.sample(upper_value)
        if lower_sample.residual <= 0.0 <= upper_sample.residual:
            return _ChargeBracket(lower_sample, upper_sample)

        step = upper_value - lower_value
        for _ in range(max_expansions):
            if lower_sample.residual > 0.0:
                lower_value -= step
                lower_sample = self.sample(lower_value)
            if upper_sample.residual < 0.0:
                upper_value += step
                upper_sample = self.sample(upper_value)
            if lower_sample.residual <= 0.0 <= upper_sample.residual:
                return _ChargeBracket(lower_sample, upper_sample)
            step *= 2.0

        raise ValueError(
            "Could not bracket the requested filling after "
            f"{max_expansions} expansions: "
            f"lower(mu={lower_value}, charge={lower_sample.charge}), "
            f"upper(mu={upper_value}, charge={upper_sample.charge}), "
            f"target={self.filling}"
        )

    def _solve_with_newton(
        self,
        bracket: _ChargeBracket,
        *,
        mu0: float,
    ) -> _ChargeSample:
        mu = float(mu0)
        while True:
            sample = self.sample(mu)
            bracket.update(sample)
            if self.accepted(sample):
                return sample

            derivative = sample.derivative
            if derivative is None or not np.isfinite(derivative) or derivative <= 0.0:
                return self._solve_with_brent(bracket)

            next_mu = sample.mu - sample.residual / derivative
            lower_mu, upper_mu = bracket.pair
            if not lower_mu < next_mu < upper_mu:
                return self._solve_with_brent(bracket)

            if abs(next_mu - sample.mu) <= self.mu_xtol:
                final = self.sample(next_mu, enforce_limit=False)
                bracket.update(final)
                if self.accepted(final):
                    return final
                return self._solve_with_brent(bracket)

            mu = next_mu

    def _solve_with_brent(self, bracket: _ChargeBracket) -> _ChargeSample:
        def residual(mu: float) -> float:
            sample = self.sample(mu)
            bracket.update(sample)
            if self.accepted(sample):
                raise _AcceptedSample(sample)
            return sample.residual

        try:
            root = brentq(
                residual,
                *bracket.pair,
                xtol=self.mu_xtol,
                rtol=np.finfo(float).eps * 4.0,
                maxiter=_BRENT_MAXITER,
                disp=False,
            )
        except _AcceptedSample as exc:
            return exc.sample
        except RuntimeError as exc:
            self._fail(
                "Brent search reached its internal iteration limit before satisfying the filling tolerance",
                self.best if self.best is not None else self.last,
            )
            raise AssertionError("unreachable") from exc

        sample = self.sample(float(root), enforce_limit=False)
        bracket.update(sample)
        return sample

    def _result(self, sample: _ChargeSample) -> FixedFillingSolve:
        return FixedFillingSolve(
            mu=sample.mu,
            charge=sample.charge,
            charge_error=sample.charge_error,
            residual=sample.residual,
            derivative=sample.derivative if self.use_derivative else None,
            charge_evaluations=self.charge_evaluations,
        )

    def _fail(self, reason: str, sample: _ChargeSample | None) -> None:
        if sample is None:
            raise RuntimeError(
                f"Chemical-potential solve failed: {reason}; no charge sample was evaluated"
            )
        raise RuntimeError(
            "Chemical-potential solve failed: "
            f"{reason}; mu={sample.mu}, residual={sample.residual}, "
            f"charge_error={sample.charge_error}, filling_tol={self.filling_tol}, "
            f"charge_evaluations={self.charge_evaluations}"
        )


def _validate_root_inputs(
    *,
    filling: float,
    mu_guess: float,
    lower: float,
    upper: float,
    filling_tol: float,
    mu_xtol: float,
    max_charge_evaluations: int | None,
) -> None:
    if not np.isfinite(filling):
        raise ValueError("Requested filling must be finite")
    if not np.isfinite(mu_guess):
        raise ValueError("Initial chemical-potential guess must be finite")
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Chemical-potential bracket endpoints must be finite")
    if not lower < upper:
        raise ValueError("Expected lower < upper for the chemical-potential bracket")
    if not np.isfinite(filling_tol) or filling_tol <= 0.0:
        raise ValueError("filling_tol must be a positive finite number")
    if not np.isfinite(mu_xtol) or mu_xtol <= 0.0:
        raise ValueError("mu_tol must be a positive finite number")
    if max_charge_evaluations is not None and max_charge_evaluations <= 0:
        raise ValueError("max_charge_evaluations must be positive when provided")


def solve_mu_in_bracket(
    evaluate_charge: ChargeEvaluation,
    *,
    filling: float,
    mu_guess: float,
    lower: float,
    upper: float,
    filling_tol: float,
    mu_xtol: float,
    max_charge_evaluations: int | None,
    use_derivative: bool = True,
) -> FixedFillingSolve:
    """Solve for the chemical potential inside an existing valid bracket.

    `evaluate_charge(mu)` must return `(charge, charge_error, derivative)`, where
    `charge` approximates the requested filling function `N(mu)`, `charge_error`
    is an absolute error estimate for that charge, and `derivative` is an optional
    `dN/dmu` estimate. The solver assumes `N(mu)` is monotone nondecreasing over
    `[lower, upper]`.
    """
    _validate_root_inputs(
        filling=filling,
        mu_guess=mu_guess,
        lower=lower,
        upper=upper,
        filling_tol=filling_tol,
        mu_xtol=mu_xtol,
        max_charge_evaluations=max_charge_evaluations,
    )
    return _ChargeRootSolver(
        evaluate_charge,
        filling=filling,
        filling_tol=filling_tol,
        mu_xtol=mu_xtol,
        max_charge_evaluations=max_charge_evaluations,
        use_derivative=use_derivative,
    ).solve_in_bracket(
        lower=lower,
        upper=upper,
        mu_guess=mu_guess,
    )


def solve_mu(
    *,
    evaluate_charge: Callable[[float], tuple[float, float, float | None]],
    initial_bracket: Callable[[], tuple[float, float]],
    filling: float,
    mu_guess: float,
    filling_tol: float,
    mu_tol: float,
    max_charge_evaluations: int | None,
    use_derivative: bool = True,
) -> FixedFillingSolve:
    """Solve for the chemical potential by first building and expanding a bracket.

    `evaluate_charge(mu)` must return `(charge, charge_error, derivative)`, where
    `charge` approximates the requested filling function `N(mu)`, `charge_error`
    is an absolute error estimate for that charge, and `derivative` is an optional
    `dN/dmu` estimate. The solver assumes `N(mu)` is monotone nondecreasing over
    the expanded bracket.
    """
    lower, upper = initial_bracket()
    return _ChargeRootSolver(
        evaluate_charge,
        filling=filling,
        filling_tol=filling_tol,
        mu_xtol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        use_derivative=use_derivative,
    ).solve_with_expansion(
        lower=lower,
        upper=upper,
        mu_guess=mu_guess,
    )
