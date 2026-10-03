"""Exact infinitesimal arithmetic executed by the proved Lean Laurent core.

The core normalizes finite Laurent expressions at both parities, checks eventual
comparisons with explicit cutoffs, and extracts rational standard parts. The
Python transport and JSON parser remain tested execution assumptions.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from .verified import LeanBackendError, LeanPeriodicSystem, PeriodicHyperreal


@dataclass(frozen=True, eq=False)
class LaurentHyperreal(PeriodicHyperreal):
    """A finite exact Laurent expression with two-periodic coefficients."""

    _system: LeanLaurentSystem

    def __add__(self, other: PeriodicHyperreal) -> LaurentHyperreal:
        self._check(other)
        return LaurentHyperreal(self._system, ("add", self._ast, other._ast))

    def __sub__(self, other: PeriodicHyperreal) -> LaurentHyperreal:
        self._check(other)
        return LaurentHyperreal(self._system, ("sub", self._ast, other._ast))

    def __mul__(self, other: PeriodicHyperreal) -> LaurentHyperreal:
        self._check(other)
        return LaurentHyperreal(self._system, ("mul", self._ast, other._ast))

    def __neg__(self) -> LaurentHyperreal:
        return self._system.constant(-1) * self

    def __pow__(self, power: int) -> LaurentHyperreal:
        if type(power) is not int or power < 0:
            raise ValueError("expression powers must be nonnegative integers")
        result = self._system.constant(1)
        factor = self
        while power:
            if power % 2:
                result = result * factor
            power //= 2
            if power:
                factor = factor * factor
        return result

    def divide_monomial(
        self, coefficient: int | float | Fraction, power: int
    ) -> LaurentHyperreal:
        """Divide by exactly ``coefficient * n**power``. Negative powers are allowed."""
        if type(power) is not int:
            raise ValueError("monomial power must be an integer")
        rational = Fraction(coefficient)
        if rational == 0:
            raise ZeroDivisionError("monomial divisor must be nonzero")
        return LaurentHyperreal(
            self._system,
            (
                "divMonomial",
                self._ast,
                str(rational.numerator),
                str(rational.denominator),
                str(power),
            ),
        )

    def __truediv__(self, other: PeriodicHyperreal) -> LaurentHyperreal:
        """Divide by a constant, n, or epsilon. Use divide_monomial for c*n**k."""
        self._check(other)
        if other._ast[0] == "const":
            return self.divide_monomial(
                Fraction(int(str(other._ast[1])), int(str(other._ast[2]))), 0
            )
        if other._ast == ("index",):
            return self.divide_monomial(1, 1)
        if other._ast == ("invn",):
            return self.divide_monomial(1, -1)
        raise ValueError(
            "division requires a primitive monomial. Use divide_monomial(coefficient, power)"
        )

    def standard_part(self) -> Fraction | None:
        """Return a proved-core rational limit shared by all remaining parities."""
        return self._system.standard_part(self)


class LeanLaurentSystem(LeanPeriodicSystem):
    """Lazy completion choices and standard parts for exact Laurent expressions.

    Build with ``lake build laurent_checker``. Comparisons commit choices.
    The ``probe`` and ``standard_part`` methods leave support unchanged. A standard part
    can become available after a choice removes disagreeing periodic branches.
    """

    def __init__(
        self, *, checker_path: str | Path | None = None, timeout: float = 30.0
    ) -> None:
        path = (
            checker_path
            if checker_path is not None
            else Path(__file__).resolve().parents[2] / ".lake/build/bin/laurent_checker"
        )
        if not Path(path).is_file():
            raise LeanBackendError(
                "Lean Laurent checker not found. Run 'lake build laurent_checker' "
                "or supply checker_path"
            )
        super().__init__(checker_path=path, timeout=timeout)
        self._last_cutoff: int | None = None

    @property
    def last_cutoff(self) -> int | None:
        """Inclusive natural-index cutoff from the latest comparison request."""
        return self._last_cutoff

    def constant(self, value: int | float | Fraction) -> LaurentHyperreal:
        rational = Fraction(value)
        return LaurentHyperreal(
            self, ("const", str(rational.numerator), str(rational.denominator))
        )

    def alt(self) -> LaurentHyperreal:
        return LaurentHyperreal(self, ("alt",))

    def infinite(self) -> LaurentHyperreal:
        """The identity sequence n."""
        return LaurentHyperreal(self, ("index",))

    def infinitesimal(self) -> LaurentHyperreal:
        """The distinguished positive infinitesimal 1/n."""
        return LaurentHyperreal(self, ("invn",))

    def _request(
        self,
        left: PeriodicHyperreal,
        right: PeriodicHyperreal,
        op: Literal["lt", "eq"],
        choice: bool | None = None,
    ) -> dict[str, Any]:
        result = super()._request(left, right, op, choice)
        cutoff = result.get("cutoff")
        if (
            not isinstance(cutoff, str)
            or not cutoff.isascii()
            or not cutoff.isdecimal()
        ):
            raise LeanBackendError("invalid cutoff in Lean response")
        self._last_cutoff = int(cutoff)
        if self._last_cutoff < 1:
            raise LeanBackendError("Lean cutoff must be positive")
        return result

    def standard_part(self, expression: LaurentHyperreal) -> Fraction | None:
        if (
            not isinstance(expression, LaurentHyperreal)
            or expression._system is not self
        ):
            raise ValueError("Laurent expression must belong to this Lean system")
        request = {
            "support": self._support,
            "op": "standardPart",
            "left": expression._ast,
        }
        try:
            process = subprocess.run(
                [str(self._checker)],
                input=json.dumps(request) + "\n",
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LeanBackendError(f"Lean checker failed: {error}") from error
        if process.returncode != 0:
            raise LeanBackendError(f"Lean checker failed: {process.stderr.strip()}")
        try:
            result = json.loads(process.stdout)
            if not isinstance(result, dict):
                raise ValueError("response must be an object")
            if "error" in result:
                raise ValueError(result["error"])
            if self._mask(result["support"]) != self._support:
                raise ValueError("standard-part extraction changed the support")
            value = result["value"]
            if value is None:
                return None
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(type(v) is not str for v in value)
            ):
                raise ValueError("invalid rational result")
            numerator, denominator = int(value[0]), int(value[1])
            if denominator <= 0:
                raise ValueError("rational denominator must be positive")
            return Fraction(numerator, denominator)
        except (KeyError, TypeError, ValueError) as error:
            raise LeanBackendError(f"invalid Lean response: {error}") from error
