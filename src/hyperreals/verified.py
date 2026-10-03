"""Python transport for the separately proved Lean periodic comparison core.

Expressions are sent to Lean without Python algebraic simplification. The
proved fragment has exact rational constants, (-1)^n, addition, subtraction,
and multiplication. It does not yet include n, 1/n, or standard-part analysis.
The Python transport, JSON parser, Lean compiler, and native runtime are part
of the execution trust boundary. See formalization.md.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

_Operation = Literal["lt", "eq"]
_Mask = tuple[bool, bool]


class LeanBackendError(RuntimeError):
    """The Lean executable is unavailable or returned an invalid response."""


@dataclass(frozen=True, eq=False)
class PeriodicHyperreal:
    """An exact periodic expression associated with one completion state."""

    _system: LeanPeriodicSystem
    _ast: tuple[object, ...]

    def _check(self, other: PeriodicHyperreal) -> None:
        if (
            not isinstance(other, PeriodicHyperreal)
            or self._system is not other._system
        ):
            raise ValueError("periodic operands must belong to the same Lean system")

    def __add__(self, other: PeriodicHyperreal) -> PeriodicHyperreal:
        self._check(other)
        return PeriodicHyperreal(self._system, ("add", self._ast, other._ast))

    def __sub__(self, other: PeriodicHyperreal) -> PeriodicHyperreal:
        self._check(other)
        return PeriodicHyperreal(self._system, ("sub", self._ast, other._ast))

    def __mul__(self, other: PeriodicHyperreal) -> PeriodicHyperreal:
        self._check(other)
        return PeriodicHyperreal(self._system, ("mul", self._ast, other._ast))

    def __lt__(self, other: PeriodicHyperreal) -> bool:
        return self._system.decide(self, other, "lt")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PeriodicHyperreal):
            return NotImplemented
        return self._system.decide(self, other, "eq")

    def __gt__(self, other: PeriodicHyperreal) -> bool:
        self._check(other)
        return self._system.decide(other, self, "lt")

    def __le__(self, other: PeriodicHyperreal) -> bool:
        return not self > other

    def __ge__(self, other: PeriodicHyperreal) -> bool:
        return not self < other


class LeanPeriodicSystem:
    """Lazy completion choices executed by the verified Lean periodic core.

    Build with ``lake build periodic_checker`` before constructing this class.
    An installed Python package can use an explicit ``checker_path``. There is
    no fallback to the Python comparison compiler if the executable is absent.
    """

    def __init__(
        self, *, checker_path: str | Path | None = None, timeout: float = 30.0
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._checker = (
            Path(checker_path).resolve()
            if checker_path is not None
            else Path(__file__).resolve().parents[2]
            / ".lake/build/bin/periodic_checker"
        )
        if not self._checker.is_file():
            raise LeanBackendError(
                "Lean periodic checker not found. Run 'lake build periodic_checker' "
                "or supply checker_path"
            )
        self._timeout = timeout
        self._support: _Mask = (True, True)

    @property
    def support(self) -> _Mask:
        """Remaining even and odd tails, in that order."""
        return self._support

    def constant(self, value: int | float | Fraction) -> PeriodicHyperreal:
        """Preserve integers/fractions exactly. Floats denote binary rationals."""
        rational = Fraction(value)
        return PeriodicHyperreal(
            self, ("const", str(rational.numerator), str(rational.denominator))
        )

    def alt(self) -> PeriodicHyperreal:
        return PeriodicHyperreal(self, ("alt",))

    @staticmethod
    def _mask(value: Any) -> _Mask:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(bit) is not bool for bit in value)
        ):
            raise LeanBackendError("invalid mask in Lean response")
        return value[0], value[1]

    def _request(
        self,
        left: PeriodicHyperreal,
        right: PeriodicHyperreal,
        op: _Operation,
        choice: bool | None = None,
    ) -> dict[str, Any]:
        left._check(right)
        if left._system is not self:
            raise ValueError("periodic operands must belong to this Lean system")
        if op not in ("lt", "eq"):
            raise ValueError("operation must be 'lt' or 'eq'")
        if choice is not None and type(choice) is not bool:
            raise ValueError("choice must be a boolean")
        request: dict[str, Any] = {
            "support": self._support,
            "op": op,
            "left": left._ast,
            "right": right._ast,
        }
        if choice is not None:
            request["choice"] = choice
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
            predicate = self._mask(result["predicate"])
            positive = self._mask(result["trueSupport"])
            negative = self._mask(result["falseSupport"])
            if positive != tuple(s and p for s, p in zip(self._support, predicate)):
                raise ValueError("positive support does not refine current state")
            if negative != tuple(s and not p for s, p in zip(self._support, predicate)):
                raise ValueError("negative support does not refine current state")
            if result["canBeTrue"] is not any(positive):
                raise ValueError("inconsistent positive feasibility")
            if result["canBeFalse"] is not any(negative):
                raise ValueError("inconsistent negative feasibility")
            if choice is not None:
                selected = positive if choice else negative
                if result["accepted"] is not any(selected):
                    raise ValueError("inconsistent acceptance")
                if result["accepted"] and self._mask(result["support"]) != selected:
                    raise ValueError("unexpected committed support")
                if not result["accepted"] and result["support"] is not None:
                    raise ValueError("rejected commit returned a state")
        except (KeyError, TypeError, ValueError) as error:
            raise LeanBackendError(f"invalid Lean response: {error}") from error
        return result

    def probe(
        self, left: PeriodicHyperreal, right: PeriodicHyperreal, op: _Operation = "lt"
    ) -> tuple[bool, bool]:
        """Return (can_be_false, can_be_true), without committing either choice."""
        result = self._request(left, right, op)
        return bool(result["canBeFalse"]), bool(result["canBeTrue"])

    def commit(
        self,
        left: PeriodicHyperreal,
        right: PeriodicHyperreal,
        op: _Operation = "lt",
        *,
        truth: bool,
    ) -> bool:
        """Apply a Lean-accepted transition. Rejection leaves support unchanged."""
        result = self._request(left, right, op, truth)
        if not result["accepted"]:
            return False
        self._support = self._mask(result["support"])
        return True

    def decide(
        self, left: PeriodicHyperreal, right: PeriodicHyperreal, op: _Operation = "lt"
    ) -> bool:
        """Choose true when feasible, otherwise choose the supported false branch."""
        if self.commit(left, right, op, truth=True):
            return True
        if self.commit(left, right, op, truth=False):
            return False
        raise LeanBackendError("neither polarity preserves nonempty support")
