"""Exact Laurent expressions with arbitrary finite periodic coefficients.

The Lean core proves LCM support refinement, comparison cutoffs, finite-trace
completion and successful standard-part extraction. Python serialization and
native execution remain tested trust assumptions.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from .replay import ReplayObservation, ReplaySnapshot
from .verified import LeanBackendError

_Operation = Literal["lt", "eq"]
_Mask = tuple[bool, ...]


@dataclass(frozen=True)
class StandardPartDiagnostic:
    """A finite value, a divergent active residue, or two unequal residue limits.

    Residues use ``period``. Finite results contain one limit and no residues.
    Divergence contains one residue and no limits. Disagreement contains two
    residues and their corresponding limits. Snapshot replay independently
    certifies whether a common value exists. Native execution remains trusted
    for the particular diagnostic witnesses reported by this interface.
    """

    kind: Literal["finite", "divergent", "disagreement"]
    period: int
    residues: tuple[int, ...]
    limits: tuple[Fraction, ...]


@dataclass(frozen=True, eq=False)
class ResidueHyperreal:
    """An exact expression belonging to one arbitrary-period completion state."""

    _system: LeanResidueSystem
    _ast: tuple[object, ...]

    def _check(self, other: ResidueHyperreal) -> None:
        if not isinstance(other, ResidueHyperreal) or self._system is not other._system:
            raise ValueError("residue operands must belong to the same Lean system")

    def __add__(self, other: ResidueHyperreal) -> ResidueHyperreal:
        self._check(other)
        return ResidueHyperreal(self._system, ("add", self._ast, other._ast))

    def __sub__(self, other: ResidueHyperreal) -> ResidueHyperreal:
        self._check(other)
        return ResidueHyperreal(self._system, ("sub", self._ast, other._ast))

    def __mul__(self, other: ResidueHyperreal) -> ResidueHyperreal:
        self._check(other)
        return ResidueHyperreal(self._system, ("mul", self._ast, other._ast))

    def __neg__(self) -> ResidueHyperreal:
        return self._system.constant(-1) * self

    def __pow__(self, power: int) -> ResidueHyperreal:
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
    ) -> ResidueHyperreal:
        """Divide by exactly ``coefficient * n**power``. Negative powers are allowed."""
        if type(power) is not int:
            raise ValueError("monomial power must be an integer")
        rational = Fraction(coefficient)
        if rational == 0:
            raise ZeroDivisionError("monomial divisor must be nonzero")
        return ResidueHyperreal(
            self._system,
            (
                "divMonomial",
                self._ast,
                str(rational.numerator),
                str(rational.denominator),
                str(power),
            ),
        )

    def __truediv__(self, other: ResidueHyperreal) -> ResidueHyperreal:
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
        """Return a proved-core rational limit shared by all remaining residues."""
        return self._system.standard_part(self)

    def explain_standard_part(self) -> StandardPartDiagnostic:
        """Explain extraction without making an observation or changing support."""
        return self._system.explain_standard_part(self)

    def __lt__(self, other: ResidueHyperreal) -> bool:
        return self._system.decide(self, other, "lt")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidueHyperreal):
            return NotImplemented
        return self._system.decide(self, other, "eq")

    def __gt__(self, other: ResidueHyperreal) -> bool:
        self._check(other)
        return self._system.decide(other, self, "lt")

    def __le__(self, other: ResidueHyperreal) -> bool:
        return not self > other

    def __ge__(self, other: ResidueHyperreal) -> bool:
        return not self < other


class LeanResidueSystem:
    """Verified Laurent choices across arbitrary positive finite periods.

    Build ``lake build residue_checker``. Support starts at period one and is
    refined to an LCM only by an accepted commitment. Probes and extraction do
    not change it, even when they inspect expressions with new periods.
    """

    def __init__(
        self, *, checker_path: str | Path | None = None, timeout: float = 30.0
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._checker = (
            Path(checker_path).resolve()
            if checker_path is not None
            else Path(__file__).resolve().parents[2] / ".lake/build/bin/residue_checker"
        )
        if not self._checker.is_file():
            raise LeanBackendError(
                "Lean residue checker not found. Run 'lake build residue_checker' "
                "or supply checker_path"
            )
        self._timeout = timeout
        self._support: _Mask = (True,)
        self._last_cutoff: int | None = None
        self._history: tuple[ReplayObservation, ...] = ()
        self._lock = RLock()

    @property
    def history(self) -> tuple[ReplayObservation, ...]:
        """Immutable accepted observations, in their original order."""
        with self._lock:
            return self._history

    @property
    def support(self) -> _Mask:
        """Active residues modulo ``period``, ordered from residue zero."""
        return self._support

    @property
    def period(self) -> int:
        return len(self._support)

    @property
    def last_cutoff(self) -> int | None:
        """Inclusive natural cutoff for the latest comparison mask."""
        return self._last_cutoff

    def constant(self, value: int | float | Fraction) -> ResidueHyperreal:
        rational = Fraction(value)
        return ResidueHyperreal(
            self, ("const", str(rational.numerator), str(rational.denominator))
        )

    def periodic(self, values: Iterable[int | float | Fraction]) -> ResidueHyperreal:
        """An exact table indexed by n modulo its nonzero length."""
        table = tuple(Fraction(value) for value in values)
        if not table:
            raise ValueError("periodic table must be nonempty")
        return ResidueHyperreal(
            self,
            (
                "periodic",
                tuple(
                    (str(value.numerator), str(value.denominator)) for value in table
                ),
            ),
        )

    def alt(self) -> ResidueHyperreal:
        return self.periodic([1, -1])

    def infinite(self) -> ResidueHyperreal:
        return ResidueHyperreal(self, ("index",))

    def infinitesimal(self) -> ResidueHyperreal:
        return ResidueHyperreal(self, ("invn",))

    @staticmethod
    def _mask(value: Any) -> _Mask:
        if (
            not isinstance(value, list)
            or not value
            or any(type(bit) is not bool for bit in value)
        ):
            raise LeanBackendError("invalid positive-period mask in Lean response")
        return tuple(value)

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
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
        except (TypeError, ValueError) as error:
            raise LeanBackendError(f"invalid Lean response: {error}") from error
        return result

    def _request(
        self,
        left: ResidueHyperreal,
        right: ResidueHyperreal,
        op: _Operation,
        choice: bool | None = None,
    ) -> dict[str, Any]:
        left._check(right)
        if left._system is not self:
            raise ValueError("residue operands must belong to this Lean system")
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
        result = self._exchange(request)
        try:
            predicate = self._mask(result["predicate"])
            positive, negative = self._mask(result["trueSupport"]), self._mask(
                result["falseSupport"]
            )
            period = math.lcm(self.period, len(predicate))
            expected_positive = tuple(
                self._support[r % self.period] and predicate[r % len(predicate)]
                for r in range(period)
            )
            expected_negative = tuple(
                self._support[r % self.period] and not predicate[r % len(predicate)]
                for r in range(period)
            )
            if positive != expected_positive or negative != expected_negative:
                raise ValueError(
                    "response does not preserve support under LCM refinement"
                )
            if result["canBeTrue"] is not any(positive) or result[
                "canBeFalse"
            ] is not any(negative):
                raise ValueError("inconsistent feasibility")
            if choice is None:
                if result["accepted"] is not None or result["support"] is not None:
                    raise ValueError("probe returned a commitment")
            else:
                selected = positive if choice else negative
                if result["accepted"] is not any(selected):
                    raise ValueError("inconsistent acceptance")
                if result["accepted"] and self._mask(result["support"]) != selected:
                    raise ValueError("unexpected committed support")
                if not result["accepted"] and result["support"] is not None:
                    raise ValueError("rejected commit returned a state")
            cutoff = result["cutoff"]
            if (
                not isinstance(cutoff, str)
                or not cutoff.isascii()
                or not cutoff.isdecimal()
            ):
                raise ValueError("invalid cutoff")
            parsed_cutoff = int(cutoff)
            if parsed_cutoff < 1:
                raise ValueError("cutoff must be positive")
        except (KeyError, TypeError, ValueError) as error:
            raise LeanBackendError(f"invalid Lean response: {error}") from error
        self._last_cutoff = parsed_cutoff
        return result

    def probe(
        self, left: ResidueHyperreal, right: ResidueHyperreal, op: _Operation = "lt"
    ) -> tuple[bool, bool]:
        with self._lock:
            result = self._request(left, right, op)
            return bool(result["canBeFalse"]), bool(result["canBeTrue"])

    def commit(
        self,
        left: ResidueHyperreal,
        right: ResidueHyperreal,
        op: _Operation = "lt",
        *,
        truth: bool,
    ) -> bool:
        with self._lock:
            left._check(right)
            observation = ReplayObservation(left._ast, right._ast, op, truth)
            result = self._request(left, right, op, truth)
            if not result["accepted"]:
                return False
            self._support = self._mask(result["support"])
            self._history = (*self._history, observation)
            return True

    def decide(
        self, left: ResidueHyperreal, right: ResidueHyperreal, op: _Operation = "lt"
    ) -> bool:
        with self._lock:
            if self.commit(left, right, op, truth=True):
                return True
            if self.commit(left, right, op, truth=False):
                return False
            raise LeanBackendError("neither polarity preserves nonempty support")

    def standard_part(self, expression: ResidueHyperreal) -> Fraction | None:
        with self._lock:
            return self._standard_part(expression)

    def explain_standard_part(
        self, expression: ResidueHyperreal
    ) -> StandardPartDiagnostic:
        """Classify the exact common limit or give active residue witnesses."""
        with self._lock:
            if (
                not isinstance(expression, ResidueHyperreal)
                or expression._system is not self
            ):
                raise ValueError("residue expression must belong to this Lean system")
            result = self._exchange(
                {
                    "support": self._support,
                    "op": "diagnoseStandardPart",
                    "left": expression._ast,
                }
            )
            try:
                if self._mask(result["support"]) != self._support:
                    raise ValueError("standard-part diagnosis changed support")

                def natural(value: object) -> int:
                    if (
                        not isinstance(value, str)
                        or not value.isascii()
                        or not value.isdecimal()
                    ):
                        raise ValueError("invalid natural number")
                    return int(value)

                period = natural(result["period"])
                if period <= 0 or period % self.period:
                    raise ValueError("invalid diagnostic period")
                diagnostic = result["diagnostic"]
                kind = diagnostic["kind"]
                if not isinstance(diagnostic["residues"], list) or not isinstance(
                    diagnostic["limits"], list
                ):
                    raise ValueError("invalid diagnostic witnesses")
                residues = tuple(natural(r) for r in diagnostic["residues"])
                limits = tuple(self._rational(v) for v in diagnostic["limits"])
                expected_sizes = {
                    "finite": (0, 1),
                    "divergent": (1, 0),
                    "disagreement": (2, 2),
                }
                if (
                    kind not in expected_sizes
                    or (len(residues), len(limits)) != expected_sizes[kind]
                ):
                    raise ValueError("invalid diagnostic kind or witness count")
                if any(
                    r >= period or not self._support[r % self.period] for r in residues
                ):
                    raise ValueError("diagnostic witness is not an active residue")
                if kind == "disagreement" and (
                    residues[0] == residues[1] or limits[0] == limits[1]
                ):
                    raise ValueError("diagnostic witnesses do not disagree")
                return StandardPartDiagnostic(kind, period, residues, limits)
            except (KeyError, TypeError, ValueError) as error:
                raise LeanBackendError(f"invalid Lean response: {error}") from error

    @staticmethod
    def _rational(value: object) -> Fraction:
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

    def snapshot(self, expression: ResidueHyperreal) -> ReplaySnapshot:
        """Capture a query result and the exact accepted history at that instant.

        Export and kernel verification are explicit operations on the immutable
        snapshot. Capturing a snapshot does not certify the native answer.
        """
        with self._lock:
            result = self._standard_part(expression)
            return ReplaySnapshot(self._history, self._support, expression._ast, result)

    def _standard_part(self, expression: ResidueHyperreal) -> Fraction | None:
        if (
            not isinstance(expression, ResidueHyperreal)
            or expression._system is not self
        ):
            raise ValueError("residue expression must belong to this Lean system")
        result = self._exchange(
            {
                "support": self._support,
                "op": "standardPart",
                "left": expression._ast,
            }
        )
        try:
            if self._mask(result["support"]) != self._support:
                raise ValueError("standard-part extraction changed support")
            value = result["value"]
            if value is None:
                return None
            return self._rational(value)
        except (KeyError, TypeError, ValueError) as error:
            raise LeanBackendError(f"invalid Lean response: {error}") from error
