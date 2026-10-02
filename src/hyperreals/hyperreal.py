"""Hyperreal number system implementation."""

from dataclasses import dataclass
import math
from typing import Dict, Literal, Optional, Tuple

from .algebra import SetExpr, complement, intersect
from .asymptotic import standard_part_extended
from .sequence import (
    Add,
    AltSign,
    Const,
    Cos,
    Cosh,
    Div,
    Exp,
    InvN,
    Log1p,
    Mul,
    NVar,
    Seq,
    Sin,
    Sinh,
    Sqrt1p,
    Sub,
    Tan,
    Tanh,
)
from .series import series_from_seq
from .ultrafilter import PartialUltrafilter, UnderdeterminedComparisonError


def _midpoint(low: float, high: float) -> float:
    """Avoid overflow at opposite extremes and underflow at equal tiny bounds."""
    if low <= 0.0 <= high:
        return low / 2.0 + high / 2.0
    return low + (high - low) / 2.0


@dataclass(frozen=True)
class StChooseResult:
    """Result of completion-sensitive standard part computation.

    Attributes:
        low: Lower bound of the approximation interval.
        high: Upper bound of the approximation interval.
        approx: Midpoint approximation of the standard part.
        bits: Number of bisection iterations performed.
        forced: Number of bits determined by forced constraints.
        chosen: Number of bits determined by ultrafilter choice.
    """

    low: float
    high: float
    approx: float
    bits: int
    forced: int
    chosen: int


class Hyperreal:
    """A hyperreal number represented as an equivalence class of sequences."""

    def __init__(self, seq: Seq, puf: PartialUltrafilter):
        self.seq = seq.simplify()
        self.puf = puf

    def _require_same_system(self, other: "Hyperreal") -> None:
        if self.puf is not other.puf:
            raise ValueError("hyperreal operands must belong to the same system")

    def __add__(self, other: "Hyperreal") -> "Hyperreal":
        self._require_same_system(other)
        return Hyperreal(Add(self.seq, other.seq).simplify(), self.puf)

    def __sub__(self, other: "Hyperreal") -> "Hyperreal":
        self._require_same_system(other)
        return Hyperreal(Sub(self.seq, other.seq).simplify(), self.puf)

    def __mul__(self, other: "Hyperreal") -> "Hyperreal":
        self._require_same_system(other)
        return Hyperreal(Mul(self.seq, other.seq).simplify(), self.puf)

    def __truediv__(self, other: "Hyperreal") -> "Hyperreal":
        self._require_same_system(other)
        return Hyperreal(Div(self.seq, other.seq).simplify(), self.puf)

    def _cmp_sets(self, other: "Hyperreal") -> Tuple[SetExpr, SetExpr, SetExpr]:
        self._require_same_system(other)
        return self.puf._install_partition(self.seq, other.seq)

    def compare_lt(self, other: "Hyperreal") -> Optional[bool]:
        """Return a semantically checked result, or ``None`` if unsupported."""
        L, _, _ = self._cmp_sets(other)
        return self.puf.decide(L)

    def compare_eq(self, other: "Hyperreal") -> Optional[bool]:
        """Return a semantically checked result, or ``None`` if unsupported."""
        _, E, _ = self._cmp_sets(other)
        return self.puf.decide(E)

    def compare_gt(self, other: "Hyperreal") -> Optional[bool]:
        """Return a semantically checked result, or ``None`` if unsupported."""
        _, _, G = self._cmp_sets(other)
        return self.puf.decide(G)

    def _require_comparison(self, other: "Hyperreal", operator: str) -> bool:
        if operator == "<":
            decision = self.compare_lt(other)
        elif operator == "=":
            decision = self.compare_eq(other)
        elif operator == ">":
            decision = self.compare_gt(other)
        else:
            raise ValueError(f"unsupported comparison operator: {operator}")
        if decision is not None:
            return decision
        if self.puf.allow_uncertified_choices:
            L, E, G = self._cmp_sets(other)
            selected = {"<": L, "=": E, ">": G}[operator]
            return self.puf.contains(selected)
        raise UnderdeterminedComparisonError(
            f"comparison {self.seq!r} {operator} {other.seq!r} is outside "
            "the supported semantic fragment"
        )

    def __lt__(self, other: "Hyperreal") -> bool:
        return self._require_comparison(other, "<")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hyperreal):
            return NotImplemented
        return self._require_comparison(other, "=")

    def __le__(self, other: "Hyperreal") -> bool:
        return not self._require_comparison(other, ">")

    def __gt__(self, other: "Hyperreal") -> bool:
        return self._require_comparison(other, ">")

    def __ge__(self, other: "Hyperreal") -> bool:
        return not self._require_comparison(other, "<")

    def standard_part(self) -> Optional[float]:
        """Return a standard-part candidate when the trusted analyzer recognizes one.

        Unlike supported comparison decisions, this value currently has no Lean
        certificate.  ``None`` means that the analyzer did not recognize the case.
        """
        return standard_part_extended(self.seq)

    def choose_standard_part(
        self,
        *,
        bits: int = 32,
        bracket: Optional[Tuple[float, float]] = None,
        tie_break: Literal["lower", "upper"] = "lower",
        max_bracket_steps: int = 64,
    ) -> Optional[StChooseResult]:
        """Completion-dependent standard part approximation.

        This method computes a standard part by selecting a completion via SAT
        commitments. Unlike standard_part(), this may return different values
        depending on which completion is chosen.

        IMPORTANT: This is explicitly completion-dependent. The result depends
        on which ultrafilter completion is selected. Use standard_part() for
        completion-invariant extraction.

        Args:
            bits: Number of bisection iterations for precision.
            bracket: Optional finite, ordered, closed (low, high) bracket.
                Both bounds are committed jointly before bisection. If not
                given, a bracket is found automatically.
            tie_break: When both directions are feasible, commit "lower" (push
                high down) or "upper" (push low up).
            max_bracket_steps: Maximum steps to find an initial bracket.

        Returns:
            StChooseResult with the approximation, or None if bracketing fails
            (likely the hyperreal is not finite).
        """
        if bits < 0:
            raise ValueError("bits must be nonnegative")
        if max_bracket_steps < 1:
            raise ValueError("max_bracket_steps must be positive")
        if tie_break not in ("lower", "upper"):
            raise ValueError("tie_break must be 'lower' or 'upper'")
        if bracket is not None:
            low, high = bracket
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError("bracket must contain finite, ordered endpoints")

        # Prefer invariant result if available. Honor supplied bounds too.
        inv = self.standard_part()
        if inv is not None:
            if bracket is not None and not bracket[0] <= inv <= bracket[1]:
                return None
            return StChooseResult(
                low=inv, high=inv, approx=inv, bits=0, forced=bits, chosen=0
            )
        choices_before = self.puf.stats.choice_commits

        # Find initial bracket if not provided
        if bracket is None:
            bracket = self._find_bracket(max_bracket_steps)
            if bracket is None:
                return None
        elif not self._commit_bracket(*bracket):
            return None

        low, high = bracket
        forced = 0
        performed = 0

        for _ in range(bits):
            mid = _midpoint(low, high)
            if mid == low or mid == high:
                break

            # Build the set {n: x < mid}
            L, _, _ = self.puf._install_partition(self.seq, Const(mid))

            # Probe without committing
            feasibility = self.puf.probe(L)
            if feasibility is None:
                return None
            can_be_false, can_be_true = feasibility

            if can_be_true and not can_be_false:
                # Forced true: x < mid eventually, so upper bound is mid
                self.puf.commit(L, True, choice=False)
                high = mid
                forced += 1
            elif can_be_false and not can_be_true:
                # Forced false: x >= mid eventually, so lower bound is mid
                self.puf.commit(L, False, choice=False)
                low = mid
                forced += 1
            elif can_be_true and can_be_false:
                # Both feasible: choose based on tie_break
                if tie_break == "lower":
                    # Commit true (x < mid), push high down
                    self.puf.commit(L, True, choice=True)
                    high = mid
                else:
                    # Commit false (x >= mid), push low up
                    self.puf.commit(L, False, choice=True)
                    low = mid
            else:
                return None
            performed += 1

        return StChooseResult(
            low=low,
            high=high,
            approx=_midpoint(low, high),
            bits=performed,
            forced=forced,
            chosen=self.puf.stats.choice_commits - choices_before,
        )

    def _bracket_set(self, low: float, high: float) -> SetExpr:
        below_low, _, _ = self.puf._install_partition(self.seq, Const(low))
        _, _, above_high = self.puf._install_partition(self.seq, Const(high))
        return intersect(complement(below_low), complement(above_high))

    def _commit_bracket(self, low: float, high: float) -> bool:
        """Commit both bounds together, so a rejected bracket changes no choices."""
        bounds = self._bracket_set(low, high)
        options = self.puf.probe(bounds)
        return options is not None and options[1] and self.puf.commit(bounds, True)

    def _find_bracket(self, max_steps: int) -> Optional[Tuple[float, float]]:
        """Find an initial bracket [low, high] for the hyperreal.

        Returns None if the hyperreal appears to be infinite (no bracket found).
        """
        low, high = -1.0, 1.0

        for _ in range(max_steps):
            bounds = self._bracket_set(low, high)
            options = self.puf.probe(bounds)
            if options is None:
                return None
            if options[1]:
                return (low, high) if self.puf.commit(bounds, True) else None
            low *= 2.0
            high *= 2.0
            if not (math.isfinite(low) and math.isfinite(high)):
                return None
        return None

    def series(self, order: int = 10) -> Optional[Dict[int, float]]:
        """Return a truncated δ=1/n series representation, if available."""
        return series_from_seq(self.seq, order=order)

    def coeff(self, k: int, *, order: int = 10) -> Optional[float]:
        """Return the coefficient of δ^k in the truncated series, if available."""
        A = series_from_seq(self.seq, order=order)
        if A is None:
            return None
        return A.get(k, 0.0)

    def value_at(self, n: int) -> float:
        """Evaluate the underlying sequence at index n."""
        return self.seq.at(n)

    def __repr__(self) -> str:
        st = self.standard_part()
        if st is not None:
            A = series_from_seq(self.seq)
            tail = None if A is None else any(k > 0 for k in A.keys())
            if tail:
                return f"{st}+ε"
            r = round(st)
            return str(int(r)) if abs(st - r) < 1e-12 else f"{st}"
        if self.seq.is_infinitesimal():
            return "ε"
        if self.seq.is_infinite():
            return "ω"
        return f"HR({self.seq})"


class HyperrealSystem:
    """Factory for semantically checked hyperreal operations.

    Unknown comparisons raise by default. ``allow_uncertified_choices`` keeps
    the legacy SAT-only behavior available for explicit experiments.
    """

    def __init__(self, *, allow_uncertified_choices: bool = False):
        self.puf = PartialUltrafilter(
            allow_uncertified_choices=allow_uncertified_choices
        )

    def constant(self, r: float) -> Hyperreal:
        """Create a hyperreal from a standard real number."""
        return Hyperreal(Const(r), self.puf)

    def infinitesimal(self) -> Hyperreal:
        """Create the infinitesimal ε = 1/n."""
        return Hyperreal(InvN(), self.puf)

    def infinite(self) -> Hyperreal:
        """Create the infinite hyperreal ω = n."""
        return Hyperreal(NVar(), self.puf)

    def alt(self) -> Hyperreal:
        """Create the alternating sequence (-1)^n."""
        return Hyperreal(AltSign(), self.puf)

    def _check_context(self, x: Hyperreal) -> None:
        if x.puf is not self.puf:
            raise ValueError("hyperreal argument must belong to this system")

    def sin(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Sin(x.seq).simplify(), self.puf)

    def cos(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Cos(x.seq).simplify(), self.puf)

    def tan(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Tan(x.seq).simplify(), self.puf)

    def tanh(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Tanh(x.seq).simplify(), self.puf)

    def exp(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Exp(x.seq).simplify(), self.puf)

    def log1p(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Log1p(x.seq).simplify(), self.puf)

    def sqrt1p(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Sqrt1p(x.seq).simplify(), self.puf)

    def cosh(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Cosh(x.seq).simplify(), self.puf)

    def sinh(self, x: Hyperreal) -> Hyperreal:
        self._check_context(x)
        return Hyperreal(Sinh(x.seq).simplify(), self.puf)
