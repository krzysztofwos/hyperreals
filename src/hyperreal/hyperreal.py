"""Hyperreal number system implementation."""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from .algebra import SetExpr
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
from .ultrafilter import PartialUltrafilter


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

    def __add__(self, other: "Hyperreal") -> "Hyperreal":
        return Hyperreal(Add(self.seq, other.seq).simplify(), self.puf)

    def __sub__(self, other: "Hyperreal") -> "Hyperreal":
        return Hyperreal(Sub(self.seq, other.seq).simplify(), self.puf)

    def __mul__(self, other: "Hyperreal") -> "Hyperreal":
        return Hyperreal(Mul(self.seq, other.seq).simplify(), self.puf)

    def __truediv__(self, other: "Hyperreal") -> "Hyperreal":
        return Hyperreal(Div(self.seq, other.seq).simplify(), self.puf)

    def _cmp_sets(self, other: "Hyperreal") -> Tuple[SetExpr, SetExpr, SetExpr]:
        return self.puf._install_partition(self.seq, other.seq)

    def __lt__(self, other: "Hyperreal") -> bool:
        L, E, G = self._cmp_sets(other)
        return self.puf.contains(L)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hyperreal):
            return NotImplemented
        L, E, G = self._cmp_sets(other)
        return self.puf.contains(E)

    def __le__(self, other: "Hyperreal") -> bool:
        return self < other or self == other

    def __gt__(self, other: "Hyperreal") -> bool:
        L, E, G = self._cmp_sets(other)
        return self.puf.contains(G)

    def __ge__(self, other: "Hyperreal") -> bool:
        return self > other or self == other

    def standard_part(self) -> Optional[float]:
        """Extract the standard part if this is a near-standard number.

        Uses the conservative extended extractor (series + bounded/limit rules).
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
            bracket: Optional initial (low, high) bracket. If not given, one
                is found automatically.
            tie_break: When both directions are feasible, commit "lower" (push
                high down) or "upper" (push low up).
            max_bracket_steps: Maximum steps to find an initial bracket.

        Returns:
            StChooseResult with the approximation, or None if bracketing fails
            (likely the hyperreal is not finite).
        """
        # Prefer invariant result if available
        inv = self.standard_part()
        if inv is not None:
            return StChooseResult(
                low=inv, high=inv, approx=inv, bits=0, forced=bits, chosen=0
            )

        # Find initial bracket if not provided
        if bracket is None:
            bracket = self._find_bracket(max_bracket_steps)
            if bracket is None:
                return None

        low, high = bracket
        forced = 0
        chosen = 0

        for _ in range(bits):
            mid = (low + high) / 2.0

            # Build the set {n: x < mid}
            L, _, _ = self.puf._install_partition(self.seq, Const(mid))

            # Probe without committing
            can_be_false, can_be_true = self.puf.probe(L)

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
                chosen += 1
            else:
                # Neither feasible: this shouldn't happen with a valid bracket
                break

        return StChooseResult(
            low=low,
            high=high,
            approx=(low + high) / 2.0,
            bits=bits,
            forced=forced,
            chosen=chosen,
        )

    def _find_bracket(self, max_steps: int) -> Optional[Tuple[float, float]]:
        """Find an initial bracket [low, high] for the hyperreal.

        Returns None if the hyperreal appears to be infinite (no bracket found).
        """
        low, high = -1.0, 1.0

        for _ in range(max_steps):
            # Check if we can commit x < high
            L_high, _, _ = self.puf._install_partition(self.seq, Const(high))
            can_be_false_high, can_be_true_high = self.puf.probe(L_high)

            if can_be_true_high:
                # x < high is feasible, commit it
                self.puf.commit(L_high, True)
                break
            else:
                # x >= high always, expand upper bound
                high *= 2.0
        else:
            return None  # Failed to find upper bound

        for _ in range(max_steps):
            # Check if we can commit x >= low (i.e., x < low is false)
            L_low, _, _ = self.puf._install_partition(self.seq, Const(low))
            can_be_false_low, can_be_true_low = self.puf.probe(L_low)

            if can_be_false_low:
                # x >= low is feasible, commit it
                self.puf.commit(L_low, False)
                break
            else:
                # x < low always, expand lower bound
                low *= 2.0
        else:
            return None  # Failed to find lower bound

        return (low, high)

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
    """Factory and operations for hyperreal numbers."""

    def __init__(self):
        self.puf = PartialUltrafilter()

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

    def sin(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Sin(x.seq).simplify(), self.puf)

    def cos(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Cos(x.seq).simplify(), self.puf)

    def tan(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Tan(x.seq).simplify(), self.puf)

    def tanh(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Tanh(x.seq).simplify(), self.puf)

    def exp(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Exp(x.seq).simplify(), self.puf)

    def log1p(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Log1p(x.seq).simplify(), self.puf)

    def sqrt1p(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Sqrt1p(x.seq).simplify(), self.puf)

    def cosh(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Cosh(x.seq).simplify(), self.puf)

    def sinh(self, x: Hyperreal) -> Hyperreal:
        return Hyperreal(Sinh(x.seq).simplify(), self.puf)
