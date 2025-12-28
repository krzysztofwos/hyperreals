"""Hyperreal number system implementation."""

from typing import Optional, Tuple

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
