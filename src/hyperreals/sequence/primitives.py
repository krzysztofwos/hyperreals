"""Primitive sequence types."""

from typing import Optional

from .base import Seq


class Const(Seq):
    """Constant sequence."""

    def __init__(self, c: float):
        self.c = float(c)

    def is_constant(self) -> bool:
        return True

    def const_value(self) -> Optional[float]:
        return self.c

    def __repr__(self) -> str:
        r = round(self.c)
        return str(int(r)) if abs(self.c - r) < 1e-12 else f"{self.c}"

    def simplify(self) -> "Seq":
        return self

    def is_infinitesimal(self) -> bool:
        return self.c == 0.0

    def at(self, n: int) -> float:
        return self.c


class NVar(Seq):
    """The sequence n."""

    def __repr__(self) -> str:
        return "n"

    def is_infinite(self) -> bool:
        return True

    def at(self, n: int) -> float:
        return float(n)


class InvN(Seq):
    """The sequence 1/n."""

    def __repr__(self) -> str:
        return "1/n"

    def is_infinitesimal(self) -> bool:
        return True

    def at(self, n: int) -> float:
        return 1.0 / float(n if n > 0 else 1)


class AltSign(Seq):
    """The alternating sequence (-1)^n."""

    def __repr__(self) -> str:
        return "(-1)^n"

    def at(self, n: int) -> float:
        return 1.0 if (n % 2 == 0) else -1.0
