"""Arithmetic operations on sequences."""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Optional, Tuple

from .base import Seq
from .domains import is_eventually_real
from .primitives import Const, InvN, NVar


def _exact_float(value: Fraction) -> Optional[Const]:
    """Fold only when a float preserves the exact rational value of the AST."""
    try:
        rounded = float(value)
    except OverflowError:
        return None
    if math.isfinite(rounded) and Fraction(rounded) == value:
        return Const(rounded)
    return None


def _finite_constants(a: Seq, b: Seq) -> bool:
    return (
        isinstance(a, Const) and isinstance(b, Const)
        and math.isfinite(a.c) and math.isfinite(b.c)
    )


@dataclass(frozen=True)
class Add(Seq):
    left: Seq
    right: Seq

    def __repr__(self) -> str:
        return f"({self.left}+{self.right})"

    def key(self) -> Tuple[object, ...]:
        return ("add", self.left.key(), self.right.key())

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        if isinstance(a, Const) and isinstance(b, Const) and _finite_constants(a, b):
            folded = _exact_float(Fraction(a.c) + Fraction(b.c))
            if folded is not None:
                return folded
        if isinstance(a, Const) and abs(a.c) == 0.0:
            return b
        if isinstance(b, Const) and abs(b.c) == 0.0:
            return a
        items = sorted([a, b], key=lambda x: x.key())
        return Add(items[0], items[1])

    def at(self, n: int) -> float:
        return self.left.at(n) + self.right.at(n)


@dataclass(frozen=True)
class Sub(Seq):
    left: Seq
    right: Seq

    def __repr__(self) -> str:
        return f"({self.left}-{self.right})"

    def key(self) -> Tuple[object, ...]:
        return ("sub", self.left.key(), self.right.key())

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        if isinstance(a, Const) and isinstance(b, Const) and _finite_constants(a, b):
            folded = _exact_float(Fraction(a.c) - Fraction(b.c))
            if folded is not None:
                return folded
        if isinstance(b, Const) and abs(b.c) == 0.0:
            return a
        if a.key() == b.key() and is_eventually_real(a):
            return Const(0.0)
        return Sub(a, b)

    def at(self, n: int) -> float:
        return self.left.at(n) - self.right.at(n)


@dataclass(frozen=True)
class Mul(Seq):
    left: Seq
    right: Seq

    def __repr__(self) -> str:
        return f"({self.left}*{self.right})"

    def key(self) -> Tuple[object, ...]:
        return ("mul", self.left.key(), self.right.key())

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        if isinstance(a, Const) and a.c == 0.0 and is_eventually_real(b):
            return Const(0.0)
        if isinstance(b, Const) and b.c == 0.0 and is_eventually_real(a):
            return Const(0.0)
        if isinstance(a, Const) and a.c == 1.0:
            return b
        if isinstance(b, Const) and b.c == 1.0:
            return a
        if isinstance(a, Const) and isinstance(b, Const) and _finite_constants(a, b):
            folded = _exact_float(Fraction(a.c) * Fraction(b.c))
            if folded is not None:
                return folded
        if (isinstance(a, NVar) and isinstance(b, InvN)) or (
            isinstance(a, InvN) and isinstance(b, NVar)
        ):
            return Const(1.0)
        if isinstance(a, Const) and isinstance(b, InvN):
            return Mul(a, b)
        if isinstance(b, Const) and isinstance(a, InvN):
            return Mul(b, a)
        items = sorted([a, b], key=lambda x: x.key())
        return Mul(items[0], items[1])

    def is_infinitesimal(self) -> bool:
        if isinstance(self.left, Const) and self.right.is_infinitesimal():
            return True
        if isinstance(self.right, Const) and self.left.is_infinitesimal():
            return True
        if self.left.is_infinitesimal() and self.right.is_infinitesimal():
            return True
        # New BV rule: bounded * infinitesimal => infinitesimal.
        if (
            self.left.is_infinitesimal()
            and self.right.abs_bound_eventually() is not None
        ):
            return True
        if (
            self.right.is_infinitesimal()
            and self.left.abs_bound_eventually() is not None
        ):
            return True
        return False

    def is_infinite(self) -> bool:
        if (
            isinstance(self.left, Const)
            and self.left.c != 0.0
            and self.right.is_infinite()
        ):
            return True
        if (
            isinstance(self.right, Const)
            and self.right.c != 0.0
            and self.left.is_infinite()
        ):
            return True
        if self.left.is_infinite() and self.right.is_infinite():
            return True
        return False

    def at(self, n: int) -> float:
        return self.left.at(n) * self.right.at(n)


@dataclass(frozen=True)
class Div(Seq):
    left: Seq
    right: Seq

    def __repr__(self) -> str:
        return f"({self.left}/{self.right})"

    def key(self) -> Tuple[object, ...]:
        return ("div", self.left.key(), self.right.key())

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        # Cancellation also requires finite real operands. Infinity is not a real constant.
        if (
            isinstance(b, Const) and isinstance(a, Const) and _finite_constants(a, b)
            and a.c == b.c and b.c != 0.0
        ):
            return Const(1.0)
        if isinstance(b, Const):
            if b.c == 0.0 or not math.isfinite(b.c):
                return Div(a, b)
            reciprocal = _exact_float(1 / Fraction(b.c))
            if reciprocal is not None:
                return Mul(a, reciprocal).simplify()
            return Div(a, b)
        if isinstance(b, InvN):
            return Mul(a, NVar()).simplify()
        if isinstance(b, NVar):
            return Mul(a, InvN()).simplify()
        return Div(a, b)

    def at(self, n: int) -> float:
        denom = self.right.at(n)
        if denom == 0.0:
            raise ZeroDivisionError("division by zero in sequence at n=%d" % n)
        return self.left.at(n) / denom
