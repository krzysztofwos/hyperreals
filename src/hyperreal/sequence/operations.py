"""Arithmetic operations on sequences."""

from dataclasses import dataclass

from .base import Seq
from .primitives import Const, InvN, NVar


@dataclass(frozen=True)
class Add(Seq):
    left: Seq
    right: Seq

    def __repr__(self) -> str:
        return f"({self.left}+{self.right})"

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.c + b.c)
        if isinstance(a, Const) and abs(a.c) == 0.0:
            return b
        if isinstance(b, Const) and abs(b.c) == 0.0:
            return a
        items = sorted([a, b], key=lambda x: repr(x))
        return Add(items[0], items[1])

    def at(self, n: int) -> float:
        return self.left.at(n) + self.right.at(n)


@dataclass(frozen=True)
class Sub(Seq):
    left: Seq
    right: Seq

    def __repr__(self) -> str:
        return f"({self.left}-{self.right})"

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.c - b.c)
        if isinstance(b, Const) and abs(b.c) == 0.0:
            return a
        if repr(a) == repr(b):
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

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(0.0)
        if isinstance(b, Const) and b.c == 0.0:
            return Const(0.0)
        if isinstance(a, Const) and a.c == 1.0:
            return b
        if isinstance(b, Const) and b.c == 1.0:
            return a
        if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.c * b.c)
        if (isinstance(a, NVar) and isinstance(b, InvN)) or (
            isinstance(a, InvN) and isinstance(b, NVar)
        ):
            return Const(1.0)
        if isinstance(a, Const) and isinstance(b, InvN):
            return Mul(a, b)
        if isinstance(b, Const) and isinstance(a, InvN):
            return Mul(b, a)
        items = sorted([a, b], key=lambda x: repr(x))
        return Mul(items[0], items[1])

    def is_infinitesimal(self) -> bool:
        if isinstance(self.left, Const) and self.right.is_infinitesimal():
            return True
        if isinstance(self.right, Const) and self.left.is_infinitesimal():
            return True
        if self.left.is_infinitesimal() and self.right.is_infinitesimal():
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

    def simplify(self) -> Seq:
        a = self.left.simplify()
        b = self.right.simplify()
        # Conservative self-cancellation: only when denominator is a known nonzero constant
        if isinstance(b, Const) and isinstance(a, Const) and a.c == b.c and b.c != 0.0:
            return Const(1.0)
        if isinstance(b, Const):
            if b.c == 0.0:
                return Div(a, b)
            return Mul(a, Const(1.0 / b.c)).simplify()
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
