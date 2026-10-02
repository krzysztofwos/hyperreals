"""Analytic functions on sequences.

Simplification preserves analytic expressions except for exact identities at
zero. Floating-point evaluation belongs in ``at`` and the approximate limit
analyzer, not in the expression tree used by exact comparison certificates.
"""

import math
from dataclasses import dataclass
from typing import Tuple

from .base import Seq
from .primitives import Const


@dataclass(frozen=True)
class Sin(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"sin({self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("sin", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(0.0)
        return Sin(a)

    def at(self, n: int) -> float:
        return math.sin(self.arg.at(n))


@dataclass(frozen=True)
class Cos(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"cos({self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("cos", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(1.0)
        return Cos(a)

    def at(self, n: int) -> float:
        return math.cos(self.arg.at(n))


@dataclass(frozen=True)
class Tan(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"tan({self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("tan", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(0.0)
        return Tan(a)

    def at(self, n: int) -> float:
        return math.tan(self.arg.at(n))


@dataclass(frozen=True)
class Tanh(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"tanh({self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("tanh", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(0.0)
        return Tanh(a)

    def at(self, n: int) -> float:
        return math.tanh(self.arg.at(n))


@dataclass(frozen=True)
class Exp(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"exp({self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("exp", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(1.0)
        return Exp(a)

    def at(self, n: int) -> float:
        return math.exp(self.arg.at(n))


@dataclass(frozen=True)
class Log1p(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"log(1+{self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("log1p", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(0.0)
        return Log1p(a)

    def at(self, n: int) -> float:
        return math.log1p(self.arg.at(n))


@dataclass(frozen=True)
class Sqrt1p(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"sqrt(1+{self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("sqrt1p", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(1.0)
        return Sqrt1p(a)

    def at(self, n: int) -> float:
        return math.sqrt(1.0 + self.arg.at(n))


@dataclass(frozen=True)
class Cosh(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"cosh({self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("cosh", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(1.0)
        return Cosh(a)

    def at(self, n: int) -> float:
        return math.cosh(self.arg.at(n))


@dataclass(frozen=True)
class Sinh(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"sinh({self.arg})"

    def key(self) -> Tuple[object, ...]:
        return ("sinh", self.arg.key())

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const) and a.c == 0.0:
            return Const(0.0)
        return Sinh(a)

    def at(self, n: int) -> float:
        return math.sinh(self.arg.at(n))
