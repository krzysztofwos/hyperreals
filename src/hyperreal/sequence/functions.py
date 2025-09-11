"""Analytic functions on sequences."""

import math
from dataclasses import dataclass

from .base import Seq
from .primitives import Const


@dataclass(frozen=True)
class Sin(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"sin({self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.sin(a.c))
        return Sin(a)

    def at(self, n: int) -> float:
        return math.sin(self.arg.at(n))


@dataclass(frozen=True)
class Cos(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"cos({self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.cos(a.c))
        return Cos(a)

    def at(self, n: int) -> float:
        return math.cos(self.arg.at(n))


@dataclass(frozen=True)
class Tan(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"tan({self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.tan(a.c))
        return Tan(a)

    def at(self, n: int) -> float:
        return math.tan(self.arg.at(n))


@dataclass(frozen=True)
class Tanh(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"tanh({self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.tanh(a.c))
        return Tanh(a)

    def at(self, n: int) -> float:
        return math.tanh(self.arg.at(n))


@dataclass(frozen=True)
class Exp(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"exp({self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.exp(a.c))
        return Exp(a)

    def at(self, n: int) -> float:
        return math.exp(self.arg.at(n))


@dataclass(frozen=True)
class Log1p(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"log(1+{self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.log1p(a.c))
        return Log1p(a)

    def at(self, n: int) -> float:
        return math.log1p(self.arg.at(n))


@dataclass(frozen=True)
class Sqrt1p(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"sqrt(1+{self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.sqrt(1.0 + a.c))
        return Sqrt1p(a)

    def at(self, n: int) -> float:
        return math.sqrt(1.0 + self.arg.at(n))


@dataclass(frozen=True)
class Cosh(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"cosh({self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.cosh(a.c))
        return Cosh(a)

    def at(self, n: int) -> float:
        return math.cosh(self.arg.at(n))


@dataclass(frozen=True)
class Sinh(Seq):
    arg: Seq

    def __repr__(self) -> str:
        return f"sinh({self.arg})"

    def simplify(self) -> Seq:
        a = self.arg.simplify()
        if isinstance(a, Const):
            return Const(math.sinh(a.c))
        return Sinh(a)

    def at(self, n: int) -> float:
        return math.sinh(self.arg.at(n))
