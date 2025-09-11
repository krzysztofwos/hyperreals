"""Base class for sequence expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


class Seq:
    """Base class for all sequence expressions."""

    def simplify(self) -> "Seq":
        return self

    def __add__(self, other: "Seq") -> "Seq":
        from .operations import Add

        return Add(self, other).simplify()

    def __sub__(self, other: "Seq") -> "Seq":
        from .operations import Sub

        return Sub(self, other).simplify()

    def __mul__(self, other: "Seq") -> "Seq":
        from .operations import Mul

        return Mul(self, other).simplify()

    def __truediv__(self, other: "Seq") -> "Seq":
        from .operations import Div

        return Div(self, other).simplify()

    def is_constant(self) -> bool:
        return False

    def const_value(self) -> Optional[float]:
        return None

    def is_infinitesimal(self) -> bool:
        return False

    def is_infinite(self) -> bool:
        return False

    def is_nonnegative_eventually(self) -> Optional[bool]:
        """Check if sequence is eventually non-negative."""
        from .operations import Add, Div, Mul, Sub
        from .primitives import Const, InvN, NVar

        # Very coarse reasoning for fast paths; enough for our demos.
        if self.is_constant():
            val = self.const_value()
            return val >= 0.0 if val is not None else None
        if isinstance(self, InvN):
            return True
        if isinstance(self, NVar):
            return True
        if isinstance(self, Add):
            a = self.left.is_nonnegative_eventually()
            b = self.right.is_nonnegative_eventually()
            if a is True and b is True:
                return True
            return None
        if isinstance(self, Mul):
            a = self.left.is_nonnegative_eventually()
            b = self.right.is_nonnegative_eventually()
            if a is True and b is True:
                return True
            if (
                a is True
                and self.right.is_constant()
                and self.right.const_value() == 0.0
            ) or (
                b is True and self.left.is_constant() and self.left.const_value() == 0.0
            ):
                return True
            return None
        if isinstance(self, Sub):
            return None
        if isinstance(self, Div):
            a = self.left.is_nonnegative_eventually()
            den_pos = None
            if isinstance(self.right, Const):
                den_pos = self.right.c > 0.0
            if isinstance(self.right, InvN):
                den_pos = True
            if isinstance(self.right, NVar):
                den_pos = True
            if a is True and den_pos is True:
                return True
            return None
        return None

    def at(self, n: int) -> float:
        raise NotImplementedError
