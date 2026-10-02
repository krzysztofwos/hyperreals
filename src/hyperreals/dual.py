"""Dual numbers for automatic differentiation (first-order)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable


@dataclass
class Dual:
    """Dual number a + bε for first-order automatic differentiation."""

    a: float  # Real part
    b: float  # Dual part

    def __add__(self, other: "Dual") -> "Dual":
        return Dual(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "Dual") -> "Dual":
        return Dual(self.a - other.a, self.b - other.b)

    def __mul__(self, other: "Dual") -> "Dual":
        return Dual(self.a * other.a, self.a * other.b + self.b * other.a)

    def __truediv__(self, other: "Dual") -> "Dual":
        if other.a == 0.0:
            raise ZeroDivisionError(
                "Dual division by zero real part (can't divide by ε or ε^2)"
            )
        real = self.a / other.a
        dual = (self.b * other.a - self.a * other.b) / (other.a * other.a)
        return Dual(real, dual)

    def __repr__(self) -> str:
        return f"{self.a} + {self.b}ε"


def ad_derivative_first(f: "Callable[[Dual], Dual]", x0: float) -> float:
    """Compute first derivative of f at x0 using dual numbers."""
    x = Dual(x0, 1.0)
    y = f(x)
    return y.b
