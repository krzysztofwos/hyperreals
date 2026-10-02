"""Exact rational arithmetic used by the comparison certificate compiler.

Float constants denote their exact binary rational values.  This module does
not simplify expressions or call the approximate series analyzer: either it
extracts the represented Laurent polynomial exactly, or it returns ``None``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Optional

from .sequence import Add, Const, Div, InvN, Mul, NVar, Seq, Sub


def exact_laurent_coefficients(sequence: Seq) -> Optional[dict[int, Fraction]]:
    """Extract a finite Laurent polynomial in ``1/n`` using exact coefficients.

    Division is supported only by a nonzero monomial, which also establishes
    that every represented expression is defined for all positive indices.
    Unsupported operands are rejected before cancellation can hide them.
    """
    if isinstance(sequence, Const):
        if not math.isfinite(sequence.c):
            return None
        coefficient = Fraction.from_float(sequence.c)
        return {0: coefficient} if coefficient else {}
    if isinstance(sequence, InvN):
        return {1: Fraction(1)}
    if isinstance(sequence, NVar):
        return {-1: Fraction(1)}
    if not isinstance(sequence, (Add, Sub, Mul, Div)):
        return None

    left = exact_laurent_coefficients(sequence.left)
    right = exact_laurent_coefficients(sequence.right)
    if left is None or right is None:
        return None

    if isinstance(sequence, (Add, Sub)):
        result = dict(left)
        direction = -1 if isinstance(sequence, Sub) else 1
        for power, coefficient in right.items():
            result[power] = result.get(power, Fraction(0)) + direction * coefficient
    elif isinstance(sequence, Mul):
        result = {}
        for left_power, left_coefficient in left.items():
            for right_power, right_coefficient in right.items():
                power = left_power + right_power
                result[power] = (
                    result.get(power, Fraction(0)) + left_coefficient * right_coefficient
                )
    else:
        if len(right) != 1:
            return None
        denominator_power, denominator_coefficient = next(iter(right.items()))
        result = {
            power - denominator_power: coefficient / denominator_coefficient
            for power, coefficient in left.items()
        }
    return {power: coefficient for power, coefficient in result.items() if coefficient}
