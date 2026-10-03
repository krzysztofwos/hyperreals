"""Exact polynomial constructors corresponding to the Lean Horner compilers.

Coefficients are in ascending degree order. These constructors preserve exact
expressions and do not supply derivative values or limit certificates to Lean.
Python-to-Lean syntax capture remains a tested boundary.
"""

from collections.abc import Iterable
from fractions import Fraction

from .verified_residue import LeanResidueSystem, ResidueHyperreal

Coefficient = int | float | Fraction


def evaluate_polynomial(
    coefficients: Iterable[Coefficient], argument: ResidueHyperreal
) -> ResidueHyperreal:
    """Construct an exact Horner expression in ``argument``."""
    system = argument._system
    result = system.constant(0)
    for coefficient in reversed(tuple(Fraction(c) for c in coefficients)):
        result = system.constant(coefficient) + argument * result
    return result


def divided_difference(
    coefficients: Iterable[Coefficient], a: Coefficient, increment: ResidueHyperreal
) -> ResidueHyperreal:
    """Construct D with h*D = P(a+h)-P(a), using only + and *.

    D equals the literal quotient wherever h is nonzero. At h=0 it is the
    polynomial extension, so this function does not establish division validity.
    """
    system = increment._system
    point = system.constant(a)
    argument = point + increment
    tail = system.constant(0)
    result = system.constant(0)
    for coefficient in reversed(tuple(Fraction(c) for c in coefficients)):
        result = tail + point * result
        tail = system.constant(coefficient) + argument * tail
    return result


def polynomial_quotient(
    coefficients: Iterable[Coefficient],
    a: Coefficient,
    c: Coefficient = 1,
    order: int = 1,
    *,
    system: LeanResidueSystem,
) -> ResidueHyperreal:
    """Construct the literal quotient at a with increment c/n**order.

    The nonzero scale and positive integer order are checked before constructing
    syntax. Every rational polynomial and such increment are covered by Lean's
    ``quotientExpr_computes_derivative`` theorem.
    """
    if type(order) is not int or order <= 0:
        raise ValueError("increment order must be a positive integer")
    scale = Fraction(c)
    if scale == 0:
        raise ZeroDivisionError("increment scale must be nonzero")
    coefficients = tuple(Fraction(coefficient) for coefficient in coefficients)
    point = system.constant(a)
    increment = system.constant(scale).divide_monomial(1, order)
    numerator = evaluate_polynomial(
        coefficients, point + increment
    ) - evaluate_polynomial(coefficients, point)
    return numerator.divide_monomial(scale, -order)
