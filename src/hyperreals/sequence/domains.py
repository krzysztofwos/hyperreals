"""Conservative domain checks for simplifications that discard operands.

Only finite constants, elementary algebra, everywhere-real functions, and
domains justified by exact Laurent signs are recognized. A false result means
that definedness has not been established. It is not evidence of a domain error.
These checks do not run the approximate asymptotic analyzer.
"""

import math
from typing import Optional

from .base import Seq
from .functions import Cos, Cosh, Exp, Log1p, Sin, Sinh, Sqrt1p, Tanh
from .primitives import AltSign, Const, InvN, NVar


def _exact_eventual_sign(sequence: Seq) -> Optional[int]:
    # Import lazily: the exact extractor imports the sequence package.
    from ..exact_arithmetic import exact_laurent_coefficients

    coefficients = exact_laurent_coefficients(sequence)
    if coefficients is None:
        return None
    if not coefficients:
        return 0
    return 1 if coefficients[min(coefficients)] > 0 else -1


def _is_eventually_nonzero(sequence: Seq) -> bool:
    from .operations import Div, Mul

    sign = _exact_eventual_sign(sequence)
    if sign is not None:
        return sign != 0
    if isinstance(sequence, AltSign):
        return True
    if isinstance(sequence, (Exp, Cosh)):
        return is_eventually_real(sequence.arg)
    if isinstance(sequence, (Mul, Div)):
        return _is_eventually_nonzero(sequence.left) and _is_eventually_nonzero(sequence.right)
    return False


def is_eventually_real(sequence: Seq) -> bool:
    """Whether a supported expression is real and defined on a cofinite tail."""
    from .operations import Add, Div, Mul, Sub

    if isinstance(sequence, Const):
        return math.isfinite(sequence.c)
    if isinstance(sequence, (NVar, InvN, AltSign)):
        return True
    if isinstance(sequence, (Add, Sub, Mul)):
        return is_eventually_real(sequence.left) and is_eventually_real(sequence.right)
    if isinstance(sequence, Div):
        return is_eventually_real(sequence.left) and _is_eventually_nonzero(sequence.right)
    if isinstance(sequence, (Sin, Cos, Exp, Sinh, Cosh, Tanh)):
        return is_eventually_real(sequence.arg)
    if isinstance(sequence, (Log1p, Sqrt1p)):
        sign = _exact_eventual_sign(Add(Const(1), sequence.arg))
        return sign == 1 or (isinstance(sequence, Sqrt1p) and sign == 0)
    return False
