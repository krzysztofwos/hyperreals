"""Conservative asymptotic reasoning for standard-part extraction.

This module extends the purely series-based standard-part recognizer with a small
set of additional, completion-invariant rules:

- infinitesimal => standard part 0
- tanh(x) => ±1 when x -> ±∞ (provable from series)
- exp(x) => 0 when x -> -∞ (provable from series)
- algebraic closure (Add/Sub/Mul/Div) when both operands have certified
  standard parts

The rules are intentionally incomplete but sound. They do not consult or modify
the partial ultrafilter.
"""

from __future__ import annotations

from typing import Optional

from .sequence import Add, Const, Div, Exp, Mul, Seq, Sub, Tanh
from .series import eventual_sign_from_seq, is_near_standard_by_series, series_from_seq


def tends_to_infty_sign(seq: Seq, *, order: int = 10) -> Optional[int]:
    """Return +1 if seq(n)->+∞, -1 if seq(n)->-∞, else None.

    This only succeeds when divergence is provable from a Laurent-like series in
    δ=1/n; i.e., when series_from_seq(seq) exists and contains at least one
    negative exponent term.
    """
    A = series_from_seq(seq, order=order)
    if A is None or not any(k < 0 for k in A.keys()):
        return None
    sign = eventual_sign_from_seq(seq, order=order)
    if sign in (1, -1):
        return sign
    return None


def standard_part_extended(seq: Seq, *, order: int = 10) -> Optional[float]:
    """Extract a standard part when it can be certified by conservative rules."""
    ok, c = is_near_standard_by_series(seq, order)
    if ok:
        return c

    seq = seq.simplify()

    # Infinitesimal => standard part 0 (completion-invariant).
    if seq.is_infinitesimal():
        return 0.0

    # Constant sequences.
    if isinstance(seq, Const):
        return seq.c

    # Non-analytic but completion-invariant limits.
    if isinstance(seq, Tanh):
        sgn = tends_to_infty_sign(seq.arg, order=order)
        if sgn == 1:
            return 1.0
        if sgn == -1:
            return -1.0
        return None

    if isinstance(seq, Exp):
        sgn = tends_to_infty_sign(seq.arg, order=order)
        if sgn == -1:
            return 0.0
        return None

    # Algebraic closure: only combine when both sides have certified standard parts.
    if isinstance(seq, Add):
        a = standard_part_extended(seq.left, order=order)
        b = standard_part_extended(seq.right, order=order)
        if a is None or b is None:
            return None
        return a + b

    if isinstance(seq, Sub):
        a = standard_part_extended(seq.left, order=order)
        b = standard_part_extended(seq.right, order=order)
        if a is None or b is None:
            return None
        return a - b

    if isinstance(seq, Mul):
        a = standard_part_extended(seq.left, order=order)
        b = standard_part_extended(seq.right, order=order)
        if a is None or b is None:
            return None
        return a * b

    if isinstance(seq, Div):
        a = standard_part_extended(seq.left, order=order)
        b = standard_part_extended(seq.right, order=order)
        if a is None or b is None or b == 0.0:
            return None
        return a / b

    return None
