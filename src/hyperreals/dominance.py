"""Small, sound dominance lemmas for conservative asymptotic reasoning.

This module deliberately does *not* attempt a full asymptotic algebra. Instead
it provides a minimal, pattern-driven library of growth/decay facts that are
both:

  (i) completion-invariant (cofinite/Fréchet truth), and
 (ii) useful for certifying standard parts in non-analytic regimes.

The functions here are used by :mod:`hyperreal.asymptotic_facts` to recognize
limits such as:

  - log(1+n)/n -> 0
  - n/exp(n) -> 0
  - (n^2+1)/(n^2+n) -> 1
  - n^k * exp(-n) -> 0   (via exp(-poly(n)) decay)

All rules are conservative: returning ``None`` means "unknown", not "false".
"""

from __future__ import annotations

from typing import Optional, Tuple

from .sequence import (
    Add,
    AltSign,
    Const,
    Cos,
    Div,
    Exp,
    InvN,
    Log1p,
    Mul,
    NVar,
    Seq,
    Sin,
    Sqrt1p,
    Sub,
    Tanh,
)
from .series import series_from_seq


def leading_laurent_term(seq: Seq, *, order: int = 10) -> Optional[Tuple[int, float]]:
    """Return (k, a) for the most negative exponent term a*δ^k in the δ-series.

    Here δ = 1/n. Negative k correspond to positive powers of n.
    Returns None if no series is available.
    """
    A = series_from_seq(seq, order=order)
    if not A:
        return None
    # Drop exact-zero coefficients defensively.
    items = [(k, a) for (k, a) in A.items() if a != 0.0]
    if not items:
        return None
    k, a = min(items, key=lambda kv: kv[0])
    return int(k), float(a)


def poly_degree_upper(seq: Seq, *, order: int = 10) -> Optional[int]:
    """Return k such that |seq(n)| = O(n^k) can be certified, else None.

    This is intentionally coarse; it exists to justify super-polynomial decay
    lemmas such as poly(n) * exp(-poly(n)) -> 0. Any certified polynomial upper
    bound is sufficient.
    """
    seq = seq.simplify()

    # Many bounded sequences can certify an eventual absolute bound.
    try:
        if seq.abs_bound_eventually() is not None:  # type: ignore[misc]
            return 0
    except Exception:
        pass

    if isinstance(seq, (Const, AltSign, Sin, Cos, Tanh)):
        return 0
    if isinstance(seq, InvN):
        return -1
    if isinstance(seq, NVar):
        return 1

    if isinstance(seq, (Add, Sub)):
        a = poly_degree_upper(seq.left, order=order)
        b = poly_degree_upper(seq.right, order=order)
        if a is not None and b is not None:
            return max(a, b)
        return None

    if isinstance(seq, Mul):
        a = poly_degree_upper(seq.left, order=order)
        b = poly_degree_upper(seq.right, order=order)
        if a is not None and b is not None:
            return a + b
        # bounded * poly -> poly
        try:
            if seq.left.abs_bound_eventually() is not None and b is not None:  # type: ignore[misc]
                return b
            if seq.right.abs_bound_eventually() is not None and a is not None:  # type: ignore[misc]
                return a
        except Exception:
            pass
        return None

    if isinstance(seq, Div):
        num = poly_degree_upper(seq.left, order=order)
        if num is None:
            return None
        # Safe special case: divide by n.
        if isinstance(seq.right, NVar):
            return num - 1
        return None

    if isinstance(seq, Log1p):
        # If arg is eventually nonnegative and polynomially bounded, log(1+arg) <= arg.
        try:
            nonneg = seq.arg.is_nonnegative_eventually()
        except Exception:
            nonneg = None
        if nonneg:
            d = poly_degree_upper(seq.arg, order=order)
            if d is not None:
                return max(0, d)
        return None

    if isinstance(seq, Sqrt1p):
        # Coarse: sqrt(1+u) <= 1+u for u>=0.
        try:
            nonneg = seq.arg.is_nonnegative_eventually()
        except Exception:
            nonneg = None
        if nonneg:
            d = poly_degree_upper(seq.arg, order=order)
            if d is not None:
                return max(0, d)
        return None

    if isinstance(seq, Exp):
        # exp(negative polynomial infinity) is bounded by 1 eventually.
        lt = leading_laurent_term(seq.arg, order=order)
        if lt is not None:
            exp_k, coeff = lt
            if exp_k < 0 and coeff < 0:
                return 0
        return None

    return None


def poly_ratio_limit(num: Seq, den: Seq, *, order: int = 10) -> Optional[float]:
    """Return lim num/den for polynomial-like sequences when certifiable.

    This handles the most important case not covered by the Laurent series
    division routine: a ratio of two Laurent polynomials in δ whose denominator
    is not a monomial.

    Returns:
      - 0.0 if deg(den) > deg(num)
      - leading_coeff(num) / leading_coeff(den) if deg equal and denominator diverges
      - None otherwise (unknown or diverges).
    """
    ln = leading_laurent_term(num, order=order)
    ld = leading_laurent_term(den, order=order)
    if ln is None or ld is None:
        return None
    k_n, a_n = ln
    k_d, a_d = ld
    if a_d == 0.0:
        return None
    # Require a genuinely diverging polynomial denominator.
    if k_d >= 0:
        return None
    if k_n > k_d:
        # numerator has less growth than denominator
        return 0.0
    if k_n == k_d:
        return a_n / a_d
    # numerator grows faster -> diverges
    return None
