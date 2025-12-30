"""Unified asymptotic analysis for completion-invariant standard-part extraction.

This module provides a compositional limit algebra that proves facts about sequences:
- Finite limits (x_n -> c)
- Divergence to ±infinity
- Boundedness
- Bounded away from zero

The analyzer is conservative: it only returns facts that are provable from classical
cofinite reasoning, without consulting the partial ultrafilter. This ensures that
any finite limit it certifies is completion-invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from .dominance import leading_laurent_term, poly_degree_upper, poly_ratio_limit
from .sequence import (
    Add,
    AltSign,
    Const,
    Cos,
    Cosh,
    Div,
    Exp,
    InvN,
    Log1p,
    Mul,
    NVar,
    Seq,
    Sin,
    Sinh,
    Sqrt1p,
    Sub,
    Tan,
    Tanh,
)
from .series import is_near_standard_by_series

LimitKind = Literal["finite", "plus_inf", "minus_inf", "abs_inf", "unknown"]


@dataclass(frozen=True)
class AsymptoticFact:
    """Result of asymptotic analysis on a sequence.

    Attributes:
        kind: The type of limit behavior.
            - "finite": sequence converges to a real number
            - "plus_inf": sequence diverges to +infinity
            - "minus_inf": sequence diverges to -infinity
            - "abs_inf": sequence diverges (|x_n| -> infinity) but sign unknown
            - "unknown": no conclusion could be drawn
        limit: For kind == "finite", the limit value. None otherwise.
        abs_bound: If known, M such that |x_n| <= M eventually.
        abs_lower: If known, m > 0 such that |x_n| >= m eventually.
        sign: Eventual sign: -1, 0, or +1. 0 only if identically zero is provable.
        reason: Human-readable explanation of how this fact was derived.
    """

    kind: LimitKind
    limit: Optional[float] = None
    abs_bound: Optional[float] = None
    abs_lower: Optional[float] = None
    sign: Optional[int] = None
    reason: str = ""

    def is_infinite(self) -> bool:
        """Return True if this fact indicates divergence to infinity."""
        return self.kind in ("plus_inf", "minus_inf", "abs_inf")

    def is_finite_limit(self) -> bool:
        """Return True if this fact indicates convergence to a finite limit."""
        return self.kind == "finite" and self.limit is not None

    def is_bounded(self) -> bool:
        """Return True if sequence is provably bounded."""
        return self.abs_bound is not None

    def is_infinitesimal(self) -> bool:
        """Return True if sequence converges to zero."""
        return self.kind == "finite" and self.limit == 0.0


def _finite(
    limit: float,
    *,
    abs_bound: Optional[float] = None,
    abs_lower: Optional[float] = None,
    sign: Optional[int] = None,
    reason: str = "",
) -> AsymptoticFact:
    """Construct a finite-limit fact."""
    if sign is None and limit != 0:
        sign = 1 if limit > 0 else -1
    if abs_bound is None:
        abs_bound = abs(limit)
    return AsymptoticFact(
        kind="finite",
        limit=limit,
        abs_bound=abs_bound,
        abs_lower=abs_lower,
        sign=sign,
        reason=reason,
    )


def _plus_inf(
    *,
    abs_lower: Optional[float] = None,
    reason: str = "",
) -> AsymptoticFact:
    """Construct a +infinity fact."""
    return AsymptoticFact(
        kind="plus_inf",
        sign=1,
        abs_lower=abs_lower,
        reason=reason,
    )


def _minus_inf(
    *,
    abs_lower: Optional[float] = None,
    reason: str = "",
) -> AsymptoticFact:
    """Construct a -infinity fact."""
    return AsymptoticFact(
        kind="minus_inf",
        sign=-1,
        abs_lower=abs_lower,
        reason=reason,
    )


def _abs_inf(*, reason: str = "") -> AsymptoticFact:
    """Construct an absolute-infinity fact (divergence, unknown sign)."""
    return AsymptoticFact(kind="abs_inf", reason=reason)


def _unknown(
    *,
    abs_bound: Optional[float] = None,
    reason: str = "",
) -> AsymptoticFact:
    """Construct an unknown fact."""
    return AsymptoticFact(kind="unknown", abs_bound=abs_bound, reason=reason)


# Cache analyze results by repr to avoid exponential recursion
_analyze_cache: dict[str, AsymptoticFact] = {}


def analyze(seq: Seq, *, order: int = 10) -> AsymptoticFact:
    """Conservative, memoized asymptotic analysis.

    This function analyzes a sequence expression and returns an AsymptoticFact
    describing its limit behavior. The analysis is:
    - Sound: any finite limit returned is provably correct (completion-invariant)
    - Conservative: returns "unknown" when unsure, never guesses
    - Memoized: results are cached by repr of the simplified sequence

    The analyzer does NOT consult the SAT solver or partial ultrafilter.

    Args:
        seq: The sequence to analyze.
        order: Truncation order for series expansion (passed to is_near_standard_by_series).

    Returns:
        An AsymptoticFact describing the limit behavior of the sequence.
    """
    seq = seq.simplify()
    key = repr(seq)

    if key in _analyze_cache:
        return _analyze_cache[key]

    result = _analyze_impl(seq, order=order)
    _analyze_cache[key] = result
    return result


def _analyze_impl(seq: Seq, *, order: int) -> AsymptoticFact:
    """Implementation of analyze (without memoization)."""
    # Try series-based analysis first (covers analytic cases)
    ok, c = is_near_standard_by_series(seq, order)
    if ok and c is not None:
        return _finite(c, reason="series")

    # Base primitives
    if isinstance(seq, Const):
        return _finite(seq.c, reason="const")

    if isinstance(seq, InvN):
        return _finite(0.0, abs_bound=1.0, reason="1/n -> 0")

    if isinstance(seq, NVar):
        return _plus_inf(reason="n -> +inf")

    if isinstance(seq, AltSign):
        # Bounded but oscillating, no limit
        return _unknown(abs_bound=1.0, reason="(-1)^n oscillates")

    # Algebraic operations
    if isinstance(seq, Add):
        return _analyze_add(seq.left, seq.right, order=order)

    if isinstance(seq, Sub):
        return _analyze_sub(seq.left, seq.right, order=order)

    if isinstance(seq, Mul):
        return _analyze_mul(seq.left, seq.right, order=order)

    if isinstance(seq, Div):
        return _analyze_div(seq.left, seq.right, order=order)

    # Function nodes
    if isinstance(seq, Sin):
        return _analyze_sin(seq.arg, order=order)

    if isinstance(seq, Cos):
        return _analyze_cos(seq.arg, order=order)

    if isinstance(seq, Tan):
        return _analyze_tan(seq.arg, order=order)

    if isinstance(seq, Tanh):
        return _analyze_tanh(seq.arg, order=order)

    if isinstance(seq, Exp):
        return _analyze_exp(seq.arg, order=order)

    if isinstance(seq, Log1p):
        return _analyze_log1p(seq.arg, order=order)

    if isinstance(seq, Sqrt1p):
        return _analyze_sqrt1p(seq.arg, order=order)

    if isinstance(seq, Cosh):
        return _analyze_cosh(seq.arg, order=order)

    if isinstance(seq, Sinh):
        return _analyze_sinh(seq.arg, order=order)

    # Fallback: try to at least prove boundedness
    bound = seq.abs_bound_eventually()
    if bound is not None:
        return _unknown(abs_bound=bound, reason="bounded but unknown limit")

    return _unknown(reason="unknown")


def _analyze_add(left: Seq, right: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Add(left, right)."""
    A = analyze(left, order=order)
    B = analyze(right, order=order)

    # Both finite: sum is finite
    if A.kind == "finite" and B.kind == "finite":
        assert A.limit is not None and B.limit is not None
        limit = A.limit + B.limit
        abs_bound = None
        if A.abs_bound is not None and B.abs_bound is not None:
            abs_bound = A.abs_bound + B.abs_bound
        return _finite(limit, abs_bound=abs_bound, reason="finite + finite")

    # Infinity + finite = infinity
    if A.kind == "plus_inf" and B.kind == "finite":
        return _plus_inf(reason="+inf + finite")
    if A.kind == "minus_inf" and B.kind == "finite":
        return _minus_inf(reason="-inf + finite")
    if B.kind == "plus_inf" and A.kind == "finite":
        return _plus_inf(reason="finite + +inf")
    if B.kind == "minus_inf" and A.kind == "finite":
        return _minus_inf(reason="finite + -inf")

    # Same-sign infinities add
    if A.kind == "plus_inf" and B.kind == "plus_inf":
        return _plus_inf(reason="+inf + +inf")
    if A.kind == "minus_inf" and B.kind == "minus_inf":
        return _minus_inf(reason="-inf + -inf")

    # Mixed infinities: indeterminate
    if A.is_infinite() and B.is_infinite():
        return _unknown(reason="inf + inf (opposite signs or unknown)")

    # Bounds propagation
    abs_bound = None
    if A.abs_bound is not None and B.abs_bound is not None:
        abs_bound = A.abs_bound + B.abs_bound

    return _unknown(abs_bound=abs_bound, reason="add: insufficient info")


def _analyze_sub(left: Seq, right: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Sub(left, right)."""
    A = analyze(left, order=order)
    B = analyze(right, order=order)

    # Both finite: difference is finite
    if A.kind == "finite" and B.kind == "finite":
        assert A.limit is not None and B.limit is not None
        limit = A.limit - B.limit
        abs_bound = None
        if A.abs_bound is not None and B.abs_bound is not None:
            abs_bound = A.abs_bound + B.abs_bound
        return _finite(limit, abs_bound=abs_bound, reason="finite - finite")

    # Infinity - finite
    if A.kind == "plus_inf" and B.kind == "finite":
        return _plus_inf(reason="+inf - finite")
    if A.kind == "minus_inf" and B.kind == "finite":
        return _minus_inf(reason="-inf - finite")
    if A.kind == "finite" and B.kind == "plus_inf":
        return _minus_inf(reason="finite - +inf")
    if A.kind == "finite" and B.kind == "minus_inf":
        return _plus_inf(reason="finite - -inf")

    # Opposite-sign infinities: determinate
    if A.kind == "plus_inf" and B.kind == "minus_inf":
        return _plus_inf(reason="+inf - -inf")
    if A.kind == "minus_inf" and B.kind == "plus_inf":
        return _minus_inf(reason="-inf - +inf")

    # Same-sign infinities: indeterminate
    if A.is_infinite() and B.is_infinite():
        return _unknown(reason="inf - inf (same sign or unknown)")

    # Bounds propagation
    abs_bound = None
    if A.abs_bound is not None and B.abs_bound is not None:
        abs_bound = A.abs_bound + B.abs_bound

    return _unknown(abs_bound=abs_bound, reason="sub: insufficient info")


def _analyze_mul(left: Seq, right: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Mul(left, right)."""
    A = analyze(left, order=order)
    B = analyze(right, order=order)

    # Both finite: product is finite
    if A.kind == "finite" and B.kind == "finite":
        assert A.limit is not None and B.limit is not None
        limit = A.limit * B.limit
        abs_bound = None
        if A.abs_bound is not None and B.abs_bound is not None:
            abs_bound = A.abs_bound * B.abs_bound
        return _finite(limit, abs_bound=abs_bound, reason="finite * finite")

    # Zero * bounded = 0 (BV closure)
    if A.kind == "finite" and A.limit == 0.0 and B.abs_bound is not None:
        return _finite(0.0, reason="0 * bounded")
    if B.kind == "finite" and B.limit == 0.0 and A.abs_bound is not None:
        return _finite(0.0, reason="bounded * 0")

    # Infinitesimal * bounded = infinitesimal (extended BV)
    if A.is_infinitesimal() and B.is_bounded():
        return _finite(0.0, reason="infinitesimal * bounded")
    if B.is_infinitesimal() and A.is_bounded():
        return _finite(0.0, reason="bounded * infinitesimal")

    # Infinity * nonzero finite
    if A.is_infinite() and B.kind == "finite" and B.limit is not None and B.limit != 0:
        if A.sign is not None:
            combined_sign = A.sign * (1 if B.limit > 0 else -1)
            if combined_sign > 0:
                return _plus_inf(reason="inf * nonzero")
            else:
                return _minus_inf(reason="inf * nonzero")
        return _abs_inf(reason="inf * nonzero (sign unknown)")

    if B.is_infinite() and A.kind == "finite" and A.limit is not None and A.limit != 0:
        if B.sign is not None:
            combined_sign = B.sign * (1 if A.limit > 0 else -1)
            if combined_sign > 0:
                return _plus_inf(reason="nonzero * inf")
            else:
                return _minus_inf(reason="nonzero * inf")
        return _abs_inf(reason="nonzero * inf (sign unknown)")

    # Bounded * bounded -> bounded.
    if A.abs_bound is not None and B.abs_bound is not None:
        return _unknown(abs_bound=A.abs_bound * B.abs_bound, reason="bounded*bounded")

    # Tier-3: log(1+poly) * 1/n -> 0 (when poly diverges polynomially).
    # This handles the simplified form of log(1+n)/n.
    if isinstance(left, Log1p) and isinstance(right, InvN):
        lt_u = leading_laurent_term(left.arg, order=order)
        if lt_u is not None:
            k_u, a_u = lt_u
            if k_u < 0 and a_u > 0:
                return _finite(0.0, reason="log(1+poly)*1/n")
    if isinstance(right, Log1p) and isinstance(left, InvN):
        lt_u = leading_laurent_term(right.arg, order=order)
        if lt_u is not None:
            k_u, a_u = lt_u
            if k_u < 0 and a_u > 0:
                return _finite(0.0, reason="1/n*log(1+poly)")

    # Tier-3: exp(-poly(n)) * poly_bounded(n) -> 0.
    #
    # We recognize this only when the exp argument has a Laurent leading term
    # with negative exponent (polynomial in n) and negative coefficient (-> -∞).
    if isinstance(left, Exp):
        lt = leading_laurent_term(left.arg, order=order)
        if lt is not None:
            k, a = lt
            if k < 0 and a < 0:
                deg = poly_degree_upper(right, order=order)
                if deg is not None:
                    return _finite(0.0, reason="exp(-poly)*poly")
    if isinstance(right, Exp):
        lt = leading_laurent_term(right.arg, order=order)
        if lt is not None:
            k, a = lt
            if k < 0 and a < 0:
                deg = poly_degree_upper(left, order=order)
                if deg is not None:
                    return _finite(0.0, reason="poly*exp(-poly)")

    return _unknown(reason="mul: insufficient info")


def _analyze_div(left: Seq, right: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Div(left, right)."""
    A = analyze(left, order=order)
    B = analyze(right, order=order)

    # Both finite, denominator nonzero
    if (
        A.kind == "finite"
        and B.kind == "finite"
        and B.limit is not None
        and B.limit != 0
    ):
        assert A.limit is not None
        limit = A.limit / B.limit
        # Denominator bounded away from zero
        abs_lower = abs(B.limit) / 2.0
        return _finite(limit, abs_lower=abs_lower, reason="finite / nonzero finite")

    # Bounded / diverging = 0 (key Tier-2 rule)
    if A.abs_bound is not None and B.is_infinite():
        return _finite(0.0, reason="bounded / diverging")

    # Tier-3: Polynomial ratio limits (including non-monomial denominators).
    pr = poly_ratio_limit(left, right, order=order)
    if pr is not None:
        return _finite(pr, reason="poly_ratio")

    # Tier-3: Polynomial numerator over exp(poly(n)) -> 0.
    if isinstance(right, Exp):
        lt = leading_laurent_term(right.arg, order=order)
        if lt is not None:
            k, a = lt
            if k < 0 and a > 0:
                deg = poly_degree_upper(left, order=order)
                if deg is not None:
                    return _finite(0.0, reason="poly/exp(poly)")

    # Tier-3: log(1+poly(n)) / poly(n) -> 0 when denominator diverges polynomially.
    if isinstance(left, Log1p):
        lt_u = leading_laurent_term(left.arg, order=order)
        lt_d = leading_laurent_term(right, order=order)
        if lt_u is not None and lt_d is not None:
            k_u, a_u = lt_u
            k_d, a_d = lt_d
            if k_u < 0 and a_u > 0 and k_d < 0 and a_d != 0.0:
                return _finite(0.0, reason="log/poly")

    # Infinitesimal / bounded-away-from-zero = infinitesimal
    if A.is_infinitesimal() and B.abs_lower is not None and B.abs_lower > 0:
        return _finite(0.0, reason="infinitesimal / bounded-away-from-zero")

    # Finite limit 0 / nonzero finite
    if A.kind == "finite" and A.limit == 0.0 and B.abs_lower is not None:
        return _finite(0.0, reason="0 / bounded-away-from-zero")

    # Diverging / finite nonzero
    if A.is_infinite() and B.kind == "finite" and B.limit is not None and B.limit != 0:
        if A.sign is not None:
            combined_sign = A.sign * (1 if B.limit > 0 else -1)
            if combined_sign > 0:
                return _plus_inf(reason="diverging / nonzero finite")
            else:
                return _minus_inf(reason="diverging / nonzero finite")
        return _abs_inf(reason="diverging / nonzero finite (sign unknown)")

    return _unknown(reason="div: insufficient info")


# Function analysis


def _analyze_sin(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Sin(arg)."""
    A = analyze(arg, order=order)

    # Always bounded by 1
    if A.kind == "finite" and A.limit is not None:
        try:
            limit = math.sin(A.limit)
            return _finite(limit, abs_bound=1.0, reason="sin(finite)")
        except (ValueError, OverflowError):
            pass

    # sin of anything is bounded
    return _unknown(abs_bound=1.0, reason="sin bounded")


def _analyze_cos(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Cos(arg)."""
    A = analyze(arg, order=order)

    if A.kind == "finite" and A.limit is not None:
        try:
            limit = math.cos(A.limit)
            return _finite(limit, abs_bound=1.0, reason="cos(finite)")
        except (ValueError, OverflowError):
            pass

    return _unknown(abs_bound=1.0, reason="cos bounded")


def _analyze_tan(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Tan(arg)."""
    A = analyze(arg, order=order)

    if A.kind == "finite" and A.limit is not None:
        # Check if limit is at a pole (odd multiple of pi/2)
        normalized = A.limit / (math.pi / 2)
        if abs(normalized - round(normalized)) > 1e-10 or round(normalized) % 2 == 0:
            try:
                limit = math.tan(A.limit)
                return _finite(limit, reason="tan(finite)")
            except (ValueError, OverflowError):
                pass

    return _unknown(reason="tan: unknown")


def _analyze_tanh(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Tanh(arg)."""
    A = analyze(arg, order=order)

    # Always bounded by 1
    if A.kind == "finite" and A.limit is not None:
        try:
            limit = math.tanh(A.limit)
            return _finite(limit, abs_bound=1.0, reason="tanh(finite)")
        except (ValueError, OverflowError):
            pass

    # Monotone limits at infinity
    if A.kind == "plus_inf":
        return _finite(1.0, abs_bound=1.0, reason="tanh(+inf) = 1")
    if A.kind == "minus_inf":
        return _finite(-1.0, abs_bound=1.0, reason="tanh(-inf) = -1")

    return _unknown(abs_bound=1.0, reason="tanh bounded")


def _analyze_exp(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Exp(arg)."""
    A = analyze(arg, order=order)

    if A.kind == "finite" and A.limit is not None:
        try:
            limit = math.exp(A.limit)
            return _finite(limit, abs_lower=limit / 2.0, reason="exp(finite)")
        except (ValueError, OverflowError):
            pass

    # Monotone limits at infinity
    if A.kind == "plus_inf":
        return _plus_inf(reason="exp(+inf) = +inf")
    if A.kind == "minus_inf":
        return _finite(0.0, reason="exp(-inf) = 0")

    # abs_inf: we know |exp(arg)| diverges in some direction
    # but can't determine which without knowing sign

    return _unknown(reason="exp: unknown")


def _analyze_log1p(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Log1p(arg) = log(1 + arg)."""
    A = analyze(arg, order=order)

    if A.kind == "finite" and A.limit is not None:
        if A.limit > -1:
            try:
                limit = math.log1p(A.limit)
                return _finite(limit, reason="log1p(finite)")
            except (ValueError, OverflowError):
                pass

    # log(1 + x) -> +inf as x -> +inf
    if A.kind == "plus_inf":
        return _plus_inf(reason="log1p(+inf) = +inf")

    # x -> -1 or worse: domain issues
    return _unknown(reason="log1p: unknown")


def _analyze_sqrt1p(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Sqrt1p(arg) = sqrt(1 + arg)."""
    A = analyze(arg, order=order)

    if A.kind == "finite" and A.limit is not None:
        if A.limit >= -1:
            try:
                limit = math.sqrt(1.0 + A.limit)
                return _finite(limit, reason="sqrt1p(finite)")
            except (ValueError, OverflowError):
                pass

    # sqrt(1 + x) -> +inf as x -> +inf
    if A.kind == "plus_inf":
        return _plus_inf(reason="sqrt1p(+inf) = +inf")

    return _unknown(reason="sqrt1p: unknown")


def _analyze_cosh(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Cosh(arg)."""
    A = analyze(arg, order=order)

    if A.kind == "finite" and A.limit is not None:
        try:
            limit = math.cosh(A.limit)
            return _finite(limit, abs_lower=1.0, reason="cosh(finite)")
        except (ValueError, OverflowError):
            pass

    # cosh -> +inf for any ±inf
    if A.kind in ("plus_inf", "minus_inf", "abs_inf"):
        return _plus_inf(reason="cosh(±inf) = +inf")

    return _unknown(reason="cosh: unknown")


def _analyze_sinh(arg: Seq, *, order: int) -> AsymptoticFact:
    """Analyze Sinh(arg)."""
    A = analyze(arg, order=order)

    if A.kind == "finite" and A.limit is not None:
        try:
            limit = math.sinh(A.limit)
            return _finite(limit, reason="sinh(finite)")
        except (ValueError, OverflowError):
            pass

    # sinh is monotone
    if A.kind == "plus_inf":
        return _plus_inf(reason="sinh(+inf) = +inf")
    if A.kind == "minus_inf":
        return _minus_inf(reason="sinh(-inf) = -inf")
    if A.kind == "abs_inf":
        return _abs_inf(reason="sinh(±inf) diverges")

    return _unknown(reason="sinh: unknown")


def clear_cache() -> None:
    """Clear the analysis cache. Useful for testing."""
    _analyze_cache.clear()
