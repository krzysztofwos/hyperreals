"""Laurent series operations for standard part computation."""

import math
from typing import Dict, List, Optional, Tuple

from .sequence import (
    Add,
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


def series_from_seq(s: Seq, order: int = 10) -> Optional[Dict[int, float]]:
    """Return a Laurent-like series in powers of δ=1/n (key = exponent).

    The `order` parameter controls the highest positive exponent retained.
    """
    s = s.simplify()
    if isinstance(s, Const):
        return {0: s.c}
    if isinstance(s, InvN):
        return {1: 1.0}
    if isinstance(s, NVar):
        return {-1: 1.0}
    if isinstance(s, Add):
        A = series_from_seq(s.left, order)
        B = series_from_seq(s.right, order)
        if A is None or B is None:
            return None
        return series_add(A, B)
    if isinstance(s, Sub):
        A = series_from_seq(s.left, order)
        B = series_from_seq(s.right, order)
        if A is None or B is None:
            return None
        return series_sub(A, B)
    if isinstance(s, Mul):
        A = series_from_seq(s.left, order)
        B = series_from_seq(s.right, order)
        if A is None or B is None:
            return None
        return series_truncate(series_mul(A, B), order)
    if isinstance(s, Div):
        A = series_from_seq(s.left, order)
        B = series_from_seq(s.right, order)
        if A is None or B is None:
            return None
        if len(B) != 1:
            return None
        (k, c) = next(iter(B.items()))
        if c == 0.0:
            return None
        return series_truncate(series_scale(series_shift(A, -k), 1.0 / c), order)
    # Analytic functions
    if isinstance(s, Sin):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_sin(A, order=order)
    if isinstance(s, Cos):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_cos(A, order=order)
    if isinstance(s, Tan):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_tan(A, order=order)
    if isinstance(s, Tanh):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_tanh(A, order=order)
    if isinstance(s, Exp):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_exp(A, order=order)
    if isinstance(s, Log1p):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_log1p(A, order=order)
    if isinstance(s, Sqrt1p):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_sqrt1p(A, order=order)
    if isinstance(s, Cosh):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_cosh(A, order=order)
    if isinstance(s, Sinh):
        A = series_from_seq(s.arg, order)
        if A is None or any(exp < 0 for exp in A.keys()):
            return None
        return series_sinh(A, order=order)
    return None


def series_add(A: Dict[int, float], B: Dict[int, float]) -> Dict[int, float]:
    out = dict(A)
    for k, v in B.items():
        out[k] = out.get(k, 0.0) + v
    return {k: v for k, v in out.items() if v != 0.0}


def series_sub(A: Dict[int, float], B: Dict[int, float]) -> Dict[int, float]:
    out = dict(A)
    for k, v in B.items():
        out[k] = out.get(k, 0.0) - v
    return {k: v for k, v in out.items() if v != 0.0}


def series_mul(A: Dict[int, float], B: Dict[int, float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k1, c1 in A.items():
        for k2, c2 in B.items():
            k = k1 + k2
            out[k] = out.get(k, 0.0) + c1 * c2
    return {k: v for k, v in out.items() if v != 0.0}


def series_scale(A: Dict[int, float], c: float) -> Dict[int, float]:
    return {k: c * v for k, v in A.items() if (c * v) != 0.0}


def series_shift(A: Dict[int, float], shift: int) -> Dict[int, float]:
    return {k + shift: v for k, v in A.items()}


def series_pow(A: Dict[int, float], p: int, max_exp: int) -> Dict[int, float]:
    if p == 0:
        return {0: 1.0}
    out = dict(A)
    for _ in range(1, p):
        out = series_mul(out, A)
        out = {k: v for k, v in out.items() if k <= max_exp}
    return out


def series_truncate(A: Dict[int, float], max_pos_exp: int) -> Dict[int, float]:
    return {k: v for k, v in A.items() if k <= max_pos_exp}


def _split_const_and_tail(A: Dict[int, float]) -> Tuple[float, Dict[int, float]]:
    c = A.get(0, 0.0)
    tail = dict(A)
    if 0 in tail:
        del tail[0]
    return c, tail


def _compose_from_constant(
    A: Dict[int, float], derivs_at_c: List[float], order: int
) -> Dict[int, float]:
    """
    Compose using Taylor around the constant term c:
        f(c + D) = sum_{k=0..order} f^{(k)}(c)/k! * D^k
    where A = {0: c} + D, and D has only positive exponents.
    derivs_at_c[k] = f^{(k)}(c) for k=0..order.
    """
    c, D = _split_const_and_tail(A)
    out: Dict[int, float] = {}
    # k = 0 term
    out = series_add(out, {0: derivs_at_c[0]})
    if not D:
        return out
    for k in range(1, order + 1):
        coef = derivs_at_c[k] / math.factorial(k)
        Dk = series_pow(D, k, order)
        out = series_add(out, series_scale(Dk, coef))
    return series_truncate(out, order)


def series_sin(A: Dict[int, float], order: int = 5) -> Dict[int, float]:
    # Taylor around constant term c with periodic derivatives
    c, _ = _split_const_and_tail(A)
    derivs: List[float] = []
    for k in range(order + 1):
        r = k % 4
        if r == 0:
            derivs.append(math.sin(c))
        elif r == 1:
            derivs.append(math.cos(c))
        elif r == 2:
            derivs.append(-math.sin(c))
        else:
            derivs.append(-math.cos(c))
    return _compose_from_constant(A, derivs, order)


def series_cos(A: Dict[int, float], order: int = 5) -> Dict[int, float]:
    c, _ = _split_const_and_tail(A)
    derivs: List[float] = []
    for k in range(order + 1):
        r = k % 4
        if r == 0:
            derivs.append(math.cos(c))
        elif r == 1:
            derivs.append(-math.sin(c))
        elif r == 2:
            derivs.append(-math.cos(c))
        else:
            derivs.append(math.sin(c))
    return _compose_from_constant(A, derivs, order)


def series_exp(A: Dict[int, float], order: int = 5) -> Dict[int, float]:
    c, _ = _split_const_and_tail(A)
    ec = math.exp(c)
    # f^{(k)}(c) = exp(c)
    derivs = [ec for _ in range(order + 1)]
    return _compose_from_constant(A, derivs, order)


def series_log1p(A: Dict[int, float], order: int = 5) -> Optional[Dict[int, float]]:
    c, _ = _split_const_and_tail(A)
    base = 1.0 + c
    if base <= 0.0:
        return None
    derivs: List[float] = [0.0 for _ in range(order + 1)]
    derivs[0] = math.log1p(c)
    for k in range(1, order + 1):
        derivs[k] = ((-1.0) ** (k - 1)) * math.factorial(k - 1) / (base**k)
    return _compose_from_constant(A, derivs, order)


def _binom(a: float, k: int) -> float:
    """Generalized binomial coefficient (a choose k)."""
    if k == 0:
        return 1.0
    num = 1.0
    for i in range(k):
        num *= a - i
    return num / math.factorial(k)


def series_sqrt1p(A: Dict[int, float], order: int = 5) -> Optional[Dict[int, float]]:
    c, _ = _split_const_and_tail(A)
    base = 1.0 + c
    if base <= 0.0:
        return None
    derivs: List[float] = []
    for k in range(order + 1):
        coef = 1.0
        for i in range(k):
            coef *= 0.5 - i
        derivs.append(coef * (base ** (0.5 - k)))
    return _compose_from_constant(A, derivs, order)


def series_cosh(A: Dict[int, float], order: int = 5) -> Dict[int, float]:
    # Compose using derivatives at c: cosh, sinh, cosh, sinh, ...
    c, _ = _split_const_and_tail(A)
    derivs: List[float] = []
    for k in range(order + 1):
        if k % 2 == 0:
            derivs.append(math.cosh(c))
        else:
            derivs.append(math.sinh(c))
    return _compose_from_constant(A, derivs, order)


def series_sinh(A: Dict[int, float], order: int = 5) -> Dict[int, float]:
    # Compose using derivatives at c: sinh, cosh, sinh, cosh, ...
    c, _ = _split_const_and_tail(A)
    derivs: List[float] = []
    for k in range(order + 1):
        if k % 2 == 0:
            derivs.append(math.sinh(c))
        else:
            derivs.append(math.cosh(c))
    return _compose_from_constant(A, derivs, order)


def series_inv(B: Dict[int, float], order: int) -> Optional[Dict[int, float]]:
    """Inverse of a power series with nonzero constant term up to `order`.

    Returns C such that (B * C) == 1 mod O(δ^{order+1}).
    """
    b0 = B.get(0, 0.0)
    if abs(b0) < 1e-15:
        return None
    C: Dict[int, float] = {0: 1.0 / b0}
    for k in range(1, order + 1):
        s = 0.0
        for i in range(1, k + 1):
            s += B.get(i, 0.0) * C.get(k - i, 0.0)
        C[k] = -s / b0
    return C


def series_tan(A: Dict[int, float], order: int = 5) -> Dict[int, float]:
    S = series_sin(A, order)
    C = series_cos(A, order)
    invC = series_inv(C, order)
    if invC is None:
        return {}
    return series_truncate(series_mul(S, invC), order)


def series_tanh(A: Dict[int, float], order: int = 5) -> Dict[int, float]:
    Sh = series_sinh(A, order)
    Ch = series_cosh(A, order)
    invCh = series_inv(Ch, order)
    if invCh is None:
        return {}
    return series_truncate(series_mul(Sh, invCh), order)


def series_standard_part(A: Dict[int, float]) -> Optional[float]:
    if any(k < 0 for k in A.keys()):
        return None
    return A.get(0, 0.0)


def is_near_standard_by_series(s: Seq, order: int = 10) -> Tuple[bool, Optional[float]]:
    A = series_from_seq(s, order)
    if A is None:
        return (False, None)
    c = series_standard_part(A)
    if c is None:
        return (False, None)
    return (True, c)


def series_from_seq_exact(s: Seq) -> Optional[Dict[int, float]]:
    """Exact Laurent polynomial in δ=1/n, when the expression admits one."""
    s = s.simplify()
    if isinstance(s, Const):
        return {0: s.c} if s.c != 0.0 else {}
    if isinstance(s, InvN):
        return {1: 1.0}
    if isinstance(s, NVar):
        return {-1: 1.0}
    if isinstance(s, Add):
        A = series_from_seq_exact(s.left)
        B = series_from_seq_exact(s.right)
        if A is None or B is None:
            return None
        return series_add(A, B)
    if isinstance(s, Sub):
        A = series_from_seq_exact(s.left)
        B = series_from_seq_exact(s.right)
        if A is None or B is None:
            return None
        return series_sub(A, B)
    if isinstance(s, Mul):
        A = series_from_seq_exact(s.left)
        B = series_from_seq_exact(s.right)
        if A is None or B is None:
            return None
        return series_mul(A, B)
    if isinstance(s, Div):
        A = series_from_seq_exact(s.left)
        B = series_from_seq_exact(s.right)
        if A is None or B is None:
            return None
        if len(B) != 1:
            return None
        (k, c) = next(iter(B.items()))
        if c == 0.0:
            return None
        return series_scale(series_shift(A, -k), 1.0 / c)
    return None


def eventual_sign_from_seq(
    s: Seq, *, order: int = 10, max_order: int = 30, tol: float = 0.0
) -> Optional[int]:
    """Return the eventual sign of `s(n)` for large n when provable from series.

    Returns:
      - `1` if `s(n) > 0` for all sufficiently large n
      - `-1` if `s(n) < 0` for all sufficiently large n
      - `0` if `s(n) == 0` for all n (exactly)
      - `None` if undetermined
    """

    def dominant_term_sign(A: Dict[int, float]) -> Optional[int]:
        if not A:
            return None
        exp = min(A.keys())
        c = A.get(exp, 0.0)
        if abs(c) <= tol:
            return None
        return 1 if c > 0.0 else -1

    exact = series_from_seq_exact(s)
    if exact is not None:
        if not exact:
            return 0
        return dominant_term_sign(exact)

    for k in range(order, max_order + 1, max(1, order // 2)):
        A = series_from_seq(s, k)
        if A is None:
            return None
        sign = dominant_term_sign(A)
        if sign is not None:
            return sign
    return None
