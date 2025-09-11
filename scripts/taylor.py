#!/usr/bin/env python3

"""
Taylor series via Hyperreals (ε = 1/n)
--------------------------------------

Demonstrates how to compute many derivatives of a function at once using the
hyperreal engine. We evaluate f(x + ε) and read off the coefficients of powers
of ε from the internal Laurent series in δ=1/n (here ε≡δ).

For analytic functions supported by the series engine (sin, cos, tan, tanh,
exp, log1p, sqrt1p) the implementation computes coefficients up to a chosen
order (default: 10).

Usage examples:
  # Default demo suite at several points
  uv run python scripts/taylor.py

  # Single function at a point with order 4
  uv run python scripts/taylor.py --func poly3 --x 2.0 --order 4

  # Multiple functions
  uv run python scripts/taylor.py --func exp --func sin --x 0.5 --order 5

"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyperreal import Hyperreal, HyperrealSystem, series_from_seq

Func = Callable[[Hyperreal, HyperrealSystem], Hyperreal]


def compute_taylor_coefficients(f: Func, x: float, max_order: int = 10) -> List[float]:
    """Return [f(x), f'(x), f''(x), ...] up to `max_order`.

    Implementation detail: builds f(x+ε) and extracts coefficients of ε^k using
    the library's series engine (δ=1/n). The coefficient of ε^k equals
    f^{(k)}(x)/k!, so we multiply by k! to get derivatives.

    Note: Order is configurable; higher orders increase compute cost roughly
    quadratically due to series convolutions.
    """
    sys = HyperrealSystem()
    eps = sys.infinitesimal()
    x_hr = sys.constant(float(x))
    # Evaluate once; this embeds all derivatives in the ε-series
    fx_eps = f(x_hr + eps, sys)

    series: Dict[int, float] | None = series_from_seq(fx_eps.seq, max_order)
    if series is None:
        # Fall back to empty series; caller sees zeros
        series = {}

    # Build derivatives list: d^k f(x) = coeff(ε^k) * k!
    out: List[float] = []
    for k in range(0, max_order + 1):
        coeff = series.get(k, 0.0)
        out.append(coeff * math.factorial(k))
    return out


# Demo functions --------------------------------------------------------------


def f_poly3(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return u * u * u


def f_poly3_minus_3u(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    three = sys.constant(3.0)
    return u * u * u - three * u


def f_exp(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return sys.exp(u)


def f_sin(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return sys.sin(u)


def f_cos(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return sys.cos(u)


def f_log1p(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return sys.log1p(u)


def f_sqrt1p(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return sys.sqrt1p(u)


def f_tan(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return sys.tan(u)


def f_tanh(u: Hyperreal, sys: HyperrealSystem) -> Hyperreal:
    return sys.tanh(u)


FUNCS: Dict[str, Tuple[str, Func, float]] = {
    # name: (label, function, default_x)
    "poly3": ("f(x)=x^3", f_poly3, 2.0),
    "poly3m3x": ("f(x)=x^3-3x", f_poly3_minus_3u, 2.0),
    "exp": ("f(x)=exp(x)", f_exp, 0.7),
    "sin": ("f(x)=sin(x)", f_sin, 0.7),
    "cos": ("f(x)=cos(x)", f_cos, 0.7),
    "log1p": ("f(x)=log(1+x)", f_log1p, 0.0),
    "sqrt1p": ("f(x)=sqrt(1+x)", f_sqrt1p, 0.0),
    "tan": ("f(x)=tan(x)", f_tan, 0.1),
    "tanh": ("f(x)=tanh(x)", f_tanh, 0.3),
}


def run_one(name: str, x: float | None, order: int) -> None:
    if name not in FUNCS:
        print(f"Unknown function '{name}'. Choices: {', '.join(sorted(FUNCS))}")
        return
    label, fn, default_x = FUNCS[name]
    xi = default_x if x is None else x
    derivs = compute_taylor_coefficients(fn, xi, max_order=order)
    # Pretty print
    print("=" * 70)
    print(f"{label}  at x={xi}  (order {order})")
    print("-" * 70)
    for k, d in enumerate(derivs):
        suffix = "(~0)" if (k > 5 and abs(d) < 1e-12) else ""
        print(f"  d^{k} f(x) = {d}{' ' + suffix if suffix else ''}")


def demo_suite(order: int) -> None:
    # A compact set showcasing polynomial and analytic cases
    sequences: Iterable[Tuple[str, float | None]] = [
        ("poly3", 2.0),
        ("poly3m3x", 2.0),
        ("exp", 0.0),
        ("sin", 0.0),
        ("cos", 0.0),
        ("log1p", 0.0),
        ("sqrt1p", 0.0),
    ]
    for name, xv in sequences:
        run_one(name, xv, order)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute many derivatives at once via Hyperreals (ε-series)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--func", "-f", action="append", help="Function name (repeatable)")
    p.add_argument(
        "--x", type=float, help="Point of evaluation for all selected functions"
    )
    p.add_argument(
        "--order",
        type=int,
        default=10,
        help="Max derivative order to report",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv)
    order = max(0, ns.order)
    if not ns.func:
        demo_suite(order)
        return 0
    for name in ns.func:
        run_one(name, ns.x, order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
