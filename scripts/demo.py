#!/usr/bin/env python3

"""Demo of the hyperreal number system."""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyperreals import Dual, HyperrealSystem, ad_derivative_first  # noqa: E402


def f_cubic(hr, sys):
    """Cubic function f(x) = x^3."""
    return hr * hr * hr


def g_poly(hr, sys):
    """Polynomial g(x) = x^3 - 3x."""
    three = sys.constant(3.0)
    return hr * hr * hr - three * hr


def demo_and_tests(
    save_ultrafilter=None,
    clause_limit: int = 10,
    var_limit: int = 30,
    committed_limit: int = 20,
    partitions_limit: int = 20,
):
    print("=" * 70)
    print("Hyperreal + Lazy Ultrafilter Demo")
    print("=" * 70)

    # Setup
    sys = HyperrealSystem()
    zero = sys.constant(0.0)
    one = sys.constant(1.0)
    two = sys.constant(2.0)
    eps = sys.infinitesimal()
    omega = sys.infinite()

    print("\n### Category 1: Basic Sanity Checks ###")
    print("--- 1.1: Infinitesimal ordering ---")
    print("ε < 1e-6 ? ", eps < sys.constant(1e-6))
    print("ε < 1e-12? ", eps < sys.constant(1e-12))

    print("\n--- 1.2: Orders of infinitesimals ---")
    eps2 = eps * eps
    print("ε² < ε ?  ", eps2 < eps)
    print("st(ε²)   = ", eps2.standard_part())

    print("\n--- 1.3: Indeterminate form ω·ε resolved ---")
    omega_eps = omega * eps
    print("ω·ε       =", omega_eps)
    print("st(ω·ε)   =", omega_eps.standard_part())

    print("\n--- 1.4: (ω+1) - ω ---")
    result = (omega + one) - omega
    print("(ω+1)-ω   =", result, "  st =", result.standard_part())

    print("\n### Category 2: Calculus & Limits ###")
    print("--- 2.1: Derivative of x^3 at x=2 ---")
    x = two
    f_x_plus_dx = f_cubic(x + eps, sys)
    f_x = f_cubic(x, sys)
    derivative_hr = (f_x_plus_dx - f_x) / eps
    print("st((f(2+ε)-f(2))/ε) =", derivative_hr.standard_part(), " (Expected: 12.0)")

    print("\n--- 2.2: Limit identities with new functions ---")
    print(
        "st((exp(ε)-1)/ε)      =",
        ((sys.exp(eps) - one) / eps).standard_part(),
        " ~ 1.0",
    )
    print("st(log(1+ε)/ε)        =", (sys.log1p(eps) / eps).standard_part(), " ~ 1.0")
    print(
        "st((sqrt(1+ε)-1)/ε)   =",
        ((sys.sqrt1p(eps) - one) / eps).standard_part(),
        " ~ 0.5",
    )
    print("st(cos(ε))            =", (sys.cos(eps)).standard_part(), " ~ 1.0")
    print("st(tan(ε)/ε)          =", (sys.tan(eps) / eps).standard_part(), " ~ 1.0")
    print("st(tanh(ε)/ε)         =", (sys.tanh(eps) / eps).standard_part(), " ~ 1.0")

    print("\n--- 2.3: Second derivative via hyperreals ---")
    # f''(x) for g(x)=x^3 - 3x equals 6x, so at x=2 it's 12
    g_x = g_poly(two, sys)
    g_xp = g_poly(two + eps, sys)
    g_xm = g_poly(two - eps, sys)
    numerator = g_xp - two * g_x + g_xm
    denom = eps * eps
    second_derivative_hr = numerator / denom
    print(
        "st([g(x+ε) - 2g(x) + g(x-ε)]/ε²) =",
        second_derivative_hr.standard_part(),
        " (Expected: 12.0)",
    )

    print("\n### Category 3: Choice-Dependence ###")
    alt = sys.alt()
    print("Is (-1)^n > 0 ? ", alt > zero)
    # Re-ask to ensure consistency
    print("Re-ask 0 < (-1)^n:", zero < alt)
    print("\n--- 3.1: Forced (finite/empty) equalities ---")
    print("sin(ω) = 0 ?        ", sys.sin(omega) == zero)
    print("sin(ω) = 0.5 ?      ", sys.sin(omega) == sys.constant(0.5))
    print("sin(ω) = cos(ω) ?   ", sys.sin(omega) == sys.cos(omega))
    print("cos(ω) = cos(-ω) ?  ", sys.cos(omega) == sys.cos(sys.constant(-1.0) * omega))

    print("\n### Category 4: AD (Dual Numbers) vs Hyperreals ###")

    def f_dual(u: Dual) -> Dual:
        return u * u * u

    ad1 = ad_derivative_first(f_dual, 2.0)
    print("AD (dual) first derivative of x^3 at 2:", ad1)
    print("Hyperreal (st)                        :", derivative_hr.standard_part())

    print("\n### Numeric Peek (n=100) for intuition only ###")
    try:
        seq_delta = f_cubic(two + eps, sys) - f_cubic(two, sys)
        val_delta = seq_delta.value_at(100)
        val_eps = eps.value_at(100)
        print(f"f(2+ε)-f(2) at n=100: {val_delta}")
        print(f"ε at n=100: {val_eps}   quotient ~ {val_delta/val_eps}")
    except Exception as e:
        print("Numeric peek failed:", e)

    print("\n### Ultrafilter Constraints (partial view) ###")
    print(
        sys.puf.human_readable_constraints(
            clause_limit=clause_limit,
            var_limit=var_limit,
            committed_limit=committed_limit,
            partitions_limit=partitions_limit,
        )
    )

    # Optional: save ultrafilter state
    if save_ultrafilter:
        sys.puf.save(save_ultrafilter)
        print(f"\nSaved ultrafilter to: {Path(save_ultrafilter).absolute()}")


def main():
    """Entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="Demonstration of the hyperreal number system"
    )
    parser.add_argument(
        "--save-ultrafilter",
        type=str,
        metavar="FILE",
        help="Save ultrafilter state to JSON file",
    )
    parser.add_argument(
        "--clause-limit",
        type=int,
        default=10,
        metavar="N",
        help="Number of clauses to show (0 for all, default: 10)",
    )
    parser.add_argument(
        "--var-limit",
        type=int,
        default=30,
        metavar="N",
        help="Number of variables to show (0 for all, default: 30)",
    )
    parser.add_argument(
        "--committed-limit",
        type=int,
        default=20,
        metavar="N",
        help="Number of committed-true sets to show (0 for all, default: 20)",
    )
    parser.add_argument(
        "--partitions-limit",
        type=int,
        default=20,
        metavar="N",
        help="Number of installed partitions to show (0 for all, default: 20)",
    )
    args = parser.parse_args()

    demo_and_tests(
        save_ultrafilter=args.save_ultrafilter,
        clause_limit=args.clause_limit,
        var_limit=args.var_limit,
        committed_limit=args.committed_limit,
        partitions_limit=args.partitions_limit,
    )


if __name__ == "__main__":
    main()
