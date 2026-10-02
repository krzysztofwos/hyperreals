#!/usr/bin/env python3
"""Run exact arithmetic through the proved Lean Laurent kernel.

Build first with ``lake build laurent_checker``, then run
``uv run python scripts/verified_demo.py``.
"""

from fractions import Fraction
from pathlib import Path
import sys


_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyperreals import LeanLaurentSystem  # noqa: E402


def main() -> None:
    system = LeanLaurentSystem()
    epsilon, n, alt = system.infinitesimal(), system.infinite(), system.alt()
    zero, one, two, three = (system.constant(i) for i in (0, 1, 2, 3))

    print("Exact Laurent arithmetic evaluated by Lean")
    print(f"0 < epsilon: {zero < epsilon}")
    print(f"epsilon < 1/1,000,000: {epsilon < system.constant(Fraction(1, 10**6))}")
    print(f"The latest comparison holds for every n >= {system.last_cutoff}.")
    print(f"n * epsilon = 1: {n * epsilon == one}")
    print(f"standard_part(n^11 * epsilon^11): {((n**11) * (epsilon**11)).standard_part()}")

    x = two
    derivative = (((x + epsilon) ** 3 - three * (x + epsilon)) - (x**3 - three * x)) / epsilon
    print(f"For f(x) = x^3 - 3x, standard_part((f(2+epsilon)-f(2))/epsilon): {derivative.standard_part()}")

    print(f"standard_part((-1)^n) before choosing: {alt.standard_part()}")
    print(f"Both answers to (-1)^n < 0 are feasible: {system.probe(alt, zero)}")
    print(f"Choose (-1)^n < 0: {system.commit(alt, zero, truth=True)}")
    print(f"Remaining (even, odd) support: {system.support}")
    print(f"standard_part((-1)^n) after choosing: {alt.standard_part()}")
    print(f"Conflicting choice (-1)^n = 1 accepted: {system.commit(alt, one, 'eq', truth=True)}")
    print(f"Support after rejection: {system.support}")
    print("The expression and completion theorems are checked in Lean. Python/JSON transport remains tested code.")


if __name__ == "__main__":
    main()
