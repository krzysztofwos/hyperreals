"""Check constructor capture against independent exact arithmetic and Lean replay."""

from fractions import Fraction
from pathlib import Path

import pytest

from hyperreals import (
    LeanResidueSystem,
    divided_difference,
    evaluate_polynomial,
    polynomial_quotient,
)
from scripts.benchmark_residues import evaluate

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".lake/build/bin/residue_checker"
requires_lean = pytest.mark.skipif(
    not CHECKER.is_file(), reason="run lake build residue_checker"
)


@pytest.fixture
def syntax_system(tmp_path):
    checker = tmp_path / "unused-checker"
    checker.touch()
    return LeanResidueSystem(checker_path=checker)


CASES = [
    ([], 2, 1, 1),
    ([7], -3, -2, 2),
    ([0, 1], 0, 3, 3),
    ([0, 0, 0, 1], 2, 1, 1),
    ([1, -3, 2, 0, -5], -2, -3, 2),
    ([Fraction(1, 3), 0, Fraction(-2, 7), 5], Fraction(2, 3), Fraction(-4, 5), 4),
]


def value(coefficients, x):
    return sum((Fraction(c) * x**k for k, c in enumerate(coefficients)), Fraction(0))


def derivative(coefficients, x):
    return sum(
        (k * Fraction(c) * x ** (k - 1) for k, c in enumerate(coefficients) if k),
        Fraction(0),
    )


@pytest.mark.parametrize("coefficients,a,c,order", CASES)
def test_literal_quotient_matches_independent_rational_evaluation(
    syntax_system, coefficients, a, c, order
):
    expression = polynomial_quotient(coefficients, a, c, order, system=syntax_system)
    assert expression._ast[0] == "divMonomial"
    assert expression._ast[1][0] == "sub"
    for n in (1, 2, 17, 1000):
        h = Fraction(c) / n**order
        expected = (
            value(coefficients, Fraction(a) + h) - value(coefficients, Fraction(a))
        ) / h
        assert evaluate(expression._ast, n) == expected


@pytest.mark.parametrize("coefficients,a,c,order", CASES)
@requires_lean
def test_extractor_returns_derivative_on_refined_support(coefficients, a, c, order):
    system = LeanResidueSystem()
    assert system.commit(
        system.periodic([0, 1, 0]), system.constant(1), "eq", truth=True
    )
    expression = polynomial_quotient(coefficients, a, c, order, system=system)
    assert expression.standard_part() == derivative(coefficients, Fraction(a))
    assert system.support == (False, True, False)


def test_divided_difference_identity_including_zero_increment(syntax_system):
    system = syntax_system
    coefficients = [3, -2, 0, 1, Fraction(1, 7)]
    a = Fraction(2, 3)
    h = system.periodic([0, 1, -2]) * (
        system.infinitesimal() + system.infinitesimal() ** 2
    )
    result = divided_difference(coefficients, a, h)
    p = evaluate_polynomial(coefficients, system.constant(a) + h)
    for n in range(1, 13):
        step = evaluate(h._ast, n)
        output = evaluate(result._ast, n)
        assert evaluate(p._ast, n) == value(coefficients, a + step)
        assert step * output == value(coefficients, a + step) - value(coefficients, a)
        if step == 0:
            assert output == derivative(coefficients, a)
        else:
            assert (
                output
                == (value(coefficients, a + step) - value(coefficients, a)) / step
            )


@pytest.mark.parametrize("order", [0, -1, True, 1.5])
def test_invalid_increment_order_rejected(syntax_system, order):
    with pytest.raises(ValueError, match="positive integer"):
        polynomial_quotient([0, 1], 0, order=order, system=syntax_system)


def test_zero_scale_rejected(syntax_system):
    with pytest.raises(ZeroDivisionError, match="nonzero"):
        polynomial_quotient([0, 1], 0, c=0, system=syntax_system)


@requires_lean
@pytest.mark.skipif(
    not (ROOT / ".lake/build/lib/lean/Hyperreals/ResidueReplay.olean").is_file(),
    reason="build Lean replay",
)
def test_nonmonomial_divided_difference_passes_kernel_replay():
    system = LeanResidueSystem()
    h = system.infinitesimal() + system.infinitesimal() ** 2
    result = divided_difference([1, -2, 0, 1], 2, h)
    snapshot = system.snapshot(result)
    assert snapshot.result == 10
    checked = snapshot.verify(project_root=ROOT, timeout=120)
    assert set(checked.axioms) <= {"propext", "Classical.choice", "Quot.sound"}


@requires_lean
def test_observation_establishes_infinitesimality_for_divided_difference():
    system = LeanResidueSystem()
    h = system.periodic([1, 0]) + system.periodic([0, 1]) * system.infinitesimal()
    result = divided_difference([0, 0, 0, 1], 2, h)
    assert h.standard_part() is None
    assert result.explain_standard_part().limits == (19, 12)
    assert system.commit(h, system.constant(Fraction(1, 2)), "lt", truth=True)
    assert h.standard_part() == 0
    assert result.standard_part() == 12
    assert system.probe(h, system.constant(0), "eq") == (True, False)
