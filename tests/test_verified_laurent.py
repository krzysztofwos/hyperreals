"""Mathematical boundary checks against the actual Lean Laurent executable.

The direct Fraction evaluations below are finite regression checks of transport
and execution. The corresponding all-indices tail statements are Lean theorems.
"""

from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess

import pytest

from hyperreals import LeanBackendError, LeanLaurentSystem


CHECKER = Path(__file__).resolve().parents[1] / ".lake/build/bin/laurent_checker"
requires_lean = pytest.mark.skipif(
    not CHECKER.is_file(), reason="run lake build laurent_checker"
)


def test_missing_laurent_checker_does_not_fall_back_to_analyzer(tmp_path):
    with pytest.raises(LeanBackendError, match="not found"):
        LeanLaurentSystem(checker_path=tmp_path / "absent")


@requires_lean
def test_infinitesimal_arithmetic_and_finite_difference_derivative():
    system = LeanLaurentSystem()
    epsilon, n = system.infinitesimal(), system.infinite()
    zero, one, two, three = (system.constant(i) for i in (0, 1, 2, 3))
    assert zero < epsilon
    assert epsilon < system.constant(Fraction(1, 10**40))
    assert system.last_cutoff is not None and system.last_cutoff > 10**40
    assert n * epsilon == one
    assert epsilon.standard_part() == 0
    assert n.standard_part() is None
    derivative = (((two + epsilon) ** 3 - three * (two + epsilon)) - (two**3 - three * two)) / epsilon
    assert derivative.standard_part() == Fraction(9)
    assert system.support == (True, True)


@requires_lean
def test_high_order_terms_survive_later_shifts():
    system = LeanLaurentSystem()
    epsilon, n = system.infinitesimal(), system.infinite()
    assert ((n**11) * (epsilon**11)).standard_part() == 1
    tiny = system.constant(7) * epsilon**12
    assert tiny.standard_part() == 0
    assert tiny.divide_monomial(1, -12).standard_part() == 7
    assert (n**12 - n**12 + system.constant(Fraction(2, 5))).standard_part() == Fraction(2, 5)


@requires_lean
@pytest.mark.parametrize("power", [-4, -1, 0, 3])
def test_division_by_signed_rational_monomials(power):
    system = LeanLaurentSystem()
    coefficient = Fraction(-7, 13)
    base = system.infinite() if power >= 0 else system.infinitesimal()
    numerator = system.constant(coefficient) * base ** abs(power)
    assert numerator.divide_monomial(coefficient, power).standard_part() == 1
    assert (system.constant(Fraction(3, 7)) / system.constant(-2)).standard_part() == Fraction(-3, 14)


@requires_lean
def test_large_constants_and_small_coefficients_remain_exact():
    system = LeanLaurentSystem()
    huge = 10**400
    a, one = system.constant(huge), system.constant(1)
    assert a < a + one
    assert (a + one - a).standard_part() == 1
    assert (system.constant(Fraction(1, 10**400)) * a).standard_part() == 1
    assert a < system.infinite()
    assert system.last_cutoff is not None and system.last_cutoff > huge
    assert system.support == (True, True)


@requires_lean
def test_standard_part_uses_remaining_support_without_choosing():
    system = LeanLaurentSystem()
    alt, zero, one = system.alt(), system.constant(0), system.constant(1)
    assert alt.standard_part() is None
    assert (one + alt * system.infinitesimal()).standard_part() == 1
    assert system.support == (True, True)
    assert system.probe(alt, zero) == (True, True)
    assert system.support == (True, True)
    assert alt < zero
    assert alt.standard_part() == -1
    assert system.support == (False, True)
    assert not system.commit(alt, one, "eq", truth=True)
    assert system.support == (False, True)


@requires_lean
def test_eliminating_a_divergent_parity_enables_standard_part():
    system = LeanLaurentSystem()
    alt, one, three = system.alt(), system.constant(1), system.constant(3)
    even = (one + alt) / system.constant(2)
    expression = even * system.infinite() + (one - even) * three
    assert expression.standard_part() is None
    assert system.support == (True, True)
    assert alt < system.constant(0)
    assert expression.standard_part() == 3
    assert system.support == (False, True)

    other = LeanLaurentSystem()
    even = (other.constant(1) + other.alt()) / other.constant(2)
    expression = even * other.infinite() + (other.constant(1) - even) * other.constant(3)
    assert other.alt() == other.constant(1)
    assert expression.standard_part() is None
    assert other.support == (True, False)


@requires_lean
def test_domain_errors_and_impossible_choices_preserve_support():
    system = LeanLaurentSystem()
    one, zero, epsilon = system.constant(1), system.constant(0), system.infinitesimal()
    with pytest.raises(ZeroDivisionError):
        _ = one / zero
    with pytest.raises(ZeroDivisionError):
        one.divide_monomial(0, -3)
    with pytest.raises(ValueError, match="monomial"):
        _ = one / (one + epsilon)
    for power in (-1, 1.5, True):
        with pytest.raises(ValueError, match="nonnegative integers"):
            _ = epsilon**power
    assert not system.commit(epsilon, zero, truth=True)
    assert system.support == (True, True)
    assert epsilon.standard_part() == 0


def _evaluate(ast, n):
    """Evaluate source syntax directly, without normalization or asymptotic rules."""
    tag = ast[0]
    if tag == "const":
        return Fraction(int(ast[1]), int(ast[2]))
    if tag == "index":
        return Fraction(n)
    if tag == "invn":
        return Fraction(1, n)
    if tag == "alt":
        return Fraction((-1) ** n)
    if tag == "divMonomial":
        divisor = Fraction(int(ast[2]), int(ast[3])) * Fraction(n) ** int(ast[4])
        return _evaluate(ast[1], n) / divisor
    left, right = _evaluate(ast[1], n), _evaluate(ast[2], n)
    if tag == "add":
        return left + right
    if tag == "sub":
        return left - right
    if tag == "mul":
        return left * right
    raise AssertionError(f"unexpected expression tag: {tag}")


def _random_ast(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([
            ["index"], ["invn"], ["alt"],
            ["const", str(rng.randint(-7, 7)), str(rng.randint(1, 13))],
        ])
    if rng.random() < 0.25:
        coefficient = rng.choice([-7, -2, 1, 3])
        return [
            "divMonomial", _random_ast(rng, depth - 1),
            str(coefficient), "5", str(rng.randint(-3, 3)),
        ]
    return [
        rng.choice(["add", "sub", "mul"]),
        _random_ast(rng, depth - 1), _random_ast(rng, depth - 1),
    ]


@requires_lean
def test_emitted_cutoffs_cover_direct_exact_source_evaluations():
    rng = random.Random(20261001)
    pairs = [(_random_ast(rng, 3), _random_ast(rng, 3)) for _ in range(30)]
    pairs += [(ast, ast) for ast, _ in pairs[:5]]
    pairs += [
        (["index"], ["const", "3", "1"]),
        (["invn"], ["const", "1", str(10**30)]),
        (["mul", ["alt"], ["index"]], ["const", "0", "1"]),
    ]
    requests = [
        {"support": [True, True], "op": op, "left": left, "right": right}
        for left, right in pairs for op in ("lt", "eq")
    ]
    completed = subprocess.run(
        [str(CHECKER)], input="".join(json.dumps(r) + "\n" for r in requests),
        text=True, capture_output=True, check=True, timeout=30,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == len(requests)
    for request, response in zip(requests, responses):
        cutoff = int(response["cutoff"])
        assert cutoff >= 1
        assert response["accepted"] is None and response["support"] is None
        for n in (cutoff, cutoff + 1, cutoff + 2, 2 * cutoff + 3, 10 * cutoff + 4):
            left, right = _evaluate(request["left"], n), _evaluate(request["right"], n)
            expected = left < right if request["op"] == "lt" else left == right
            assert response["predicate"][n % 2] is expected, (request, response, n)


@requires_lean
def test_invalid_protocol_lines_are_rejected_without_poisoning_following_requests():
    valid = {"support": [True, True], "op": "standardPart", "left": ["invn"]}
    invalid = [
        dict(valid, support=[False, False]),
        dict(valid, support=[True, 1]),
        dict(valid, left=["const", "1", "0"]),
        dict(valid, left=["divMonomial", ["index"], "0", "1", "1"]),
        dict(valid, left=["divMonomial", ["index"], "1", "1", "1.5"]),
        dict(valid, left=["index", "unexpected"]),
        dict(valid, left=["exp", ["invn"]]),
        {"support": [True, True], "op": "lt", "left": ["invn"],
         "right": ["const", "0", "1"], "choice": "true"},
    ]
    lines = ["{not json}"] + [json.dumps(item) for item in invalid] + [json.dumps(valid)]
    completed = subprocess.run(
        [str(CHECKER)], input="\n".join(lines) + "\n", text=True,
        capture_output=True, check=True, timeout=30,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == len(lines)
    assert all("error" in response for response in responses[:-1])
    assert responses[-1] == {"value": ["0", "1"], "support": [True, True]}
