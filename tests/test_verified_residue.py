"""Arbitrary-period integration checks against the actual Lean executable."""

import json
import random
import subprocess
from fractions import Fraction
from math import lcm
from pathlib import Path

import pytest

from hyperreals import LeanResidueSystem

CHECKER = Path(__file__).resolve().parents[1] / ".lake/build/bin/residue_checker"
pytestmark = pytest.mark.skipif(
    not CHECKER.is_file(), reason="run lake build residue_checker"
)


def test_modulo_two_and_three_select_residue_five_modulo_six():
    system = LeanResidueSystem()
    assert system.support == (True,)
    assert system.period == 1
    modulo_two = system.periodic([0, 1])
    modulo_three = system.periodic([0, 1, 2])
    assert modulo_two == system.constant(1)
    assert system.support == (False, True)
    assert modulo_three == system.constant(2)
    assert system.support == (False, False, False, False, False, True)
    assert system.period == 6
    assert modulo_two.standard_part() == 1
    assert modulo_three.standard_part() == 2
    assert not system.commit(modulo_two, system.constant(0), "eq", truth=True)
    assert system.support == (False, False, False, False, False, True)


def test_new_periods_preserve_every_previous_commitment():
    system = LeanResidueSystem()
    expressions = [
        (system.periodic(range(period)), residue)
        for period, residue in [(3, 2), (5, 3), (7, 4)]
    ]
    for expression, residue in expressions:
        assert expression == system.constant(residue)
    assert system.period == 105
    assert [i for i, selected in enumerate(system.support) if selected] == [53]
    for expression, residue in expressions:
        assert expression.standard_part() == residue
        assert expression == system.constant(residue)
    assert system.period == 105


def test_shared_factors_reject_incompatible_residues_without_state_change():
    system = LeanResidueSystem()
    assert system.periodic(range(4)) == system.constant(1)
    before = system.support
    modulo_six = system.periodic(range(6))
    assert system.probe(modulo_six, system.constant(2), "eq") == (True, False)
    assert system.support == before
    assert not system.commit(modulo_six, system.constant(2), "eq", truth=True)
    assert system.support == before
    assert modulo_six == system.constant(3)
    assert system.period == 12
    assert [i for i, selected in enumerate(system.support) if selected] == [9]


def test_standard_part_checks_expression_period_beyond_current_state():
    system = LeanResidueSystem()
    period_three = system.periodic([10, 20, 30])
    assert period_three.standard_part() is None
    assert system.support == (True,)
    assert system.periodic([0, 1]) == system.constant(1)
    assert period_three.standard_part() is None
    assert system.support == (False, True)
    assert (
        period_three * system.infinitesimal() + system.constant(7)
    ).standard_part() == 7
    assert system.support == (False, True)
    assert period_three == system.constant(30)
    assert period_three.standard_part() == 30
    assert system.period == 6


def test_divergent_residues_can_be_eliminated_before_extraction():
    system = LeanResidueSystem()
    selector = system.periodic([0, 1, 1])
    expression = selector * system.infinite() + system.constant(Fraction(7, 3))
    assert expression.standard_part() is None
    assert system.support == (True,)
    assert selector == system.constant(0)
    assert expression.standard_part() == Fraction(7, 3)
    assert system.support == (True, False, False)


def test_exact_tables_and_existing_laurent_arithmetic():
    system = LeanResidueSystem()
    huge = 10**100
    table = system.periodic([Fraction(1, 3), huge, Fraction(-1, 7)])
    assert table == system.constant(Fraction(1, 3))
    assert table.standard_part() == Fraction(1, 3)
    assert (table * system.constant(3)).standard_part() == 1
    epsilon = system.infinitesimal()
    assert (system.infinite() ** 11 * epsilon**11).standard_part() == 1
    assert (table * epsilon**12).divide_monomial(1, -12).standard_part() == Fraction(
        1, 3
    )


def test_invalid_tables_and_mixed_contexts_are_rejected():
    first, second = LeanResidueSystem(), LeanResidueSystem()
    with pytest.raises(ValueError):
        first.periodic([])
    with pytest.raises((ValueError, OverflowError)):
        first.periodic([float("nan")])
    with pytest.raises(ValueError, match="same Lean system"):
        _ = first.periodic([1, 2, 3]) + second.constant(1)
    with pytest.raises(ZeroDivisionError):
        first.constant(1).divide_monomial(0, 2)
    assert first.support == (True,)


def _evaluate(ast, n):
    tag = ast[0]
    if tag == "const":
        return Fraction(int(ast[1]), int(ast[2]))
    if tag == "periodic":
        numerator, denominator = ast[1][n % len(ast[1])]
        return Fraction(int(numerator), int(denominator))
    if tag == "index":
        return Fraction(n)
    if tag == "invn":
        return Fraction(1, n)
    if tag == "divMonomial":
        return _evaluate(ast[1], n) / (
            Fraction(int(ast[2]), int(ast[3])) * Fraction(n) ** int(ast[4])
        )
    a, b = _evaluate(ast[1], n), _evaluate(ast[2], n)
    return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[tag]()


def _period(ast):
    if ast[0] == "periodic":
        return len(ast[1])
    if ast[0] in ("const", "index", "invn"):
        return 1
    if ast[0] == "divMonomial":
        return _period(ast[1])
    return lcm(_period(ast[1]), _period(ast[2]))


def _random_ast(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        values = [
            [str(rng.randint(-3, 3)), str(rng.randint(1, 7))]
            for _ in range(rng.randint(1, 5))
        ]
        return rng.choice(
            [["periodic", values], ["index"], ["invn"], ["const", "2", "3"]]
        )
    if rng.random() < 0.2:
        return [
            "divMonomial",
            _random_ast(rng, depth - 1),
            "-2",
            "3",
            str(rng.randint(-2, 2)),
        ]
    return [
        rng.choice(["add", "sub", "mul"]),
        _random_ast(rng, depth - 1),
        _random_ast(rng, depth - 1),
    ]


def test_direct_exact_evaluations_validate_each_emitted_residue_at_cutoff():
    rng = random.Random(7302026)
    requests = []
    for _ in range(20):
        left, right = _random_ast(rng, 2), _random_ast(rng, 2)
        support = [True, False, True]
        for op in ("lt", "eq"):
            requests.append(
                {
                    "support": support,
                    "op": op,
                    "left": left,
                    "right": right,
                    "choice": True,
                }
            )
    completed = subprocess.run(
        [str(CHECKER)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == len(requests)
    for request, response in zip(requests, responses):
        period = lcm(_period(request["left"]), _period(request["right"]))
        assert len(response["predicate"]) == period
        cutoff = int(response["cutoff"])
        assert cutoff >= 1
        for residue in range(period):
            n = cutoff + (residue - cutoff) % period
            for index in (n, n + period, n + 11 * period):
                a, b = _evaluate(request["left"], index), _evaluate(
                    request["right"], index
                )
                assert response["predicate"][residue] is (
                    a < b if request["op"] == "lt" else a == b
                )
        common_period = lcm(len(request["support"]), period)
        positive = [
            request["support"][r % 3] and response["predicate"][r % period]
            for r in range(common_period)
        ]
        negative = [
            request["support"][r % 3] and not response["predicate"][r % period]
            for r in range(common_period)
        ]
        assert response["trueSupport"] == positive
        assert response["falseSupport"] == negative
        assert response["accepted"] is any(positive)
        assert response["support"] == (positive if any(positive) else None)


def test_parser_rejects_empty_tables_and_invalid_states_then_recovers():
    valid = {
        "support": [True],
        "op": "standardPart",
        "left": ["periodic", [["2", "3"]]],
    }
    invalid = [
        dict(valid, support=[]),
        dict(valid, support=[False, False, False]),
        dict(valid, support=[1]),
        dict(valid, left=["periodic", []]),
        dict(valid, left=["periodic", [["1", "0"]]]),
        dict(valid, left=["periodic", [["1", "2", "3"]]]),
    ]
    completed = subprocess.run(
        [str(CHECKER)],
        input="".join(json.dumps(r) + "\n" for r in invalid + [valid]),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert all("error" in response for response in responses[:-1])
    assert responses[-1] == {"value": ["2", "3"], "support": [True]}
