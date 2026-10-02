"""Integration checks of the actual Lean executable, not a Python mock."""

from fractions import Fraction
import itertools
import json
from pathlib import Path
import subprocess

import pytest

from hyperreals import LeanBackendError, LeanPeriodicSystem


CHECKER = Path(__file__).resolve().parents[1] / ".lake/build/bin/periodic_checker"
requires_lean = pytest.mark.skipif(not CHECKER.is_file(), reason="run lake build periodic_checker")


def test_missing_checker_does_not_fall_back_to_python(tmp_path):
    with pytest.raises(LeanBackendError, match="not found"):
        LeanPeriodicSystem(checker_path=tmp_path / "absent")


@requires_lean
def test_lean_choices_share_one_completion_and_rejection_preserves_state():
    system = LeanPeriodicSystem()
    x, zero, one = system.alt(), system.constant(0), system.constant(1)
    assert system.probe(x, zero) == (True, True)
    assert system.support == (True, True)
    assert x < zero
    assert system.support == (False, True)
    assert not system.commit(x, one, "eq", truth=True)
    assert system.support == (False, True)
    assert not (x == one)
    assert x == system.constant(-1)


@requires_lean
def test_lean_query_order_can_select_the_other_completion():
    system = LeanPeriodicSystem()
    x = system.alt()
    assert x == system.constant(1)
    assert system.support == (True, False)
    assert not (x < system.constant(0))


@requires_lean
def test_lean_arithmetic_avoids_float_rounding():
    system = LeanPeriodicSystem()
    a = system.constant(2**53)
    assert a < a + system.constant(1)
    assert not (a == a + system.constant(1))
    third = system.constant(Fraction(1, 3))
    assert third * system.constant(3) == system.constant(1)
    x = system.alt()
    assert x * x == system.constant(1)
    assert system.support == (True, True)


@requires_lean
def test_lean_contexts_cannot_be_mixed():
    first, second = LeanPeriodicSystem(), LeanPeriodicSystem()
    with pytest.raises(ValueError, match="same Lean system"):
        _ = first.alt() + second.alt()


@requires_lean
def test_exhaustive_finite_protocol_cases_match_rational_semantics():
    expressions = [
        (["alt"], (Fraction(1), Fraction(-1))),
        (["const", "0", "1"], (Fraction(0), Fraction(0))),
        (["add", ["alt"], ["const", "1", "3"]], (Fraction(4, 3), Fraction(-2, 3))),
        (["mul", ["alt"], ["alt"]], (Fraction(1), Fraction(1))),
    ]
    requests, expectations = [], []
    for support, op, (left, lv), (right, rv), choice in itertools.product(
        [(True, True), (True, False), (False, True)],
        ["lt", "eq"], expressions, expressions, [True, False],
    ):
        predicate = [a < b if op == "lt" else a == b for a, b in zip(lv, rv)]
        selected = [s and (p == choice) for s, p in zip(support, predicate)]
        requests.append({"support": support, "op": op, "left": left, "right": right, "choice": choice})
        expectations.append((predicate, selected))
    completed = subprocess.run(
        [str(CHECKER)], input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True, text=True, check=True, timeout=30,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == len(expectations)
    for response, (predicate, selected) in zip(responses, expectations):
        assert response["predicate"] == predicate
        assert response["accepted"] is any(selected)
        assert response["support"] == (selected if any(selected) else None)


@requires_lean
@pytest.mark.parametrize("bad_constant", [["const", "1", "0"], ["const", "1", "-2"], ["const", "nan", "1"]])
def test_lean_parser_rejects_invalid_rationals(bad_constant):
    request = {"support": [True, True], "op": "lt", "left": bad_constant, "right": ["alt"]}
    completed = subprocess.run(
        [str(CHECKER)], input=json.dumps(request) + "\n", capture_output=True,
        text=True, check=True, timeout=30,
    )
    assert "error" in json.loads(completed.stdout)
