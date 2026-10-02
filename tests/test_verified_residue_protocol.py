"""Protocol rejection and rollback tests for the arbitrary-period adapter.

Mocked subprocess responses exercise the unproved Python transport boundary.
The native parser test separately sends malformed input to the actual checker.
"""

from fractions import Fraction
import json
import operator
from pathlib import Path
import subprocess

import pytest

from hyperreals import LeanBackendError, LeanResidueSystem


CHECKER = Path(__file__).resolve().parents[1] / ".lake/build/bin/residue_checker"
requires_lean = pytest.mark.skipif(not CHECKER.is_file(), reason="run lake build residue_checker")


def _comparison_response():
    """A valid refinement from odd indices to residues 3 and 5 modulo 6."""
    positive = [False, False, False, True, False, True]
    return {
        "predicate": [True, False, True],
        "trueSupport": positive,
        "falseSupport": [False, True, False, False, False, False],
        "canBeTrue": True,
        "canBeFalse": True,
        "accepted": True,
        "support": positive,
        "cutoff": "23",
    }


@pytest.fixture
def primed_transport(tmp_path, monkeypatch):
    checker = tmp_path / "checker"
    checker.touch()
    system = LeanResidueSystem(checker_path=checker)
    wire = {"returncode": 0, "stdout": "", "stderr": ""}

    def exchange(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], **wire)

    monkeypatch.setattr("hyperreals.verified_residue.subprocess.run", exchange)
    wire["stdout"] = json.dumps({
        "predicate": [False, True], "trueSupport": [False, True],
        "falseSupport": [True, False], "canBeTrue": True, "canBeFalse": True,
        "accepted": True, "support": [False, True], "cutoff": "17",
    })
    assert system.commit(system.constant(0), system.periodic([0, 1]), truth=True)
    assert (system.support, system.last_cutoff) == ((False, True), 17)
    return system, wire


def test_missing_residue_checker_has_no_python_fallback(tmp_path):
    with pytest.raises(LeanBackendError, match="not found"):
        LeanResidueSystem(checker_path=tmp_path / "absent")


@pytest.mark.parametrize("key,value", [
    ("predicate", []),
    ("predicate", [True, 1, True]),
    ("trueSupport", [True, False, False, True, False, True]),
    ("falseSupport", [False, True]),
    ("canBeTrue", 1),
    ("canBeFalse", False),
    ("accepted", 1),
    ("accepted", False),
    ("support", [False, True]),
    ("cutoff", 23),
    ("cutoff", "0"),
    ("cutoff", "-1"),
    ("cutoff", "1.5"),
    ("cutoff", " 23"),
    ("cutoff", "٢٣"),
])
def test_malformed_commit_preserves_support_and_cutoff(primed_transport, key, value):
    system, wire = primed_transport
    response = _comparison_response()
    response[key] = value
    wire["stdout"] = json.dumps(response)
    with pytest.raises(LeanBackendError):
        system.commit(system.constant(0), system.periodic([1, 0, 1]), truth=True)
    assert (system.support, system.last_cutoff) == ((False, True), 17)


@pytest.mark.parametrize("key", ["predicate", "accepted", "support", "cutoff"])
def test_missing_comparison_fields_preserve_state(primed_transport, key):
    system, wire = primed_transport
    response = _comparison_response()
    del response[key]
    wire["stdout"] = json.dumps(response)
    with pytest.raises(LeanBackendError):
        system.commit(system.constant(0), system.periodic([1, 0, 1]), truth=True)
    assert (system.support, system.last_cutoff) == ((False, True), 17)


def test_probe_rejects_a_response_that_commits(primed_transport):
    system, wire = primed_transport
    wire["stdout"] = json.dumps(_comparison_response())
    with pytest.raises(LeanBackendError, match="probe returned a commitment"):
        system.probe(system.constant(0), system.periodic([1, 0, 1]))
    assert (system.support, system.last_cutoff) == ((False, True), 17)


def test_rejected_commit_cannot_return_a_state(primed_transport):
    system, wire = primed_transport
    wire["stdout"] = json.dumps({
        "predicate": [True], "trueSupport": [False, True],
        "falseSupport": [False, False], "canBeTrue": True, "canBeFalse": False,
        "accepted": False, "support": [False, True], "cutoff": "23",
    })
    with pytest.raises(LeanBackendError, match="rejected commit returned a state"):
        system.commit(system.constant(0), system.constant(1), truth=False)
    assert (system.support, system.last_cutoff) == ((False, True), 17)


@pytest.mark.parametrize("value", [[], ["1"], [1, 2], ["1", "0"], ["1", "-2"],
                                   ["nan", "1"], ["1", "2.0"]])
def test_invalid_standard_part_response_preserves_state(primed_transport, value):
    system, wire = primed_transport
    wire["stdout"] = json.dumps({"support": [False, True], "value": value})
    with pytest.raises(LeanBackendError):
        system.constant(1).standard_part()
    assert (system.support, system.last_cutoff) == ((False, True), 17)


def test_standard_part_cannot_refine_or_expand_support(primed_transport):
    system, wire = primed_transport
    wire["stdout"] = json.dumps({"support": [False, True, False, True], "value": ["1", "1"]})
    with pytest.raises(LeanBackendError, match="extraction changed support"):
        system.constant(1).standard_part()
    assert (system.support, system.last_cutoff) == ((False, True), 17)


@pytest.mark.parametrize("stdout", ["not JSON", "[]", "null", '{"error":"invalid AST"}',
                                    '{}\n{}'])
def test_transport_rejects_malformed_envelopes_without_state_change(primed_transport, stdout):
    system, wire = primed_transport
    wire["stdout"] = stdout
    with pytest.raises(LeanBackendError):
        system.commit(system.constant(0), system.periodic([1, 0, 1]), truth=True)
    assert (system.support, system.last_cutoff) == ((False, True), 17)


@pytest.mark.parametrize("failure", ["exit", "timeout", "oserror"])
def test_native_process_failures_preserve_state(primed_transport, monkeypatch, failure):
    system, wire = primed_transport
    if failure == "exit":
        wire.update(returncode=1, stderr="checker failed")
    else:
        def fail(*args, **kwargs):
            if failure == "timeout":
                raise subprocess.TimeoutExpired("checker", 0.01)
            raise OSError("checker unavailable")
        monkeypatch.setattr("hyperreals.verified_residue.subprocess.run", fail)
    with pytest.raises(LeanBackendError, match="checker failed"):
        system.commit(system.constant(0), system.periodic([1, 0, 1]), truth=True)
    assert (system.support, system.last_cutoff) == ((False, True), 17)


def test_successful_probe_and_extraction_keep_original_support(primed_transport):
    system, wire = primed_transport
    response = _comparison_response()
    response.update(accepted=None, support=None)
    wire["stdout"] = json.dumps(response)
    assert system.probe(system.constant(0), system.periodic([1, 0, 1])) == (True, True)
    assert (system.support, system.last_cutoff) == ((False, True), 23)
    expression = system.constant(Fraction(7, 3)) + system.periodic([0, 0, 0])
    for value, expected in [(None, None), (["7", "3"], Fraction(7, 3))]:
        wire["stdout"] = json.dumps({"support": [False, True], "value": value})
        assert expression.standard_part() == expected
        assert (system.support, system.last_cutoff) == ((False, True), 23)


@pytest.mark.parametrize("operation", [operator.sub, operator.mul, operator.truediv,
                                       operator.lt, operator.eq, operator.gt,
                                       operator.le, operator.ge])
def test_mixed_system_operators_reject_before_transport(primed_transport, monkeypatch, operation):
    system, _ = primed_transport
    other = LeanResidueSystem(checker_path=system._checker)

    def unexpected(*args, **kwargs):
        pytest.fail("mixed-context operands must not reach native transport")

    monkeypatch.setattr("hyperreals.verified_residue.subprocess.run", unexpected)
    with pytest.raises(ValueError, match="same Lean system"):
        operation(system.constant(1), other.constant(1))
    assert (system.support, system.last_cutoff) == ((False, True), 17)
    assert (other.support, other.last_cutoff) == ((True,), None)


@pytest.mark.parametrize("method", ["probe", "commit", "decide", "standard_part"])
def test_system_methods_reject_foreign_owned_expressions(primed_transport, monkeypatch, method):
    system, _ = primed_transport
    other = LeanResidueSystem(checker_path=system._checker)
    foreign = other.constant(1)

    def unexpected(*args, **kwargs):
        pytest.fail("foreign-owned operands must not reach native transport")

    monkeypatch.setattr("hyperreals.verified_residue.subprocess.run", unexpected)
    with pytest.raises(ValueError, match="this Lean system"):
        if method == "standard_part":
            system.standard_part(foreign)
        elif method == "commit":
            system.commit(foreign, foreign, truth=True)
        else:
            getattr(system, method)(foreign, foreign)
    assert (system.support, system.last_cutoff) == ((False, True), 17)
    assert (other.support, other.last_cutoff) == ((True,), None)


@requires_lean
def test_negative_choices_lift_complements_to_the_common_period():
    system = LeanResidueSystem()
    modulo_three = system.periodic([0, 1, 2])
    assert system.commit(modulo_three, system.constant(2), "eq", truth=False)
    assert system.support == (True, True, False)
    assert system.commit(system.periodic([0, 1]), system.constant(0), "eq", truth=False)
    assert system.support == (False, True, False, True, False, False)
    assert modulo_three.standard_part() is None
    assert system.commit(modulo_three, system.constant(1), "eq", truth=False)
    assert system.support == (False, False, False, True, False, False)
    assert modulo_three.standard_part() == 0


@requires_lean
def test_native_parser_rejects_invalid_expressions_and_recovers_after_each_line():
    valid = {"support": [False, True], "op": "standardPart", "left": ["const", "2", "3"]}
    bad_expressions = [
        ["const", 1, "2"], ["const", "1", "-2"], ["const", "nan", "1"],
        ["divMonomial", ["index"], "0", "1", "1"],
        ["divMonomial", ["index"], "1", "1", "1.5"],
        ["divMonomial", ["index"], "1", "1", 1],
        ["index", "extra"], ["add", ["index"]],
        ["mul", ["index"], ["invn"], ["index"]], ["exp", ["invn"]],
    ]
    comparison = dict(valid, op="lt", right=["index"])
    invalid = ["{not json}", "[]"] + [
        json.dumps(request) for request in [
            *(dict(valid, left=expression) for expression in bad_expressions),
            dict(valid, op="unknown"), dict(valid, op="lt"),
            *(dict(comparison, choice=choice) for choice in ["true", 1, None]),
        ]
    ]
    lines = [line for bad in invalid for line in (bad, json.dumps(valid))]
    completed = subprocess.run(
        [str(CHECKER)], input="\n".join(lines) + "\n", text=True,
        capture_output=True, check=True, timeout=30,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == len(lines)
    for error, recovered in zip(responses[::2], responses[1::2]):
        assert isinstance(error.get("error"), str) and error["error"]
        assert recovered == {"value": ["2", "3"], "support": [False, True]}
