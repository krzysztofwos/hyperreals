"""The cubic example uses its original AST and needs no residue choice."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from hyperreals import ReplaySnapshot
from scripts import infinitesimal_case
from scripts.benchmark_residues import evaluate
from scripts.infinitesimal_case import CaseResult, export_case, run_adaptive, run_case

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".lake/build/bin/residue_checker"
pytestmark = pytest.mark.skipif(
    not CHECKER.is_file(), reason="run lake build residue_checker"
)


@pytest.fixture(scope="module")
def case() -> CaseResult:
    return run_case()


def test_original_difference_quotient_is_sent_to_the_extractor(case):
    expression = case.before.expression
    assert expression[0] == "divMonomial"
    assert expression[2:] == ("1", "1", "-1")
    assert expression[1][0] == "sub"
    assert expression[1][1][0] == "mul"
    assert case.before.result == Fraction(12)
    assert case.error.result == Fraction(0)


def test_ordinary_derivative_is_not_the_exact_difference_quotient(case):
    for n in (1, 2, 97, 10**6):
        epsilon = Fraction(1, n)
        quotient = evaluate(case.before.expression, n)
        assert quotient == 12 + 6 * epsilon + epsilon**2
        assert quotient > 12


def test_later_parity_choice_preserves_result_and_earlier_snapshot(case):
    assert case.before.support == (True,)
    assert case.before.observations == ()
    assert case.error.support == (True,)
    assert case.error.observations == ()
    assert case.after is not None
    assert case.after.support == (False, True)
    assert len(case.after.observations) == 1
    assert case.after.result == case.before.result == Fraction(12)
    assert case.after.expression == case.before.expression
    assert ReplaySnapshot.from_json(case.before.to_json()) == case.before


def test_parity_demonstration_is_optional():
    result = run_case(include_parity=False, include_adaptive=False)
    assert result.after is None
    assert result.before.observations == ()
    assert set(result.snapshots()) == {"quotient", "error"}


def test_adaptive_choices_select_distinct_raw_quotients_with_same_standard_part(case):
    backward, forward = case.adaptive
    assert (backward.choice, backward.branch) == (True, "backward")
    assert (forward.choice, forward.branch) == (False, "forward")
    assert backward.snapshot.expression != forward.snapshot.expression
    assert forward.snapshot.expression == case.before.expression
    assert backward.snapshot.result == forward.snapshot.result == Fraction(12)
    for n in (1, 2, 97, 10**6):
        epsilon = Fraction(1, n)
        backward_value = evaluate(backward.snapshot.expression, n)
        forward_value = evaluate(forward.snapshot.expression, n)
        assert backward_value == (2**3 - (2 - epsilon) ** 3) / epsilon
        assert forward_value == ((2 + epsilon) ** 3 - 2**3) / epsilon
        assert backward_value == 12 - 6 * epsilon + epsilon**2
        assert forward_value == 12 + 6 * epsilon + epsilon**2
        assert backward_value < 12 < forward_value


def test_adaptive_runs_have_separate_systems_and_exact_accepted_traces(monkeypatch):
    systems = []
    original = infinitesimal_case.LeanResidueSystem

    def fresh_system():
        system = original()
        systems.append(system)
        return system

    monkeypatch.setattr(infinitesimal_case, "LeanResidueSystem", fresh_system)
    runs = [run_adaptive(True), run_adaptive(False)]
    assert len(systems) == 2 and systems[0] is not systems[1]
    for run, system in zip(runs, systems):
        expected_support = (False, True) if run.choice else (True, False)
        assert run.snapshot.support == system.support == expected_support
        assert run.snapshot.observations == system.history
        assert len(system.history) == 1
        observation = system.history[0]
        assert observation.op == "lt"
        assert observation.truth is run.choice
        assert observation.left_ast == ("periodic", (("1", "1"), ("-1", "1")))
        assert observation.right_ast == ("const", "0", "1")
        assert run.snapshot.result == Fraction(12)


def test_rejected_adaptive_choice_does_not_capture_a_branch(monkeypatch):
    class RejectedSystem(infinitesimal_case.LeanResidueSystem):
        def commit(self, *args, **kwargs):
            return False

        def snapshot(self, *args, **kwargs):
            pytest.fail("a rejected adaptive choice must not reach extraction")

    monkeypatch.setattr(infinitesimal_case, "LeanResidueSystem", RejectedSystem)
    with pytest.raises(
        infinitesimal_case.LeanBackendError, match="unexpectedly rejected"
    ):
        run_adaptive(True)


def test_export_without_replay_does_not_claim_verification(case, tmp_path):
    report = export_case(case, tmp_path, verify=False)
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == report
    for record in report["snapshots"]:
        assert record["verification"] == "not-run"
        directory = tmp_path / "snapshots" / record["name"]
        snapshot = ReplaySnapshot.from_json(
            (directory / "snapshot.json").read_text(encoding="utf-8")
        )
        assert snapshot == case.snapshots()[record["name"]]
        assert (directory / "Replay.lean").read_text(
            encoding="utf-8"
        ) == snapshot.to_lean()
    adaptive_records = {
        record["name"]: record
        for record in report["snapshots"]
        if record["name"].startswith("adaptive-")
    }
    assert adaptive_records["adaptive-backward"]["choice"] is True
    assert adaptive_records["adaptive-forward"]["choice"] is False
    assert all(record["result"] == "12" for record in adaptive_records.values())


@pytest.mark.skipif(
    not (ROOT / ".lake/build/lib/lean/Hyperreals/ResidueReplay.olean").is_file(),
    reason="run lake build Hyperreals.ResidueReplay",
)
def test_choice_free_difference_quotient_passes_kernel_replay(case):
    checked = case.before.verify(project_root=ROOT, timeout=120)
    assert "Hyperreals.GeneratedReplay.replayed_standard_part" in checked.stdout
    assert set(checked.axioms) <= {"propext", "Classical.choice", "Quot.sound"}
