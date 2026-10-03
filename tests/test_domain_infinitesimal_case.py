"""The observation supplies a needed denominator condition, not just a label."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from hyperreals import LeanResidueSystem, ReplaySnapshot
from scripts import domain_infinitesimal_case as domain
from scripts.benchmark_residues import evaluate

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".lake/build/bin/residue_checker"
pytestmark = pytest.mark.skipif(
    not CHECKER.is_file(), reason="run lake build residue_checker"
)


@pytest.fixture(scope="module")
def runs():
    return domain.run_case()


def test_observation_selects_nonzero_step_and_both_results_are_twelve(runs):
    fallback, nonzero = runs
    assert (fallback.choice, fallback.branch) == (True, "fallback")
    assert (nonzero.choice, nonzero.branch) == (False, "nonzero")
    assert fallback.snapshot.support == (True, False)
    assert nonzero.snapshot.support == (False, True)
    for run in runs:
        assert run.snapshot.result == Fraction(12)
        assert run.unrefined.result is None
        assert run.unrefined.support == (True,)
        assert run.unrefined.observations == ()
        assert len(run.snapshot.observations) == 1
        observation = run.snapshot.observations[0]
        assert observation.op == "eq"
        assert observation.truth is run.choice
        assert observation.left_ast == (
            "mul",
            ("periodic", (("0", "1"), ("1", "1"))),
            ("invn",),
        )
        assert observation.right_ast == ("const", "0", "1")
    assert fallback.snapshot.expression[0] == "divMonomial"
    assert fallback.snapshot.expression[2:] == ("1", "1", "-2")
    assert nonzero.snapshot.expression == nonzero.unrefined.expression
    assert nonzero.snapshot.expression[0] == "mul"
    assert nonzero.snapshot.expression[2] == (
        "mul",
        ("periodic", (("0", "1"), ("1", "1"))),
        ("index",),
    )


def test_finite_index_quotients_and_inverse_domain_independently(runs):
    fallback, nonzero = runs
    system = LeanResidueSystem()
    expressions = domain.model(system)
    for n in (1, 2, 3, 98, 99, 10**6, 10**6 + 1):
        h = Fraction(n % 2, n)
        reciprocal = (n % 2) * n
        assert evaluate(expressions.step._ast, n) == h
        assert evaluate(expressions.reciprocal._ast, n) == reciprocal
        guarded = evaluate(nonzero.snapshot.expression, n)
        if n % 2:
            assert h * reciprocal == 1
            assert guarded == ((2 + h) ** 3 - 2**3) / h
            assert guarded > 12
        else:
            assert h == reciprocal == guarded == 0
            step = Fraction(1, n**2)
            quotient = evaluate(fallback.snapshot.expression, n)
            assert quotient == ((2 + step) ** 3 - 2**3) / step
            assert quotient == 12 + 6 * step + step**2
            assert quotient > 12


def test_opposite_observation_rejected_without_mutation():
    for choice in (True, False):
        system = LeanResidueSystem()
        expressions = domain.model(system)
        zero, one = system.constant(0), system.constant(1)
        assert system.commit(expressions.step, zero, "eq", truth=choice)
        state = system.history, system.support
        assert not system.commit(expressions.step, zero, "eq", truth=not choice)
        assert (system.history, system.support) == state
        assert system.probe(expressions.step * expressions.reciprocal, one, "eq") == (
            (True, False) if choice else (False, True)
        )
        assert expressions.guarded.standard_part() == (0 if choice else 12)
        assert (system.history, system.support) == state


def test_rejected_observation_cannot_capture_a_branch(monkeypatch):
    class RejectedSystem(LeanResidueSystem):
        captures = 0

        def commit(self, *args, **kwargs):
            return False

        def snapshot(self, *args, **kwargs):
            self.captures += 1
            assert self.captures == 1, "only the pre-observation snapshot is permitted"
            return super().snapshot(*args, **kwargs)

    monkeypatch.setattr(domain, "LeanResidueSystem", RejectedSystem)
    with pytest.raises(domain.LeanBackendError, match="unexpectedly rejected"):
        domain.run_branch(False)


def test_non_monomial_division_remains_outside_the_grammar():
    system = LeanResidueSystem()
    expressions = domain.model(system)
    with pytest.raises(ValueError, match="division requires a primitive monomial"):
        system.constant(1) / expressions.step


def test_export_roundtrip_and_unverified_label(runs, tmp_path):
    report = domain.export_case(runs, tmp_path, verify=False)
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == report
    for run, record in zip(runs, report["snapshots"]):
        assert record["verification"] == "not-run"
        assert record["choice"] is run.choice
        assert record["result"] == "12"
        directory = tmp_path / "snapshots" / run.name
        assert (
            ReplaySnapshot.from_json(
                (directory / "snapshot.json").read_text(encoding="utf-8")
            )
            == run.snapshot
        )
        assert (directory / "Replay.lean").read_text(
            encoding="utf-8"
        ) == run.snapshot.to_lean()
    unrefined = report["snapshots"][2]
    assert unrefined["name"] == "domain-unrefined"
    assert unrefined["result"] is None
    assert unrefined["verification"] == "not-run"
    source = (tmp_path / "snapshots/domain-unrefined/Replay.lean").read_text(
        encoding="utf-8"
    )
    assert source == runs[0].unrefined.to_lean()
    assert "theorem no_common_standard_part" in source
    assert "theorem failure_classified" in source


@pytest.mark.skipif(
    not (ROOT / ".lake/build/lib/lean/Hyperreals/ResidueReplay.olean").is_file(),
    reason="run lake build Hyperreals.ResidueReplay",
)
def test_both_accepted_domain_branches_pass_kernel_replay(runs):
    for run in runs:
        checked = run.snapshot.verify(project_root=ROOT, timeout=120)
        assert "Hyperreals.GeneratedReplay.replayed_standard_part" in checked.stdout
        assert set(checked.axioms) <= {"propext", "Classical.choice", "Quot.sound"}
