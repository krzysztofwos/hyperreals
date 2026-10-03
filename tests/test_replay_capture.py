"""Actual session history and immutable extraction snapshot boundaries."""

from fractions import Fraction
from pathlib import Path

import pytest

from hyperreals import LeanBackendError, LeanResidueSystem, ReplayVerificationError

CHECKER = Path(__file__).resolve().parents[1] / ".lake/build/bin/residue_checker"
pytestmark = pytest.mark.skipif(
    not CHECKER.is_file(), reason="run lake build residue_checker"
)
requires_replay = pytest.mark.skipif(
    not (CHECKER.parents[1] / "lib/lean/Hyperreals/ResidueReplay.olean").is_file(),
    reason="run lake build Hyperreals.ResidueReplay",
)


def test_only_accepted_observations_enter_history():
    system = LeanResidueSystem()
    a, zero = system.alt(), system.constant(0)
    assert system.history == ()
    assert system.probe(a, zero) == (True, True)
    unknown = system.snapshot(a)
    assert unknown.result is None
    assert unknown.observations == ()
    assert system.history == ()
    assert a < zero
    accepted = system.history
    assert len(accepted) == 1
    assert accepted[0].truth is True
    assert not system.commit(a, zero, truth=False)
    assert system.history is accepted
    assert a.standard_part() == Fraction(-1)
    assert system.history is accepted


def test_snapshot_does_not_follow_later_state():
    system = LeanResidueSystem()
    a, b = system.alt(), system.periodic([0, 1, 2])
    assert a < system.constant(0)
    snapshot = system.snapshot(a)
    before = snapshot.to_json()
    assert b == system.constant(2)
    assert system.period == 6
    assert snapshot.support == (False, True)
    assert snapshot.result == Fraction(-1)
    assert len(snapshot.observations) == 1
    assert len(system.history) == 2
    assert snapshot.to_json() == before


def test_failed_native_query_preserves_history_and_support(monkeypatch):
    system = LeanResidueSystem()
    assert system.alt() < system.constant(0)
    history, support = system.history, system.support

    def failed(request):
        raise LeanBackendError("simulated transport failure")

    monkeypatch.setattr(system, "_exchange", failed)
    with pytest.raises(LeanBackendError, match="simulated"):
        system.snapshot(system.constant(1))
    with pytest.raises(LeanBackendError, match="simulated"):
        system.commit(system.constant(0), system.constant(1), truth=True)
    assert system.history is history
    assert system.support == support


def test_negative_decision_records_only_accepted_polarity():
    system = LeanResidueSystem()
    assert not system.constant(1) < system.constant(0)
    assert len(system.history) == 1
    assert system.history[0].truth is False
    assert system.support == (True,)


def test_incompatible_context_cannot_enter_transcript():
    first, second = LeanResidueSystem(), LeanResidueSystem()
    with pytest.raises(ValueError):
        first.snapshot(second.constant(1))
    with pytest.raises(ValueError):
        first.commit(second.constant(0), second.constant(1), truth=True)
    assert first.history == ()
    assert first.support == (True,)


@requires_replay
def test_actual_session_exports_the_requested_completion_invariant_result():
    system = LeanResidueSystem()
    a, b = system.alt(), system.periodic([0, 1, 2])
    assert a < system.constant(0)
    assert b == system.constant(2)
    snapshot = system.snapshot(system.constant(1) + (a + b) / system.infinite())
    assert snapshot.result == 1
    verification = snapshot.verify(timeout=120)
    assert "Hyperreals.GeneratedReplay.replayed_standard_part" in verification.stdout


@requires_replay
def test_retained_powers_and_large_exact_rationals_replay():
    system = LeanResidueSystem()
    exact = Fraction(9007199254740993, 97)
    expression = (
        system.infinite() ** 11 * system.infinitesimal() ** 11 + system.constant(exact)
    )
    snapshot = system.snapshot(expression)
    assert snapshot.result == 1 + exact
    snapshot.verify(timeout=120)


@requires_replay
def test_wrong_native_result_cannot_become_a_verified_snapshot(monkeypatch):
    system = LeanResidueSystem()
    monkeypatch.setattr(
        system,
        "_exchange",
        lambda request: {
            "support": [True],
            "value": ["2", "1"],
        },
    )
    snapshot = system.snapshot(system.constant(1))
    assert snapshot.result == 2  # Structurally valid transport is not a proof.
    with pytest.raises(ReplayVerificationError, match="did not verify"):
        snapshot.verify(timeout=120)
