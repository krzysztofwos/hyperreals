"""Failure witnesses distinguish divergence from completion disagreement."""

from pathlib import Path

import pytest

from hyperreals import LeanBackendError, LeanResidueSystem

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / ".lake/build/bin/residue_checker"
requires_lean = pytest.mark.skipif(
    not CHECKER.is_file(), reason="run lake build residue_checker"
)


@requires_lean
def test_finite_divergent_and_disagreeing_limits():
    system = LeanResidueSystem()
    finite = (system.constant(3) + system.infinitesimal()).explain_standard_part()
    assert (finite.kind, finite.period, finite.residues, finite.limits) == (
        "finite",
        1,
        (),
        (3,),
    )
    divergent = (system.periodic([0, 1]) * system.infinite()).explain_standard_part()
    assert (divergent.kind, divergent.period, divergent.residues, divergent.limits) == (
        "divergent",
        2,
        (1,),
        (),
    )
    expression = system.periodic([1, 2]) + system.infinitesimal()
    disagreement = expression.explain_standard_part()
    assert (
        disagreement.kind,
        disagreement.period,
        disagreement.residues,
        disagreement.limits,
    ) == ("disagreement", 2, (0, 1), (1, 2))
    assert expression.standard_part() is None
    assert system.support == (True,) and system.history == ()
    assert system.commit(system.periodic([0, 1]), system.constant(1), "eq", truth=True)
    assert expression.explain_standard_part().limits == (2,)
    assert expression.standard_part() == 2


@requires_lean
def test_divergence_has_priority_and_uses_common_period():
    system = LeanResidueSystem()
    assert system.commit(system.periodic([0, 1]), system.constant(1), "eq", truth=True)
    expression = (
        system.periodic([1, 2, 3]) + system.periodic([0, 1, 0]) * system.infinite()
    )
    diagnostic = expression.explain_standard_part()
    assert (diagnostic.kind, diagnostic.period, diagnostic.residues) == (
        "divergent",
        6,
        (1,),
    )
    assert system.support == (False, True)


@pytest.mark.parametrize(
    "updates",
    [
        {"period": "0"},
        {"period": 2},
        {"period": "٢"},
        {"support": [False, True]},
        {"diagnostic": {"kind": "invalidInput", "residues": [], "limits": []}},
        {"diagnostic": {"kind": "divergent", "residues": ["2"], "limits": []}},
        {"diagnostic": {"kind": "finite", "residues": [], "limits": [["1", "0"]]}},
        {
            "diagnostic": {
                "kind": "disagreement",
                "residues": ["0", "1"],
                "limits": [["1", "1"], ["1", "1"]],
            }
        },
    ],
)
def test_invalid_diagnostic_does_not_change_state(tmp_path, monkeypatch, updates):
    checker = tmp_path / "checker"
    checker.touch()
    system = LeanResidueSystem(checker_path=checker)
    response = {
        "support": [True],
        "period": "2",
        "diagnostic": {"kind": "divergent", "residues": ["1"], "limits": []},
    }
    response.update(updates)
    monkeypatch.setattr(system, "_exchange", lambda request: response)
    with pytest.raises(LeanBackendError):
        system.periodic([0, 1]).explain_standard_part()
    assert system.support == (True,) and system.history == ()
