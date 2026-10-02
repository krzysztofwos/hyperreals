"""Replay data/transport boundaries plus small actual Lean kernel replays."""

import hashlib
import json
import os
import subprocess
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import hyperreals.replay as replay_module
from hyperreals import LeanResidueSystem
from hyperreals.replay import (
    MAX_DEPTH,
    MAX_PERIOD,
    ReplayObservation,
    ReplaySnapshot,
    ReplayVerificationError,
    verify_export,
)

ROOT = Path(__file__).resolve().parents[1]
REPLAY_OLEAN = ROOT / ".lake/build/lib/lean/Hyperreals/ResidueReplay.olean"
requires_lean = pytest.mark.skipif(
    not REPLAY_OLEAN.is_file(), reason="build Hyperreals.ResidueReplay"
)


def constant(value):
    value = Fraction(value)
    return ("const", str(value.numerator), str(value.denominator))


def periodic(values):
    return ("periodic", tuple((str(value), "1") for value in values))


def numeric_snapshot():
    return ReplaySnapshot((), (True,), constant(Fraction(7, 3)), Fraction(7, 3))


def mixed_snapshot():
    observations = (
        ReplayObservation(periodic([0, 1]), constant(1), "eq", True),
        ReplayObservation(periodic([0, 1, 2]), constant(2), "eq", True),
    )
    expression = ("add", periodic(range(6)), ("invn",))
    return ReplaySnapshot(
        observations, (False, False, False, False, False, True), expression, Fraction(5)
    )


def audit_output(snapshot, *, empty=False):
    roots = [
        line.removeprefix("#print axioms ")
        for line in snapshot.to_lean().splitlines()
        if line.startswith("#print axioms ")
    ]
    if empty:
        return "\n".join(f"'{root}' does not depend on any axioms" for root in roots)
    return "\n".join(
        f"'{root}' depends on axioms: [propext, Classical.choice, Quot.sound]"
        for root in roots
    )


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "Hyperreals").mkdir(parents=True)
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.33.0\n", encoding="utf-8")
    (root / "Hyperreals/ResidueReplay.lean").write_text(
        "-- Mock project for transport tests only.\n", encoding="utf-8"
    )
    return root


def test_snapshot_is_deeply_immutable_and_canonicalizes_rationals():
    left = ["periodic", [["2", "4"], ["-3", "6"]]]
    observation = ReplayObservation(left, ["const", "1", "2"], "eq", True)
    observations, support = [observation], [True, False]
    expression = ["add", left, ["const", "0", "3"]]
    snapshot = ReplaySnapshot(observations, support, expression, Fraction(1, 2))
    original = snapshot.to_json()
    left[1][0][0] = "999"
    observations.clear()
    support[0] = False
    expression[0] = "sub"
    exported = snapshot.to_dict()
    exported["expression"][1][1][0][0] = "888"
    assert snapshot.to_json() == original
    assert snapshot.observations[0].left_ast[1] == (("1", "2"), ("-1", "2"))
    assert ReplaySnapshot.from_json(original) == snapshot
    with pytest.raises(FrozenInstanceError):
        snapshot.result = Fraction(99)


@pytest.mark.parametrize(
    "bad",
    [
        ["const", "0); axiom injected : False; --", "1"],
        ["const", "01", "1"],
        ["const", "-0", "1"],
        ["const", "١", "1"],
        ["const", "1", "0"],
        ["const", 1, "2"],
        ["index", "extra"],
        ["periodic", []],
        ["periodic", [["1", "2", "3"]]],
        ["divMonomial", ["index"], "0", "1", "1"],
        ["divMonomial", ["index"], "1", "1", "1.0"],
        ["exp", ["invn"]],
    ],
)
def test_schema_rejects_invalid_ast_and_source_injection(bad):
    payload = numeric_snapshot().to_dict()
    payload["expression"] = bad
    with pytest.raises(ValueError):
        ReplaySnapshot.from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("format", "other"),
        ("support", []),
        ("support", [False]),
        ("support", [1]),
        ("result", ["1", "-2"]),
        ("result", [1, 2]),
        ("result", "1/2"),
        (
            "observations",
            [{"left": ["index"], "right": ["invn"], "op": "lt", "truth": 1}],
        ),
    ],
)
def test_strict_snapshot_schema(field, value):
    payload = numeric_snapshot().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        ReplaySnapshot.from_dict(payload)


def test_unknown_keys_duplicates_and_non_json_constants_are_rejected():
    payload = numeric_snapshot().to_dict()
    payload["lean_source"] = "axiom injected : False"
    with pytest.raises(ValueError):
        ReplaySnapshot.from_dict(payload)
    with pytest.raises(ValueError, match="duplicate"):
        ReplaySnapshot.from_json('{"version":1,"version":1}')
    with pytest.raises(ValueError, match="non-JSON"):
        ReplaySnapshot.from_json('{"version":NaN}')


def test_replay_limits_do_not_restrict_observation_capture():
    large = periodic(range(MAX_PERIOD + 1))
    observation = ReplayObservation(large, constant(0), "eq", True)
    assert len(observation.left_ast[1]) == MAX_PERIOD + 1
    with pytest.raises(ValueError, match="periodic tables"):
        ReplaySnapshot((observation,), (True,), constant(0), Fraction(0))
    deep = constant(1)
    for _ in range(MAX_DEPTH + 1):
        deep = ("add", deep, constant(0))
    observation = ReplayObservation(deep, constant(1), "eq", True)
    with pytest.raises(ValueError, match="depth"):
        ReplaySnapshot((observation,), (True,), constant(0), Fraction(0))


def test_lcm_and_polynomial_growth_are_bounded():
    with pytest.raises(ValueError, match="common period"):
        ReplaySnapshot(
            (), (True,), ("add", periodic(range(67)), periodic(range(71))), None
        )
    with pytest.raises(ValueError, match="polynomial/shift"):
        ReplaySnapshot((), (True,), ("divMonomial", ("index",), "1", "1", "4096"), None)


def test_numeric_and_unknown_sources_use_only_fixed_kernel_proofs():
    numeric = numeric_snapshot().to_lean()
    unknown = ReplaySnapshot((), (True,), periodic([1, -1]), None).to_lean()
    for source in (numeric, unknown):
        assert "by decide +kernel" in source
        assert "native_decide" not in source and "axiom " not in source
        assert "theorem trace_consistent" in source
        assert "theorem support_correspondence" in source
    assert "theorem replayed_standard_part" in numeric
    assert "theorem extractor_unknown" in unknown
    assert "theorem replayed_standard_part" not in unknown


def test_export_binds_snapshot_source_and_project_without_verifying(
    tmp_path, project, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("export must not execute Lean")

    monkeypatch.setattr("hyperreals.replay.subprocess.run", unexpected)
    snapshot = numeric_snapshot()
    exported = snapshot.export(tmp_path / "certificate", project_root=project)
    manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
    assert (exported / "snapshot.json").read_text(
        encoding="utf-8"
    ) == snapshot.to_json()
    assert (exported / "Replay.lean").read_text(encoding="utf-8") == snapshot.to_lean()
    assert (
        manifest["snapshot_sha256"]
        == hashlib.sha256(snapshot.to_json().encode()).hexdigest()
    )
    assert (
        manifest["source_sha256"]
        == hashlib.sha256(snapshot.to_lean().encode()).hexdigest()
    )
    assert "Hyperreals/ResidueReplay.lean" in manifest["provenance"]["files"]


@pytest.mark.parametrize("empty", [False, True])
def test_verify_regenerates_temporary_source_and_audits_all_headlines(
    project, monkeypatch, empty
):
    snapshot = numeric_snapshot()

    def checked(command, **kwargs):
        if command == ["lake", "build", "Hyperreals.ResidueReplay"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command[:3] == ["lake", "env", "lean"]
        assert kwargs["cwd"] == project
        assert Path(command[-1]).read_text(encoding="utf-8") == snapshot.to_lean()
        return subprocess.CompletedProcess(
            command, 0, audit_output(snapshot, empty=empty), ""
        )

    monkeypatch.setattr("hyperreals.replay.subprocess.run", checked)
    result = snapshot.verify(project)
    assert (
        result.snapshot_sha256
        == hashlib.sha256(snapshot.to_json().encode()).hexdigest()
    )
    assert result.axioms == (
        () if empty else ("Classical.choice", "Quot.sound", "propext")
    )


@pytest.mark.parametrize(
    "failure",
    ["missing", "wrong_root", "sorry", "native", "error", "exit", "timeout", "oserror"],
)
def test_verifier_fails_closed_on_execution_or_audit_failure(
    project, monkeypatch, failure
):
    snapshot = numeric_snapshot()

    def failed(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "oserror":
            raise OSError("missing lake")
        output = audit_output(snapshot)
        if failure == "missing":
            output = output.splitlines()[0]
        elif failure == "wrong_root":
            output = output.replace("GeneratedReplay", "UnrelatedReplay")
        elif failure == "sorry":
            output = output.replace("Quot.sound", "sorryAx")
        elif failure == "native":
            output = output.replace("Quot.sound", "Lean.ofReduceBool")
        elif failure == "error":
            output += "\nReplay.lean:1:1: error: failed check"
        return subprocess.CompletedProcess(command, int(failure == "exit"), output, "")

    monkeypatch.setattr("hyperreals.replay.subprocess.run", failed)
    with pytest.raises(ReplayVerificationError):
        snapshot.verify(project)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 301, True])
def test_timeout_bounds_are_checked_before_execution(project, timeout):
    with pytest.raises(ValueError, match="timeout"):
        numeric_snapshot().verify(project, timeout=timeout)


def test_build_and_kernel_share_one_timeout_budget(project, monkeypatch):
    snapshot, timeouts = numeric_snapshot(), []
    ticks = iter([100.0, 101.0, 108.0])
    monkeypatch.setattr(
        replay_module, "time", SimpleNamespace(monotonic=lambda: next(ticks))
    )

    def checked(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, audit_output(snapshot), "")

    monkeypatch.setattr("hyperreals.replay.subprocess.run", checked)
    snapshot.verify(project, timeout=20)
    assert timeouts == [19.0, 12.0]


def test_elapsed_build_budget_prevents_starting_kernel(project, monkeypatch):
    ticks, commands = iter([100.0, 101.0, 121.0]), []
    monkeypatch.setattr(
        replay_module, "time", SimpleNamespace(monotonic=lambda: next(ticks))
    )

    def checked(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("hyperreals.replay.subprocess.run", checked)
    with pytest.raises(ReplayVerificationError, match="total timeout"):
        numeric_snapshot().verify(project, timeout=20)
    assert commands == [["lake", "build", "Hyperreals.ResidueReplay"]]


@pytest.mark.parametrize(
    "tamper",
    ["snapshot", "source", "manifest", "manifest_version", "project", "noncanonical"],
)
def test_export_tampering_rejected_before_any_source_executes(
    tmp_path, project, monkeypatch, tamper
):
    snapshot = numeric_snapshot()
    exported = snapshot.export(tmp_path / "certificate", project_root=project)
    if tamper == "snapshot":
        (exported / "snapshot.json").write_text(
            replace(snapshot, result=Fraction(8, 3)).to_json(), encoding="utf-8"
        )
    elif tamper == "source":
        source = snapshot.to_lean() + '\n#eval IO.println "unexpected execution"\n'
        (exported / "Replay.lean").write_text(source, encoding="utf-8")
        manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
        manifest["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        (exported / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "manifest":
        (exported / "manifest.json").write_text("{}", encoding="utf-8")
    elif tamper == "manifest_version":
        path = exported / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["version"] = True
        path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    elif tamper == "project":
        (project / "Hyperreals/ResidueReplay.lean").write_text(
            "-- Changed project.\n", encoding="utf-8"
        )
    else:
        (exported / "snapshot.json").write_text(
            json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8"
        )

    def unexpected(*args, **kwargs):
        pytest.fail("artifact association failures must not invoke Lean")

    monkeypatch.setattr("hyperreals.replay.subprocess.run", unexpected)
    with pytest.raises(ReplayVerificationError):
        verify_export(exported, project)


def test_verify_export_compiles_regenerated_source_not_the_supplied_file(
    tmp_path, project, monkeypatch
):
    snapshot = numeric_snapshot()
    exported = snapshot.export(tmp_path / "certificate", project_root=project)

    def checked(command, **kwargs):
        if command == ["lake", "build", "Hyperreals.ResidueReplay"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        assert Path(command[-1]) != exported / "Replay.lean"
        assert Path(command[-1]).read_text(encoding="utf-8") == snapshot.to_lean()
        return subprocess.CompletedProcess(command, 0, audit_output(snapshot), "")

    monkeypatch.setattr("hyperreals.replay.subprocess.run", checked)
    assert (
        verify_export(exported, project).source_sha256
        == hashlib.sha256(snapshot.to_lean().encode()).hexdigest()
    )


def test_verify_export_detects_project_changes_during_verification(
    tmp_path, project, monkeypatch
):
    snapshot = numeric_snapshot()
    exported = snapshot.export(tmp_path / "certificate", project_root=project)

    def checked(command, **kwargs):
        if command[1] == "env":
            (project / "Hyperreals/ResidueReplay.lean").write_text(
                "-- Changed while verifying.\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0, audit_output(snapshot), "")

    monkeypatch.setattr("hyperreals.replay.subprocess.run", checked)
    with pytest.raises(ReplayVerificationError, match="provenance changed"):
        verify_export(exported, project)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX named pipes")
def test_verify_export_rejects_nonregular_files_without_reading_them(tmp_path, project):
    exported = numeric_snapshot().export(tmp_path / "certificate", project_root=project)
    source = exported / "Replay.lean"
    source.unlink()
    os.mkfifo(source)
    with pytest.raises(ReplayVerificationError, match="regular files"):
        verify_export(exported, project)


def test_snapshot_does_not_follow_later_live_state(tmp_path, monkeypatch):
    checker = tmp_path / "checker"
    checker.touch()
    system = LeanResidueSystem(checker_path=checker)
    replies = iter(
        [
            {"value": None, "support": [True]},
            {
                "predicate": [False, True],
                "trueSupport": [False, True],
                "falseSupport": [True, False],
                "canBeTrue": True,
                "canBeFalse": True,
                "accepted": True,
                "support": [False, True],
                "cutoff": "1",
            },
            {"value": ["1", "1"], "support": [False, True]},
        ]
    )
    monkeypatch.setattr(system, "_exchange", lambda request: next(replies))
    expression = system.periodic([0, 1])
    before = system.snapshot(expression)
    assert expression == system.constant(1)
    after = system.snapshot(expression)
    assert (before.support, before.observations, before.result) == ((True,), (), None)
    assert (after.support, len(after.observations), after.result) == (
        (False, True),
        1,
        Fraction(1),
    )
    assert before != after


@requires_lean
@pytest.mark.parametrize("unknown", [False, True])
def test_actual_kernel_replay_for_mixed_periods_and_unknown(unknown):
    snapshot = mixed_snapshot()
    if unknown:
        snapshot = ReplaySnapshot((), (True,), periodic([1, -1]), None)
    report = snapshot.verify(ROOT, timeout=120)
    assert set(report.axioms) <= {"propext", "Classical.choice", "Quot.sound"}


@requires_lean
@pytest.mark.parametrize("tamper", ["support", "result", "expression"])
def test_actual_kernel_rejects_forged_claims(tamper):
    snapshot = numeric_snapshot()
    if tamper == "support":
        snapshot = replace(snapshot, support=(True, False))
    elif tamper == "result":
        snapshot = replace(snapshot, result=Fraction(8, 3))
    else:
        snapshot = replace(snapshot, expression=constant(9))
    with pytest.raises(ReplayVerificationError, match="did not verify"):
        snapshot.verify(ROOT, timeout=120)
