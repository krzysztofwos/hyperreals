#!/usr/bin/env python3
"""Exact phase-calibration example with delayed choices and replay artifacts.

Build the residue checker first, then run:
  uv run python scripts/delayed_choice_case_study.py --repeat 5

The physical interpretation is illustrative. All model arithmetic is rational.
Timings compare these concrete implementations, not general choice algorithms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import Any

from hyperreals import LeanResidueSystem

ROOT = Path(__file__).resolve().parents[1]
FULL_PERIOD = 60
OFFSETS_A = (Fraction(1, 3), Fraction(-2, 5), Fraction(7, 4), Fraction(0))
OFFSETS_B = (
    Fraction(0),
    Fraction(1, 7),
    Fraction(-2, 3),
    Fraction(5, 2),
    Fraction(-1, 4),
)
BIASES = (3, -2, 5, 1, -4, 7)


@dataclass(frozen=True)
class Observation:
    name: str
    values: tuple[int, ...]
    target: int

    def accepts(self, phase: int) -> bool:
        return self.values[phase % len(self.values)] == self.target


OBSERVATIONS = (
    Observation("coarse: phase mod 4 is odd", (0, 1, 0, 1), 1),
    Observation("shared clock: phase mod 6 = 5", tuple(range(6)), 5),
    Observation("independent clock: phase mod 5 = 3", tuple(range(5)), 3),
    Observation("fine phase: phase mod 4 = 3", tuple(range(4)), 3),
)


def rational(value: Fraction | None) -> str | None:
    return None if value is None else str(value)


def common_value(values: set[Fraction]) -> Fraction | None:
    return next(iter(values)) if len(values) == 1 else None


def model(system: LeanResidueSystem) -> dict[str, Any]:
    """Direct syntax for F+(x,n)=x³+b(n)x+a(n), F-(x,n)=x³-b(n)x+c(n)."""
    epsilon, x = system.infinitesimal(), system.constant(2)
    a, c = system.periodic(OFFSETS_A), system.periodic(OFFSETS_B)
    bias = system.periodic(BIASES)
    shifted = x + epsilon
    base_a, shifted_a = x**3 + bias * x + a, shifted**3 + bias * shifted + a
    base_b, shifted_b = x**3 - bias * x + c, shifted**3 - bias * shifted + c
    raw_a, raw_b = (shifted_a - base_a) / epsilon, (shifted_b - base_b) / epsilon
    return {
        "raw": raw_a,
        "fused": (raw_a + raw_b) / system.constant(2),
        "phase": system.periodic(range(FULL_PERIOD)),
    }


def stage_record(
    name: str,
    accepted: bool | None,
    period: int,
    active: list[int],
    candidates: list[int],
    raw: Fraction | None,
    fused: Fraction | None,
) -> dict[str, Any]:
    return {
        "stage": name,
        "accepted": accepted,
        "stored_period": period,
        "stored_active_residues": active,
        "remaining_phases": candidates,
        "raw_standard_part": rational(raw),
        "fused_standard_part": rational(fused),
    }


def reference_record(
    name: str,
    accepted: bool | None,
    candidates: list[int],
    period: int,
    active: list[int],
) -> dict[str, Any]:
    raw = common_value({Fraction(12 + BIASES[r % 6]) for r in candidates})
    return stage_record(name, accepted, period, active, candidates, raw, Fraction(12))


def run_native(
    *, snapshots: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the public wrapper, including one native process launch per request."""
    system = LeanResidueSystem()
    expressions = model(system)
    captures: dict[str, Any] = {}

    def record(name: str, accepted: bool | None) -> dict[str, Any]:
        support = system.support
        return stage_record(
            name,
            accepted,
            system.period,
            [r for r, bit in enumerate(support) if bit],
            [r for r in range(FULL_PERIOD) if support[r % system.period]],
            expressions["raw"].standard_part(),
            expressions["fused"].standard_part(),
        )

    records = [record("initial", None)]
    for index, observation in enumerate(OBSERVATIONS):
        accepted = system.commit(
            system.periodic(observation.values),
            system.constant(observation.target),
            "eq",
            truth=True,
        )
        records.append(record(observation.name, accepted))
        if not accepted:
            break
        if snapshots and index == 0:
            captures["early-fused"] = system.snapshot(expressions["fused"])
        elif snapshots and index == 1:
            captures["partial-raw"] = system.snapshot(expressions["raw"])
        elif snapshots and index == 3:
            captures["final-phase"] = system.snapshot(expressions["phase"])
    return records, captures


def run_delayed_python() -> list[dict[str, Any]]:
    """Independent dynamic Boolean masks. Limits use the expanded model formula."""
    support: tuple[bool, ...] = (True,)

    def record(name: str, accepted: bool | None) -> dict[str, Any]:
        candidates = [r for r in range(FULL_PERIOD) if support[r % len(support)]]
        return reference_record(
            name,
            accepted,
            candidates,
            len(support),
            [r for r, bit in enumerate(support) if bit],
        )

    records = [record("initial", None)]
    for observation in OBSERVATIONS:
        period = lcm(len(support), len(observation.values))
        proposed = tuple(
            support[r % len(support)] and observation.accepts(r) for r in range(period)
        )
        accepted = any(proposed)
        if accepted:
            support = proposed
        records.append(record(observation.name, accepted))
        if not accepted:
            break
    return records


def run_enumerated(*, eager: bool = False) -> list[dict[str, Any]]:
    """Enumerate all 60 recurring phases. Optionally overcommit after an update."""
    candidates = list(range(FULL_PERIOD))
    records = [reference_record("initial", None, candidates, FULL_PERIOD, candidates)]
    for observation in OBSERVATIONS:
        proposed = [r for r in candidates if observation.accepts(r)]
        accepted = bool(proposed)
        if accepted:
            candidates = proposed[:1] if eager else proposed
        records.append(
            reference_record(
                observation.name, accepted, candidates, FULL_PERIOD, candidates
            )
        )
        if not accepted:
            break
    return records


def validate_model() -> dict[str, Any]:
    """Exact sampled-index checks of the independent algebra, covering every phase."""
    checks = 0
    for phase in range(FULL_PERIOD):
        for cycle in (1, 2, 5):
            n = phase + FULL_PERIOD * cycle
            epsilon, x = Fraction(1, n), Fraction(2)
            a, c, bias = OFFSETS_A[n % 4], OFFSETS_B[n % 5], Fraction(BIASES[n % 6])
            raw_a = (
                (x + epsilon) ** 3 + bias * (x + epsilon) + a - (x**3 + bias * x + a)
            ) / epsilon
            raw_b = (
                (x + epsilon) ** 3 - bias * (x + epsilon) + c - (x**3 - bias * x + c)
            ) / epsilon
            assert raw_a == 12 + bias + 6 * epsilon + epsilon**2
            assert raw_b == 12 - bias + 6 * epsilon + epsilon**2
            assert (raw_a + raw_b) / 2 == 12 + 6 * epsilon + epsilon**2
            checks += 1
    return {
        "sampled_indices": checks,
        "phases_covered": FULL_PERIOD,
        "scope": "Exact sampled-index checks, exhaustive over 60 residue classes. These checks do not prove a limit.",
    }


def semantic_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "stage",
        "accepted",
        "remaining_phases",
        "raw_standard_part",
        "fused_standard_part",
    )
    return [{field: record[field] for field in fields} for record in records]


def accepted_count(records: list[dict[str, Any]]) -> int:
    return sum(record["accepted"] is True for record in records)


def measure(repeats: int) -> dict[str, Any]:
    reference = run_enumerated()
    eager = run_enumerated(eager=True)
    assert [len(row["remaining_phases"]) for row in reference] == [60, 30, 10, 2, 1]
    assert reference[-1]["remaining_phases"] == [23]
    assert [row["raw_standard_part"] for row in reference] == [
        None,
        None,
        "19",
        "19",
        "19",
    ]
    assert [row["fused_standard_part"] for row in reference] == ["12"] * 5
    assert eager[-1]["accepted"] is False and eager[-1]["remaining_phases"] == [1]
    runners: dict[str, Callable[[], list[dict[str, Any]]]] = {
        "lean_delayed_public_api": lambda: run_native()[0],
        "python_delayed_mask": run_delayed_python,
        "python_eager_single_phase": lambda: run_enumerated(eager=True),
        "python_exhaustive_reference": run_enumerated,
    }
    measurements = []
    traces = {}
    for name, run in runners.items():
        expected = eager if name == "python_eager_single_phase" else reference
        warmup = run()
        assert semantic_records(warmup) == semantic_records(expected), name
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            observed = run()
            samples.append(time.perf_counter_ns() - start)
            assert semantic_records(observed) == semantic_records(expected), name
        traces[name] = warmup
        measurements.append(
            {
                "implementation": name,
                "samples_ns": samples,
                "median_ns": statistics.median(samples),
                "min_ns": min(samples),
                "max_ns": max(samples),
                "accepted_observations": accepted_count(warmup),
                "attempted_observations": len(warmup) - 1,
                "remaining_phases": len(warmup[-1]["remaining_phases"]),
            }
        )
        print(f"Validated and measured {name}", flush=True)
    return {"measurements": measurements, "traces": traces}


def replay(
    output: Path, expected: list[dict[str, Any]], timeout: float
) -> list[dict[str, Any]]:
    records, snapshots = run_native(snapshots=True)
    assert semantic_records(records) == semantic_records(expected)
    reports = []
    for name, snapshot in snapshots.items():
        directory = output / "snapshots" / name
        snapshot.export(directory, project_root=ROOT)
        start = time.perf_counter_ns()
        verification = snapshot.verify(project_root=ROOT, timeout=timeout)
        elapsed = time.perf_counter_ns() - start
        hashes = {
            "snapshot_sha256": verification.snapshot_sha256,
            "source_sha256": verification.source_sha256,
        }
        assert (
            hashlib.sha256((directory / "snapshot.json").read_bytes()).hexdigest()
            == hashes["snapshot_sha256"]
        )
        assert (
            hashlib.sha256((directory / "Replay.lean").read_bytes()).hexdigest()
            == hashes["source_sha256"]
        )
        reports.append(
            {
                "snapshot": name,
                "elapsed_ns": elapsed,
                "stdout": verification.stdout,
                "axioms": verification.axioms,
                "hashes": hashes,
            }
        )
        print(f"Kernel replay checked {name}", flush=True)
    return reports


def write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Measured delayed-choice case study",
        "",
        f"Generated {report['generated_utc']} on {report['environment']['platform']} with Python {report['environment']['python']}. Each implementation ran {report['repeats']} repetitions after one warmup.",
        "",
        "| Implementation | Median ms | Min–max ms | Accepted / attempted evidence | Remaining phases |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["measurements"]:
        lines.append(
            f"| {row['implementation']} | {row['median_ns'] / 1e6:.4f} | {row['min_ns'] / 1e6:.4f}–{row['max_ns'] / 1e6:.4f} | {row['accepted_observations']} / {row['attempted_observations']} | {row['remaining_phases']} |"
        )
    lines += [
        "",
        "The native row uses the public per-request wrapper, including checker launches, syntax construction, normalization, JSON transport, and stage reporting. The Python rows use exact expanded model limits, without general expression normalization or native transport. All rows enumerate the 60 candidate phases for reporting. The eager row stops after its second attempted observation, so it performs less work. These are implementation costs, not evidence of a general speed advantage for any choice strategy.",
        "",
        "## Evidence and outcomes",
        "",
        "| Stage | Stored mask period / active | Compatible phases out of 60 | Raw standard part | Fused standard part |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["traces"]["lean_delayed_public_api"]:
        raw = (
            row["raw_standard_part"]
            if row["raw_standard_part"] is not None
            else "unknown"
        )
        lines.append(
            f"| {row['stage']} | {row['stored_period']} / {len(row['stored_active_residues'])} | {len(row['remaining_phases'])} | {raw} | {row['fused_standard_part']} |"
        )
    lines += [
        "",
        "The no-backtracking eager policy chooses residue 1 after the coarse observation. The later condition n mod 6 = 5 is compatible with the original evidence but not with that extra commitment. Its rejection is sound relative to its chosen branch. Accepting all supplied evidence would require backtracking. The retained support and exhaustive reference instead end at residue 23 modulo 60.",
        "",
        "## Kernel replay",
        "",
    ]
    if report.get("replays"):
        lines += [
            "The early snapshots preceded later state changes. All three were checked after the full trace completed. Full verification wall times include the dependency-freshness build, Lean startup/elaboration, kernel checking, and axiom audit. They are measured once with existing build artifacts, separately from native execution timings.",
            "",
            "| Snapshot | Full verification seconds |",
            "|---|---:|",
        ]
        for replay_report in report["replays"]:
            lines.append(
                f"| [{replay_report['snapshot']}](snapshots/{replay_report['snapshot']}/Replay.lean) | {replay_report['elapsed_ns'] / 1e9:.3f} |"
            )
        lines += [
            "",
            "Full checker output, axiom reports, hashes, and timing samples are retained in results.json and the snapshot manifests.",
        ]
    else:
        lines.append(
            "Replay verification was not requested in this run. Native answers are not described as kernel-verified artifacts."
        )
    (output / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/delayed-choice")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--replay-timeout", type=float, default=180)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.replay_only and args.skip_replay:
        parser.error("--replay-only and --skip-replay are mutually exclusive")
    if args.replay_only:
        report = json.loads((args.output / "results.json").read_text())
    else:
        checker = ROOT / ".lake/build/bin/residue_checker"
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "repeats": args.repeat,
            "model_checks": validate_model(),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "lean_toolchain": (ROOT / "lean-toolchain").read_text().strip(),
                "checker_sha256": hashlib.sha256(checker.read_bytes()).hexdigest(),
            },
            **measure(args.repeat),
        }
    if not args.skip_replay:
        report["replays"] = replay(args.output, run_enumerated(), args.replay_timeout)
        report["replays_generated_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(args.output, report)
    print(f"Saved case-study artifacts in {args.output}")


if __name__ == "__main__":
    main()
