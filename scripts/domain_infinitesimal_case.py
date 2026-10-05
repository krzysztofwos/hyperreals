#!/usr/bin/env python3
"""A checked observation establishes whether an infinitesimal is invertible.

The step h is zero at even indices and 1/n at odd indices. A false answer to
h = 0 allows its represented reciprocal [0, 1] * n. A true answer requires a
fallback step, 1/n**2. Both resulting cubic quotients have standard part 12.
The residue language has only monomial division.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from hyperreals import (
    LeanBackendError,
    LeanResidueSystem,
    ReplaySnapshot,
    ResidueHyperreal,
    verify_export,
)
from hyperreals.replay import MAX_TIMEOUT

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DomainExpressions:
    step: ResidueHyperreal
    reciprocal: ResidueHyperreal
    guarded: ResidueHyperreal
    fallback: ResidueHyperreal
    fallback_step: ResidueHyperreal


@dataclass(frozen=True)
class DomainRun:
    choice: bool
    branch: Literal["fallback", "nonzero"]
    snapshot: ReplaySnapshot
    unrefined: ReplaySnapshot

    @property
    def name(self) -> str:
        return f"domain-{self.branch}"


def model(system: LeanResidueSystem) -> DomainExpressions:
    """Keep the numerator intact and explicitly represent the partial inverse."""
    indicator = system.periodic([0, 1])
    epsilon, x = system.infinitesimal(), system.constant(2)
    step = indicator * epsilon
    reciprocal = indicator * system.infinite()
    guarded = ((x + step) ** 3 - x**3) * reciprocal
    fallback_step = epsilon * epsilon
    fallback = ((x + fallback_step) ** 3 - x**3).divide_monomial(1, -2)
    return DomainExpressions(step, reciprocal, guarded, fallback, fallback_step)


def run_branch(choice: bool) -> DomainRun:
    """Use h = 0 to choose a valid denominator in one fresh completion state."""
    if type(choice) is not bool:
        raise ValueError("the domain choice must be a boolean")
    system = LeanResidueSystem()
    expressions = model(system)
    zero, one = system.constant(0), system.constant(1)
    if expressions.step.standard_part() != Fraction(0):
        raise LeanBackendError(
            "the original step must be infinitesimal before choosing"
        )
    if system.probe(expressions.step, zero, "eq") != (True, True):
        raise LeanBackendError("both zero and nonzero steps should remain possible")
    unrefined = system.snapshot(expressions.guarded)
    if unrefined.result is not None:
        raise LeanBackendError(
            "the unrefined represented quotient has no common standard part"
        )
    if not system.commit(expressions.step, zero, "eq", truth=choice):
        raise LeanBackendError("the domain observation was unexpectedly rejected")
    if choice:
        branch: Literal["fallback", "nonzero"] = "fallback"
        selected_step, quotient = expressions.fallback_step, expressions.fallback
        expected_inverse = (True, False)
    else:
        branch = "nonzero"
        selected_step, quotient = expressions.step, expressions.guarded
        expected_inverse = (False, True)
    history, support = system.history, system.support
    if system.probe(selected_step, zero, "eq") != (True, False):
        raise LeanBackendError("the selected step is not provably nonzero")
    if (
        system.probe(expressions.step * expressions.reciprocal, one, "eq")
        != expected_inverse
    ):
        raise LeanBackendError(
            "the represented inverse does not match the selected domain"
        )
    snapshot = system.snapshot(quotient)
    if system.history != history or system.support != support:
        raise LeanBackendError(
            "domain checking or extraction changed the accepted trace"
        )
    if snapshot.result != Fraction(12):
        raise LeanBackendError("the selected quotient has an unexpected standard part")
    return DomainRun(choice, branch, snapshot, unrefined)


def run_case() -> tuple[DomainRun, DomainRun]:
    return run_branch(True), run_branch(False)


def export_case(
    runs: tuple[DomainRun, DomainRun],
    output: Path,
    *,
    verify: bool = True,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Export both branches and the failure before their domain observation."""
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for run in runs:
        directory = run.snapshot.export(
            output / "snapshots" / run.name, project_root=ROOT
        )
        record: dict[str, Any] = {
            "name": run.name,
            "query": "h = 0",
            "choice": run.choice,
            "selected_step": "epsilon^2" if run.choice else "h",
            "support": list(run.snapshot.support),
            "accepted_observations": len(run.snapshot.observations),
            "result": str(run.snapshot.result),
            "verification": "not-run",
        }
        if verify:
            checked = verify_export(directory, project_root=ROOT, timeout=timeout)
            record.update(
                {
                    "verification": "kernel-checked",
                    "snapshot_sha256": checked.snapshot_sha256,
                    "source_sha256": checked.source_sha256,
                    "axioms": list(checked.axioms),
                    "lean_output": checked.stdout,
                }
            )
        reports.append(record)
    unrefined = runs[0].unrefined
    directory = unrefined.export(
        output / "snapshots" / "domain-unrefined", project_root=ROOT
    )
    record = {
        "name": "domain-unrefined",
        "support": list(unrefined.support),
        "accepted_observations": 0,
        "result": None,
        "verification": "not-run",
    }
    if verify:
        checked = verify_export(directory, project_root=ROOT, timeout=timeout)
        record.update(
            {
                "verification": "kernel-checked",
                "snapshot_sha256": checked.snapshot_sha256,
                "source_sha256": checked.source_sha256,
                "axioms": list(checked.axioms),
                "lean_output": checked.stdout,
            }
        )
    reports.append(record)
    report = {
        "case": "observation-dependent-infinitesimal-domain",
        "h": "periodic([0, 1]) * epsilon",
        "represented_reciprocal": "periodic([0, 1]) * n",
        "ordinary_derivative": "12",
        "scope": (
            "The formal domain program proves that the selected step is nonzero and the "
            "returned expression denotes its literal cubic quotient in every completion of "
            "the accepted trace. The represented reciprocal is justified only after the "
            "negative observation. This does not add general division to the grammar. "
            "Replay checks exported instances. Python-to-Lean capture is tested, not proved."
        ),
        "snapshots": reports,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/domain-infinitesimal"
    )
    parser.add_argument(
        "--skip-replay", action="store_true", help="export without kernel checking"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="seconds per snapshot verification"
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= MAX_TIMEOUT:
        parser.error(f"--timeout must be positive, finite, and at most {MAX_TIMEOUT:g}")
    try:
        runs = run_case()
        export_case(
            runs, args.output, verify=not args.skip_replay, timeout=args.timeout
        )
    except (LeanBackendError, ValueError, OSError) as error:
        parser.exit(1, f"Domain example failed: {error}\n")
    print(
        "h = periodic([0, 1]) * epsilon is infinitesimal before making an observation."
    )
    print("If h = 0, replace it with epsilon^2 before dividing.")
    print("If h != 0, periodic([0, 1]) * n is a proved represented reciprocal.")
    print("Both selected nonzero steps give a cubic quotient with standard part 12.")
    print(
        "Snapshots exported without verification."
        if args.skip_replay
        else "Both branches and the unrefined failure passed Lean kernel replay."
    )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
