#!/usr/bin/env python3
"""An exact cubic difference quotient with optional kernel replay.

The original expression ((2 + epsilon)**3 - 2**3) / epsilon is sent to
LeanResidueSystem. Its standard part is 12, while its nonzero error has
standard part zero. Separate adaptive runs use an accepted parity answer to
choose the backward or forward quotient. Both branches have standard part 12.
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
class CubicExpressions:
    epsilon: ResidueHyperreal
    dy: ResidueHyperreal
    quotient: ResidueHyperreal
    error: ResidueHyperreal


@dataclass(frozen=True)
class AdaptiveResult:
    choice: bool
    branch: Literal["backward", "forward"]
    snapshot: ReplaySnapshot

    @property
    def name(self) -> str:
        return f"adaptive-{self.branch}"


@dataclass(frozen=True)
class CaseResult:
    before: ReplaySnapshot
    error: ReplaySnapshot
    after: ReplaySnapshot | None
    adaptive: tuple[AdaptiveResult, ...] = ()

    def snapshots(self) -> dict[str, ReplaySnapshot]:
        result = {"quotient": self.before, "error": self.error}
        if self.after is not None:
            result["after-odd-choice"] = self.after
        result.update({run.name: run.snapshot for run in self.adaptive})
        return result


def model(system: LeanResidueSystem) -> CubicExpressions:
    """Construct the unexpanded finite difference using exact constants."""
    epsilon, x = system.infinitesimal(), system.constant(Fraction(2))
    dy = (x + epsilon) ** 3 - x**3
    quotient = dy / epsilon
    return CubicExpressions(
        epsilon, dy, quotient, quotient - system.constant(Fraction(12))
    )


def run_adaptive(choice: bool) -> AdaptiveResult:
    """Use one accepted answer to select the quotient in a fresh system."""
    if type(choice) is not bool:
        raise ValueError("the adaptive choice must be a boolean")
    system = LeanResidueSystem()
    epsilon, x = system.infinitesimal(), system.constant(Fraction(2))
    if not system.commit(system.alt(), system.constant(Fraction(0)), truth=choice):
        raise LeanBackendError("the adaptive parity choice was unexpectedly rejected")
    if choice:
        branch: Literal["backward", "forward"] = "backward"
        quotient = (x**3 - (x - epsilon) ** 3) / epsilon
    else:
        branch = "forward"
        quotient = ((x + epsilon) ** 3 - x**3) / epsilon
    history, support = system.history, system.support
    snapshot = system.snapshot(quotient)
    if system.history != history or system.support != support:
        raise LeanBackendError(
            "adaptive extraction changed the accepted trace or support"
        )
    if snapshot.result != Fraction(12):
        raise LeanBackendError("the adaptive quotient has an unexpected standard part")
    return AdaptiveResult(choice, branch, snapshot)


def run_case(
    *, include_parity: bool = True, include_adaptive: bool = True
) -> CaseResult:
    """Capture choice-free results and optional fixed and adaptive parity runs."""
    system = LeanResidueSystem()
    expressions = model(system)
    zero, derivative = system.constant(Fraction(0)), system.constant(Fraction(12))
    # Probes return (can_be_false, can_be_true) and record no observations.
    if system.probe(zero, expressions.epsilon) != (False, True):
        raise LeanBackendError("the checker did not establish epsilon > 0")
    if system.probe(derivative, expressions.quotient) != (False, True):
        raise LeanBackendError("the checker did not establish quotient > 12")
    before, error = system.snapshot(expressions.quotient), system.snapshot(
        expressions.error
    )
    if before.result != Fraction(12) or error.result != Fraction(0):
        raise LeanBackendError(
            "the cubic standard parts differ from the expected exact values"
        )
    after = None
    if include_parity:
        if not system.commit(system.alt(), zero, truth=True):
            raise LeanBackendError("the odd parity choice was unexpectedly rejected")
        after = system.snapshot(expressions.quotient)
        if after.result != Fraction(12):
            raise LeanBackendError("the parity choice changed the cubic standard part")
    adaptive = (run_adaptive(True), run_adaptive(False)) if include_adaptive else ()
    return CaseResult(before, error, after, adaptive)


def export_case(
    case: CaseResult,
    output: Path,
    *,
    verify: bool = True,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Export immutable snapshots, optionally checking their generated claims."""
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    adaptive_runs = {run.name: run for run in case.adaptive}
    for name, snapshot in case.snapshots().items():
        directory = snapshot.export(output / "snapshots" / name, project_root=ROOT)
        record: dict[str, Any] = {
            "name": name,
            "result": str(snapshot.result),
            "support": list(snapshot.support),
            "accepted_observations": len(snapshot.observations),
            "verification": "not-run",
        }
        if name in adaptive_runs:
            run = adaptive_runs[name]
            record.update(
                {
                    "query": "(-1)^n < 0",
                    "choice": run.choice,
                    "branch": run.branch,
                    "quotient": (
                        "(2^3 - (2 - epsilon)^3) / epsilon"
                        if run.choice
                        else "((2 + epsilon)^3 - 2^3) / epsilon"
                    ),
                }
            )
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
        "case": "cubic-difference-quotient",
        "epsilon": "1/n",
        "dy": "(2 + epsilon)^3 - 2^3",
        "quotient": "dy / epsilon",
        "identity_at_positive_indices": "12 + 6*epsilon + epsilon^2",
        "ordinary_derivative": "12",
        "scope": (
            "An exact symbolic example. Replay proves the exported mathematical instances. "
            "The Lean adaptive-program theorem concerns its formal decision tree. "
            "The Python branch and expression mapping is tested, not formally refined."
        ),
        "snapshots": reports,
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/infinitesimal")
    parser.add_argument(
        "--no-parity", action="store_true", help="omit the later fixed odd choice"
    )
    parser.add_argument(
        "--no-adaptive", action="store_true", help="omit both adaptive branch runs"
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
        case = run_case(
            include_parity=not args.no_parity, include_adaptive=not args.no_adaptive
        )
        export_case(
            case, args.output, verify=not args.skip_replay, timeout=args.timeout
        )
    except (LeanBackendError, ValueError, OSError) as error:
        parser.exit(1, f"Infinitesimal example failed: {error}\n")
    print("epsilon = 1/n is positive and nonzero in every free completion.")
    print("dy/dx = 12 + 6*epsilon + epsilon^2, which is greater than 12.")
    print("st(dy/dx) = 12 and st(dy/dx - 12) = 0, without making a choice.")
    if case.after is not None:
        print("After choosing odd parity, st(dy/dx) is still 12.")
    for run in case.adaptive:
        print(
            f"Adaptive choice {str(run.choice).lower()}: {run.branch} quotient, "
            f"standard part {run.snapshot.result}."
        )
    print(
        "Snapshots exported without verification."
        if args.skip_replay
        else "All exported snapshots passed Lean kernel replay."
    )
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
