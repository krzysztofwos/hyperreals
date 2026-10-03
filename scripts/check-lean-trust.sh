#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
cd "$project_root"

lean_sources=(Hyperreals.lean PeriodicChecker.lean LaurentChecker.lean ResidueChecker.lean Hyperreals/*.lean)

for dependency in rg python3 lake; do
  if ! command -v "$dependency" >/dev/null; then
    echo "Lean trust audit failed: required command '$dependency' is not installed." >&2
    exit 1
  fi
done

if rg -n '\b(sorry|admit)\b' "${lean_sources[@]}"; then
  echo "Lean trust audit failed: unfinished proof found." >&2
  exit 1
fi

if rg -n '^[[:space:]]*((private|protected|public|noncomputable|unsafe)[[:space:]]+)*(axiom|constant)[[:space:]]' "${lean_sources[@]}"; then
  echo "Lean trust audit failed: project-local axiom or constant declaration found." >&2
  exit 1
fi

lake build

audited_modules=(
  Hyperreals/Completion.lean
  Hyperreals/Certificates.lean
  Hyperreals/EventuallyPeriodic.lean
  Hyperreals/StandardPart.lean
  Hyperreals/Counterexample.lean
  Hyperreals/Periodic.lean
  Hyperreals/RuntimeInvariants.lean
  Hyperreals/LaurentExpr.lean
  Hyperreals/LaurentSign.lean
  Hyperreals/LaurentLimit.lean
  Hyperreals/LaurentStandardPart.lean
  Hyperreals/LaurentRuntime.lean
  Hyperreals/LaurentTrace.lean
  Hyperreals/ResidueSupport.lean
  Hyperreals/ResidueExpr.lean
  Hyperreals/ResidueComparison.lean
  Hyperreals/ResidueLimit.lean
  Hyperreals/ResidueRuntime.lean
  Hyperreals/ResidueTrace.lean
  Hyperreals/ResidueReplay.lean
  Hyperreals/ObservationPrograms.lean
  Hyperreals/ObservationProgramExamples.lean
  Hyperreals/InfinitesimalCase.lean
  Hyperreals/AdaptiveInfinitesimal.lean
  Hyperreals/LaurentLimitCompleteness.lean
  Hyperreals/ResidueLimitCompleteness.lean
  Hyperreals/ResidueLimitDiagnostic.lean
  Hyperreals/PolynomialDifferentiation.lean
  Hyperreals/DomainInfinitesimal.lean
)

for module in "${audited_modules[@]}"; do
  audit_output="$(lake env lean "$module" 2>&1)"
  printf '%s\n' "$audit_output"
  printf '%s\n' "$audit_output" | python3 -c '
import re
import sys

reports = re.findall(r"depends on axioms:\s*\[([^]]*)\]", sys.stdin.read())
if not reports:
    sys.exit("Lean trust audit failed: audited module has no axiom reports.")
allowed = {"propext", "Classical.choice", "Quot.sound"}
dependencies = {name.strip() for report in reports for name in report.split(",") if name.strip()}
unexpected = dependencies - allowed
if unexpected:
    sys.exit("Lean trust audit failed: unexpected axioms: " + ", ".join(sorted(unexpected)))
'
done

echo "Lean trust audit passed."
