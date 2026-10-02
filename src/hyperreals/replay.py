"""Immutable replay data and generated, kernel-checked Lean proof artifacts.

The JSON and manifest record a claimed session, not its historical provenance.
Verification regenerates fixed Lean syntax from validated data and proves the
claim by kernel reduction. It never executes an exported source file directly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

from .verified import LeanBackendError

FORMAT = "hyperreals-residue-replay"
VERSION = 1
MAX_DEPTH = 64
MAX_NODES = 10_000
MAX_PERIOD = 4096
MAX_DENSE_SIZE = 4096
MAX_OBSERVATIONS = 128
MAX_DECIMAL_DIGITS = 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_TIMEOUT = 300.0
_ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z", re.ASCII)
_AST = tuple[object, ...]


class ReplayVerificationError(LeanBackendError):
    """An artifact association, Lean execution, or theorem audit failed."""


@dataclass(frozen=True)
class ReplayVerification:
    snapshot_sha256: str
    source_sha256: str
    axioms: tuple[str, ...]
    stdout: str


def _array(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be an array")
    return cast(list[Any] | tuple[Any, ...], value)


def _object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or value.keys() != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")
    return value


def _integer(value: Any, *, limited: bool = True) -> int:
    if (
        type(value) is not str
        or (limited and len(value.lstrip("-")) > MAX_DECIMAL_DIGITS)
        or _INTEGER.fullmatch(value) is None
    ):
        raise ValueError("integers must be bounded canonical ASCII decimal strings")
    return int(value)


def _rational(numerator: Any, denominator: Any, *, limited: bool = True) -> Fraction:
    n, d = _integer(numerator, limited=limited), _integer(denominator, limited=limited)
    if d <= 0:
        raise ValueError("rational denominator must be positive")
    return Fraction(n, d)


def _pair(value: Fraction) -> tuple[str, str]:
    return str(value.numerator), str(value.denominator)


def _period(left: int, right: int, *, limited: bool = True) -> int:
    result = math.lcm(left, right)
    if limited and result > MAX_PERIOD:
        raise ValueError(f"replay common period exceeds {MAX_PERIOD}")
    return result


@dataclass
class _Budget:
    nodes: int = 0
    limited: bool = True

    def spend(self, count: int = 1) -> None:
        self.nodes += count
        if self.limited and self.nodes > MAX_NODES:
            raise ValueError(f"replay exceeds {MAX_NODES} AST nodes/table entries")


def _normalize_ast(
    value: Any, budget: _Budget, depth: int = 0
) -> tuple[_AST, int, int, int]:
    """Return immutable syntax, coefficient period, numerator degree, and shift."""
    if budget.limited and depth > MAX_DEPTH:
        raise ValueError(f"replay AST depth exceeds {MAX_DEPTH}")
    budget.spend()
    node = _array(value, "expression")
    if not node or type(node[0]) is not str:
        raise ValueError("expression must start with an operator string")
    tag = node[0]
    if tag == "const" and len(node) == 3:
        return (
            (tag, *_pair(_rational(node[1], node[2], limited=budget.limited))),
            1,
            0,
            0,
        )
    if tag in ("index", "invn") and len(node) == 1:
        return (tag,), 1, int(tag == "index"), int(tag == "invn")
    if tag == "periodic" and len(node) == 2:
        table = _array(node[1], "periodic table")
        if not table or (budget.limited and len(table) > MAX_PERIOD):
            raise ValueError(f"periodic tables need 1 to {MAX_PERIOD} entries")
        budget.spend(len(table))
        entries = []
        for entry in table:
            pair = _array(entry, "periodic rational")
            if len(pair) != 2:
                raise ValueError("periodic entries must be rational pairs")
            entries.append(_pair(_rational(*pair, limited=budget.limited)))
        return (tag, tuple(entries)), len(entries), 0, 0
    if tag in ("add", "sub", "mul") and len(node) == 3:
        left, lp, ld, ls = _normalize_ast(node[1], budget, depth + 1)
        right, rp, rd, rs = _normalize_ast(node[2], budget, depth + 1)
        period, shift = _period(lp, rp, limited=budget.limited), ls + rs
        degree = ld + rd if tag == "mul" else max(ld + rs, rd + ls)
        result: _AST = (tag, left, right)
    elif tag == "divMonomial" and len(node) == 5:
        argument, period, degree, shift = _normalize_ast(node[1], budget, depth + 1)
        coefficient = _rational(node[2], node[3], limited=budget.limited)
        power = _integer(node[4], limited=budget.limited)
        if not coefficient:
            raise ValueError("monomial divisor must be nonzero")
        shift += max(power, 0)
        degree += max(-power, 0)
        result = (tag, argument, *_pair(coefficient), str(power))
    else:
        raise ValueError(f"unsupported expression operator or arity: {tag!r}")
    if budget.limited and (degree >= MAX_DENSE_SIZE or shift >= MAX_DENSE_SIZE):
        raise ValueError(f"replay dense polynomial/shift exceeds {MAX_DENSE_SIZE - 1}")
    return result, period, degree, shift


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    )


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(text: str) -> Any:
    if type(text) is not str or len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("replay JSON exceeds the size limit or is not text")

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"non-JSON numeric constant: {value}")

    try:
        return json.loads(
            text, object_pairs_hook=_reject_duplicates, parse_constant=invalid_constant
        )
    except (RecursionError, UnicodeError) as error:
        raise ValueError("invalid or excessively nested replay JSON") from error


@dataclass(frozen=True)
class ReplayObservation:
    left_ast: _AST
    right_ast: _AST
    op: Literal["lt", "eq"]
    truth: bool

    def __post_init__(self) -> None:
        if type(self.op) is not str or self.op not in ("lt", "eq"):
            raise ValueError("replay operation must be lt or eq")
        if type(self.truth) is not bool:
            raise ValueError("replay choice must be a boolean")
        budget = _Budget(limited=False)
        left, lp, _, _ = _normalize_ast(self.left_ast, budget)
        right, rp, _, _ = _normalize_ast(self.right_ast, budget)
        _period(lp, rp, limited=False)
        object.__setattr__(self, "left_ast", left)
        object.__setattr__(self, "right_ast", right)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": _json_value(self.left_ast),
            "right": _json_value(self.right_ast),
            "op": self.op,
            "truth": self.truth,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ReplayObservation:
        obj = _object(value, {"left", "right", "op", "truth"}, "observation")
        budget = _Budget()
        _normalize_ast(obj["left"], budget)
        _normalize_ast(obj["right"], budget)
        return cls(obj["left"], obj["right"], obj["op"], obj["truth"])


@dataclass(frozen=True)
class ReplaySnapshot:
    observations: tuple[ReplayObservation, ...]
    support: tuple[bool, ...]
    expression: _AST
    result: Fraction | None

    def __post_init__(self) -> None:
        observations = _array(self.observations, "observations")
        if len(observations) > MAX_OBSERVATIONS:
            raise ValueError(f"replay exceeds {MAX_OBSERVATIONS} observations")
        support = _array(self.support, "support")
        if (
            not support
            or len(support) > MAX_PERIOD
            or any(type(bit) is not bool for bit in support)
            or not any(support)
        ):
            raise ValueError("support must be a bounded nonempty Boolean mask")
        if self.result is not None and type(self.result) is not Fraction:
            raise ValueError("replay result must be Fraction or None")
        if self.result is not None:
            _rational(*_pair(self.result))
        budget, period = _Budget(), len(support)
        immutable = []
        for observation in observations:
            if type(observation) is not ReplayObservation:
                raise ValueError("observations must be ReplayObservation objects")
            copied = ReplayObservation(
                observation.left_ast,
                observation.right_ast,
                observation.op,
                observation.truth,
            )
            _, lp, _, _ = _normalize_ast(copied.left_ast, budget)
            _, rp, _, _ = _normalize_ast(copied.right_ast, budget)
            period = _period(period, _period(lp, rp))
            immutable.append(copied)
        expression, expression_period, _, _ = _normalize_ast(self.expression, budget)
        _period(period, expression_period)
        object.__setattr__(self, "observations", tuple(immutable))
        object.__setattr__(self, "support", tuple(support))
        object.__setattr__(self, "expression", expression)
        if len(self.to_json().encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("replay JSON exceeds the size limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "support": list(self.support),
            "expression": _json_value(self.expression),
            "result": None if self.result is None else list(_pair(self.result)),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ReplaySnapshot:
        obj = _object(
            value,
            {"format", "version", "observations", "support", "expression", "result"},
            "snapshot",
        )
        if (
            obj["format"] != FORMAT
            or type(obj["version"]) is not int
            or obj["version"] != VERSION
        ):
            raise ValueError("unsupported replay format or version")
        observations = _array(obj["observations"], "observations")
        if len(observations) > MAX_OBSERVATIONS:
            raise ValueError(f"replay exceeds {MAX_OBSERVATIONS} observations")
        result = obj["result"]
        if result is not None:
            pair = _array(result, "result")
            if len(pair) != 2:
                raise ValueError("result must be a rational pair or null")
            result = _rational(*pair)
        return cls(
            tuple(ReplayObservation.from_dict(item) for item in observations),
            obj["support"],
            obj["expression"],
            result,
        )

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> ReplaySnapshot:
        return cls.from_dict(_load_json(text))

    def to_lean(self) -> str:
        """Generate fixed declarations. No caller-supplied Lean text is accepted."""
        observations = ",\n    ".join(
            f"⟨.{item.op}, {_lean_ast(item.left_ast)}, {_lean_ast(item.right_ast)}, "
            f"{str(item.truth).lower()}⟩"
            for item in self.observations
        )
        result = "none" if self.result is None else f"some {_lean_rat(self.result)}"
        source = f"""import Hyperreals.ResidueReplay

set_option autoImplicit false
set_option relaxedAutoImplicit false
set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

open Hyperreals Hyperreals.Residue Hyperreals.Residue.Replay

namespace Hyperreals.GeneratedReplay

def snapshot : Snapshot := {{
  observations := [{observations}]
  support := [{', '.join(str(bit).lower() for bit in self.support)}]
  expression := {_lean_ast(self.expression)}
  result := {result}
}}

theorem replay_check : snapshot.check = true := by decide +kernel

theorem trace_consistent : Extendible snapshot.commitments :=
  Snapshot.trace_extendible replay_check

theorem extraction_matches : standardPart snapshot.support snapshot.expression = snapshot.result :=
  Snapshot.checked_result replay_check

theorem support_correspondence (U : Ultrafilter ℕ)
    (hfree : (U : Filter ℕ) ≤ Filter.cofinite) :
    snapshot.support.carrier ∈ U ↔
      ∀ observation ∈ snapshot.observations, observation.denote ∈ U :=
  Snapshot.support_mem_iff replay_check U hfree

"""
        if self.result is None:
            source += """theorem extractor_unknown : standardPart snapshot.support snapshot.expression = none :=
  Snapshot.unknown_result replay_check rfl

"""
        else:
            rational = _lean_rat(self.result)
            source += f"""theorem replayed_standard_part :
    ∀ C : Completion snapshot.commitments,
      NearStandardAt C.ultrafilter snapshot.expression.denote (({rational} : Rat) : ℝ) :=
  Snapshot.standardPart_sound replay_check rfl

"""
        source += "\n".join(f"#print axioms {name}" for name in _roots(self.result))
        return source + "\n\nend Hyperreals.GeneratedReplay\n"

    def export(
        self, directory: str | Path, *, project_root: str | Path | None = None
    ) -> Path:
        """Write data, generated source, and provenance. This does not verify them."""
        root = _project_root(project_root)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        snapshot, source = self.to_json(), self.to_lean()
        manifest = _manifest(root, snapshot, source)
        for name, text in (
            ("snapshot.json", snapshot),
            ("Replay.lean", source),
            ("manifest.json", _canonical_json(manifest)),
        ):
            path = directory / name
            if path.is_symlink():
                raise ReplayVerificationError(
                    f"refusing to overwrite symbolic link: {path}"
                )
            path.write_text(text, encoding="utf-8")
        return directory

    def verify(
        self, project_root: str | Path | None = None, *, timeout: float = 60.0
    ) -> ReplayVerification:
        """Rebuild imports, then kernel-check regenerated source within one timeout."""
        _validate_timeout(timeout)
        root = _project_root(project_root)
        source = self.to_lean()
        deadline = time.monotonic() + timeout
        _run_lean(["lake", "build", "Hyperreals.ResidueReplay"], root, deadline)
        with tempfile.TemporaryDirectory(prefix="hyperreals-replay-") as temporary:
            path = Path(temporary) / "Replay.lean"
            path.write_text(source, encoding="utf-8")
            output = _run_lean(["lake", "env", "lean", str(path)], root, deadline)
        axioms = _audit(output, _roots(self.result))
        return ReplayVerification(_sha(self.to_json()), _sha(source), axioms, output)


def _lean_rat(value: Fraction) -> str:
    return f"(({value.numerator} : Rat) / ({value.denominator} : Rat))"


def _lean_ast(ast: _AST) -> str:
    tag = ast[0]
    if tag == "const":
        return f"(.constant {_lean_rat(_rational(ast[1], ast[2]))})"
    if tag == "index":
        return ".index"
    if tag == "invn":
        return ".reciprocalIndex"
    if tag == "periodic":
        table = _array(ast[1], "periodic table")
        return (
            "(.periodic ["
            + ", ".join(_lean_rat(_rational(*pair)) for pair in table)
            + "])"
        )
    if tag == "divMonomial":
        return (
            f"(.divMonomial {_lean_ast(ast[1])} "  # type: ignore[arg-type]
            f"{_lean_rat(_rational(ast[2], ast[3]))} ({ast[4]} : Int))"
        )
    return f"(.{tag} {_lean_ast(ast[1])} {_lean_ast(ast[2])})"  # type: ignore[arg-type]


def _roots(result: Fraction | None) -> tuple[str, ...]:
    names = (
        "replay_check",
        "trace_consistent",
        "extraction_matches",
        "support_correspondence",
        "extractor_unknown" if result is None else "replayed_standard_part",
    )
    return tuple(f"Hyperreals.GeneratedReplay.{name}" for name in names)


def _audit(output: str, roots: tuple[str, ...]) -> tuple[str, ...]:
    reports = re.findall(
        r"'([^']+)'\s+(?:depends on axioms:\s*\[([^]]*)\]|(does not depend on any axioms))",
        output,
    )
    found: dict[str, set[str]] = {}
    for name, dependencies, _ in reports:
        axioms = {item.strip() for item in dependencies.split(",") if item.strip()}
        if axioms - _ALLOWED_AXIOMS:
            raise ReplayVerificationError(
                f"unexpected replay axioms: {sorted(axioms - _ALLOWED_AXIOMS)}"
            )
        if name in found:
            raise ReplayVerificationError(f"duplicate theorem audit report: {name}")
        found[name] = axioms
    if any(root not in found for root in roots):
        raise ReplayVerificationError("missing generated theorem axiom reports")
    return tuple(sorted(set().union(*(found[root] for root in roots))))


def _validate_timeout(timeout: float) -> None:
    if (
        type(timeout) not in (float, int)
        or not math.isfinite(timeout)
        or timeout <= 0
        or timeout > MAX_TIMEOUT
    ):
        raise ValueError(
            f"timeout must be positive, finite, and at most {MAX_TIMEOUT:g} seconds"
        )


def _run_lean(command: list[str], root: Path, deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ReplayVerificationError("Lean replay exceeded its total timeout")
    try:
        process = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=remaining,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReplayVerificationError(f"Lean replay failed: {error}") from error
    output = process.stdout + process.stderr
    if process.returncode != 0 or re.search(r"\berror:", output):
        raise ReplayVerificationError(f"Lean replay did not verify:\n{output}")
    return output


def _project_root(project_root: str | Path | None) -> Path:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if (
        not (root / "lean-toolchain").is_file()
        or not (root / "Hyperreals/ResidueReplay.lean").is_file()
    ):
        raise ReplayVerificationError(
            "project_root must contain lean-toolchain and ResidueReplay.lean"
        )
    return root


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest(root: Path, snapshot: str, source: str) -> dict[str, Any]:
    paths = sorted(
        set(root.glob("Hyperreals/*.lean"))
        | set(root.glob("*.lean"))
        | {
            root / name
            for name in (
                "Hyperreals.lean",
                "lean-toolchain",
                "lakefile.toml",
                "lake-manifest.json",
                "src/hyperreals/replay.py",
                "src/hyperreals/verified_residue.py",
            )
            if (root / name).is_file()
        }
    )
    return {
        "format": FORMAT + "-manifest",
        "version": VERSION,
        "snapshot_sha256": _sha(snapshot),
        "source_sha256": _sha(source),
        "provenance": {
            "lean_toolchain": (root / "lean-toolchain")
            .read_text(encoding="utf-8")
            .strip(),
            "files": {
                path.relative_to(root)
                .as_posix(): hashlib.sha256(path.read_bytes())
                .hexdigest()
                for path in paths
            },
        },
    }


def verify_export(
    directory: str | Path,
    project_root: str | Path | None = None,
    *,
    timeout: float = 60.0,
) -> ReplayVerification:
    """Validate artifact association, then regenerate the typed claim for Lean.

    Hashes bind the local artifact files to one snapshot and current project.
    They are not signatures or proof that the snapshot records an external run.
    """
    _validate_timeout(timeout)
    root, directory = _project_root(project_root), Path(directory).resolve()
    try:
        paths = [
            directory / name
            for name in ("snapshot.json", "Replay.lean", "manifest.json")
        ]
        if any(
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 2 * MAX_JSON_BYTES
            for path in paths
        ):
            raise ValueError(
                "replay artifacts must be bounded regular files, not symbolic links"
            )
        snapshot_text, source, manifest_text = (
            path.read_text(encoding="utf-8") for path in paths
        )
        snapshot = ReplaySnapshot.from_json(snapshot_text)
        if snapshot_text != snapshot.to_json():
            raise ValueError("snapshot JSON is not in canonical export form")
        if source != snapshot.to_lean():
            raise ValueError("exported Lean source differs from regenerated snapshot")
        manifest = _load_json(manifest_text)
        if manifest_text != _canonical_json(_manifest(root, snapshot_text, source)):
            raise ValueError("manifest hashes or project provenance do not match")
    except (OSError, UnicodeError, ValueError) as error:
        raise ReplayVerificationError(f"invalid replay export: {error}") from error
    result = snapshot.verify(root, timeout=timeout)
    try:
        if manifest != _manifest(root, snapshot_text, source):
            raise ValueError("project provenance changed during verification")
        for path, text in zip(paths, (snapshot_text, source, manifest_text)):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_text(encoding="utf-8") != text
            ):
                raise ValueError("exported artifacts changed during verification")
    except (OSError, UnicodeError, ValueError) as error:
        raise ReplayVerificationError(f"invalid replay export: {error}") from error
    return result
