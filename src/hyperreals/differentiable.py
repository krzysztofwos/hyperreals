"""Symbolic forward differentiation with optional Lean kernel verification.

The exact output is an expression, including elementary functions. Numerical
samples are explicitly approximate. Domain obligations are retained even when
algebraic cancellation could hide a singularity. This language is separate from
the complete periodic Laurent standard-part decision procedure.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from .replay import _audit, _project_root, _run_lean, _sha, _validate_timeout

Scalar = int | Fraction
_BINARY = {"add", "sub", "mul", "div"}
_UNARY = {"neg", "sin", "cos", "exp", "log", "sqrt"}
_MAX_DIMENSION = 64
_MAX_NODES = 20_000
_MAX_DEPTH = 128


def _dimension(value: int) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_DIMENSION:
        raise ValueError(f"dimension must be an integer from 0 to {_MAX_DIMENSION}")


def _scalar(value: Scalar) -> Fraction:
    if type(value) not in (int, Fraction):
        raise TypeError("use int or Fraction for exact constants")
    result = Fraction(value)
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError("rational constants are limited to 4096 bits")
    return result


@dataclass(frozen=True)
class DifferentiableExpr:
    """An immutable, dimension-checked expression in real input variables."""

    inputs: int
    op: str
    args: tuple[DifferentiableExpr, ...] = ()
    value: Fraction | int | None = None

    def __post_init__(self) -> None:
        _dimension(self.inputs)
        if type(self.args) is not tuple:
            raise TypeError("expression children must be an immutable tuple")
        if type(self.op) is not str:
            raise TypeError("operator must be a string")
        if self.op == "constant":
            if self.args:
                raise ValueError("a constant has no children")
            object.__setattr__(self, "value", _scalar(self.value))  # type: ignore[arg-type]
        elif self.op == "var":
            if (
                self.args
                or type(self.value) is not int
                or not 0 <= self.value < self.inputs
            ):
                raise ValueError("variable index is outside its input dimension")
        elif self.op in _BINARY | _UNARY:
            if self.value is not None or len(self.args) != (
                2 if self.op in _BINARY else 1
            ):
                raise ValueError("invalid operator arity or payload")
        else:
            raise ValueError("unknown differentiable expression operator")
        if any(
            not isinstance(a, DifferentiableExpr) or a.inputs != self.inputs
            for a in self.args
        ):
            raise ValueError("expression input dimensions must agree")

    @classmethod
    def constant(cls, inputs: int, value: Scalar) -> DifferentiableExpr:
        return cls(inputs, "constant", value=_scalar(value))

    def _coerce(self, other: DifferentiableExpr | Scalar) -> DifferentiableExpr:
        return (
            other
            if isinstance(other, DifferentiableExpr)
            else self.constant(self.inputs, other)
        )

    def _binary(
        self, op: str, other: DifferentiableExpr | Scalar
    ) -> DifferentiableExpr:
        return DifferentiableExpr(self.inputs, op, (self, self._coerce(other)))

    def __add__(self, other: DifferentiableExpr | Scalar) -> DifferentiableExpr:
        return self._binary("add", other)

    def __radd__(self, other: Scalar) -> DifferentiableExpr:
        return self._coerce(other) + self

    def __sub__(self, other: DifferentiableExpr | Scalar) -> DifferentiableExpr:
        return self._binary("sub", other)

    def __rsub__(self, other: Scalar) -> DifferentiableExpr:
        return self._coerce(other) - self

    def __mul__(self, other: DifferentiableExpr | Scalar) -> DifferentiableExpr:
        return self._binary("mul", other)

    def __rmul__(self, other: Scalar) -> DifferentiableExpr:
        return self._coerce(other) * self

    def __truediv__(self, other: DifferentiableExpr | Scalar) -> DifferentiableExpr:
        return self._binary("div", other)

    def __rtruediv__(self, other: Scalar) -> DifferentiableExpr:
        return self._coerce(other) / self

    def __neg__(self) -> DifferentiableExpr:
        return self._unary("neg")

    def __bool__(self) -> bool:
        raise TypeError("symbolic expressions cannot control Python branches")

    def _unary(self, op: str) -> DifferentiableExpr:
        return DifferentiableExpr(self.inputs, op, (self,))

    def sin(self) -> DifferentiableExpr:
        return self._unary("sin")

    def cos(self) -> DifferentiableExpr:
        return self._unary("cos")

    def exp(self) -> DifferentiableExpr:
        return self._unary("exp")

    def log(self) -> DifferentiableExpr:
        return self._unary("log")

    def sqrt(self) -> DifferentiableExpr:
        return self._unary("sqrt")

    @property
    def domain_conditions(self) -> tuple[DomainCondition, ...]:
        conditions = [
            condition for arg in self.args for condition in arg.domain_conditions
        ]
        if self.op == "div":
            conditions.append(DomainCondition("nonzero", self.args[1]))
        elif self.op in {"log", "sqrt"}:
            conditions.append(DomainCondition("positive", self.args[0]))
        return tuple(dict.fromkeys(conditions))

    def approximate(self, point: Sequence[float | int | Fraction]) -> float:
        """Evaluate with floats, checking the smooth domain. This is not a proof."""
        if len(point) != self.inputs:
            raise ValueError("point dimension does not match the expression")
        values = tuple(float(value) for value in point)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("numerical inputs must be finite")
        return self._approximate(values)

    def _approximate(self, point: tuple[float, ...]) -> float:
        if self.op == "constant":
            return float(self.value)  # type: ignore[arg-type]
        if self.op == "var":
            return point[int(self.value)]  # type: ignore[arg-type]
        values = tuple(arg._approximate(point) for arg in self.args)
        a = values[0]
        if self.op == "add":
            result = a + values[1]
        elif self.op == "sub":
            result = a - values[1]
        elif self.op == "mul":
            result = a * values[1]
        elif self.op == "div":
            if values[1] == 0:
                raise ValueError("division requires a nonzero denominator")
            result = a / values[1]
        elif self.op == "neg":
            result = -a
        else:
            if self.op in {"log", "sqrt"} and a <= 0:
                raise ValueError(
                    f"{self.op} requires a positive argument in the smooth domain"
                )
            result = getattr(math, self.op)(a)
        if not math.isfinite(result):
            raise ValueError("numerical evaluation overflowed")
        return float(result)

    def __str__(self) -> str:
        if self.op == "constant":
            return str(self.value)
        if self.op == "var":
            return f"x[{self.value}]"
        if self.op in _BINARY:
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
            return f"({self.args[0]} {symbol} {self.args[1]})"
        return f"{self.op}({self.args[0]})"


@dataclass(frozen=True)
class DomainCondition:
    kind: Literal["positive", "nonzero"]
    expression: DifferentiableExpr

    def __str__(self) -> str:
        return f"{self.expression} {'> 0' if self.kind == 'positive' else '≠ 0'}"


def variables(inputs: int) -> tuple[DifferentiableExpr, ...]:
    """Construct the coordinates of a finite real input vector."""
    _dimension(inputs)
    return tuple(DifferentiableExpr(inputs, "var", value=i) for i in range(inputs))


def _differentiate(
    e: DifferentiableExpr, direction: tuple[DifferentiableExpr, ...]
) -> DifferentiableExpr:
    """Mirror Expr.jvp exactly. Kernel verification checks the resulting syntax."""
    if e.op == "constant":
        return e.constant(e.inputs, 0)
    if e.op == "var":
        return direction[int(e.value)]  # type: ignore[arg-type]
    a = e.args[0]
    da = _differentiate(a, direction)
    if e.op in _BINARY:
        b = e.args[1]
        db = _differentiate(b, direction)
        if e.op == "add":
            return da + db
        if e.op == "sub":
            return da - db
        if e.op == "mul":
            return da * b + a * db
        return (da * b - a * db) / (b * b)
    if e.op == "neg":
        return -da
    if e.op == "sin":
        return a.cos() * da
    if e.op == "cos":
        return -a.sin() * da
    if e.op == "exp":
        return a.exp() * da
    if e.op == "log":
        return da / a
    return da / (2 * a.sqrt())


@dataclass(frozen=True)
class DifferentiableProgram:
    inputs: int
    outputs: tuple[DifferentiableExpr, ...]

    def __post_init__(self) -> None:
        _dimension(self.inputs)
        if type(self.outputs) is not tuple:
            raise TypeError("program outputs must be an immutable tuple")
        _dimension(len(self.outputs))
        if any(
            not isinstance(e, DifferentiableExpr) or e.inputs != self.inputs
            for e in self.outputs
        ):
            raise ValueError("program input dimensions must agree")

    @property
    def domain_conditions(self) -> tuple[DomainCondition, ...]:
        return tuple(
            dict.fromkeys(c for e in self.outputs for c in e.domain_conditions)
        )

    def approximate(self, point: Sequence[float | int | Fraction]) -> tuple[float, ...]:
        if len(point) != self.inputs:
            raise ValueError("point dimension does not match the program")
        return tuple(e.approximate(point) for e in self.outputs)

    def jvp(self, direction: Iterable[DifferentiableExpr | Scalar]) -> CompiledJVP:
        tangents = tuple(
            (
                d
                if isinstance(d, DifferentiableExpr)
                else DifferentiableExpr.constant(self.inputs, d)
            )
            for d in direction
        )
        if len(tangents) != self.inputs or any(
            d.inputs != self.inputs for d in tangents
        ):
            raise ValueError("tangent dimension does not match the program")
        output = DifferentiableProgram(
            self.inputs, tuple(_differentiate(e, tangents) for e in self.outputs)
        )
        return CompiledJVP(self, tangents, output)

    def jacobian(self) -> tuple[tuple[DifferentiableExpr, ...], ...]:
        """Return symbolic rows (outputs) by columns (inputs), using basis JVPs."""
        columns = [
            self.jvp(int(i == j) for i in range(self.inputs)).derivative.outputs
            for j in range(self.inputs)
        ]
        return tuple(
            tuple(column[i] for column in columns) for i in range(len(self.outputs))
        )

    def gradient(self) -> tuple[DifferentiableExpr, ...]:
        if len(self.outputs) != 1:
            raise ValueError("gradient requires exactly one scalar output")
        return self.jacobian()[0]


@dataclass(frozen=True)
class JVPVerification:
    source_sha256: str
    axioms: tuple[str, ...]
    domain_verified: bool
    stdout: str


@dataclass(frozen=True)
class CompiledJVP:
    source: DifferentiableProgram
    direction: tuple[DifferentiableExpr, ...]
    derivative: DifferentiableProgram

    def __post_init__(self) -> None:
        if (
            type(self.direction) is not tuple
            or len(self.direction) != self.source.inputs
            or any(d.inputs != self.source.inputs for d in self.direction)
            or self.derivative.inputs != self.source.inputs
            or len(self.derivative.outputs) != len(self.source.outputs)
        ):
            raise ValueError("compiled JVP dimensions must agree")

    def lean_source(self, point: Sequence[Scalar] | None = None) -> str:
        """Generate fixed proof syntax. A point additionally requests a domain proof.

        No caller-supplied Lean source or tactics are accepted. The bounded domain
        tactic is incomplete. Failure to prove a domain is not proof of invalidity.
        """
        n, m = self.source.inputs, len(self.source.outputs)
        budget = [0]
        source = _lean_vector(tuple(_lean_expr(e, budget) for e in self.source.outputs))
        direction = _lean_vector(tuple(_lean_expr(e, budget) for e in self.direction))
        derivative = _lean_vector(
            tuple(_lean_expr(e, budget) for e in self.derivative.outputs)
        )
        text = f"""import Hyperreals.DifferentiableInfinitesimal
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Positivity

set_option autoImplicit false
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

open Filter Topology
open Hyperreals Hyperreals.Differentiable

namespace Hyperreals.GeneratedJVP

def source : Program {n} {m} := {source}
def direction : Fin {n} → Expr {n} := {direction}
def result : Program {n} {m} := {derivative}

theorem compiler_matches : ∀ i, result i = (source.jvp direction) i := by
  decide +kernel

theorem result_domain (x : Fin {n} → ℝ) (hdomain : source.Domain x)
    (hdirection : ∀ j, (direction j).Domain x) : result.Domain x := by
  intro i
  rw [compiler_matches i]
  exact (source i).domain_jvp direction x (hdomain i) hdirection

theorem derivative_correct (x : Fin {n} → ℝ) (hdomain : source.Domain x) :
    ∀ i, (result i).eval x = fderiv ℝ (source i).eval x (fun j => (direction j).eval x) := by
  intro i
  rw [compiler_matches i]
  exact (source i).jvp_eq_fderiv direction x (hdomain i)

theorem quotient_correct (x : Fin {n} → ℝ) (hdomain : source.Domain x)
    {{Γ : Commitments}} (C : Completion Γ) (h : Sequence)
    (hzero : NearStandardAt C.ultrafilter h 0)
    (hnonzero : ∀ᶠ k in (C.ultrafilter : Filter ℕ), h k ≠ 0) :
    ∀ i, NearStandardAt C.ultrafilter ((source i).quotient direction x h) ((result i).eval x) := by
  intro i
  rw [compiler_matches i]
  exact (source i).quotient_standardPart direction x (hdomain i) C h hzero hnonzero
"""
        if point is not None:
            if len(point) != n:
                raise ValueError("point dimension does not match the program")
            values = _lean_vector(tuple(_lean_rational(_scalar(x)) for x in point))
            text += f"""
noncomputable def point : Fin {n} → ℝ := {values}

theorem domain_at_point : source.Domain point := by
  intro i
  fin_cases i <;> norm_num [source, point, Program.Domain, Expr.Domain, Expr.eval] <;> positivity

theorem direction_domain_at_point : ∀ j, (direction j).Domain point := by
  intro j
  fin_cases j <;> norm_num [direction, point, Expr.Domain, Expr.eval] <;> positivity

theorem result_domain_at_point : result.Domain point :=
  result_domain point domain_at_point direction_domain_at_point

theorem quotient_at_point {{Γ : Commitments}} (C : Completion Γ) (h : Sequence)
    (hzero : NearStandardAt C.ultrafilter h 0)
    (hnonzero : ∀ᶠ k in (C.ultrafilter : Filter ℕ), h k ≠ 0) :
    ∀ i, NearStandardAt C.ultrafilter ((source i).quotient direction point h)
      ((result i).eval point) :=
  quotient_correct point domain_at_point C h hzero hnonzero
"""
        text += "\n" + "\n".join(
            f"#print axioms {root}" for root in _roots(point is not None)
        )
        return text + "\n\nend Hyperreals.GeneratedJVP\n"

    def verify(
        self,
        project_root: str | Path | None = None,
        *,
        point: Sequence[Scalar] | None = None,
        timeout: float = 60.0,
    ) -> JVPVerification:
        """Check compiler correspondence and semantics in Lean's kernel.

        Without a point the result is universally quantified over valid domains.
        With a point this also requires Lean to discharge its domain obligations.
        """
        _validate_timeout(timeout)
        source = self.lean_source(point)
        root = _project_root(project_root)
        deadline = time.monotonic() + timeout
        _run_lean(
            ["lake", "build", "Hyperreals.DifferentiableInfinitesimal"], root, deadline
        )
        with tempfile.TemporaryDirectory(prefix="hyperreals-jvp-") as directory:
            proof_file = Path(directory) / "JVP.lean"
            proof_file.write_text(source, encoding="utf-8")
            output = _run_lean(["lake", "env", "lean", str(proof_file)], root, deadline)
        axioms = _audit(output, _roots(point is not None))
        return JVPVerification(_sha(source), axioms, point is not None, output)


def _roots(at_point: bool) -> tuple[str, ...]:
    names: tuple[str, ...] = (
        "compiler_matches",
        "derivative_correct",
        "quotient_correct",
        "result_domain",
    )
    if at_point:
        names += (
            "domain_at_point",
            "direction_domain_at_point",
            "result_domain_at_point",
            "quotient_at_point",
        )
    return tuple(f"Hyperreals.GeneratedJVP.{name}" for name in names)


def _lean_rational(value: Fraction) -> str:
    return f"(({value.numerator}) / {value.denominator})"


def _lean_vector(entries: tuple[str, ...]) -> str:
    return "![" + ", ".join(entries) + "]" if entries else "(fun i => Fin.elim0 i)"


def _lean_expr(e: DifferentiableExpr, budget: list[int], depth: int = 0) -> str:
    budget[0] += 1
    if budget[0] > _MAX_NODES or depth > _MAX_DEPTH:
        raise ValueError("generated Lean syntax exceeds the node or depth budget")
    if e.op == "constant":
        return f"(.constant {_lean_rational(Fraction(e.value))})"  # type: ignore[arg-type]
    if e.op == "var":
        return f"(.var ⟨{e.value}, by decide⟩)"
    return (
        "(."
        + e.op
        + " "
        + " ".join(_lean_expr(a, budget, depth + 1) for a in e.args)
        + ")"
    )
