"""Exact eventual set semantics for a decidable comparison fragment.

Free ultrafilters identify sets that differ at only finitely many indices. An
EventuallyPeriodicSet therefore records only a represented periodic tail: for
every n at or beyond the cutoff, membership is determined by n modulo period.

The representation is closed under complement and intersection, and infinitude
is decidable from the recurring residues. This gives the runtime a semantic
finite-intersection check for finite/cofinite and periodic comparisons without
claiming to decide arbitrary sequence predicates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Optional, Tuple

from .algebra import Atom, Complement, Empty, FiniteSetExpr, Intersect, SetExpr, Universe
from .exact_arithmetic import exact_laurent_coefficients
from .sequence import Add, AltSign, Const, Div, Exp, InvN, Mul, NVar, Seq, Sub


@dataclass(frozen=True)
class EventuallyPeriodicSet:
    """A subset of the naturals represented exactly from some cutoff onward."""

    cutoff: int
    period: int
    residues: frozenset[int]

    def __post_init__(self) -> None:
        if self.cutoff < 0:
            raise ValueError("cutoff must be nonnegative")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if any(residue < 0 or residue >= self.period for residue in self.residues):
            raise ValueError("residues must lie in [0, period)")

    @classmethod
    def universe(cls, *, cutoff: int = 0) -> "EventuallyPeriodicSet":
        return cls(cutoff=cutoff, period=1, residues=frozenset({0}))

    @classmethod
    def empty(cls, *, cutoff: int = 0) -> "EventuallyPeriodicSet":
        return cls(cutoff=cutoff, period=1, residues=frozenset())

    def complement(self) -> "EventuallyPeriodicSet":
        return EventuallyPeriodicSet(
            cutoff=self.cutoff,
            period=self.period,
            residues=frozenset(set(range(self.period)) - set(self.residues)),
        )

    def intersect(self, other: "EventuallyPeriodicSet") -> "EventuallyPeriodicSet":
        period = lcm(self.period, other.period)
        residues = frozenset(
            residue
            for residue in range(period)
            if residue % self.period in self.residues
            and residue % other.period in other.residues
        )
        return EventuallyPeriodicSet(
            cutoff=max(self.cutoff, other.cutoff),
            period=period,
            residues=residues,
        )

    def is_infinite(self) -> bool:
        """Whether the represented set contains infinitely many indices."""
        return bool(self.residues)

    def is_cofinite(self) -> bool:
        """Whether the represented set contains every sufficiently large index."""
        return len(self.residues) == self.period


def _combine_periodic_values(
    left: Tuple[Fraction, ...],
    right: Tuple[Fraction, ...],
    operator: str,
) -> Optional[Tuple[Fraction, ...]]:
    period = lcm(len(left), len(right))
    values = []
    for index in range(period):
        a = left[index % len(left)]
        b = right[index % len(right)]
        if operator == "add":
            value = a + b
        elif operator == "sub":
            value = a - b
        elif operator == "mul":
            value = a * b
        elif operator == "div":
            if b == 0:
                return None
            value = a / b
        else:
            raise ValueError(f"unsupported periodic operator: {operator}")
        values.append(value)
    return tuple(values)


def _periodic_values(sequence: Seq) -> Optional[Tuple[Fraction, ...]]:
    """Evaluate the exact finite pattern of the algebraic periodic fragment."""
    if isinstance(sequence, Const):
        return (Fraction.from_float(sequence.c),) if math.isfinite(sequence.c) else None
    if isinstance(sequence, AltSign):
        return (Fraction(1), Fraction(-1))
    if isinstance(sequence, (Add, Sub, Mul, Div)):
        left = _periodic_values(sequence.left)
        right = _periodic_values(sequence.right)
        if left is None or right is None:
            return None
        operator = {
            Add: "add",
            Sub: "sub",
            Mul: "mul",
            Div: "div",
        }[type(sequence)]
        return _combine_periodic_values(left, right, operator)
    return None


def _eventual_laurent_sign(sequence: Seq) -> Optional[Tuple[int, int]]:
    """Return an eventual sign and a conservative cutoff for an exact Laurent polynomial."""
    coefficients = exact_laurent_coefficients(sequence)
    if coefficients is None:
        return None
    if not coefficients:
        return (0, 1)

    leading_power = min(coefficients)
    leading = coefficients[leading_power]
    tail_bound = sum(
        abs(value)
        for power, value in coefficients.items()
        if power > leading_power
    )
    ratio = 2 * tail_bound / abs(leading)
    cutoff = max(1, math.floor(ratio) + 1)
    return (1 if leading > 0 else -1, cutoff)


def _total_cutoff(sequence: Seq) -> Optional[int]:
    """Recognize a tail on which a supported expression is real and defined."""
    if _periodic_values(sequence) is not None:
        return 0
    if exact_laurent_coefficients(sequence) is not None:
        return 1
    if isinstance(sequence, Exp):
        return _total_cutoff(sequence.arg)
    return None


def _constant_index_set(
    op: str,
    constant: float,
    *,
    constant_on_left: bool,
) -> Optional[EventuallyPeriodicSet]:
    if not math.isfinite(constant):
        return None
    cutoff = max(0, math.floor(constant) + 1)
    if op == "EQ":
        return EventuallyPeriodicSet.empty(cutoff=cutoff)
    if op != "LT":
        return None
    if constant_on_left:
        return EventuallyPeriodicSet.universe(cutoff=cutoff)
    return EventuallyPeriodicSet.empty(cutoff=cutoff)


def _constant_reciprocal_set(
    op: str,
    constant: float,
    *,
    constant_on_left: bool,
) -> Optional[EventuallyPeriodicSet]:
    if not math.isfinite(constant):
        return None
    exact_constant = Fraction.from_float(constant)
    if op == "EQ":
        cutoff = 1 if constant <= 0.0 else max(1, math.floor(1 / exact_constant) + 2)
        return EventuallyPeriodicSet.empty(cutoff=cutoff)
    if op != "LT":
        return None

    if constant_on_left:
        # c < 1/n
        if constant <= 0.0:
            return EventuallyPeriodicSet.universe(cutoff=1)
        return EventuallyPeriodicSet.empty(cutoff=max(1, math.ceil(1 / exact_constant)))

    # 1/n < c
    if constant <= 0.0:
        return EventuallyPeriodicSet.empty(cutoff=1)
    return EventuallyPeriodicSet.universe(
        cutoff=max(1, math.floor(1 / exact_constant) + 1)
    )


def _index_reciprocal_set(
    op: str,
    *,
    index_on_left: bool,
) -> Optional[EventuallyPeriodicSet]:
    if op == "EQ":
        return EventuallyPeriodicSet.empty(cutoff=2)
    if op != "LT":
        return None
    if index_on_left:
        return EventuallyPeriodicSet.empty(cutoff=1)
    return EventuallyPeriodicSet.universe(cutoff=2)


def _exp_index_constant_set(
    op: str,
    constant: float,
    *,
    constant_on_left: bool,
) -> Optional[EventuallyPeriodicSet]:
    if op != "LT" or not math.isfinite(constant):
        return None
    if constant <= 0.0:
        cutoff = 0
    else:
        cutoff = max(0, math.floor(math.log(constant)) + 2)
    if constant_on_left:
        return EventuallyPeriodicSet.universe(cutoff=cutoff)
    return EventuallyPeriodicSet.empty(cutoff=cutoff)


def _atom_semantics(atom: Atom) -> Optional[EventuallyPeriodicSet]:
    left = atom.a
    right = atom.b

    if left.key() == right.key():
        cutoff = _total_cutoff(left)
        if cutoff is not None:
            if atom.op == "EQ":
                return EventuallyPeriodicSet.universe(cutoff=cutoff)
            if atom.op == "LT":
                return EventuallyPeriodicSet.empty(cutoff=cutoff)
            return None

    left_values = _periodic_values(left)
    right_values = _periodic_values(right)
    if left_values is not None and right_values is not None:
        period = lcm(len(left_values), len(right_values))
        if atom.op == "EQ":
            residues = frozenset(
                index
                for index in range(period)
                if left_values[index % len(left_values)]
                == right_values[index % len(right_values)]
            )
        elif atom.op == "LT":
            residues = frozenset(
                index
                for index in range(period)
                if left_values[index % len(left_values)]
                < right_values[index % len(right_values)]
            )
        else:
            return None
        return EventuallyPeriodicSet(cutoff=0, period=period, residues=residues)

    difference_sign = _eventual_laurent_sign(Sub(right, left))
    if difference_sign is not None:
        sign, cutoff = difference_sign
        if atom.op == "EQ":
            if sign == 0:
                return EventuallyPeriodicSet.universe(cutoff=cutoff)
            return EventuallyPeriodicSet.empty(cutoff=cutoff)
        if atom.op == "LT":
            if sign == 1:
                return EventuallyPeriodicSet.universe(cutoff=cutoff)
            return EventuallyPeriodicSet.empty(cutoff=cutoff)

    if isinstance(left, NVar) and isinstance(right, Const):
        return _constant_index_set(atom.op, right.c, constant_on_left=False)
    if isinstance(left, Const) and isinstance(right, NVar):
        return _constant_index_set(atom.op, left.c, constant_on_left=True)
    if isinstance(left, InvN) and isinstance(right, Const):
        return _constant_reciprocal_set(atom.op, right.c, constant_on_left=False)
    if isinstance(left, Const) and isinstance(right, InvN):
        return _constant_reciprocal_set(atom.op, left.c, constant_on_left=True)
    if isinstance(left, NVar) and isinstance(right, InvN):
        return _index_reciprocal_set(atom.op, index_on_left=True)
    if isinstance(left, InvN) and isinstance(right, NVar):
        return _index_reciprocal_set(atom.op, index_on_left=False)

    if atom.op == "LT":
        if isinstance(left, Const) and left.c == 0.0 and isinstance(right, Exp):
            cutoff = _total_cutoff(right.arg)
            if cutoff is not None:
                return EventuallyPeriodicSet.universe(cutoff=cutoff)
        if isinstance(left, Exp) and isinstance(right, Const) and right.c == 0.0:
            cutoff = _total_cutoff(left.arg)
            if cutoff is not None:
                return EventuallyPeriodicSet.empty(cutoff=cutoff)
        if (
            isinstance(left, Exp)
            and isinstance(left.arg, NVar)
            and isinstance(right, Const)
        ):
            return _exp_index_constant_set(
                atom.op, right.c, constant_on_left=False
            )
        if (
            isinstance(left, Const)
            and isinstance(right, Exp)
            and isinstance(right.arg, NVar)
        ):
            return _exp_index_constant_set(
                atom.op, left.c, constant_on_left=True
            )

    return None


def eventually_periodic_set(
    expression: SetExpr,
) -> Optional[EventuallyPeriodicSet]:
    """Compile a supported set expression to its trusted eventual semantics."""
    if isinstance(expression, Universe):
        return EventuallyPeriodicSet.universe()
    if isinstance(expression, Empty):
        return EventuallyPeriodicSet.empty()
    if isinstance(expression, FiniteSetExpr):
        cutoff = max(expression.indices, default=-1) + 1
        return EventuallyPeriodicSet.empty(cutoff=cutoff)
    if isinstance(expression, Atom):
        return _atom_semantics(expression)
    if isinstance(expression, Complement):
        inner = eventually_periodic_set(expression.s)
        return None if inner is None else inner.complement()
    if isinstance(expression, Intersect):
        result = EventuallyPeriodicSet.universe()
        for part in expression.parts:
            semantics = eventually_periodic_set(part)
            if semantics is None:
                return None
            result = result.intersect(semantics)
        return result
    return None
