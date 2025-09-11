"""Set algebra for representing index sets."""

from dataclasses import dataclass
from typing import List, Tuple

from .sequence import Seq


class SetExpr:
    """Base class for set expressions."""

    def key(self) -> Tuple:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SetExpr) and self.key() == other.key()

    def __repr__(self) -> str:
        raise NotImplementedError


class Universe(SetExpr):
    """The universe set ℕ."""

    def key(self) -> Tuple:
        return ("UNIV",)

    def __repr__(self) -> str:
        return "ℕ"


class Empty(SetExpr):
    """The empty set."""

    def key(self) -> Tuple:
        return ("EMPTY",)

    def __repr__(self) -> str:
        return "∅"


@dataclass(frozen=True, eq=False)
class Atom(SetExpr):
    """Atomic set expression {n: a op b} where op is 'LT' or 'EQ'."""

    op: str  # 'LT' or 'EQ'
    a: Seq
    b: Seq

    def key(self) -> Tuple:
        return ("ATOM", self.op, repr(self.a.simplify()), repr(self.b.simplify()))

    def __repr__(self) -> str:
        sym = "<" if self.op == "LT" else "="
        return f"{{n: {self.a} {sym} {self.b}}}"


@dataclass(frozen=True, eq=False)
class Complement(SetExpr):
    """Complement of a set."""

    s: SetExpr

    def key(self) -> Tuple:
        inner = self.s
        if isinstance(inner, Complement):
            return inner.s.key()
        return ("NOT", inner.key())

    def __repr__(self) -> str:
        return f"¬({self.s})"


def complement(s: SetExpr) -> SetExpr:
    """Smart constructor for complement that eliminates double negation."""
    if isinstance(s, Complement):
        return s.s
    return Complement(s)


@dataclass(frozen=True, eq=False)
class Intersect(SetExpr):
    """Intersection of sets."""

    parts: Tuple[SetExpr, ...]

    def key(self) -> Tuple:
        def flat_parts(p: SetExpr):
            if isinstance(p, Intersect):
                for q in p.parts:
                    yield from flat_parts(q)
            else:
                yield p

        lits: List[SetExpr] = []
        for p in self.parts:
            for q in flat_parts(p):
                if isinstance(q, Universe):
                    continue
                if isinstance(q, Empty):
                    return ("EMPTY",)
                lits.append(q)
        lit_keys = [x.key() for x in lits]
        lit_set = set(lit_keys)
        for k in list(lit_set):
            if k[0] == "NOT":
                compk = k[1]
            else:
                compk = ("NOT", k)
            if compk in lit_set:
                return ("EMPTY",)
        uniq = sorted(set(lits), key=lambda s: s.key())
        if not uniq:
            return ("UNIV",)
        if len(uniq) == 1:
            return uniq[0].key()
        return ("ANDN", tuple(s.key() for s in uniq))

    def __repr__(self) -> str:
        return " ∩ ".join(repr(p) for p in self.parts)


def intersect(a: SetExpr, b: SetExpr) -> SetExpr:
    """Smart constructor for intersection."""
    if isinstance(a, Empty) or isinstance(b, Empty):
        return Empty()
    if isinstance(a, Universe):
        return b
    if isinstance(b, Universe):
        return a
    it = Intersect((a, b))
    k = it.key()
    if k == ("EMPTY",):
        return Empty()
    if k == ("UNIV",):
        return Universe()
    if k[0] != "ANDN":
        for candidate in (a, b):
            if candidate.key() == k:
                return candidate
        return a
    return it


@dataclass(frozen=True, eq=False)
class FiniteSetExpr(SetExpr):
    """Finite set of indices."""

    indices: Tuple[int, ...]

    def key(self) -> Tuple:
        return ("FIN", self.indices)

    def __repr__(self) -> str:
        return "{" + ",".join(str(k) for k in self.indices) + "}"
