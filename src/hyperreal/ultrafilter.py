"""Partial ultrafilter implementation with SAT-based consistency."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from .algebra import Atom, Complement, Empty, SetExpr, Universe, complement, intersect
from .sat import SAT
from .sequence import Add, AltSign, Const, Cos, Div, InvN, Mul, NVar, Seq, Sin, Sub, Tan
from .series import eventual_sign_from_seq, series_from_seq_exact


@dataclass
class PUFStats:
    """Statistics for partial ultrafilter operations."""

    puf_contains: int = 0
    unit_true: int = 0
    unit_false: int = 0
    fip_checks: int = 0
    fip_blocks: int = 0
    meets: int = 0
    cofinite_includes: int = 0
    finite_excludes: int = 0
    fastpath_true: int = 0
    fastpath_false: int = 0
    cache_true_hits: int = 0
    cache_false_hits: int = 0
    sat_calls: int = 0
    choice_commits: int = 0
    max_committed_true: int = 0


class PartialUltrafilter:
    """
    Finite, constraint-maintaining partial ultrafilter:
      - XOR for complements
      - ∅ ∉ U, ℕ ∈ U
      - Finite sets excluded, cofinite included (when provable)
      - Trichotomy partitions L/E/G for each compared pair
      - Incremental closure under finite intersections among committed-true sets
    """

    def __init__(self) -> None:
        self.sat = SAT()
        self.stats = PUFStats()
        self.var_of: Dict[SetExpr, int] = {}
        self._committed_true: Set[SetExpr] = set()
        self._installed_partitions: Set[Tuple[str, str]] = set()
        # Track canonical Seq objects by repr for cross-link/transitivity clauses
        self._seq_by_repr: Dict[str, Seq] = {}
        self._ensure_var(Empty())
        self._ensure_var(Universe())
        self._unit_true(Universe())
        self._unit_false(Empty())

    def _name(self, s: SetExpr) -> str:
        return repr(s)

    def _ensure_var(self, s: SetExpr) -> int:
        if s in self.var_of:
            return self.var_of[s]
        v = self.sat.new_var(self._name(s))
        self.var_of[s] = v
        comp = complement(s)
        if comp not in self.var_of:
            vc = self.sat.new_var(self._name(comp))
            self.var_of[comp] = vc
            self.sat.add_clause(
                [
                    self.sat.lit(self._name(s), True),
                    self.sat.lit(self._name(comp), True),
                ]
            )
            self.sat.add_clause(
                [
                    self.sat.lit(self._name(s), False),
                    self.sat.lit(self._name(comp), False),
                ]
            )
        else:
            self.sat.add_clause(
                [
                    self.sat.lit(self._name(s), True),
                    self.sat.lit(self._name(comp), True),
                ]
            )
            self.sat.add_clause(
                [
                    self.sat.lit(self._name(s), False),
                    self.sat.lit(self._name(comp), False),
                ]
            )
        return v

    def _unit_true(self, s: SetExpr, *, choice: bool = False) -> None:
        v = self._ensure_var(s)
        self.sat.add_clause([v])
        self.stats.unit_true += 1
        if choice:
            self.stats.choice_commits += 1

        # Track intersection closure (meets)
        self.stats.meets += len(self._committed_true)

        for t in list(self._committed_true):
            if isinstance(t, Universe):
                continue
            intersection = intersect(s, t)
            # Guard against contradictory unit clauses
            if self._is_finite(intersection) is True:
                # Skip adding intersection as unit-true if it's provably finite
                continue
            iv = self._ensure_var(intersection)
            self.sat.add_clause([iv])
        self._committed_true.add(s)
        if len(self._committed_true) > self.stats.max_committed_true:
            self.stats.max_committed_true = len(self._committed_true)

    def _unit_false(self, s: SetExpr) -> None:
        v = self._ensure_var(s)
        self.sat.add_clause([-v])
        self.stats.unit_false += 1

    def _is_finite(self, s: SetExpr) -> Optional[bool]:
        """Check if a set is provably finite."""
        if isinstance(s, Empty):
            return True
        if isinstance(s, Universe):
            return False
        if isinstance(s, Complement):
            return None
        if isinstance(s, Atom):
            a = s.a.simplify()
            b = s.b.simplify()
            if s.op == "EQ":
                # For integer-valued affine arguments, trig equalities are finite:
                # sin(k*n+b)=c, cos(k*n+b)=c, tan(k*n+b)=c have at most one integer solution.
                if isinstance(a, (Sin, Cos, Tan)) and isinstance(b, Const):
                    affine = self._affine_int_in_n(a.arg)
                    if affine is not None and affine[0] != 0:
                        return True
                if isinstance(b, (Sin, Cos, Tan)) and isinstance(a, Const):
                    affine = self._affine_int_in_n(b.arg)
                    if affine is not None and affine[0] != 0:
                        return True
                # For integer-valued affine arguments, trig is injective enough to make most equalities
                # either tautological (cofinite) or finite. We use that for obvious cases.
                if isinstance(a, Sin) and isinstance(b, Sin):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        return True if aa != bb else False
                if isinstance(a, Tan) and isinstance(b, Tan):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        return True if aa != bb else False
                if isinstance(a, Cos) and isinstance(b, Cos):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        if aa == bb or aa == (-bb[0], -bb[1]):
                            return False
                        return True
                if (isinstance(a, Sin) and isinstance(b, Cos)) or (
                    isinstance(a, Cos) and isinstance(b, Sin)
                ):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        return True
                if isinstance(a, Const) and isinstance(b, Const):
                    return True if a.c != b.c else False
                if isinstance(a, AltSign) and isinstance(b, Const):
                    if b.c not in (-1.0, 1.0):
                        return True
                if isinstance(b, AltSign) and isinstance(a, Const):
                    if a.c not in (-1.0, 1.0):
                        return True
                if (isinstance(a, NVar) and isinstance(b, Const)) or (
                    isinstance(b, NVar) and isinstance(a, Const)
                ):
                    return True
                if (isinstance(a, InvN) and isinstance(b, Const)) or (
                    isinstance(b, InvN) and isinstance(a, Const)
                ):
                    return True
                diff_exact = series_from_seq_exact(Sub(a, b).simplify())
                if diff_exact is not None:
                    return False if not diff_exact else True
                # If the difference is eventually nonzero, equality can only hold finitely often.
                diff_sign = eventual_sign_from_seq(Sub(b, a).simplify())
                if diff_sign in (-1, 1):
                    return True
                return None
            elif s.op == "LT":
                diff_sign = eventual_sign_from_seq(Sub(b, a).simplify())
                if diff_sign in (-1, 0):
                    return True
                if diff_sign == 1:
                    return False
                if isinstance(a, NVar) and isinstance(b, Const):
                    return True
                if isinstance(a, Const) and isinstance(b, NVar):
                    return False
                if isinstance(a, InvN) and isinstance(b, Const):
                    return True if b.c <= 0.0 else False
                if isinstance(a, Const) and isinstance(b, InvN):
                    # {n : a < 1/n} is finite iff a > 0
                    return True if a.c > 0.0 else False
                if isinstance(a, Const) and isinstance(b, Const):
                    return True if not (a.c < b.c) else False
                return None
        return None

    def _affine_int_in_n(self, s: Seq) -> Optional[Tuple[int, int]]:
        """Recognize sequences of the form k*n + b with integer k,b."""
        s = s.simplify()
        if isinstance(s, NVar):
            return (1, 0)
        if isinstance(s, Const):
            r = round(s.c)
            if abs(s.c - r) < 1e-12:
                return (0, int(r))
            return None
        if isinstance(s, Add):
            left = self._affine_int_in_n(s.left)
            right = self._affine_int_in_n(s.right)
            if left is None or right is None:
                return None
            return (left[0] + right[0], left[1] + right[1])
        if isinstance(s, Sub):
            left = self._affine_int_in_n(s.left)
            right = self._affine_int_in_n(s.right)
            if left is None or right is None:
                return None
            return (left[0] - right[0], left[1] - right[1])
        if isinstance(s, Mul):
            if isinstance(s.left, Const):
                factor_seq = s.left
                rest_seq = s.right
            elif isinstance(s.right, Const):
                factor_seq = s.right
                rest_seq = s.left
            else:
                return None
            factor = self._affine_int_in_n(factor_seq)
            rest = self._affine_int_in_n(rest_seq)
            if factor is None or rest is None:
                return None
            if factor[0] != 0:
                return None
            factor_int = factor[1]
            return (factor_int * rest[0], factor_int * rest[1])
        if isinstance(s, Div):
            num = self._affine_int_in_n(s.left)
            den = self._affine_int_in_n(s.right)
            if num is None or den is None:
                return None
            if den[0] != 0:
                return None
            d = den[1]
            if d == 1:
                return num
            if d == -1:
                return (-num[0], -num[1])
            return None
        return None

    def _is_cofinite(self, s: SetExpr) -> Optional[bool]:
        """Check if a set is provably cofinite."""
        if isinstance(s, Universe):
            return True
        if isinstance(s, Empty):
            return False
        if isinstance(s, Complement):
            f = self._is_finite(s.s)
            return True if f is True else (False if f is False else None)
        if isinstance(s, Atom):
            a = s.a.simplify()
            b = s.b.simplify()
            if s.op == "EQ":
                if isinstance(a, (Sin, Cos, Tan)) and isinstance(b, Const):
                    affine = self._affine_int_in_n(a.arg)
                    if affine is not None and affine[0] != 0:
                        return False
                if isinstance(b, (Sin, Cos, Tan)) and isinstance(a, Const):
                    affine = self._affine_int_in_n(b.arg)
                    if affine is not None and affine[0] != 0:
                        return False
                if isinstance(a, Sin) and isinstance(b, Sin):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        return True if aa == bb else False
                if isinstance(a, Tan) and isinstance(b, Tan):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        return True if aa == bb else False
                if isinstance(a, Cos) and isinstance(b, Cos):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        if aa == bb or aa == (-bb[0], -bb[1]):
                            return True
                        return False
                if (isinstance(a, Sin) and isinstance(b, Cos)) or (
                    isinstance(a, Cos) and isinstance(b, Sin)
                ):
                    aa = self._affine_int_in_n(a.arg)
                    bb = self._affine_int_in_n(b.arg)
                    if aa is not None and bb is not None and aa[0] != 0 and bb[0] != 0:
                        return False
                if isinstance(a, Const) and isinstance(b, Const):
                    return True if a.c == b.c else False
                diff_exact = series_from_seq_exact(Sub(a, b).simplify())
                if diff_exact is not None:
                    return True if not diff_exact else False
                return None
            elif s.op == "LT":
                diff_sign = eventual_sign_from_seq(Sub(b, a).simplify())
                if diff_sign == 1:
                    return True
                if diff_sign in (-1, 0):
                    return False
                if isinstance(a, NVar) and isinstance(b, Const):
                    return False
                if isinstance(a, Const) and isinstance(b, NVar):
                    return True
                if isinstance(a, InvN) and isinstance(b, Const):
                    return True if b.c > 0.0 else False
                if isinstance(a, Const) and isinstance(b, InvN):
                    # {n : a < 1/n} is cofinite iff a <= 0
                    return True if a.c <= 0.0 else False
                if isinstance(a, Const) and isinstance(b, Const):
                    return True if a.c < b.c else False
                diff = Sub(b, a).simplify()
                nonneg = diff.is_nonnegative_eventually()
                if diff.is_infinitesimal() and nonneg is True:
                    return True
                return None
        return None

    def _partition_neighbors(self, seq_repr: str) -> Set[str]:
        """Return sequence reprs that have a partition with `seq_repr`."""
        neighbors: Set[str] = set()
        for a_repr, b_repr in self._installed_partitions:
            if a_repr == seq_repr:
                neighbors.add(b_repr)
            elif b_repr == seq_repr:
                neighbors.add(a_repr)
        return neighbors

    def _install_transitivity_triangle(self, a: Seq, b: Seq) -> None:
        """Install LT-transitivity clauses for every triangle involving {a,b}."""
        aS = a.simplify()
        bS = b.simplify()
        a_repr = repr(aS)
        b_repr = repr(bS)

        common = self._partition_neighbors(a_repr) & self._partition_neighbors(b_repr)
        for c_repr in common:
            cS = self._seq_by_repr.get(c_repr)
            if cS is None:
                continue

            v_ab = self._ensure_var(Atom("LT", aS, bS))
            v_ba = self._ensure_var(Atom("LT", bS, aS))
            v_ac = self._ensure_var(Atom("LT", aS, cS))
            v_ca = self._ensure_var(Atom("LT", cS, aS))
            v_bc = self._ensure_var(Atom("LT", bS, cS))
            v_cb = self._ensure_var(Atom("LT", cS, bS))

            # For x,y,z ∈ {a,b,c}: (x<y) ∧ (y<z) → (x<z)
            self.sat.add_clause([-v_ab, -v_bc, v_ac])  # a<b ∧ b<c → a<c
            self.sat.add_clause([-v_ac, -v_cb, v_ab])  # a<c ∧ c<b → a<b
            self.sat.add_clause([-v_ba, -v_ac, v_bc])  # b<a ∧ a<c → b<c
            self.sat.add_clause([-v_bc, -v_ca, v_ba])  # b<c ∧ c<a → b<a
            self.sat.add_clause([-v_ca, -v_ab, v_cb])  # c<a ∧ a<b → c<b
            self.sat.add_clause([-v_cb, -v_ba, v_ca])  # c<b ∧ b<a → c<a

    def _install_partition(self, a: Seq, b: Seq) -> Tuple[SetExpr, SetExpr, SetExpr]:
        """Install trichotomy partition for sequences a and b."""
        aS = a.simplify()
        bS = b.simplify()
        a_repr = repr(aS)
        b_repr = repr(bS)
        self._seq_by_repr[a_repr] = aS
        self._seq_by_repr[b_repr] = bS
        key = (repr(aS), repr(bS))
        L = Atom("LT", aS, bS)
        E = Atom("EQ", aS, bS)
        G = Atom("LT", bS, aS)
        if key not in self._installed_partitions:
            self._installed_partitions.add(key)
            vL = self._ensure_var(L)
            vE = self._ensure_var(E)
            vG = self._ensure_var(G)
            # Trichotomy: exactly-one of L, E, G
            self.sat.add_clause([vL, vE, vG])
            self.sat.add_clause([-vL, -vE])
            self.sat.add_clause([-vL, -vG])
            self.sat.add_clause([-vE, -vG])

            # Cross-links (safe) and transitivity encodings

            # Symmetry of equality: a=b <-> b=a
            vE_ab = self._ensure_var(E)
            vE_ba = self._ensure_var(Atom("EQ", bS, aS))
            self.sat.add_clause([-vE_ab, vE_ba])
            self.sat.add_clause([-vE_ba, vE_ab])

            # Transitivity of LT, added only when triangles close.
            self._install_transitivity_triangle(aS, bS)
        else:
            self._ensure_var(L)
            self._ensure_var(E)
            self._ensure_var(G)
        return L, E, G

    def _fip_allows(self, s: SetExpr, truth: bool) -> bool:
        """Check if adding this set maintains finite intersection property."""
        self.stats.fip_checks += 1
        lit = self.sat.lit(self._name(s), truth)
        trial = self.sat.with_unit(lit)
        model = trial.solve()

        # Accumulate statistics from the temporary SAT instance.
        self.sat.solves = trial.solves
        self.sat.unit_props = trial.unit_props
        self.sat.decisions = trial.decisions

        return model is not None

    def contains(self, s: SetExpr) -> bool:
        """Check if a set is in the ultrafilter."""
        self.stats.puf_contains += 1
        v = self._ensure_var(s)
        # Short-circuit if already committed
        if s in self._committed_true:
            self.stats.cache_true_hits += 1
            return True
        from .algebra import complement

        if complement(s) in self._committed_true:
            self.stats.cache_false_hits += 1
            return False
        f = self._is_finite(s)
        c = self._is_cofinite(s)

        if f is True:
            self.stats.finite_excludes += 1
            self.stats.fastpath_false += 1
            self._unit_false(s)
            return False

        if c is True:
            self.stats.cofinite_includes += 1
            # Cofinite sets are always in any non-principal ultrafilter (they form the Fréchet filter).
            # If this ever leads to inconsistency, prior commitments were already unsound.
            self.stats.fastpath_true += 1
            self._unit_true(s)
            return True

        # Try with negative unit clause
        sat_neg = self.sat.with_unit(-v)
        self.stats.sat_calls += 1
        model_neg = sat_neg.solve()

        # Accumulate statistics from the temporary SAT instance
        self.sat.solves = sat_neg.solves
        self.sat.unit_props = sat_neg.unit_props
        self.sat.decisions = sat_neg.decisions

        if model_neg is None:
            if self._fip_allows(s, True):
                self._unit_true(s)
                return True
            else:
                self.stats.fip_blocks += 1
                return False

        # Try with positive unit clause
        sat_pos = self.sat.with_unit(v)
        self.stats.sat_calls += 1
        model_pos = sat_pos.solve()

        # Accumulate statistics from the temporary SAT instance
        self.sat.solves = sat_pos.solves
        self.sat.unit_props = sat_pos.unit_props
        self.sat.decisions = sat_pos.decisions

        if model_pos is None:
            self._unit_false(s)
            return False

        # Choice-dependent -> commit True (consistent extension)
        if self._fip_allows(s, True):
            self._unit_true(s, choice=True)
            return True
        else:
            self.stats.fip_blocks += 1
            return False

    def human_readable_constraints(
        self,
        clause_limit: Optional[int] = None,
        *,
        var_limit: Optional[int] = None,
        committed_limit: Optional[int] = None,
        partitions_limit: Optional[int] = None,
    ) -> str:
        """Generate a human-readable constraint description.

        Limits are optional; pass `0` (or `None`) to show all entries.
        """
        if clause_limit == 0:
            clause_limit = None
        if var_limit == 0:
            var_limit = None
        if committed_limit == 0:
            committed_limit = None
        if partitions_limit == 0:
            partitions_limit = None

        lines = []
        var_ids = sorted(self.sat.var_to_name.keys())
        lines.append(f"Variables ({len(var_ids)}):")
        shown_var_ids = var_ids if var_limit is None else var_ids[:var_limit]
        for vid in shown_var_ids:
            lines.append(f"  v{vid}: {self.sat.var_to_name[vid]}")
        if var_limit is not None and len(var_ids) > var_limit:
            lines.append(f"  ... ({len(var_ids) - var_limit} more)")

        lines.append(f"Clauses (CNF) ({len(self.sat.cnf)}):")
        count = 0
        for clause in self.sat.cnf:
            syms = []
            for lit in clause:
                v = abs(lit)
                pol = lit > 0
                name = self.sat.var_to_name.get(v, f"v{v}")
                syms.append(name if pol else f"¬{name}")
            lines.append("  (" + " ∨ ".join(syms) + ")")
            count += 1
            if clause_limit is not None and count >= clause_limit:
                remaining = len(self.sat.cnf) - clause_limit
                if remaining > 0:
                    lines.append(f"  ... ({remaining} more)")
                break

        committed = sorted((repr(s) for s in self._committed_true))
        lines.append(f"Committed True sets ({len(committed)}):")
        shown_committed = (
            committed if committed_limit is None else committed[:committed_limit]
        )
        for name in shown_committed:
            lines.append(f"  {name}")
        if committed_limit is not None and len(committed) > committed_limit:
            lines.append(f"  ... ({len(committed) - committed_limit} more)")

        partitions = sorted(self._installed_partitions)
        lines.append(f"Installed partitions (a,b) ({len(partitions)}):")
        shown_partitions = (
            partitions if partitions_limit is None else partitions[:partitions_limit]
        )
        for a, b in shown_partitions:
            lines.append(f"  ({a}, {b})")
        if partitions_limit is not None and len(partitions) > partitions_limit:
            lines.append(f"  ... ({len(partitions) - partitions_limit} more)")
        return "\n".join(lines)

    def serialize(self) -> Dict[str, Any]:
        """Serialize ultrafilter state to dict."""
        # Implementation simplified for brevity
        return {
            "version": 1,
            "cnf": self.sat.export_cnf(),
            "committed_true": [repr(s) for s in self._committed_true],
            "installed_partitions": list(self._installed_partitions),
        }

    def save(self, path: str) -> None:
        """Save ultrafilter state to JSON file."""
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)
