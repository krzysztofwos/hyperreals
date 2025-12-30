#!/usr/bin/env python3

"""
Transitivity and direction-swap smoke test
------------------------------------------

This script exercises two behaviors in the lazy-ultrafilter hyperreal engine:

1) Direction-swap invariance: queries `a<b` and `b>a` refer to the same set (the Atom
   `{n: a < b}`), so they are consistent and even share the same SAT variable id.

2) Transitivity of `<` (requires transitivity CNF encoding):
   After choosing `a<b` and `b<c`, asking `c<a` should be UNSAT (i.e., evaluate to False).
   Without this encoding, `c<a` may be arbitrarily chosen True, because nothing links the
   three comparisons.

Usage:
  uv run python scripts/transitivity-smoke-test.py [--clauses N] [--verbose]

- `--clauses` prints the first N CNF clauses for a quick audit.
- `--verbose` prints extra details, including SAT stats before/after queries.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hyperreals import HyperrealSystem  # noqa: E402


def sat_stats(sys: HyperrealSystem):
    s = sys.puf.sat
    return {"solves": s.solves, "unit_props": s.unit_props, "decisions": s.decisions}


def print_stats(tag: str, sys: HyperrealSystem):
    s = sat_stats(sys)
    print(
        f"[{tag}] SAT solves={s['solves']}, unit_props={s['unit_props']}, decisions={s['decisions']}"
    )


def show_clauses(
    sys: HyperrealSystem,
    *,
    clause_limit: Optional[int],
    var_limit: int,
    committed_limit: int,
    partitions_limit: int,
):
    print("\n--- CNF snapshot ---")
    print(
        sys.puf.human_readable_constraints(
            clause_limit=clause_limit,
            var_limit=var_limit,
            committed_limit=committed_limit,
            partitions_limit=partitions_limit,
        )
    )


def has_transitivity_clause(
    sys: HyperrealSystem, a_label: str, b_label: str, c_label: str
) -> bool:
    """Detect the transitivity clause (a<b) ∧ (b<c) → (a<c), if installed."""
    aS = sys.puf._seq_by_repr.get(a_label)
    bS = sys.puf._seq_by_repr.get(b_label)
    cS = sys.puf._seq_by_repr.get(c_label)
    if aS is None or bS is None or cS is None:
        return False

    from hyperreals.algebra import Atom  # local import for scripts-only usage

    vL_ab = sys.puf._ensure_var(Atom("LT", aS, bS))
    vL_bc = sys.puf._ensure_var(Atom("LT", bS, cS))
    vL_ac = sys.puf._ensure_var(Atom("LT", aS, cS))
    needle = {-vL_ab, -vL_bc, vL_ac}
    return any(len(clause) == 3 and set(clause) == needle for clause in sys.puf.sat.cnf)


def demo_direction_swap(sys: HyperrealSystem):
    print("== Direction-swap invariance ==")
    n = sys.infinite()
    five = sys.constant(5.0)
    # a<b is 5<n; b>a is n>5 (same set)
    a, b = five, n
    # Probe variable identity through internal API
    L_ab = a._cmp_sets(b)[0]  # {n: a<b}
    v1 = sys.puf._ensure_var(L_ab)
    # For b>a, Hyperreal.__gt__ refers to the same set {n: a<b}
    v2 = sys.puf._ensure_var(b._cmp_sets(a)[2])  # G in (b,a) partition is {n: a<b}
    print("Var id for {n: 5 < n}:", v1)
    print("Var id obtained from (n > 5) path:", v2)
    print("Same SAT var id? ->", v1 == v2)
    print("5 < n ? ->", a < b)
    print("n > 5 ? ->", b > a)
    print_stats("after direction-swap", sys)


def demo_transitivity(sys: HyperrealSystem, verbose: bool = False):
    print("\n== Transitivity of '<' (requires transitivity encoding) ==")
    n = sys.infinite()
    a = sys.sin(n)  # opaque / not decided by fast-paths
    b = sys.cos(n)  # opaque / not decided by fast-paths
    c = sys.tan(n)  # opaque / not decided by fast-paths

    if verbose:
        print_stats("start", sys)

    # Force two True choices: a<b and b<c (both are choice-dependent)
    print("Commit a<b  (sin(n) < cos(n)) ->", a < b)
    print("Commit b<c  (cos(n) < tan(n)) ->", b < c)

    if verbose:
        print_stats("after a<b, b<c", sys)

    # With transitivity, c<a should be impossible now
    print("Query c<a  (tan(n) < sin(n)) ->", c < a)

    if verbose:
        print_stats("after c<a", sys)

    # Heuristic check for the transitivity clause in CNF
    tri = has_transitivity_clause(sys, repr(a.seq), repr(b.seq), repr(c.seq))
    print("Transitivity clause present? ->", tri)


def demo_gt_transitivity(sys: HyperrealSystem, verbose: bool = False):
    print("\n== Transitivity of '>' (mixed direction) ==")
    n = sys.infinite()
    # Use different functions to avoid collision with previous demo
    x = sys.sin(n + sys.constant(1.0))
    y = sys.cos(n + sys.constant(1.0))
    z = sys.tan(n + sys.constant(1.0))

    if verbose:
        print_stats("start gt", sys)

    # Force x > y and y > z
    # These install partitions (x,y) and (y,z) but check G
    print("Commit x>y  (sin > cos) ->", x > y)
    print("Commit y>z  (cos > tan) ->", y > z)

    if verbose:
        print_stats("after x>y, y>z", sys)

    # With proper transitivity, z > x should be impossible (x > z holds)
    # If the system misses the G-transitivity, it might allow z > x
    res = z > x
    print("Query z>x  (tan > sin) ->", res)

    if res:
        print("FAIL: System allowed cycle x > y > z > x")
    else:
        print("PASS: System prevented cycle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clauses",
        "--clause-limit",
        type=int,
        default=20,
        help="print first N CNF clauses (default: 20; 0=all)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="print SAT stats at each step"
    )
    ap.add_argument(
        "--var-limit",
        type=int,
        default=40,
        help="print first N SAT variables in the constraint dump (0=all)",
    )
    ap.add_argument(
        "--committed-limit",
        type=int,
        default=20,
        help="print first N committed-true sets in the constraint dump (0=all)",
    )
    ap.add_argument(
        "--partitions-limit",
        type=int,
        default=20,
        help="print first N installed partitions in the constraint dump (0=all)",
    )
    ns = ap.parse_args()

    sys = HyperrealSystem()

    demo_direction_swap(sys)
    demo_transitivity(sys, verbose=ns.verbose)
    demo_gt_transitivity(sys, verbose=ns.verbose)

    if ns.clauses >= 0:
        show_clauses(
            sys,
            clause_limit=None if ns.clauses == 0 else ns.clauses,
            var_limit=ns.var_limit,
            committed_limit=ns.committed_limit,
            partitions_limit=ns.partitions_limit,
        )


if __name__ == "__main__":
    main()
