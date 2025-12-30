from hyperreals import HyperrealSystem
from hyperreals.algebra import Atom


def test_direction_swap_invariance():
    """Test that a<b and b>a refer to the same set/variable."""
    sys = HyperrealSystem()
    n = sys.infinite()
    five = sys.constant(5.0)

    # a<b is 5<n; b>a is n>5 (same set)
    a, b = five, n

    # Probe variable identity through internal API
    L_ab = a._cmp_sets(b)[0]  # {n: a<b}
    v1 = sys.puf._ensure_var(L_ab)

    # For b>a, Hyperreal.__gt__ refers to the same set {n: a<b}
    # G in (b,a) partition is {n: a<b}
    v2 = sys.puf._ensure_var(b._cmp_sets(a)[2])

    assert v1 == v2, "SAT variable IDs should match for direction swap"
    assert a < b, "5 < n should be true"
    assert b > a, "n > 5 should be true"


def test_transitivity_of_lt():
    """Test transitivity of < using sin/cos/tan."""
    sys = HyperrealSystem()
    n = sys.infinite()
    a = sys.sin(n)
    b = sys.cos(n)
    c = sys.tan(n)

    # Force two True choices: a<b and b<c (both are choice-dependent)
    # The act of comparing them commits the choice if not decided
    assert a < b
    assert b < c

    # With transitivity, c<a should be impossible now
    assert not (c < a), "Cycle a < b < c < a should be prevented"

    # Check for transitivity clause presence
    aS = sys.puf._seq_by_repr.get(repr(a.seq))
    bS = sys.puf._seq_by_repr.get(repr(b.seq))
    cS = sys.puf._seq_by_repr.get(repr(c.seq))

    assert aS is not None and bS is not None and cS is not None

    vL_ab = sys.puf._ensure_var(Atom("LT", aS, bS))
    vL_bc = sys.puf._ensure_var(Atom("LT", bS, cS))
    vL_ac = sys.puf._ensure_var(Atom("LT", aS, cS))
    needle = {-vL_ab, -vL_bc, vL_ac}

    clause_found = any(
        len(clause) == 3 and set(clause) == needle for clause in sys.puf.sat.cnf
    )
    assert clause_found, "Transitivity clause should be installed"


def test_transitivity_of_gt_mixed():
    """Test transitivity of > mixed with other ops."""
    sys = HyperrealSystem()
    n = sys.infinite()
    # Use different functions to avoid collision with previous test
    x = sys.sin(n + sys.constant(1.0))
    y = sys.cos(n + sys.constant(1.0))
    z = sys.tan(n + sys.constant(1.0))

    # Force x > y and y > z
    assert x > y
    assert y > z

    # With proper transitivity, z > x should be impossible (x > z holds)
    assert not (z > x), "Cycle x > y > z > x should be prevented"
