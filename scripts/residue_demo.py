"""Two independent periodic choices with a verified common standard part."""

from fractions import Fraction

from hyperreals import LeanResidueSystem


def main() -> None:
    system = LeanResidueSystem()
    zero, one = system.constant(0), system.constant(1)
    a, b = system.alt(), system.periodic([0, 1, 2])
    assert a < zero
    assert system.support == (False, True)
    assert b == system.constant(2)
    assert system.support == (False, False, False, False, False, True)
    assert (one + (a + b) / system.infinite()).standard_part() == Fraction(1)
    assert not system.commit(b, one, "eq", truth=True)
    assert a.standard_part() == Fraction(-1)
    assert b.standard_part() == Fraction(2)
    print("Accepted n ≡ 5 (mod 6). The earlier parity choice is preserved.")
    print("st(1 + (a + b)/n) = 1. The incompatible b = 1 is rejected.")


if __name__ == "__main__":
    main()
