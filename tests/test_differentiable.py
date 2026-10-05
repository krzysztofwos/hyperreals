"""The symbolic compiler's numerical interface, boundaries, and kernel checks."""

# trunk-ignore-all(bandit/B101): pytest assertions are test oracles, not runtime validation.

import math
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from hyperreals import (
    DifferentiableExpr,
    DifferentiableProgram,
    ReplayVerificationError,
    variables,
)

ROOT = Path(__file__).resolve().parents[1]
KERNEL_AVAILABLE = (
    shutil.which("lake") is not None
    and (ROOT / ".lake/packages/mathlib/Mathlib.lean").exists()
)
kernel = pytest.mark.skipif(
    not KERNEL_AVAILABLE, reason="requires the pinned Lean checkout"
)


def vector_example():
    x, y = variables(2)
    return DifferentiableProgram(
        2,
        (
            x.sin() * y.exp(),
            (1 + x * x + y * y).log(),
            (1 + x * x).sqrt() / (1 + y * y),
        ),
    )


def test_vector_jvp_and_jacobian():
    p = vector_example()
    jvp = p.jvp((1, 2))
    assert jvp.derivative.approximate((1, 0)) == pytest.approx(
        (math.cos(1) + 2 * math.sin(1), 1, 1 / math.sqrt(2))
    )
    jacobian = tuple(tuple(e.approximate((1, 0)) for e in row) for row in p.jacobian())
    assert jacobian[0] == pytest.approx((math.cos(1), math.sin(1)))
    assert jacobian[1] == pytest.approx((1, 0))
    assert jacobian[2] == pytest.approx((1 / math.sqrt(2), 0))
    # Independently sample the literal quotient. This is a regression check, not proof.
    step = 1e-6
    base = p.approximate((1, 0))
    perturbed = p.approximate((1 + step, 2 * step))
    quotient = tuple((a - b) / step for a, b in zip(perturbed, base, strict=True))
    assert quotient == pytest.approx(jvp.derivative.approximate((1, 0)), rel=1e-5)


def test_scalar_gradient_and_composition():
    x, y = variables(2)
    p = DifferentiableProgram(2, (((x.cos() - y.sin()) / (1 + x * x)).exp(),))
    point = (Fraction(1, 2), Fraction(1, 3))
    gradient = tuple(e.approximate(point) for e in p.gradient())
    expected = p.jvp((2, -3)).derivative.approximate(point)[0]
    assert expected == pytest.approx(2 * gradient[0] - 3 * gradient[1])
    # Tangents may depend on the base point. They are held fixed in the line quotient.
    assert p.jvp((x, y)).derivative.approximate(point)[0] == pytest.approx(
        gradient[0] / 2 + gradient[1] / 3
    )


@pytest.mark.parametrize(
    "make", [lambda x: x / x, lambda x: 0 * x.log(), lambda x: x.sqrt()]
)
def test_domain_obligations_survive_cancellation(make):
    (x,) = variables(1)
    e = make(x)
    assert e.domain_conditions
    with pytest.raises(ValueError):
        e.approximate((0,))
    with pytest.raises(ValueError):
        DifferentiableProgram(1, (e,)).jvp((1,)).derivative.approximate((0,))


def test_dimensions_exact_constants_and_branch_boundary():
    (x,) = variables(1)
    y, _ = variables(2)
    for operation in (
        lambda: x + y,
        lambda: DifferentiableProgram(2, (x,)),
        lambda: DifferentiableProgram(1, (x,)).jvp((1, 2)),
        lambda: DifferentiableProgram(1, (x, x)).gradient(),
    ):
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(TypeError, match="int or Fraction"):
        x + 0.1
    with pytest.raises(TypeError, match="branches"):
        bool(x)
    with pytest.raises(ValueError):
        variables(-1)
    assert DifferentiableExpr.constant(1, Fraction(2, 3)).value == Fraction(2, 3)


def test_empty_dimensions():
    p = DifferentiableProgram(0, (DifferentiableExpr.constant(0, 7),))
    assert p.jvp(()).derivative.approximate(()) == (0,)
    assert p.gradient() == ()
    assert DifferentiableProgram(2, ()).jvp((1, 2)).derivative.outputs == ()


def test_proof_source_is_bounded_and_contains_no_numeric_certificate():
    p = vector_example()
    source = p.jvp((1, 2)).lean_source((1, 0))
    assert "decide +kernel" in source
    assert "fderiv" in source and "quotient_standardPart" in source
    assert "domain_at_point" in source and "result_domain_at_point" in source
    assert "native_decide" not in source
    assert "2.223" not in source
    (x,) = variables(1)
    e = x
    for _ in range(130):
        e = e.sin()
    with pytest.raises(ValueError, match="budget"):
        DifferentiableProgram(1, (e,)).jvp((1,)).lean_source()


@kernel
def test_kernel_checks_all_operators_and_concrete_domain():
    p = vector_example()
    x, y = variables(2)
    p = DifferentiableProgram(2, (*p.outputs, -x.cos() - Fraction(2, 3) * y))
    result = p.jvp((1, 2)).verify(ROOT, point=(1, 0), timeout=180)
    assert result.domain_verified
    assert set(result.axioms) <= {"propext", "Classical.choice", "Quot.sound"}


@kernel
def test_kernel_rejects_tampered_derivative():
    (x,) = variables(1)
    jvp = DifferentiableProgram(1, (x.sin(),)).jvp((1,))
    forged = replace(jvp, derivative=DifferentiableProgram(1, (x,)))
    with pytest.raises(ReplayVerificationError, match="did not verify"):
        forged.verify(ROOT, timeout=180)


@kernel
def test_generic_proof_does_not_certify_invalid_domain():
    (x,) = variables(1)
    jvp = DifferentiableProgram(1, (x.log(),)).jvp((1,))
    assert not jvp.verify(ROOT, timeout=180).domain_verified
    with pytest.raises(ReplayVerificationError, match="did not verify"):
        jvp.verify(ROOT, point=(0,), timeout=180)
    tangent = DifferentiableProgram(1, (x,)).jvp((1 / x,))
    with pytest.raises(ReplayVerificationError, match="did not verify"):
        tangent.verify(ROOT, point=(0,), timeout=180)


@kernel
def test_kernel_checks_empty_dimensions():
    DifferentiableProgram(0, (DifferentiableExpr.constant(0, 1),)).jvp(()).verify(
        ROOT, point=(), timeout=180
    )
    DifferentiableProgram(1, ()).jvp((1,)).verify(ROOT, point=(0,), timeout=180)
