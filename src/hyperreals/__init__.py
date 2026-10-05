"""Exact infinitesimal computation through finite observations.

LeanResidueSystem exposes the verified periodic Laurent core. DifferentiableProgram
compiles elementary vector derivatives. Concrete exported residue computations
can be checked independently through verify_export.
"""

from .differentiable import (
    CompiledJVP,
    DifferentiableExpr,
    DifferentiableProgram,
    DomainCondition,
    JVPVerification,
    variables,
)
from .errors import LeanBackendError
from .polynomial import divided_difference, evaluate_polynomial, polynomial_quotient
from .replay import (
    ReplayObservation,
    ReplaySnapshot,
    ReplayVerification,
    ReplayVerificationError,
    verify_export,
)
from .verified_residue import (
    LeanResidueSystem,
    ResidueHyperreal,
    StandardPartDiagnostic,
)

__version__ = "0.1.0"

__all__ = [
    "CompiledJVP",
    "DifferentiableExpr",
    "DifferentiableProgram",
    "DomainCondition",
    "JVPVerification",
    "variables",
    "LeanResidueSystem",
    "ResidueHyperreal",
    "StandardPartDiagnostic",
    "evaluate_polynomial",
    "divided_difference",
    "polynomial_quotient",
    "ReplayObservation",
    "ReplaySnapshot",
    "ReplayVerification",
    "ReplayVerificationError",
    "verify_export",
    "LeanBackendError",
]
