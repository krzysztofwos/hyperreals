"""Exact infinitesimal computation through finite observations.

LeanResidueSystem exposes the verified periodic Laurent core. The periodic
and parity Laurent interfaces provide its smaller verified fragments.
Concrete exported results can be checked independently through verify_export.
"""

from .replay import (
    ReplayObservation,
    ReplaySnapshot,
    ReplayVerification,
    ReplayVerificationError,
    verify_export,
)
from .verified import LeanBackendError, LeanPeriodicSystem, PeriodicHyperreal
from .verified_laurent import LaurentHyperreal, LeanLaurentSystem
from .verified_residue import LeanResidueSystem, ResidueHyperreal

__version__ = "0.1.0"

__all__ = [
    "LeanPeriodicSystem",
    "LeanLaurentSystem",
    "LaurentHyperreal",
    "LeanResidueSystem",
    "ResidueHyperreal",
    "ReplayObservation",
    "ReplaySnapshot",
    "ReplayVerification",
    "ReplayVerificationError",
    "verify_export",
    "PeriodicHyperreal",
    "LeanBackendError",
]
