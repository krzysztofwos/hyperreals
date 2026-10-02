"""Finite observations and exact infinitesimal computation.

LeanResidueSystem exposes the verified exact core. HyperrealSystem retains the
experimental Python analytic frontend. Concrete exported results can be checked
independently through verify_export.
"""

from .asymptotic_facts import AsymptoticFact, analyze
from .dual import Dual, ad_derivative_first
from .hyperreal import Hyperreal, HyperrealSystem, StChooseResult
from .verified import LeanBackendError, LeanPeriodicSystem, PeriodicHyperreal
from .verified_laurent import LaurentHyperreal, LeanLaurentSystem
from .verified_residue import LeanResidueSystem, ResidueHyperreal
from .replay import (
    ReplayObservation,
    ReplaySnapshot,
    ReplayVerification,
    ReplayVerificationError,
    verify_export,
)

# Sequence types for advanced usage
from .sequence import (
    Add,
    AltSign,
    Const,
    Cos,
    Cosh,
    Div,
    Exp,
    InvN,
    Log1p,
    Mul,
    NVar,
    Seq,
    Sin,
    Sinh,
    Sqrt1p,
    Sub,
    Tan,
    Tanh,
)

# Series operations for advanced usage
from .series import is_near_standard_by_series, series_from_seq
from .semantic_sets import EventuallyPeriodicSet, eventually_periodic_set
from .ultrafilter import (
    PartialUltrafilter,
    SemanticInconsistencyError,
    UnderdeterminedComparisonError,
)

__version__ = "0.1.0"

__all__ = [
    # Verified exact interfaces and replay
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
    # Experimental Python analytic frontend and supporting tools
    "Hyperreal",
    "HyperrealSystem",
    "PartialUltrafilter",
    "EventuallyPeriodicSet",
    "eventually_periodic_set",
    "SemanticInconsistencyError",
    "UnderdeterminedComparisonError",
    "Dual",
    "ad_derivative_first",
    "StChooseResult",
    "AsymptoticFact",
    "analyze",
    # Sequence types
    "Seq",
    "Const",
    "NVar",
    "InvN",
    "AltSign",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Sin",
    "Cos",
    "Tan",
    "Tanh",
    "Exp",
    "Log1p",
    "Sqrt1p",
    "Cosh",
    "Sinh",
    # Series operations
    "series_from_seq",
    "is_near_standard_by_series",
]
