"""Finite observations and symbolic sequence computation."""

from .asymptotic_facts import AsymptoticFact, analyze
from .dual import Dual, ad_derivative_first
from .hyperreal import Hyperreal, HyperrealSystem, StChooseResult
from .verified import LeanBackendError, LeanPeriodicSystem, PeriodicHyperreal
from .verified_laurent import LaurentHyperreal, LeanLaurentSystem
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
from .series import is_near_standard_by_series, series_from_seq
from .semantic_sets import EventuallyPeriodicSet, eventually_periodic_set
from .ultrafilter import (
    PartialUltrafilter,
    SemanticInconsistencyError,
    UnderdeterminedComparisonError,
)

__version__ = "0.1.0"

__all__ = ['LeanPeriodicSystem', 'LeanLaurentSystem', 'LaurentHyperreal', 'PeriodicHyperreal', 'LeanBackendError', 'Hyperreal', 'HyperrealSystem', 'PartialUltrafilter', 'EventuallyPeriodicSet', 'eventually_periodic_set', 'SemanticInconsistencyError', 'UnderdeterminedComparisonError', 'Dual', 'ad_derivative_first', 'StChooseResult', 'AsymptoticFact', 'analyze', 'Seq', 'Const', 'NVar', 'InvN', 'AltSign', 'Add', 'Sub', 'Mul', 'Div', 'Sin', 'Cos', 'Tan', 'Tanh', 'Exp', 'Log1p', 'Sqrt1p', 'Cosh', 'Sinh', 'series_from_seq', 'is_near_standard_by_series']
