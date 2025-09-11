"""Hyperreal number system with lazy ultrafilters."""

from .dual import Dual, ad_derivative_first
from .hyperreal import Hyperreal, HyperrealSystem

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
from .ultrafilter import PartialUltrafilter

__version__ = "0.1.0"

__all__ = [
    # Main API
    "Hyperreal",
    "HyperrealSystem",
    "PartialUltrafilter",
    "Dual",
    "ad_derivative_first",
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
