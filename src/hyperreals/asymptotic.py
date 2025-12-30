"""Completion-invariant standard-part extraction helpers.

Historically, ``Hyperreal.standard_part`` was derived purely from the Laurent/
Taylor recognizer in :mod:`hyperreal.series`.

This file keeps the public helper ``standard_part_extended`` but delegates the
actual analysis to :mod:`hyperreal.asymptotic_facts`, which provides a more
compositional and extensible (still conservative) rule set.
"""

from __future__ import annotations

from typing import Optional

from .asymptotic_facts import analyze
from .sequence import Seq


def standard_part_extended(seq: Seq, *, order: int = 10) -> Optional[float]:
    """Extract a standard part when it can be certified without choices."""
    fact = analyze(seq, order=order)
    if fact.kind == "finite":
        return fact.limit
    return None
