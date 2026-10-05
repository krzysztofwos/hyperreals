"""Errors shared by native execution and Lean kernel verification."""


class LeanBackendError(RuntimeError):
    """Lean is unavailable, execution failed, or its response is invalid."""
