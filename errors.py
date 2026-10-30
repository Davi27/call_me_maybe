class NoLegalContinuationError(ValueError):
    """Raised when no token can continue the constrained output."""


class StepLimitExceededError(Exception):
    """Raised when generation does not finish within the step limit."""
