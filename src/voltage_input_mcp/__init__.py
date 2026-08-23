"""VoltageInputMcp -- a two-layer input engine.

Layer 1 is the orchestrator: a frontier model that writes a Playbook, a state machine
with guards, limits and a safety policy.

Layer 2 is two small local models that execute it -- one reads the screen, one emits
timed input bursts. Neither of them plans. The orchestrator does the thinking once, up
front, and the local pair chains inputs at a rate a remote model could never reach.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import (
    Aborted,
    BackendError,
    BurstParseError,
    CaptureError,
    ConfigError,
    ExpressionError,
    InputDeviceError,
    PlaybookError,
    SafetyViolation,
    SessionError,
    VoltageError,
)

__all__ = [
    "__version__",
    "VoltageError", "ConfigError", "PlaybookError", "ExpressionError", "BurstParseError",
    "SafetyViolation", "CaptureError", "InputDeviceError", "BackendError", "SessionError",
    "Aborted",
]
