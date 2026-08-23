"""Exception hierarchy.

Every failure mode that can reach an MCP client is a `VoltageError` subclass with a
stable `code`. The MCP layer serialises `code` + `detail` so the orchestrating model
can branch on the failure without string-matching prose.
"""

from __future__ import annotations

from typing import Any


class VoltageError(Exception):
    code = "voltage_error"

    def __init__(self, detail: str, **context: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "detail": self.detail, **self.context}


class ConfigError(VoltageError):
    code = "config_error"


class PlaybookError(VoltageError):
    """Playbook failed schema or static validation. Always fixable by the orchestrator."""

    code = "playbook_invalid"


class ExpressionError(VoltageError):
    """A guard expression is malformed or used a disallowed construct."""

    code = "expression_invalid"


class BurstParseError(VoltageError):
    """A burst string could not be parsed into actions."""

    code = "burst_parse_error"


class SafetyViolation(VoltageError):
    """The governor refused an action. Carries the rule that fired."""

    code = "safety_violation"

    def __init__(self, detail: str, rule: str, action: str | None = None, **context: Any) -> None:
        super().__init__(detail, rule=rule, action=action, **context)
        self.rule = rule
        self.action = action


class CaptureError(VoltageError):
    code = "capture_error"


class InputDeviceError(VoltageError):
    code = "input_device_error"


class BackendError(VoltageError):
    """A local model backend was unreachable or returned something unusable."""

    code = "backend_error"


class SessionError(VoltageError):
    code = "session_error"


class Aborted(VoltageError):
    """Raised inside the loop when a panic stop or deadman timer fires."""

    code = "aborted"
