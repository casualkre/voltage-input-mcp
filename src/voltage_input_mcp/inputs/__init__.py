"""Input injection: a platform sink plus the burst executor."""

from __future__ import annotations

import sys

from .executor import ExecutionReport, Executor, PointerMode, TextMode
from .sink import InputSink
from .uinput import DeviceSet, UInputDevice, probe_uinput

__all__ = [
    "Executor", "ExecutionReport", "TextMode", "PointerMode", "InputSink",
    "DeviceSet", "UInputDevice", "probe_uinput", "create_sink", "probe_input",
]


def create_sink(screen: tuple[int, int] | None = None) -> InputSink:
    """Build the input sink for this platform."""
    if sys.platform == "win32":
        from .win32 import Win32Sink

        return Win32Sink(screen=screen)
    return DeviceSet(screen=screen or (1920, 1080))


def probe_input() -> dict[str, object]:
    """Whether input injection can work here, and the fix if not."""
    if sys.platform == "win32":
        from .win32 import probe_win32

        return probe_win32()
    return probe_uinput()
