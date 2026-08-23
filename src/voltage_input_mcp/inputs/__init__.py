"""Input injection: virtual evdev devices and the burst executor."""

from __future__ import annotations

from .executor import ExecutionReport, Executor, PointerMode, TextMode
from .uinput import DeviceSet, UInputDevice, probe_uinput

__all__ = [
    "Executor", "ExecutionReport", "TextMode", "PointerMode",
    "DeviceSet", "UInputDevice", "probe_uinput",
]
