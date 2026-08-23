"""The platform-neutral input interface.

Five operations are enough to express every burst action. Everything above this line --
scheduling, timing, held-key tracking, abort handling, the safety governor -- is shared
across platforms; everything below it is per-platform and small.

Implementations:

    DeviceSet   Linux, via /dev/uinput (kernel evdev layer, works under X11, Wayland,
                the console, and in games reading raw input)
    Win32Sink   Windows, via SendInput (Win32 message layer)

`text()` is optional. It exists because Windows can type a UTF-16 code unit directly with
no keyboard layout involved, which is both simpler and more correct than the scancode
path Linux is stuck with. A sink that omits it gets the executor's keystroke/clipboard
handling instead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["InputSink"]


@runtime_checkable
class InputSink(Protocol):
    """What the executor needs from a platform."""

    screen: tuple[int, int]

    def open(self) -> None: ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...

    def key(self, name: str, down: bool) -> None:
        """Press or release a key, named canonically (see keymap.canonical_key)."""

    def button(self, name: str, down: bool) -> None:
        """Press or release a pointer button: l, r, m, 4, 5."""

    def move_abs(self, x: int, y: int) -> None:
        """Move the pointer to a desktop pixel."""

    def move_rel(self, dx: int, dy: int) -> None:
        """Move the pointer by a delta."""

    def scroll(self, amount: int, axis: str = "v") -> None:
        """Scroll by `amount` detents; axis is "v" or "h"."""
