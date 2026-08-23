"""Capture backend selection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..errors import CaptureError
from .base import CaptureBackend, Frame, downscale, encode_png
from .probes import ProbeEngine

__all__ = [
    "CaptureBackend", "Frame", "downscale", "encode_png", "ProbeEngine",
    "create_backend", "detect_backends", "AUTO_ORDER",
]

# Ordered by how well each works on a modern compositor, not by ease of use. The first
# entry that reports available() wins when `preference="auto"`.
#
# `portal` leads because it is the only backend that works on both KDE and GNOME under
# Wayland. `kwin` would be cheaper per frame -- raw pixels over a pipe, no dialog -- but
# KWin authorises ScreenShot2 against an allowlist of executables, so a third-party
# process gets NoAuthorized. Its available() performs a real capture to detect that, and
# it stays in the list because it *does* work when this runs from an authorised binary.
AUTO_ORDER: tuple[str, ...] = (
    ("windows",) if sys.platform == "win32" else ("portal", "kwin", "grim", "x11")
)


def detect_backends() -> dict[str, bool]:
    """Which backends could work here. Used by `voltage.doctor`."""
    if sys.platform == "win32":
        from . import windows

        return {"windows": windows.available()}

    from . import kwin, portal, tools

    return {
        "kwin": kwin.available(),
        "portal": portal.available(),
        "grim": tools.grim_available(),
        "x11": tools.x11_available(),
    }


def create_backend(
    preference: str = "auto", *, state_dir: Path | None = None, cursor: bool = True
) -> CaptureBackend:
    """Build a capture backend.

    `preference` is one of AUTO_ORDER, or "auto" to pick the best available. An explicit
    name is honoured even if `available()` says no, so a user can force a backend and
    see the real error rather than a generic "nothing available".
    """
    def build(name: str) -> CaptureBackend:
        if name == "windows":
            from . import windows

            return windows.WindowsBackend()

        from . import kwin, portal, tools

        if name == "kwin":
            return kwin.KWinBackend()
        if name == "portal":
            return portal.PortalBackend(
                state_dir=state_dir,
                cursor_mode=portal.CURSOR_EMBEDDED if cursor else portal.CURSOR_HIDDEN,
            )
        if name == "grim":
            return tools.GrimBackend()
        if name == "x11":
            return tools.X11Backend()
        raise CaptureError(
            f"unknown capture backend {name!r}; expected one of {list(AUTO_ORDER)} or 'auto'"
        )

    if preference != "auto":
        return build(preference)

    detected = detect_backends()
    for name in AUTO_ORDER:
        if detected.get(name):
            return build(name)

    if sys.platform == "win32":
        raise CaptureError(
            "screen capture failed on Windows. GDI BitBlt needs a desktop session -- it "
            "cannot capture from a service or a disconnected RDP session.",
            detected=detected,
        )
    session = os.environ.get("XDG_SESSION_TYPE", "unknown")
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "unknown")
    raise CaptureError(
        f"no capture backend works on this session (type={session}, desktop={desktop}). "
        f"Probed: {detected}. On KDE Wayland, check that KWin is running and that "
        f"xdg-desktop-portal-kde is installed.",
        detected=detected,
    )
