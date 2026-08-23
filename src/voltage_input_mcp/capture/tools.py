"""Subprocess capture fallbacks: grim (wlroots) and ImageMagick import (X11).

Neither of these is a good option on KDE Wayland -- `grim` needs wlr-screencopy, which
KWin does not implement, and `import` only sees XWayland surfaces. They exist so this
runs unmodified on Sway/Hyprland and on real X11 sessions, and so `voltage.doctor` can
report *why* a session has no working backend rather than just failing.

Both shell out and decode a PNG per frame, which costs 60-200 ms. Treat them as
correctness fallbacks, not as something to run a 2 Hz loop on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import numpy as np

from ..errors import CaptureError
from .base import CaptureBackend, Frame

__all__ = ["GrimBackend", "X11Backend", "grim_available", "x11_available"]


def _decode_png(data: bytes) -> np.ndarray:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def grim_available() -> bool:
    if not shutil.which("grim") or not os.environ.get("WAYLAND_DISPLAY"):
        return False
    # grim exits non-zero on KWin because wlr-screencopy is absent, so actually try it.
    try:
        proc = subprocess.run(["grim", "-"], capture_output=True, timeout=4.0, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and proc.stdout[:8] == b"\x89PNG\r\n\x1a\n"


def x11_available() -> bool:
    return bool(os.environ.get("DISPLAY")) and bool(shutil.which("import"))


class GrimBackend(CaptureBackend):
    """wlroots screencopy via grim."""

    name = "grim"

    def __init__(self, timeout_s: float = 5.0) -> None:
        self._timeout = timeout_s
        self._frame_id = 0

    def grab(self, region: tuple[int, int, int, int] | None = None) -> Frame:
        cmd = ["grim"]
        origin = (0, 0)
        if region is not None:
            x, y, w, h = region
            cmd += ["-g", f"{x},{y} {w}x{h}"]
            origin = (x, y)
        cmd.append("-")
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=self._timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CaptureError(f"grim failed: {exc}") from exc
        if proc.returncode != 0:
            raise CaptureError(
                f"grim exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace').strip()}. "
                "On KWin this is expected -- grim needs wlr-screencopy, which KDE does "
                "not implement. Use the kwin or portal backend instead."
            )
        self._frame_id += 1
        return Frame(
            pixels=_decode_png(proc.stdout), origin=origin,
            ts=time.monotonic(), frame_id=self._frame_id, backend=self.name,
        )


class X11Backend(CaptureBackend):
    """X11 root-window capture via ImageMagick `import`.

    Under XWayland this sees only X11 clients, which on a modern KDE desktop is usually
    nothing at all -- the capture succeeds and returns a black or stale image. That
    silent-wrongness is why this backend is last in the auto-detect order.
    """

    name = "x11"

    def __init__(self, timeout_s: float = 6.0) -> None:
        self._timeout = timeout_s
        self._frame_id = 0

    def grab(self, region: tuple[int, int, int, int] | None = None) -> Frame:
        cmd = ["import", "-silent", "-window", "root"]
        origin = (0, 0)
        if region is not None:
            x, y, w, h = region
            cmd += ["-crop", f"{w}x{h}+{x}+{y}"]
            origin = (x, y)
        cmd += ["png:-"]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=self._timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CaptureError(f"import failed: {exc}") from exc
        if proc.returncode != 0:
            raise CaptureError(
                f"import exited {proc.returncode}: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        self._frame_id += 1
        return Frame(
            pixels=_decode_png(proc.stdout), origin=origin,
            ts=time.monotonic(), frame_id=self._frame_id, backend=self.name,
        )
