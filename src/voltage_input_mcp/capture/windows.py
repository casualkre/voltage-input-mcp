"""Windows screen capture via GDI BitBlt, ctypes only.

Chosen over `Windows.Graphics.Capture` deliberately. WGC is the modern API, is faster,
and handles hardware-overlay content that GDI misses -- but it needs WinRT projection
(`winsdk`/`pywinrt`), which is a heavyweight dependency that fails to install on plenty
of systems. BitBlt is in every Windows since NT, needs nothing but ctypes, and captures a
1080p desktop in roughly 10-25 ms, which is well inside the budget here given that decode
dominates the loop by two orders of magnitude.

The known GDI limitation is worth stating plainly: it cannot see content rendered by some
hardware-accelerated video overlays and certain full-screen exclusive games, which come
back black. `CAPTUREBLT` is set to include layered windows, which covers most desktop
cases. If a specific game captures black, that is this limitation and not a bug -- run it
in borderless windowed mode.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from typing import Final

import numpy as np

from ..errors import CaptureError
from .base import CaptureBackend, Frame

__all__ = ["WindowsBackend", "available"]

SRCCOPY: Final = 0x00CC0020
CAPTUREBLT: Final = 0x40000000
DIB_RGB_COLORS: Final = 0
BI_RGB: Final = 0

SM_XVIRTUALSCREEN: Final = 76
SM_YVIRTUALSCREEN: Final = 77
SM_CXVIRTUALSCREEN: Final = 78
SM_CYVIRTUALSCREEN: Final = 79


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def available() -> bool:
    return sys.platform == "win32"


class WindowsBackend(CaptureBackend):
    """Per-call desktop capture. Spans all monitors as one virtual desktop."""

    name = "windows"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise CaptureError("the windows capture backend requires Windows")
        from ..inputs.win32 import _declare_dpi_awareness

        # Same reason as the input side: without this, a scaled display reports
        # virtualised metrics and the captured image does not match real coordinates.
        _declare_dpi_awareness()
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._frame_id = 0

    def geometry(self) -> tuple[int, int]:
        return (
            self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )

    def grab(self, region: tuple[int, int, int, int] | None = None) -> Frame:
        user32, gdi32 = self._user32, self._gdi32

        if region is not None:
            x, y, width, height = region
        else:
            x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

        if width <= 0 or height <= 0:
            raise CaptureError(f"invalid capture region {width}x{height}")

        screen_dc = user32.GetDC(None)
        if not screen_dc:
            raise CaptureError("GetDC(NULL) failed")

        memory_dc = bitmap = None
        try:
            memory_dc = gdi32.CreateCompatibleDC(screen_dc)
            bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
            if not memory_dc or not bitmap:
                raise CaptureError("could not create a compatible DC or bitmap")
            gdi32.SelectObject(memory_dc, bitmap)

            # CAPTUREBLT includes layered/transparent windows, which are otherwise
            # missing -- and a missing dropdown is exactly the thing a playbook is
            # usually waiting for.
            if not gdi32.BitBlt(
                memory_dc, 0, 0, width, height, screen_dc, x, y, SRCCOPY | CAPTUREBLT
            ):
                raise CaptureError(f"BitBlt failed (error {ctypes.get_last_error()})")

            info = BITMAPINFO()
            info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            info.bmiHeader.biWidth = width
            # Negative height requests a top-down DIB. Without it the rows come back
            # bottom-up and the image is vertically mirrored.
            info.bmiHeader.biHeight = -height
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = BI_RGB

            buffer = ctypes.create_string_buffer(width * height * 4)
            copied = gdi32.GetDIBits(
                memory_dc, bitmap, 0, height, buffer, ctypes.byref(info), DIB_RGB_COLORS
            )
            if copied == 0:
                raise CaptureError("GetDIBits returned no scanlines")

            # 32-bit BI_RGB is BGRX in memory; drop the padding byte and reverse.
            arr = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
            pixels = np.ascontiguousarray(arr[:, :, 2::-1])
        finally:
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if memory_dc:
                gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(None, screen_dc)

        self._frame_id += 1
        return Frame(
            pixels=pixels,
            origin=(x, y),
            ts=time.monotonic(),
            frame_id=self._frame_id,
            backend=self.name,
        )
