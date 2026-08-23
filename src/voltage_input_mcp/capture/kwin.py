"""KWin ScreenShot2 capture.

KDE exposes `org.kde.KWin.ScreenShot2` on the session bus. The caller passes the *write*
end of a pipe as a DBus unix-fd; KWin writes raw image bytes into it and returns metadata
(width, height, stride, format) in the reply. No PNG round-trip, no temp file, no
per-shot permission dialog.

The one non-obvious hazard is deadlock. A 1080p RGBA image is ~8 MB and a pipe holds
64 KB, so KWin blocks on write long before it can send its reply. Calling
`call_with_unix_fd_list_sync` and *then* reading the pipe deadlocks reliably. The read
therefore happens on a separate thread that is started before the DBus call.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np

from ..errors import CaptureError
from .base import CaptureBackend, Frame

__all__ = ["KWinBackend", "available"]

_BUS_NAME = "org.kde.KWin"
_OBJECT_PATH = "/org/kde/KWin/ScreenShot2"
_INTERFACE = "org.kde.KWin.ScreenShot2"

# QImage::Format values KWin may hand back. The names describe *memory* byte order on a
# little-endian machine, which is what matters for slicing the buffer.
_FMT_RGB32 = 4              # BGRX
_FMT_ARGB32 = 5             # BGRA
_FMT_ARGB32_PREMUL = 6      # BGRA, premultiplied
_FMT_RGBA8888 = 17          # RGBA
_FMT_RGBA8888_PREMUL = 19   # RGBA, premultiplied
_FMT_RGBX8888 = 18          # RGBX


def available() -> bool:
    """Whether this process can actually use ScreenShot2 -- not merely reach it.

    KWin authorises ScreenShot2 by inspecting the *calling executable* against an
    allowlist (Spectacle, krfb, and friends). Every other process gets
    `Error.NoAuthorized`, no matter what it asks for. The interface being present on the
    bus therefore says nothing about whether it is usable, so this performs a real 1x1
    capture and reports the answer. That costs about a millisecond and shows no dialog.

    In practice this returns False on a stock KDE session and the portal backend is used
    instead, which is the correct path there anyway.
    """
    try:
        from gi.repository import Gio  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    try:
        backend = KWinBackend(timeout_ms=2500)
        backend.grab((0, 0, 1, 1))
        return True
    except Exception:  # noqa: BLE001
        return False


def _connection():
    from gi.repository import Gio

    return Gio.bus_get_sync(Gio.BusType.SESSION, None)


def _call_flags():
    from gi.repository import Gio

    return Gio.DBusCallFlags.NONE


def _variant(signature: str, value: Any):
    from gi.repository import GLib

    return GLib.Variant(signature, value)


def _drain(fd: int, out: list[bytes], limit: int) -> None:
    """Read the pipe to EOF or `limit` bytes. Runs on its own thread."""
    total = 0
    try:
        while total < limit:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            out.append(chunk)
            total += len(chunk)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _to_rgb(buf: bytes, width: int, height: int, stride: int, fmt: int) -> np.ndarray:
    """Reinterpret KWin's raw buffer as an (H, W, 3) RGB array."""
    bytes_per_px = 4
    expected = stride * height
    if len(buf) < expected:
        raise CaptureError(
            f"short read from KWin: got {len(buf)} bytes, expected {expected} "
            f"({width}x{height}, stride {stride})"
        )

    arr = np.frombuffer(buf[:expected], dtype=np.uint8).reshape(height, stride)
    # Strides are padded to a 4-byte (often 32-byte) boundary, so slice to the real width
    # before reshaping into pixels -- otherwise every row is skewed by the padding.
    arr = arr[:, : width * bytes_per_px].reshape(height, width, bytes_per_px)

    if fmt in (_FMT_RGBA8888, _FMT_RGBA8888_PREMUL, _FMT_RGBX8888):
        rgb = arr[:, :, :3]
    elif fmt in (_FMT_ARGB32, _FMT_ARGB32_PREMUL, _FMT_RGB32):
        # ARGB32 on little-endian is B,G,R,A in memory.
        rgb = arr[:, :, 2::-1]
    else:
        raise CaptureError(f"unsupported QImage format {fmt} from KWin ScreenShot2")

    return np.ascontiguousarray(rgb)


class KWinBackend(CaptureBackend):
    """Per-call capture through KWin's screenshot service."""

    name = "kwin"

    def __init__(self, timeout_ms: int = 4000, max_bytes: int = 256 << 20) -> None:
        self._timeout_ms = timeout_ms
        self._max_bytes = max_bytes
        self._frame_id = 0
        self._conn = None
        self._lock = threading.Lock()

    def _bus(self):
        if self._conn is None:
            try:
                self._conn = _connection()
            except Exception as exc:  # noqa: BLE001
                raise CaptureError(f"cannot reach the session bus: {exc}") from exc
        return self._conn

    def grab(self, region: tuple[int, int, int, int] | None = None) -> Frame:
        # KWin can crop server-side via CaptureArea, which is strictly cheaper than
        # grabbing the workspace and slicing: less to serialise through the pipe.
        if region is not None:
            x, y, w, h = region
            method = "CaptureArea"
            params = _variant("(iiuua{sv}h)", (x, y, w, h, {}, 0))
            origin = (x, y)
        else:
            method = "CaptureWorkspace"
            params = _variant("(a{sv}h)", ({}, 0))
            origin = (0, 0)

        meta, buf = self._call(method, params)

        try:
            width = int(meta["width"])
            height = int(meta["height"])
            stride = int(meta["stride"])
            fmt = int(meta["format"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(f"KWin returned unusable metadata: {meta}") from exc

        pixels = _to_rgb(buf, width, height, stride, fmt)
        self._frame_id += 1
        return Frame(
            pixels=pixels, origin=origin, ts=time.monotonic(),
            frame_id=self._frame_id, backend=self.name,
        )

    def _call(self, method: str, params: Any) -> tuple[dict[str, Any], bytes]:
        from gi.repository import Gio, GLib

        with self._lock:
            read_fd, write_fd = os.pipe()
            chunks: list[bytes] = []
            reader = threading.Thread(
                target=_drain, args=(read_fd, chunks, self._max_bytes),
                name="kwin-screenshot-reader", daemon=True,
            )
            reader.start()

            fd_list = Gio.UnixFDList.new()
            try:
                fd_list.append(write_fd)  # dups; our copy is closed below
            except Exception as exc:  # noqa: BLE001
                os.close(write_fd)
                raise CaptureError(f"could not pass the pipe to KWin: {exc}") from exc
            finally:
                # Must close *our* write end or the reader never sees EOF, because the
                # pipe stays open as long as any writer fd exists in any process.
                try:
                    os.close(write_fd)
                except OSError:
                    pass

            try:
                reply, _ = self._bus().call_with_unix_fd_list_sync(
                    _BUS_NAME, _OBJECT_PATH, _INTERFACE, method, params,
                    GLib.VariantType("(a{sv})"), Gio.DBusCallFlags.NONE,
                    self._timeout_ms, fd_list, None,
                )
            except Exception as exc:  # noqa: BLE001
                reader.join(timeout=1.0)
                hint = ""
                if "NoAuthorized" in str(exc):
                    hint = (
                        " KWin only authorises ScreenShot2 for a fixed allowlist of "
                        "executables (Spectacle and similar), so a third-party process "
                        "can never use it. Use the 'portal' backend instead -- set "
                        "capture_backend = \"portal\" in voltage.toml, or leave it on "
                        "\"auto\", which now detects this correctly."
                    )
                raise CaptureError(f"KWin {method} failed: {exc}.{hint}") from exc

            reader.join(timeout=self._timeout_ms / 1000.0 + 2.0)
            if reader.is_alive():
                raise CaptureError("timed out reading the screenshot pipe from KWin")

            meta = reply.unpack()[0]
            return meta, b"".join(chunks)

    def geometry(self) -> tuple[int, int]:
        return self.grab().size
