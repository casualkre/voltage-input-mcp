"""Frames and the capture backend interface.

A `Frame` is a plain RGB numpy array plus the geometry needed to map coordinates back
onto the desktop. Everything downstream -- probes, the vision model, the coordinate
mapper -- consumes this and nothing else, so adding a capture backend means implementing
one method.

Backends are ordered by how well they work on the session at hand, not by preference.
On KDE Wayland the honest ordering is:

    kwin      org.kde.KWin.ScreenShot2 over a pipe fd. Raw pixels, no re-encode, no
              per-shot permission prompt. ~15-40 ms for a 1080p workspace grab.
    portal    xdg-desktop-portal ScreenCast -> PipeWire -> GStreamer appsink. One
              permission dialog per session (persisted via restore token), then a
              continuous stream. Higher setup cost, much cheaper per frame, and the
              only option that keeps up with probe-rate capture.
    grim      wlroots only. Present on this box but does not work under KWin.
    x11       XWayland only, so it sees almost nothing on a modern KDE desktop. Last
              resort, and mostly there for people running this on actual X11.
"""

from __future__ import annotations

import abc
import io
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..errors import CaptureError

__all__ = ["CaptureBackend", "Frame", "downscale", "encode_png"]

BackendName = Literal["kwin", "portal", "grim", "x11", "stub"]


@dataclass(slots=True)
class Frame:
    """One captured image in RGB, plus where it sits on the desktop."""

    pixels: np.ndarray  # (H, W, 3) uint8, RGB
    origin: tuple[int, int] = (0, 0)
    ts: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    backend: str = "stub"

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise CaptureError(
                f"frame must be (H, W, 3) RGB, got shape {self.pixels.shape}"
            )
        if self.pixels.dtype != np.uint8:
            self.pixels = self.pixels.astype(np.uint8, copy=False)

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def age_s(self) -> float:
        return time.monotonic() - self.ts

    def crop(self, x: int, y: int, w: int, h: int) -> Frame:
        """Crop in frame-local coordinates, carrying the origin forward."""
        x0 = max(0, min(x, self.width))
        y0 = max(0, min(y, self.height))
        x1 = max(x0, min(x + w, self.width))
        y1 = max(y0, min(y + h, self.height))
        return Frame(
            pixels=self.pixels[y0:y1, x0:x1],
            origin=(self.origin[0] + x0, self.origin[1] + y0),
            ts=self.ts,
            frame_id=self.frame_id,
            backend=self.backend,
        )

    def downscaled(self, target: tuple[int, int]) -> np.ndarray:
        return downscale(self.pixels, target)

    def to_png(self, target: tuple[int, int] | None = None, quality: int = 6) -> bytes:
        pixels = self.downscaled(target) if target else self.pixels
        return encode_png(pixels, compress_level=quality)

    def luma(self) -> np.ndarray:
        """Rec. 601 luma as float32. Used by the diff and brightness probes."""
        p = self.pixels
        return (
            0.299 * p[:, :, 0].astype(np.float32)
            + 0.587 * p[:, :, 1].astype(np.float32)
            + 0.114 * p[:, :, 2].astype(np.float32)
        )


def downscale(pixels: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Box-filter downscale to fit inside `target`, preserving aspect ratio.

    Deliberately not PIL: this runs on the hot path before every vision call, and a
    strided mean over an integer block factor is several times faster than a Lanczos
    resample while being entirely good enough as input to a vision tower that is about
    to patchify the image anyway.

    Falls back to PIL for non-integer ratios, where naive striding would alias badly.
    """
    src_h, src_w = pixels.shape[:2]
    dst_w, dst_h = target
    if dst_w <= 0 or dst_h <= 0 or (src_w <= dst_w and src_h <= dst_h):
        return pixels

    scale = min(dst_w / src_w, dst_h / src_h)
    out_w = max(1, int(src_w * scale))
    out_h = max(1, int(src_h * scale))

    fx, fy = src_w // out_w, src_h // out_h
    if fx >= 1 and fy >= 1 and src_w % fx == 0 and src_h % fy == 0 and (fx > 1 or fy > 1):
        trimmed = pixels[: (src_h // fy) * fy, : (src_w // fx) * fx]
        blocks = trimmed.reshape(src_h // fy, fy, src_w // fx, fx, 3)
        return blocks.mean(axis=(1, 3)).astype(np.uint8)

    from PIL import Image  # local import: only needed on the non-integer path

    img = Image.fromarray(pixels, mode="RGB").resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def encode_png(pixels: np.ndarray, compress_level: int = 6) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


class CaptureBackend(abc.ABC):
    """One way of getting pixels off the screen."""

    name: BackendName = "stub"

    @abc.abstractmethod
    def grab(self, region: tuple[int, int, int, int] | None = None) -> Frame:
        """Capture the desktop, or `region` as (x, y, w, h) in desktop coordinates."""

    def start(self) -> None:
        """Optional: open a persistent stream. Idempotent."""

    def stop(self) -> None:
        """Optional: tear down a persistent stream. Idempotent."""

    @property
    def streaming(self) -> bool:
        """True if `grab()` reads from a live stream rather than doing a fresh capture."""
        return False

    def geometry(self) -> tuple[int, int]:
        """Full desktop size. Derived from a capture unless a backend knows better."""
        return self.grab().size

    def health(self) -> dict[str, object]:
        try:
            t0 = time.perf_counter()
            frame = self.grab()
            dt = (time.perf_counter() - t0) * 1000.0
            return {
                "backend": self.name,
                "ok": True,
                "size": list(frame.size),
                "latency_ms": round(dt, 1),
                "streaming": self.streaming,
            }
        except Exception as exc:  # noqa: BLE001 - health checks report, never raise
            return {"backend": self.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
