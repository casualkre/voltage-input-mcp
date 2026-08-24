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

# Rec. 601 luma weights, as float32 so the dot below stays in single precision.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


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
        """Rec. 601 luma as float32.

        A single float32 matmul rather than three weighted channel adds: numpy hands the
        dot to BLAS, and it measures about eight times faster than either the three-term
        form or `.mean(axis=2)`, which is not luma anyway.
        """
        return self.pixels.astype(np.float32) @ _LUMA_WEIGHTS


def downscale(pixels: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Box-filter downscale to fit inside `target`, preserving aspect ratio.

    Two things here were wrong for a long time, and they hid each other.

    This used to reduce by an integer block factor with a strided numpy mean, on the
    stated grounds that it was "several times faster than a resample". Measured on a
    1080p frame: the numpy block mean costs ~103 ms and PIL's BOX resize to the same size
    costs ~11 ms. The premise was backwards by an order of magnitude -- the 5-D reshape
    reduction is cache-hostile, and Pillow's is SIMD C over rows. BOX *is* a box filter,
    and at an exact 2:1 ratio it agrees with the block mean to the last bit, so there is
    no quality argument either.

    The integer factor also meant the requested size was quietly ignored across most of
    its range: 1920x1080 asked for 896x504 came back 960x540, and asked for 644x364 also
    came back 960x540. That is the documented cost knob doing nothing over the whole band
    anyone would tune in, and it discarded `Perception.downscale_to`'s snapping to
    multiples of 28 -- the vision model's token block size -- on the way past.

    Note that fitting inside `target` while preserving aspect means the result lands on
    the 28-pixel grid only when the aspect ratio cooperates. On a 16:9 source that is the
    multiples of 448x252 (so 448x252, 896x504, 1344x756); other sizes come out a few
    pixels off it and the model resizes internally. Grounding is unaffected either way --
    `CoordinateMapper` is built from the array that was actually sent, not the request.
    """
    src_h, src_w = pixels.shape[:2]
    dst_w, dst_h = target
    if dst_w <= 0 or dst_h <= 0 or (src_w <= dst_w and src_h <= dst_h):
        return pixels

    scale = min(dst_w / src_w, dst_h / src_h)
    out_w = max(1, int(src_w * scale))
    out_h = max(1, int(src_h * scale))

    from PIL import Image

    resized = Image.fromarray(pixels, mode="RGB").resize((out_w, out_h), Image.BOX)
    return np.asarray(resized, dtype=np.uint8)


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
        """Report readiness and the *steady-state* cost of a frame.

        The first grab on a streaming backend pays for the whole session -- the portal
        dialog, the PipeWire negotiation, the first buffer -- which on a working setup is
        a couple of hundred milliseconds and on the second grab is about two. Timing the
        cold one and calling it the frame cost makes a streaming backend look ten times
        worse than a per-call one, and the number is then used to decide what reflex rate
        is sustainable. So warm up first, then measure, and report both.
        """
        try:
            t0 = time.perf_counter()
            frame = self.grab()
            cold_ms = (time.perf_counter() - t0) * 1000.0

            samples: list[float] = []
            for _ in range(3):
                t1 = time.perf_counter()
                frame = self.grab()
                samples.append((time.perf_counter() - t1) * 1000.0)
            warm_ms = sorted(samples)[len(samples) // 2]

            return {
                "backend": self.name,
                "ok": True,
                "size": list(frame.size),
                # Two decimals: a warm read from a streaming slot is well under a
                # millisecond, and rounding that to 0.0 reads as "unknown" rather than
                # "too fast to matter" everywhere it is displayed or divided by.
                "latency_ms": round(warm_ms, 2),
                "first_frame_ms": round(cold_ms, 1),
                "streaming": self.streaming,
            }
        except Exception as exc:  # noqa: BLE001 - health checks report, never raise
            return {"backend": self.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
