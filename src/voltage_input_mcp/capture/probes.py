"""Model-free screen measurements.

Probes are the reason the loop can run faster than the vision model. A 3B VLM answering
"is the health bar low" costs a few hundred milliseconds; sampling one pixel costs about
two microseconds. Anything reducible to a pixel, a region average, or a change detector
belongs here, and reflex rules can then fire on it with no model in the path at all.

Return contract -- every probe yields a float in [0, 1] so guards can threshold uniformly:

    pixel        1.0 if the sampled colour is within `tolerance`, else 0.0
    region_mean  1.0 if the region's mean colour is within `tolerance`, else 0.0
    brightness   mean luma, normalised
    region_diff  fraction of pixels in the region that changed since the last frame
    template     best normalised cross-correlation score in the search region

Two probes are always published, whether or not the playbook declares any:

    __frame_delta__  fraction of the whole frame that changed since the previous frame
    __static_for__   seconds the screen has been materially unchanged

They back the `changed()` and `stalled()` guard functions and, more importantly, gate
whether the vision model runs at all this cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..models.playbook import ProbeSpec, Rect
from .base import Frame

__all__ = ["ProbeEngine", "parse_hex_colour"]

# The whole-frame change detector runs on a thumbnail. Full-resolution differencing of a
# 1080p frame costs ~4 ms; at 160x90 it costs ~40 us and detects exactly the same
# "something moved on screen" events, which is all it is used for.
_DELTA_W, _DELTA_H = 160, 90
# Per-pixel luma delta above which a pixel counts as changed. Below ~6 it trips on
# compositor dithering and font antialiasing.
_PIXEL_CHANGE_THRESHOLD = 8.0


def parse_hex_colour(value: str) -> np.ndarray:
    text = value.lstrip("#")
    return np.array(
        [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)], dtype=np.float32
    )


@dataclass(slots=True)
class ProbeEngine:
    """Evaluates a playbook's probes against each frame, holding per-probe history."""

    specs: list[ProbeSpec] = field(default_factory=list)
    _prev_thumb: np.ndarray | None = field(default=None, init=False)
    _prev_regions: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _templates: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _static_since: float = field(default_factory=time.monotonic, init=False)
    _last_values: dict[str, float] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._prev_thumb = None
        self._prev_regions.clear()
        self._static_since = time.monotonic()
        self._last_values.clear()

    @property
    def last(self) -> dict[str, float]:
        return dict(self._last_values)

    def evaluate(self, frame: Frame) -> dict[str, float]:
        values: dict[str, float] = {}

        delta = self._frame_delta(frame)
        values["__frame_delta__"] = delta
        now = time.monotonic()
        if delta > 0.004:
            self._static_since = now
        values["__static_for__"] = now - self._static_since

        for spec in self.specs:
            try:
                values[spec.id] = self._evaluate_one(spec, frame)
            except Exception:  # noqa: BLE001 - a broken probe must not stop the loop
                values[spec.id] = 0.0

        self._last_values = values
        return values

    # -- individual probes -----------------------------------------------------------

    def _evaluate_one(self, spec: ProbeSpec, frame: Frame) -> float:
        if spec.type == "pixel":
            return self._pixel(spec, frame)
        if spec.type == "region_mean":
            return self._region_mean(spec, frame)
        if spec.type == "brightness":
            return self._brightness(spec, frame)
        if spec.type == "region_diff":
            return self._region_diff(spec, frame)
        if spec.type == "template":
            return self._template(spec, frame)
        return 0.0

    def _pixel(self, spec: ProbeSpec, frame: Frame) -> float:
        assert spec.at is not None and spec.expect is not None
        x, y = self._to_local(spec.at[0], spec.at[1], frame)
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            return 0.0
        actual = frame.pixels[y, x].astype(np.float32)
        return self._colour_match(actual, spec)

    def _region_mean(self, spec: ProbeSpec, frame: Frame) -> float:
        assert spec.region is not None and spec.expect is not None
        patch = self._patch(spec.region, frame)
        if patch.size == 0:
            return 0.0
        return self._colour_match(patch.reshape(-1, 3).mean(axis=0), spec)

    def _brightness(self, spec: ProbeSpec, frame: Frame) -> float:
        assert spec.region is not None
        patch = self._patch(spec.region, frame)
        if patch.size == 0:
            return 0.0
        luma = (
            0.299 * patch[:, :, 0].astype(np.float32)
            + 0.587 * patch[:, :, 1].astype(np.float32)
            + 0.114 * patch[:, :, 2].astype(np.float32)
        )
        return float(luma.mean() / 255.0)

    def _region_diff(self, spec: ProbeSpec, frame: Frame) -> float:
        assert spec.region is not None
        patch = self._patch(spec.region, frame)
        if patch.size == 0:
            return 0.0
        luma = patch.mean(axis=2).astype(np.float32)
        prev = self._prev_regions.get(spec.id)
        self._prev_regions[spec.id] = luma
        if prev is None or prev.shape != luma.shape:
            return 0.0
        changed = np.abs(luma - prev) > _PIXEL_CHANGE_THRESHOLD
        return float(changed.mean())

    def _template(self, spec: ProbeSpec, frame: Frame) -> float:
        assert spec.template is not None and spec.region is not None
        template = self._load_template(spec.template)
        if template is None:
            return 0.0
        patch = self._patch(spec.region, frame)
        if patch.size == 0:
            return 0.0
        return _ncc_peak(patch.mean(axis=2).astype(np.float32),
                         template.mean(axis=2).astype(np.float32))

    # -- helpers ---------------------------------------------------------------------

    def _colour_match(self, actual: np.ndarray, spec: ProbeSpec) -> float:
        assert spec.expect is not None
        target = parse_hex_colour(spec.expect)
        if spec.channel == "rgb":
            distance = float(np.abs(actual - target).max())
        elif spec.channel == "luma":
            to_luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            distance = float(abs(actual @ to_luma - target @ to_luma))
        else:
            idx = {"r": 0, "g": 1, "b": 2}[spec.channel]
            distance = float(abs(actual[idx] - target[idx]))
        return 1.0 if distance <= spec.tolerance else 0.0

    def _to_local(self, x: int, y: int, frame: Frame) -> tuple[int, int]:
        """Desktop coordinates -> frame-local, accounting for a cropped capture."""
        ox, oy = frame.origin
        return x - ox, y - oy

    def _patch(self, rect: Rect, frame: Frame) -> np.ndarray:
        x, y = self._to_local(rect.x, rect.y, frame)
        x0 = max(0, min(x, frame.width))
        y0 = max(0, min(y, frame.height))
        x1 = max(x0, min(x + rect.w, frame.width))
        y1 = max(y0, min(y + rect.h, frame.height))
        return frame.pixels[y0:y1, x0:x1]

    def _load_template(self, path: str) -> np.ndarray | None:
        cached = self._templates.get(path)
        if cached is not None:
            return cached
        try:
            from PIL import Image

            with Image.open(Path(path).expanduser()) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        except Exception:  # noqa: BLE001
            return None
        self._templates[path] = arr
        return arr

    def _frame_delta(self, frame: Frame) -> float:
        thumb = _thumbnail_luma(frame.pixels)
        prev = self._prev_thumb
        self._prev_thumb = thumb
        if prev is None or prev.shape != thumb.shape:
            return 1.0  # first frame counts as fully changed, so perception runs
        return float((np.abs(thumb - prev) > _PIXEL_CHANGE_THRESHOLD).mean())


def _thumbnail_luma(pixels: np.ndarray) -> np.ndarray:
    """Cheap fixed-size luma thumbnail via integer-factor strided subsampling."""
    h, w = pixels.shape[:2]
    step_y = max(1, h // _DELTA_H)
    step_x = max(1, w // _DELTA_W)
    sub = pixels[::step_y, ::step_x]
    return sub.mean(axis=2).astype(np.float32)


def _ncc_peak(image: np.ndarray, template: np.ndarray) -> float:
    """Peak normalised cross-correlation of `template` within `image`.

    Uses `sliding_window_view`, which is O(search_area * template_area). That is fine for
    the small search regions template probes are meant for (a button, an icon) and
    catastrophic for a full-screen search -- hence the requirement that template probes
    declare a `region`.
    """
    ih, iw = image.shape
    th, tw = template.shape
    if th > ih or tw > iw or th == 0 or tw == 0:
        return 0.0

    t = template - template.mean()
    t_norm = float(np.sqrt((t * t).sum()))
    if t_norm == 0.0:
        return 0.0

    windows = np.lib.stride_tricks.sliding_window_view(image, (th, tw))
    # windows: (ih-th+1, iw-tw+1, th, tw)
    means = windows.mean(axis=(2, 3), keepdims=True)
    centred = windows - means
    numerator = (centred * t).sum(axis=(2, 3))
    denominator = np.sqrt((centred * centred).sum(axis=(2, 3))) * t_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, numerator / denominator, 0.0)
    return float(np.clip(scores.max(), 0.0, 1.0))
