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

Three probes are always published, whether or not the playbook declares any:

    __frame_delta__      fraction of the whole frame that changed since the previous call
    __static_for__       seconds the screen has been materially unchanged
    __delta_vs_vision__  fraction that has changed since the vision model last looked

The first two back the `changed()` and `stalled()` guard functions. The third gates
whether the vision model runs at all, and it is separate for a reason that only became
apparent once reflexes moved into their own loop: with two loops calling `evaluate()`,
`__frame_delta__` means "changed in the last 50 ms", which is not the question
`perception.mode = on_change` is asking. That question is "does the screen still look
like what the VLM last saw", and it can only be answered against a baseline captured at
that moment -- so one is kept.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..models.playbook import ProbeSpec, Rect
from .base import Frame

__all__ = ["ProbeEngine", "parse_hex_colour", "ocr_available"]

# The whole-frame change detector runs on a thumbnail. Full-resolution differencing of a
# 1080p frame costs ~4 ms; at 160x90 it costs ~40 us and detects exactly the same
# "something moved on screen" events, which is all it is used for.
_DELTA_W, _DELTA_H = 160, 90
# Per-pixel luma delta above which a pixel counts as changed. Below ~6 it trips on
# compositor dithering and font antialiasing.
_PIXEL_CHANGE_THRESHOLD = 8.0

# Region probes stride down to at most this many pixels before measuring. The argument is
# the same one that justifies the thumbnail above, and the numbers are stark: measured on a
# 1080p frame, an 800x500 region_diff cost 10.9 ms at full resolution and 0.24 ms
# subsampled. At 20 Hz that is the difference between a probe costing 20% of a core and
# 0.5% of one -- and the reflex loop is the one thing that must never be expensive.
#
# What it costs: nothing that matters. A `region_diff` returns the *fraction* of sampled
# pixels that changed, and subsampling scales numerator and denominator together, so the
# fraction is preserved in expectation. A `brightness` or `region_mean` over a large area
# is an average, and an average over a regular sample of it is the same number to within
# noise. Regions smaller than this are not subsampled at all, which covers every HUD
# element and progress bar -- the cases where precision is actually wanted.
_MAX_PROBE_PIXELS = 4096

# Rec. 601 luma. Used as a float32 dot rather than `.mean(axis=2)`, which is both eight
# times slower (float64 promotion of every channel) and not actually luma -- a flat channel
# average treats blue as though the eye weighed it the same as green.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _luma(pixels: np.ndarray) -> np.ndarray:
    return pixels.astype(np.float32) @ _LUMA_WEIGHTS


def _subsample(patch: np.ndarray, max_pixels: int = _MAX_PROBE_PIXELS) -> np.ndarray:
    """Stride `patch` down to at most `max_pixels`, or return it unchanged if it fits."""
    height, width = patch.shape[:2]
    total = height * width
    if total <= max_pixels or total == 0:
        return patch
    step = max(1, int((total / max_pixels) ** 0.5))
    return patch[::step, ::step]


def parse_hex_colour(value: str) -> np.ndarray:
    text = value.lstrip("#")
    return np.array(
        [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)], dtype=np.float32
    )


@dataclass(slots=True)
class ProbeEngine:
    """Evaluates a playbook's probes against each frame, holding per-probe history.

    One engine serves both loops. The reflex loop calls `evaluate()` at reflex rate and
    the decision loop reads the result, so probe history is single-sourced and the two
    loops cannot disagree about what the screen is doing. Everything on this path is
    bounded and non-blocking: the slowest measurement is a template match over a small
    region, and OCR -- the one genuinely slow probe -- is handed to a background worker
    and never awaited.
    """

    specs: list[ProbeSpec] = field(default_factory=list)
    _prev_thumb: np.ndarray | None = field(default=None, init=False)
    _vision_thumb: np.ndarray | None = field(default=None, init=False)
    _prev_regions: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _templates: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _static_since: float = field(default_factory=time.monotonic, init=False)
    _last_values: dict[str, float] = field(default_factory=dict, init=False)
    _numbers: dict[str, float] = field(default_factory=dict, init=False)
    _number_at: dict[str, float] = field(default_factory=dict, init=False)
    _rates: dict[str, float] = field(default_factory=dict, init=False)
    _ocr_asked: dict[str, float] = field(default_factory=dict, init=False)
    _ocr_inflight: dict[str, Future] = field(default_factory=dict, init=False)
    _ocr_reads: int = field(default=0, init=False)
    _ocr_misses: int = field(default=0, init=False)
    _pool: ThreadPoolExecutor | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def reset(self) -> None:
        self._prev_thumb = None
        self._vision_thumb = None
        with self._lock:
            self._numbers.clear()
            self._number_at.clear()
            self._rates.clear()
        self._ocr_asked.clear()
        self._ocr_inflight.clear()
        self._ocr_reads = 0
        self._ocr_misses = 0
        self._prev_regions.clear()
        self._static_since = time.monotonic()
        self._last_values.clear()

    def close(self) -> None:
        """Drop the OCR worker. Safe to call more than once."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    @property
    def last(self) -> dict[str, float]:
        return dict(self._last_values)

    def ocr_stats(self) -> dict[str, int]:
        """Successful and failed OCR reads, for diagnosis.

        A number probe that never resolves looks exactly like one reading a steady zero,
        and a guard cannot tell the difference. Counting the failures is what lets
        `voltage_diagnose` say "this probe has never produced a number" instead of the
        playbook author concluding the HUD really does say 0.
        """
        return {"reads": self._ocr_reads, "misses": self._ocr_misses}

    def mark_perceived(self, frame: Frame) -> None:
        """Record that the vision model has just looked at this frame.

        Sets the baseline `__delta_vs_vision__` measures against. Called by the session
        immediately after a successful vision call, and only then -- a skipped or failed
        call must leave the old baseline in place, or the loop concludes the screen is
        unchanged relative to an observation it never made.
        """
        self._vision_thumb = _thumbnail_luma(frame.pixels)

    def evaluate(self, frame: Frame) -> dict[str, float]:
        values: dict[str, float] = {}

        thumb = _thumbnail_luma(frame.pixels)
        prev, self._prev_thumb = self._prev_thumb, thumb
        delta = _thumb_delta(thumb, prev)
        values["__frame_delta__"] = delta
        values["__delta_vs_vision__"] = _thumb_delta(thumb, self._vision_thumb)

        now = time.monotonic()
        if delta > 0.004:
            self._static_since = now
        values["__static_for__"] = now - self._static_since

        for spec in self.specs:
            try:
                values[spec.id] = self._evaluate_one(spec, frame)
            except Exception:  # noqa: BLE001 - a broken probe must not stop the loop
                values[spec.id] = 0.0

        # Derivatives ride alongside, so `rate('meters')` is available without the
        # playbook differencing anything itself.
        with self._lock:
            for key, rate in self._rates.items():
                values[f"{key}__rate"] = rate

        self._last_values = values
        return values

    # -- individual probes -----------------------------------------------------------

    def _number(self, spec: ProbeSpec, frame: Frame) -> float:
        """Read a number off the HUD, without ever waiting for the read.

        OCR is 80-200 ms. Doing it inline would make a 20 Hz reflex loop into a 5 Hz one
        the moment a playbook declares a number probe -- the probe meant to make control
        possible would be the thing that made it impossible. So this never runs tesseract
        on the calling thread. It returns the last value it has and, if that value is
        older than `ocr_interval_ms`, hands a copy of the patch to a single background
        worker to refresh in the meantime.

        The consequence is that a number is always slightly stale, by one OCR round trip.
        For a falling character at 170 m/s that is around 20 m of error, which is why
        `rate()` exists: extrapolating from a rate is more accurate than waiting for a
        fresher absolute value would have been, and costs nothing.
        """
        assert spec.region is not None
        with self._lock:
            cached = self._numbers.get(spec.id)

        now = time.monotonic()
        asked = self._ocr_asked.get(spec.id, 0.0)
        due = (now - asked) * 1000.0 >= spec.ocr_interval_ms
        inflight = self._ocr_inflight.get(spec.id)
        if inflight is not None and inflight.done():
            self._ocr_inflight.pop(spec.id, None)
            inflight = None

        if due and inflight is None:
            patch = self._patch(spec.region, frame)
            if patch.size:
                self._ocr_asked[spec.id] = now
                # .copy() because the frame buffer is reused by streaming backends; the
                # worker would otherwise OCR whatever the screen looks like when it gets
                # around to it rather than what it was handed.
                self._ocr_inflight[spec.id] = self._worker().submit(
                    self._ocr_into, spec, patch.copy()
                )

        return cached if cached is not None else 0.0

    def _worker(self) -> ThreadPoolExecutor:
        if self._pool is None:
            # One worker, deliberately. Number probes are read off small fixed HUD
            # regions, so several in parallel would contend for the same cores that the
            # capture and the two model servers need, to shave latency off a value that
            # is already only used for thresholds.
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voltage-ocr")
        return self._pool

    def _ocr_into(self, spec: ProbeSpec, patch: np.ndarray) -> None:
        value = _ocr_number(patch, invert=spec.invert)
        if value is None:
            self._ocr_misses += 1
            return
        self._ocr_reads += 1
        value *= spec.scale
        now = time.monotonic()
        with self._lock:
            previous = self._numbers.get(spec.id)
            previous_at = self._number_at.get(spec.id)
            if previous is not None and previous_at and now > previous_at:
                # Rate of change, published as `<id>__rate`. This is what makes descent
                # speed, income per second and progress bars usable in a guard without
                # the playbook having to difference them by hand.
                self._rates[spec.id] = (value - previous) / (now - previous_at)
            self._numbers[spec.id] = value
            self._number_at[spec.id] = now

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
        if spec.type == "number":
            return self._number(spec, frame)
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
        patch = _subsample(self._patch(spec.region, frame))
        if patch.size == 0:
            return 0.0
        return self._colour_match(
            patch.reshape(-1, 3).mean(axis=0, dtype=np.float32), spec
        )

    def _brightness(self, spec: ProbeSpec, frame: Frame) -> float:
        assert spec.region is not None
        patch = _subsample(self._patch(spec.region, frame))
        if patch.size == 0:
            return 0.0
        return float(_luma(patch).mean() / 255.0)

    def _region_diff(self, spec: ProbeSpec, frame: Frame) -> float:
        assert spec.region is not None
        patch = _subsample(self._patch(spec.region, frame))
        if patch.size == 0:
            return 0.0
        luma = _luma(patch)
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


def _thumb_delta(thumb: np.ndarray, baseline: np.ndarray | None) -> float:
    """Fraction of the thumbnail that differs from a baseline.

    A missing or differently shaped baseline counts as fully changed, so the first frame
    -- and the frame after a resolution change -- always triggers perception rather than
    being quietly treated as "nothing moved".
    """
    if baseline is None or baseline.shape != thumb.shape:
        return 1.0
    return float((np.abs(thumb - baseline) > _PIXEL_CHANGE_THRESHOLD).mean())


_NUMBER_RE = re.compile(r"-?\d[\d,.]*")


_OCR_STATUS: dict[str, str] = {}


def ocr_available() -> tuple[bool, str]:
    """Whether OCR can actually read anything, and why not if it cannot.

    Checking only for the tesseract binary is not enough: the language data is a separate
    package on most distributions, and without it tesseract exits with an error for every
    call. A number probe would then return 0 forever, which a playbook reads as a real
    measurement -- a guard on `probe('health') < 0.25` would fire permanently.
    """
    import shutil
    import subprocess

    if cached := _OCR_STATUS.get("state"):
        return cached == "ok", _OCR_STATUS.get("detail", "")

    if shutil.which("tesseract") is None:
        _OCR_STATUS.update(
            state="missing",
            detail="tesseract is not installed; number probes will not work. "
                   "Arch: sudo pacman -S tesseract tesseract-data-eng  |  "
                   "Debian/Ubuntu: sudo apt install tesseract-ocr",
        )
        return False, _OCR_STATUS["detail"]

    try:
        proc = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, timeout=5.0, check=False
        )
        langs = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        _OCR_STATUS.update(state="broken", detail=f"tesseract failed to run: {exc}")
        return False, _OCR_STATUS["detail"]

    if "eng" not in langs.split():
        _OCR_STATUS.update(
            state="nolang",
            detail="tesseract is installed but has no English language data, so every "
                   "read fails. Arch: sudo pacman -S tesseract-data-eng  |  "
                   "Debian/Ubuntu: sudo apt install tesseract-ocr-eng",
        )
        return False, _OCR_STATUS["detail"]

    _OCR_STATUS.update(state="ok", detail="")
    return True, ""


def _ocr_number(patch: np.ndarray, *, invert: bool = False) -> float | None:
    """Extract the first number in a small image region via tesseract.

    Three preprocessing steps matter far more than the OCR settings do. Game HUD text is
    small, so it is upscaled 4x before tesseract sees it; it is usually light-on-dark,
    which tesseract handles badly, so it is inverted to dark-on-light; and it is
    thresholded, because anti-aliased glyphs over a moving 3D background confuse the
    segmenter more than low contrast does.
    """
    import subprocess
    import tempfile

    ready, _ = ocr_available()
    if not ready:
        return None

    grey = patch.mean(axis=2).astype(np.float32)
    if not invert:
        grey = 255.0 - grey                      # light text -> dark text
    threshold = grey.mean() * 0.85
    binary = np.where(grey < threshold, 0, 255).astype(np.uint8)
    binary = np.repeat(np.repeat(binary, 4, axis=0), 4, axis=1)

    try:
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            Image.fromarray(binary, mode="L").save(handle, format="PNG")
            path = handle.name
        proc = subprocess.run(
            ["tesseract", path, "stdout", "--psm", "7", "-c",
             "tessedit_char_whitelist=0123456789.,-"],
            capture_output=True, timeout=3.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            import os

            os.unlink(path)
        except (OSError, NameError, UnboundLocalError):
            pass

    match = _NUMBER_RE.search(proc.stdout.decode("utf-8", "replace"))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _thumbnail_luma(pixels: np.ndarray) -> np.ndarray:
    """Cheap fixed-size luma thumbnail via integer-factor strided subsampling."""
    h, w = pixels.shape[:2]
    step_y = max(1, h // _DELTA_H)
    step_x = max(1, w // _DELTA_W)
    return _luma(pixels[::step_y, ::step_x])


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
