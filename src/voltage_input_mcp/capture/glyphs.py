"""Read HUD numbers by matching glyphs, not by running OCR.

Why not OCR
-----------
Tesseract is built to read documents: unknown fonts, unknown layout, arbitrary text, all
of it worth 80-200 ms a call. A game HUD is none of those things. It is *one font, at one
size, in one place, drawing one of about twelve possible glyphs*. Using a document OCR
engine for that is not merely slow, it is inaccurate in a way that matters: measured
against the real Broken Bones HUD, tesseract read `414` as `4114` and `0` as `636`. A
reflex guarded on `probe('meters') < 90` cannot survive its input being wrong by an order
of magnitude, and no amount of tuning the guard fixes a lying sensor.

So: learn the twelve glyphs once, then classify by normalised cross-correlation. That is
a few hundred microseconds instead of a couple hundred milliseconds, it is exact rather
than probabilistic, and it removes tesseract from the hot path entirely.

How the glyphs get learned
--------------------------
Bootstrapping is the only hard part, and it is done once per HUD region:

  1. Watch the region over some seconds while the number changes.
  2. Binarise each frame and segment it into glyphs by column-wise ink projection --
     digits in a HUD are separated by blank columns, which is exactly the assumption a
     proportional document font would break and a HUD font does not.
  3. Cluster the glyph bitmaps. A counter passing through a range of values will show
     every digit, and the clusters converge on the true glyph set.
  4. Label the clusters *once*, using OCR on the frames where it is most likely to be
     right: those whose glyph count matches the OCR'd digit count. Positional votes
     across many frames outvote any single misread.

After that, OCR is never called again for that region. The expensive, unreliable thing
runs during calibration, where being slow does not matter and being wrong is caught by
the voting.

What this deliberately does not handle
--------------------------------------
Anti-aliased text over a *moving* background, glyphs that touch, and proportional fonts
with kerning that closes the column gaps. HUD numerals are usually drawn over a solid or
strongly-contrasting panel and are usually monospaced, which is why this works at all.
`confidence` is reported per read so a caller can tell a clean match from a guess, and
the probe engine falls back rather than inventing a number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["GlyphSet", "GlyphLearner", "segment_glyphs", "binarise", "glyphs_path"]

# Every glyph is resampled to this height before matching, so a HUD that scales with
# resolution still classifies. Width is left free and carried as an aspect ratio, because
# a `1` is genuinely narrower than an `8` and that is a useful discriminator.
_NORM_H = 16
_NORM_W = 12
# Below this correlation a glyph is not confidently any known shape.
_MIN_SCORE = 0.72
# A column with less than this fraction of the region's peak ink counts as a gap.
_GAP_FRACTION = 0.06
_MIN_GLYPH_W = 2

_NUMBER_RE = re.compile(r"-?\d[\d,.]*")


def glyphs_path(name: str) -> Path:
    from ..config import state_dir

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64]
    return state_dir() / "glyphs" / f"{safe}.json"


def binarise(patch: np.ndarray, *, invert: bool = False) -> np.ndarray:
    """Ink mask for a HUD patch: True where the glyph is.

    Uses saturation *and* luma rather than either alone. HUD text is usually a saturated
    colour over a desaturated background -- green on snow, white on a dark panel -- and
    whichever of the two separates better varies by scene. Taking the stronger separation
    per patch means one code path covers both, instead of a per-probe flag that a caller
    has to get right by trial and error.
    """
    f = patch.astype(np.float32)
    luma = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    saturation = f.max(axis=2) - f.min(axis=2)

    # Try every plausible separation rather than deciding up front which one this HUD
    # needs. Both luma polarities are offered because "bright text on dark" and "dark
    # text on bright" both occur, often in the same game, and picking wrong does not
    # produce a bad mask -- it produces the *background* as the mask, which then segments
    # into one enormous blob instead of digits.
    candidates = []
    for channel in (saturation, luma, 255.0 - luma):
        span = float(channel.max() - channel.min())
        if span < 12.0:
            continue  # flat: separates nothing
        mask = channel > (channel.min() + span * 0.55)
        covered = float(mask.mean())
        if not 0.01 < covered < 0.6:
            continue
        # Score by how much glyph-like structure the mask actually has. Text is a row of
        # separated column runs; a background mask is one solid region. Counting the runs
        # is what tells them apart, and it is the thing we are about to rely on anyway.
        runs = len(segment_glyphs(mask))
        if runs == 0:
            continue
        candidates.append((runs, -covered, mask))

    if not candidates:
        return np.zeros(patch.shape[:2], dtype=bool)
    # Most separated runs wins; ties go to the sparser mask, since ink is sparser than
    # background.
    return max(candidates, key=lambda c: (c[0], c[1]))[2]


def segment_glyphs(mask: np.ndarray) -> list[np.ndarray]:
    """Split an ink mask into per-glyph bitmaps by column-wise projection."""
    if not mask.any():
        return []
    columns = mask.sum(axis=0)
    threshold = max(1.0, columns.max() * _GAP_FRACTION)
    inked = columns >= threshold

    spans: list[tuple[int, int]] = []
    start = None
    for x, on in enumerate(inked):
        if on and start is None:
            start = x
        elif not on and start is not None:
            spans.append((start, x))
            start = None
    if start is not None:
        spans.append((start, len(inked)))

    out = []
    for x0, x1 in spans:
        if x1 - x0 < _MIN_GLYPH_W:
            continue
        column = mask[:, x0:x1]
        rows = np.flatnonzero(column.any(axis=1))
        if rows.size == 0:
            continue
        out.append(column[rows[0]: rows[-1] + 1])
    return out


def _normalise(glyph: np.ndarray) -> np.ndarray:
    """Resample a glyph bitmap to a fixed box, as float32 in [0, 1]."""
    from PIL import Image

    img = Image.fromarray((glyph.astype(np.uint8) * 255), mode="L")
    img = img.resize((_NORM_W, _NORM_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised correlation of two same-shape glyph boxes."""
    x = a - a.mean()
    y = b - b.mean()
    denominator = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denominator <= 1e-9:
        return 0.0
    return float((x * y).sum() / denominator)


@dataclass(slots=True)
class GlyphSet:
    """Learned templates for one HUD region."""

    templates: dict[str, list[list[float]]]
    source: str = ""

    def _matrix(self) -> list[tuple[str, np.ndarray]]:
        return [
            (label, np.asarray(data, dtype=np.float32))
            for label, data in self.templates.items()
        ]

    def read_glyphs(
        self, patch: np.ndarray, *, invert: bool = False
    ) -> list[tuple[str, float]]:
        """Per-glyph (label, correlation), left to right. Empty if nothing segmented.

        Exposed separately because it is what makes a bad read diagnosable: a number that
        came back wrong is almost always one glyph scoring poorly, and the average hides
        exactly that.
        """
        glyphs = segment_glyphs(binarise(patch, invert=invert))
        known = self._matrix()
        if not glyphs or not known:
            return []
        out = []
        for glyph in glyphs:
            box = _normalise(glyph)
            out.append(
                max(
                    ((lab, _correlate(box, tpl)) for lab, tpl in known),
                    key=lambda pair: pair[1],
                )
            )
        return out

    def read(
        self, patch: np.ndarray, *, invert: bool = False
    ) -> tuple[float | None, float]:
        """Classify the digits in `patch`. Returns (value, confidence in [0, 1]).

        Confidence is the *weakest* glyph match, not the average. One misclassified digit
        ruins the whole number -- reading 414 as 814 is not 2/3 right, it is wrong -- so
        the number is only as trustworthy as its worst character.
        """
        scored = self.read_glyphs(patch, invert=invert)
        if not scored:
            return None, 0.0

        worst = min(score for _, score in scored)
        if worst < _MIN_SCORE:
            # Refuse rather than guess. A sensor that invents a number is worse than one
            # that admits it cannot see, because a guard cannot tell the two apart.
            return None, worst

        match = _NUMBER_RE.search("".join(label for label, _ in scored))
        if not match:
            return None, worst
        try:
            return float(match.group(0).replace(",", "")), worst
        except ValueError:
            return None, worst

    # -- persistence ------------------------------------------------------------------

    def save(self, name: str) -> Path:
        path = glyphs_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"source": self.source, "templates": self.templates}, indent=1
        ))
        return path

    @classmethod
    def load(cls, name: str) -> GlyphSet | None:
        path = glyphs_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            templates = data["templates"]
        except (OSError, ValueError, KeyError):
            return None
        if not isinstance(templates, dict) or not templates:
            return None
        return cls(templates=templates, source=str(data.get("source", "")))


class GlyphLearner:
    """Collects glyph shapes across frames and labels them once, using OCR as a teacher.

    The teacher is only consulted where it is most likely to be right -- frames whose
    glyph count matches the digit count it returned -- and its answers are voted on
    across many frames. A single misread cannot decide a label, which matters because the
    whole point is that the teacher is unreliable.
    """

    def __init__(self, *, invert: bool = False) -> None:
        self.invert = invert
        self._clusters: list[np.ndarray] = []
        self._votes: list[dict[str, int]] = []
        self.frames = 0
        self.taught = 0

    def observe(self, patch: np.ndarray, *, teacher_text: str | None) -> None:
        """Add one frame. `teacher_text` is OCR's reading of it, if any."""
        self.frames += 1
        glyphs = segment_glyphs(binarise(patch, invert=self.invert))
        if not glyphs:
            return

        indices = [self._assign(_normalise(g)) for g in glyphs]

        digits = re.sub(r"[^0-9]", "", teacher_text or "")
        # Only trust the teacher when its digit count matches what segmentation found;
        # otherwise the positional alignment between the two is meaningless.
        if not digits or len(digits) != len(indices):
            return
        self.taught += 1
        for index, digit in zip(indices, digits, strict=True):
            self._votes[index][digit] = self._votes[index].get(digit, 0) + 1

    def _assign(self, box: np.ndarray) -> int:
        for i, centroid in enumerate(self._clusters):
            if _correlate(box, centroid) >= 0.88:
                # Running mean, so the template converges on the glyph's typical
                # rendering rather than whichever frame happened to arrive first.
                n = sum(self._votes[i].values()) + 1
                self._clusters[i] = centroid + (box - centroid) / max(2, n)
                return i
        self._clusters.append(box)
        self._votes.append({})
        return len(self._clusters) - 1

    def result(self, *, min_votes: int = 3) -> GlyphSet | None:
        """Build a GlyphSet from clusters that got a clear, well-supported label."""
        templates: dict[str, list[list[float]]] = {}
        best: dict[str, tuple[int, np.ndarray]] = {}
        for centroid, votes in zip(self._clusters, self._votes, strict=True):
            if not votes:
                continue
            label, count = max(votes.items(), key=lambda kv: kv[1])
            total = sum(votes.values())
            # A cluster whose votes are split is a segmentation failure, not a glyph.
            if count < min_votes or count / total < 0.7:
                continue
            if label not in best or count > best[label][0]:
                best[label] = (count, centroid)
        for label, (_, centroid) in best.items():
            templates[label] = centroid.tolist()
        if len(templates) < 2:
            return None
        return GlyphSet(templates=templates, source=f"learned from {self.frames} frames")

    def report(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "frames_with_teacher": self.taught,
            "clusters": len(self._clusters),
            "labelled": sum(1 for v in self._votes if v),
        }
