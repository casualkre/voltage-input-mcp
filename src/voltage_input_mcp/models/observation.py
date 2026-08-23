"""What the vision layer reports, and how its coordinates get onto the screen.

The vision model's output is grammar-constrained to a compact JSON shape::

    {"s":"file manager, list view","e":[{"l":"address bar","b":[120,44,890,72],"c":0.9}],
     "t":["Documents"],"f":["dialog"]}

Short keys are not cosmetic: at ~15 tokens per element, key names are a meaningful
fraction of decode time, and decode is the actuator loop's critical path.

`l` (label) is restricted by the generated grammar to the current state's `watch` list
plus a small generic set. That is the single most important guardrail on the vision
model: it cannot invent an element name, so a downstream `sees("address bar")` guard is
comparing against a closed vocabulary rather than whatever prose the model felt like.

Coordinate spaces
-----------------
This is the classic footgun with VLM grounding. Depending on model and prompt, boxes
come back either normalised to 0-1000 or in the pixel space of the *resized* image the
vision tower actually saw -- which is neither the screen nor the file you sent. Getting
this wrong produces clicks that are subtly and consistently offset, which looks like a
model quality problem and is not.

`CoordinateMapper` handles all three cases explicitly, with `auto` inferring the space
from the magnitude of the observed boxes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Element", "Observation", "CoordinateMapper", "parse_vision_output"]

CoordSpace = Literal["norm1000", "image_px", "screen_px", "auto"]

# Labels the vision grammar always permits, on top of the state's `watch` list.
GENERIC_LABELS: tuple[str, ...] = (
    "dialog",
    "button",
    "text field",
    "menu",
    "list item",
    "window",
    "icon",
    "checkbox",
    "scrollbar",
    "loading indicator",
    "error message",
)

# Flags the vision model may raise. Kept tiny so the grammar enum stays cheap.
FLAGS: tuple[str, ...] = (
    "dialog",
    "loading",
    "error",
    "empty",
    "fullscreen",
    "occluded",
    "unchanged",
)


class Element(BaseModel):
    """A grounded UI element in **screen pixels**."""

    model_config = ConfigDict(frozen=True)

    label: str
    x: int
    y: int
    w: int
    h: int
    conf: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "box": [self.x, self.y, self.w, self.h],
            "center": list(self.center),
            "conf": round(self.conf, 3),
        }

    def render(self) -> str:
        """Compact form fed back into the actuator prompt."""
        cx, cy = self.center
        return f"{self.label}@{cx},{cy}[{self.w}x{self.h}]"


@dataclass(slots=True)
class Observation:
    """One perception result. Immutable in practice; the session keeps the latest."""

    scene: str = ""
    elements: list[Element] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    frame_id: int = 0
    ts: float = field(default_factory=time.monotonic)
    latency_ms: float = 0.0
    source: Literal["vlm", "cache", "probe", "manual", "empty"] = "empty"
    raw: str = ""

    def find(self, label: str, min_conf: float = 0.0) -> Element | None:
        """Highest-confidence element with this label, breaking ties by area."""
        candidates = [e for e in self.elements if e.label == label and e.conf >= min_conf]
        if not candidates:
            return None
        return max(candidates, key=lambda e: (e.conf, e.area))

    def at(self, px: int, py: int) -> Element | None:
        """Smallest element containing the point -- the most specific hit target."""
        hits = [e for e in self.elements if e.contains(px, py)]
        return min(hits, key=lambda e: e.area) if hits else None

    @property
    def labels(self) -> list[str]:
        return [e.label for e in self.elements]

    def render(self, max_elements: int = 8, max_text: int = 3) -> str:
        """The block injected into the actuator prompt. Kept under ~60 tokens."""
        lines = [f"SCENE: {self.scene or '(none)'}"]
        if self.elements:
            top = sorted(self.elements, key=lambda e: -e.conf)[:max_elements]
            lines.append("SEEN: " + "; ".join(e.render() for e in top))
        else:
            lines.append("SEEN: (nothing matched)")
        if self.texts:
            joined = " | ".join(t[:60] for t in self.texts[:max_text])
            lines.append(f"TEXT: {joined}")
        if self.flags:
            lines.append("FLAGS: " + ",".join(sorted(self.flags)))
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "elements": [e.as_dict() for e in self.elements],
            "texts": self.texts,
            "flags": sorted(self.flags),
            "frame_id": self.frame_id,
            "latency_ms": round(self.latency_ms, 1),
            "source": self.source,
        }

    def is_stale(self, max_age_s: float) -> bool:
        return (time.monotonic() - self.ts) > max_age_s


@dataclass(slots=True)
class CoordinateMapper:
    """Maps vision-model boxes into screen pixels.

    There are three distinct rectangles in play and conflating any two of them produces a
    constant click offset:

      capture_size  the downscaled image actually handed to the model (e.g. 896x504)
      region_size   the desktop region that image depicts, at full scale (e.g. 1920x1080)
      screen_size   the whole desktop, used only for clamping
      origin        top-left of the region within the desktop; non-zero when capturing a
                    secondary monitor or a cropped area

    For a full-screen capture, region_size == screen_size and origin == (0, 0).
    """

    capture_size: tuple[int, int]
    screen_size: tuple[int, int]
    region_size: tuple[int, int] | None = None
    origin: tuple[int, int] = (0, 0)
    space: CoordSpace = "auto"

    def __post_init__(self) -> None:
        if self.region_size is None:
            self.region_size = self.screen_size

    def _resolve(self, boxes: list[list[float]]) -> CoordSpace:
        """Infer the coordinate space from the magnitude of the boxes.

        The prompt asks for 0-1000 normalised coordinates, so that is the default. A
        value above 1000 cannot be normalised, which means the model ignored the
        instruction and emitted resized-image pixels -- a known Qwen-VL behaviour.
        """
        if self.space != "auto":
            return self.space
        if not boxes:
            return "norm1000"
        peak = max((max(abs(v) for v in b) for b in boxes), default=0.0)
        return "image_px" if peak > 1000.0 else "norm1000"

    def map_box(self, box: list[float], space: CoordSpace) -> tuple[int, int, int, int]:
        """Convert one [x1,y1,x2,y2] box to screen-space (x, y, w, h)."""
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        screen_w, screen_h = self.screen_size
        cap_w, cap_h = self.capture_size
        reg_w, reg_h = self.region_size
        ox, oy = self.origin

        if space == "norm1000":
            # Normalised -> captured-region pixels, directly. No need to detour through
            # the downscaled image's own pixel grid.
            sx, sy = reg_w / 1000.0, reg_h / 1000.0
            x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        elif space == "image_px":
            # Downscaled image pixels -> captured-region pixels.
            fx = reg_w / cap_w if cap_w else 1.0
            fy = reg_h / cap_h if cap_h else 1.0
            x1, y1, x2, y2 = x1 * fx, y1 * fy, x2 * fx, y2 * fy
        # "screen_px": already in desktop pixels, nothing to do.

        x, y = int(round(x1)) + ox, int(round(y1)) + oy
        w, h = int(round(x2 - x1)), int(round(y2 - y1))

        # Clamp into the desktop. A box clipped to zero area is dropped by the caller.
        x = max(0, min(x, screen_w - 1))
        y = max(0, min(y, screen_h - 1))
        w = max(0, min(w, screen_w - x))
        h = max(0, min(h, screen_h - y))
        return x, y, w, h

    def map_all(
        self,
        raw_elements: list[Any],
        vocabulary: Sequence[str] | None = None,
    ) -> list[Element]:
        """Map raw vision output to screen-space elements.

        Accepts two encodings:

          compact  ``[label_index, x1, y1, x2, y2]`` -- what the grammar emits. The index
                   refers to `vocabulary`, which the caller must build with
                   `llm.grammar.vision_vocabulary` so the ordering matches the grammar's.
          verbose  ``{"l": "address bar", "b": [x1, y1, x2, y2], "c": 0.9}`` -- what a
                   backend without grammar support (Ollama) produces from a JSON schema.

        The compact form exists because decode is the vision model's bottleneck at roughly
        22 ms per output token; it costs about 11 tokens per element against the verbose
        form's 20.
        """
        normalised: list[tuple[str, list[float], float]] = []
        for item in raw_elements:
            if isinstance(item, list) and len(item) >= 5:
                try:
                    index = int(item[0])
                    box = [float(v) for v in item[1:5]]
                except (TypeError, ValueError):
                    continue
                if not vocabulary or not 0 <= index < len(vocabulary):
                    continue
                normalised.append((vocabulary[index], box, 1.0))
            elif isinstance(item, dict):
                label = item.get("l") or item.get("label")
                box_raw = item.get("b") or item.get("box")
                if not label or not isinstance(box_raw, list) or len(box_raw) < 4:
                    continue
                try:
                    box = [float(v) for v in box_raw[:4]]
                except (TypeError, ValueError):
                    continue
                try:
                    conf = max(0.0, min(1.0, float(item.get("c", item.get("conf", 1.0)))))
                except (TypeError, ValueError):
                    conf = 1.0
                normalised.append((str(label), box, conf))

        space = self._resolve([box for _, box, _ in normalised])
        out: list[Element] = []
        for label, box, conf in normalised:
            x, y, w, h = self.map_box(box, space)
            if w <= 0 or h <= 0:
                continue
            out.append(Element(label=label, x=x, y=y, w=w, h=h, conf=conf))
        return out


def parse_vision_output(
    raw: str,
    mapper: CoordinateMapper,
    *,
    vocabulary: Sequence[str] | None = None,
    frame_id: int = 0,
    latency_ms: float = 0.0,
    max_elements: int = 8,
) -> Observation:
    """Parse grammar-constrained vision JSON into a screen-space `Observation`.

    The grammar guarantees well-formed JSON, so this does not need to be defensive about
    syntax -- but it stays defensive about *semantics* (missing keys, wrong types, junk
    boxes) because a grammar constrains shape, not sense.
    """
    text = (raw or "").strip()
    if not text:
        return Observation(source="empty", raw=raw, frame_id=frame_id, latency_ms=latency_ms)

    # Tolerate a model that wrapped the object in prose despite the grammar.
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return Observation(source="empty", raw=raw, frame_id=frame_id, latency_ms=latency_ms)
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Observation(source="empty", raw=raw, frame_id=frame_id, latency_ms=latency_ms)
    if not isinstance(data, dict):
        return Observation(source="empty", raw=raw, frame_id=frame_id, latency_ms=latency_ms)

    raw_elements = data.get("e") or data.get("elements") or []
    if not isinstance(raw_elements, list):
        raw_elements = []
    elements = mapper.map_all(raw_elements[: max_elements * 2], vocabulary)[:max_elements]

    texts = data.get("t") or data.get("texts") or []
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t)[:200] for t in texts if isinstance(t, (str, int, float))][:6]

    flags = data.get("f") or data.get("flags") or []
    if isinstance(flags, str):
        flags = [flags]
    flag_set = {str(f) for f in flags if str(f) in FLAGS}

    scene = str(data.get("s") or data.get("scene") or "")[:200]

    return Observation(
        scene=scene,
        elements=elements,
        texts=texts,
        flags=flag_set,
        frame_id=frame_id,
        latency_ms=latency_ms,
        source="vlm",
        raw=raw,
    )
