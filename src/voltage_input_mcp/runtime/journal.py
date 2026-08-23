"""Run journal: the record the orchestrator reads back.

The orchestrator is not watching the screen. When a run ends badly its only evidence is
this file, so the journal is written for *diagnosis*, not for logging hygiene. Every
cycle records what was seen, what was decided, what the governor said, and what actually
happened -- enough to answer "why did it click there" without re-running anything.

Two surfaces:

  * A JSONL file, append-only, one object per event. Survives the process.
  * An in-memory ring buffer that `voltage.status` reads, so polling a live run costs
    nothing and never touches the disk.

Frames are stored separately as PNGs and only when `keep_frames` is on, because a
2 Hz run for three minutes is 360 screenshots and nobody wants that by default. When it
is on, the frame filename is recorded on the cycle so a post-mortem can line up an image
with the decision it produced.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Journal", "CycleRecord", "default_journal_dir"]


def default_journal_dir() -> Path:
    from ..config import state_dir

    return state_dir() / "runs"


@dataclass(slots=True)
class CycleRecord:
    """One pass of the loop."""

    cycle: int = 0
    t: float = 0.0
    state: str = ""
    scene: str = ""
    elements: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    probes: dict[str, float] = field(default_factory=dict)
    perception: str = ""          # "vlm" | "cache" | "skipped"
    burst: str = ""
    burst_source: str = ""        # "actuator" | "reflex:<id>" | "on_enter" | "manual"
    proposed_state: str | None = None
    transition: str | None = None
    allowed: bool = True
    violations: list[dict[str, Any]] = field(default_factory=list)
    executed: bool = False
    note: str = ""
    frame: str | None = None
    timing: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "cycle", **asdict(self)}

    def summary(self) -> str:
        """One line, for a human skimming `voltage journal`."""
        mark = "x" if not self.allowed else ("!" if self.error else " ")
        arrow = f" -> {self.transition}" if self.transition else ""
        return (
            f"[{self.cycle:>4}]{mark} {self.state:<18} "
            f"{self.perception:<7} {self.burst[:52]:<52}{arrow}"
        )


class Journal:
    def __init__(
        self,
        run_id: str,
        *,
        directory: Path | None = None,
        keep_frames: bool = False,
        ring_size: int = 400,
    ) -> None:
        self.run_id = run_id
        self.directory = (directory or default_journal_dir()) / run_id
        self.keep_frames = keep_frames
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._file = None
        self._frame_dir = self.directory / "frames"
        self._counts: dict[str, int] = {}
        self._started = time.time()
        self._open()

    def _open(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.keep_frames:
                self._frame_dir.mkdir(parents=True, exist_ok=True)
            self._file = (self.directory / "journal.jsonl").open("a", encoding="utf-8")
        except OSError:
            # A journal that cannot be written must not stop a run; the ring buffer
            # still serves `voltage.status`.
            self._file = None

    # -- writing ---------------------------------------------------------------------

    def event(self, kind: str, **payload: Any) -> None:
        record = {"kind": kind, "t": round(time.time() - self._started, 3), **payload}
        self._append(record)

    def cycle(self, record: CycleRecord) -> None:
        self._counts[record.burst_source or "none"] = (
            self._counts.get(record.burst_source or "none", 0) + 1
        )
        self._append(record.as_dict())

    def _append(self, record: dict[str, Any]) -> None:
        self._ring.append(record)
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            self._file.flush()
        except (OSError, TypeError, ValueError):
            pass

    def save_frame(self, cycle: int, png: bytes) -> str | None:
        if not self.keep_frames:
            return None
        name = f"{cycle:06d}.png"
        try:
            (self._frame_dir / name).write_bytes(png)
        except OSError:
            return None
        return name

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    # -- reading ---------------------------------------------------------------------

    def tail(self, limit: int = 20, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(kinds) if kinds else None
        items = [r for r in self._ring if wanted is None or r.get("kind") in wanted]
        return items[-limit:]

    def cycles(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.tail(limit, kinds={"cycle"})

    @property
    def path(self) -> Path:
        return self.directory / "journal.jsonl"

    def stats(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "records": len(self._ring),
            "by_source": dict(self._counts),
            "frames": self.keep_frames,
        }
