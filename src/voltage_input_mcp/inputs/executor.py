"""Burst execution: turning a parsed programme into precisely timed kernel events.

This is the layer that makes the whole two-tier idea pay off. A decision costs 150-600 ms
of model time; a burst of 40 inputs costs whatever the burst says it costs, scheduled to
about a millisecond, with no model in the loop. Effective input rate is set by the burst,
not by the model.

Timing
------
Actions are scheduled against a single monotonic cursor rather than by sleeping for each
action's duration in turn. Sleeping accumulates drift -- `time.sleep(0.008)` typically
returns after 8.5-9.5 ms, so a 40-action burst of nominal 320 ms would land at 380 ms and
the error compounds across cycles. Anchoring every action to `t0 + offset` keeps total
burst time within a few ms of nominal.

`_wait_until` is a hybrid: it blocks on the abort Event while there is comfortable time
left (cheap, and makes panic-stop responsive within a millisecond), then busy-spins for
the last ~1.5 ms where the scheduler cannot be trusted. The spin is bounded and only
happens at the tail of each wait, so CPU cost is negligible.

Text
----
uinput sends scancodes, so typing via keystrokes produces whatever the compositor's
active XKB layout maps those scancodes to. On a non-US layout, punctuation comes out
wrong -- silently, which is the worst kind of wrong. `TextMode.AUTO` therefore routes
anything outside the plain-ASCII-alphanumeric range through the clipboard instead, which
is layout-independent, unicode-safe, and O(1) rather than O(len(text)) in wall time.
Clipboard contents are saved and restored around the paste.
"""

from __future__ import annotations

import enum
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from ..errors import InputDeviceError
from ..models.burst import (
    Burst,
    ButtonDown,
    ButtonUp,
    Click,
    KeyChord,
    KeyDown,
    KeyUp,
    MoveAbs,
    MoveRel,
    Scroll,
    TypeText,
    Wait,
)
from . import keymap as km
from .uinput import DeviceSet

__all__ = ["Executor", "ExecutionReport", "TextMode", "PointerMode"]

# Below this, a blocking wait is unreliable and we spin instead.
_SPIN_THRESHOLD_S: Final = 0.0015
# Longest single blocking wait, so abort latency stays bounded even inside a long w:.
_WAIT_SLICE_S: Final = 0.02


class TextMode(enum.StrEnum):
    KEYS = "keys"
    CLIPBOARD = "clipboard"
    AUTO = "auto"


class PointerMode(enum.StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass(slots=True)
class ExecutionReport:
    ok: bool = True
    executed: int = 0
    inputs: int = 0
    duration_ms: float = 0.0
    aborted: bool = False
    dry_run: bool = False
    error: str | None = None
    held_keys: list[str] = field(default_factory=list)
    held_buttons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "executed": self.executed,
            "inputs": self.inputs,
            "duration_ms": round(self.duration_ms, 2),
            "aborted": self.aborted,
            "dry_run": self.dry_run,
            "error": self.error,
            "held": {"keys": self.held_keys, "buttons": self.held_buttons},
            "notes": self.notes,
        }


class Executor:
    """Owns the virtual devices and runs bursts on the calling thread.

    The session loop calls `run()` via `asyncio.to_thread` so the event loop stays
    responsive while a burst is in flight -- status polls and stop requests are served
    during execution rather than after it.

    A single `threading.Lock` serialises execution: two overlapping bursts would
    interleave key presses and produce garbage. Reflex bursts contend for the same lock,
    which is correct -- a reflex that fires mid-burst waits for it to finish rather than
    corrupting it.
    """

    def __init__(
        self,
        devices: DeviceSet,
        *,
        dry_run: bool = True,
        text_mode: TextMode = TextMode.AUTO,
        pointer_mode: PointerMode = PointerMode.ABSOLUTE,
        max_hold_ms: int = 4000,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.devices = devices
        self.dry_run = dry_run
        self.text_mode = text_mode
        self.pointer_mode = pointer_mode
        self.max_hold_ms = max_hold_ms
        self._on_event = on_event

        self._lock = threading.Lock()
        self._abort = threading.Event()
        # name -> monotonic time it went down, for the hold watchdog and panic release.
        self._held_keys: dict[str, float] = {}
        self._held_buttons: dict[str, float] = {}
        self._cursor: tuple[int, int] = (0, 0)
        self._clipboard_tool = _detect_clipboard_tool()

    # -- control ---------------------------------------------------------------------

    @property
    def abort_event(self) -> threading.Event:
        return self._abort

    def request_abort(self) -> None:
        """Interrupt any burst in flight. Safe to call from any thread."""
        self._abort.set()

    def clear_abort(self) -> None:
        self._abort.clear()

    @property
    def cursor(self) -> tuple[int, int]:
        """Last position we commanded. Not the real cursor -- the user may have moved it."""
        return self._cursor

    def held(self) -> tuple[list[str], list[str]]:
        return sorted(self._held_keys), sorted(self._held_buttons)

    def release_all(self) -> list[str]:
        """Release every key and button we are holding.

        Called on abort, on session end, and by the hold watchdog. This is the single
        most important cleanup path in the project: a burst interrupted between `d:shift`
        and `u:shift` leaves the compositor believing Shift is physically down, and the
        user's keyboard then behaves bizarrely until they press and release it manually.
        """
        released: list[str] = []
        if self.dry_run:
            self._held_keys.clear()
            self._held_buttons.clear()
            return released
        try:
            events: list[tuple[int, int, int]] = []
            for name in list(self._held_keys):
                code = km.resolve_key(name)
                if code is not None:
                    events.append((km.EV_KEY, code, 0))
                    released.append(name)
            if events:
                self.devices.keyboard.emit(events)
            self._held_keys.clear()

            btn_events: list[tuple[int, int, int]] = []
            for name in list(self._held_buttons):
                code = km.BUTTONS.get(name)
                if code is not None:
                    btn_events.append((km.EV_KEY, code, 0))
                    released.append(f"btn:{name}")
            if btn_events:
                self.devices.pointer_abs.emit(btn_events)
            self._held_buttons.clear()
        except InputDeviceError:
            # Best-effort: if the device is already gone there is nothing to release.
            self._held_keys.clear()
            self._held_buttons.clear()
        return released

    def enforce_hold_watchdog(self) -> list[str]:
        """Release anything held longer than `max_hold_ms`. Called once per cycle."""
        if self.max_hold_ms <= 0:
            return []
        now = time.monotonic()
        limit = self.max_hold_ms / 1000.0
        stale = [k for k, t in self._held_keys.items() if now - t > limit]
        stale_btn = [b for b, t in self._held_buttons.items() if now - t > limit]
        if not stale and not stale_btn:
            return []
        released: list[str] = []
        for name in stale:
            if self._key_event(name, 0):
                released.append(name)
            self._held_keys.pop(name, None)
        for name in stale_btn:
            if self._button_event(name, 0):
                released.append(f"btn:{name}")
            self._held_buttons.pop(name, None)
        self._emit("hold_watchdog", {"released": released})
        return released

    # -- execution -------------------------------------------------------------------

    def run(self, burst: Burst, *, label: str = "") -> ExecutionReport:
        report = ExecutionReport(dry_run=self.dry_run)
        if not burst.actions:
            return report

        with self._lock:
            if self._abort.is_set():
                report.ok = False
                report.aborted = True
                report.error = "aborted before start"
                return report

            if not self.dry_run and not self.devices.is_open:
                try:
                    self.devices.open()
                except InputDeviceError as exc:
                    report.ok = False
                    report.error = str(exc)
                    return report

            t0 = time.perf_counter()
            cursor = t0
            try:
                for action in burst.actions:
                    if self._abort.is_set():
                        report.aborted = True
                        report.ok = False
                        report.error = "aborted mid-burst"
                        break
                    cursor = self._dispatch(action, cursor)
                    report.executed += 1
                    if not isinstance(action, Wait):
                        report.inputs += 1
                else:
                    # Settle on the final action's own duration before returning, so the
                    # caller's notion of "burst finished" matches reality.
                    self._wait_until(cursor)
            except InputDeviceError as exc:
                report.ok = False
                report.error = str(exc)
            except Exception as exc:  # noqa: BLE001 - never let the device leak held keys
                report.ok = False
                report.error = f"{type(exc).__name__}: {exc}"

            report.duration_ms = (time.perf_counter() - t0) * 1000.0

            if report.aborted or not report.ok:
                released = self.release_all()
                if released:
                    report.notes.append(f"released on failure: {', '.join(released)}")

            keys, buttons = self.held()
            report.held_keys, report.held_buttons = keys, buttons

        self._emit("burst", {"label": label, "source": burst.source, **report.as_dict()})
        return report

    def _dispatch(self, action: object, cursor: float) -> float:
        if isinstance(action, Wait):
            return cursor + action.ms / 1000.0

        if isinstance(action, KeyChord):
            return self._do_chord(action, cursor)
        if isinstance(action, KeyDown):
            self._wait_until(cursor)
            self._key_event(action.key, 1)
            self._held_keys[km.canonical_key(action.key)] = time.monotonic()
            return cursor
        if isinstance(action, KeyUp):
            self._wait_until(cursor)
            self._key_event(action.key, 0)
            self._held_keys.pop(km.canonical_key(action.key), None)
            return cursor
        if isinstance(action, TypeText):
            return self._do_text(action, cursor)
        if isinstance(action, MoveAbs):
            self._wait_until(cursor)
            self._do_move_abs(action.x, action.y)
            return cursor
        if isinstance(action, MoveRel):
            self._wait_until(cursor)
            self._do_move_rel(action.dx, action.dy)
            return cursor
        if isinstance(action, Click):
            return self._do_click(action, cursor)
        if isinstance(action, ButtonDown):
            self._wait_until(cursor)
            self._button_event(action.button, 1)
            self._held_buttons[action.button] = time.monotonic()
            return cursor
        if isinstance(action, ButtonUp):
            self._wait_until(cursor)
            self._button_event(action.button, 0)
            self._held_buttons.pop(action.button, None)
            return cursor
        if isinstance(action, Scroll):
            return self._do_scroll(action, cursor)

        raise InputDeviceError(f"no handler for action {type(action).__name__}")

    # -- action handlers -------------------------------------------------------------

    def _do_chord(self, action: KeyChord, cursor: float) -> float:
        codes = []
        for name in action.keys:
            code = km.resolve_key(name)
            if code is None:
                raise InputDeviceError(f"unknown key {name!r}")
            codes.append(code)

        self._wait_until(cursor)
        if not self.dry_run:
            # Press in order (modifiers first, as written), one atomic frame.
            self.devices.keyboard.emit([(km.EV_KEY, c, 1) for c in codes])
        hold_end = cursor + action.hold_ms / 1000.0
        self._wait_until(hold_end)
        if not self.dry_run:
            # Release in reverse, so modifiers outlive the key they modified.
            self.devices.keyboard.emit([(km.EV_KEY, c, 0) for c in reversed(codes)])
        return hold_end

    def _do_text(self, action: TypeText, cursor: float) -> float:
        mode = self.text_mode
        if mode is TextMode.AUTO:
            mode = (
                TextMode.KEYS
                if km.is_typeable(action.text) and len(action.text) <= 48
                else TextMode.CLIPBOARD
            )

        if mode is TextMode.CLIPBOARD:
            self._wait_until(cursor)
            if self._paste(action.text):
                # Paste is one keystroke plus compositor round-trip; give it a beat.
                return cursor + 0.06
            # Fall through to keystrokes if no clipboard tool is available.

        t = cursor
        step = action.interval_ms / 1000.0
        shift_code = km.KEYS["leftshift"]
        for ch in action.text:
            mapped = km.ascii_to_keys(ch)
            if mapped is None:
                # Unreachable via AUTO, but a playbook can force KEYS mode.
                continue
            code, needs_shift = mapped
            self._wait_until(t)
            if not self.dry_run:
                events: list[tuple[int, int, int]] = []
                if needs_shift:
                    events.append((km.EV_KEY, shift_code, 1))
                events.append((km.EV_KEY, code, 1))
                self.devices.keyboard.emit(events)
                release: list[tuple[int, int, int]] = [(km.EV_KEY, code, 0)]
                if needs_shift:
                    release.append((km.EV_KEY, shift_code, 0))
                self.devices.keyboard.emit(release)
            t += step
        return t

    def _do_click(self, action: Click, cursor: float) -> float:
        t = cursor
        press = action.press_ms / 1000.0
        gap = action.gap_ms / 1000.0
        for i in range(action.count):
            self._wait_until(t)
            self._button_event(action.button, 1)
            t += press
            self._wait_until(t)
            self._button_event(action.button, 0)
            if i < action.count - 1:
                t += gap
        return t

    def _do_move_abs(self, x: int, y: int) -> None:
        self._cursor = (x, y)
        if self.dry_run:
            return
        if self.pointer_mode is PointerMode.RELATIVE:
            # No absolute channel available: approximate by homing to the top-left with
            # a deliberately over-large delta, then stepping out to the target. Pointer
            # acceleration does not apply to the clamp at 0,0, so this is exact.
            w, h = self.devices.screen
            self.devices.pointer_rel.emit(
                [(km.EV_REL, km.REL_X, -w * 2), (km.EV_REL, km.REL_Y, -h * 2)]
            )
            if x or y:
                self.devices.pointer_rel.emit(
                    [(km.EV_REL, km.REL_X, x), (km.EV_REL, km.REL_Y, y)]
                )
            return
        ax, ay = self.devices.to_abs(x, y)
        self.devices.pointer_abs.emit([(km.EV_ABS, km.ABS_X, ax), (km.EV_ABS, km.ABS_Y, ay)])

    def _do_move_rel(self, dx: int, dy: int) -> None:
        cx, cy = self._cursor
        w, h = self.devices.screen
        self._cursor = (max(0, min(cx + dx, w - 1)), max(0, min(cy + dy, h - 1)))
        if self.dry_run:
            return
        events: list[tuple[int, int, int]] = []
        if dx:
            events.append((km.EV_REL, km.REL_X, dx))
        if dy:
            events.append((km.EV_REL, km.REL_Y, dy))
        if events:
            self.devices.pointer_rel.emit(events)

    def _do_scroll(self, action: Scroll, cursor: float) -> float:
        t = cursor
        step = action.step_ms / 1000.0
        direction = 1 if action.amount > 0 else -1
        lo_axis = km.REL_WHEEL if action.axis == "v" else km.REL_HWHEEL
        hi_axis = km.REL_WHEEL_HI_RES if action.axis == "v" else km.REL_HWHEEL_HI_RES
        for _ in range(abs(action.amount)):
            self._wait_until(t)
            if not self.dry_run:
                # Emit both resolutions in one frame. Modern toolkits read HI_RES (120
                # units per detent) for smooth scrolling; older ones only see REL_WHEEL.
                # Sending only one of the two makes scrolling work in some apps and not
                # others, which is a miserable bug to chase.
                self.devices.pointer_abs.emit(
                    [
                        (km.EV_REL, hi_axis, 120 * direction),
                        (km.EV_REL, lo_axis, direction),
                    ]
                )
            t += step
        return t

    # -- primitives ------------------------------------------------------------------

    def _key_event(self, name: str, value: int) -> bool:
        code = km.resolve_key(name)
        if code is None:
            raise InputDeviceError(f"unknown key {name!r}")
        if not self.dry_run:
            self.devices.keyboard.emit([(km.EV_KEY, code, value)])
        return True

    def _button_event(self, name: str, value: int) -> bool:
        code = km.BUTTONS.get(name)
        if code is None:
            raise InputDeviceError(f"unknown button {name!r}")
        if not self.dry_run:
            self.devices.pointer_abs.emit([(km.EV_KEY, code, value)])
        return True

    def _wait_until(self, deadline: float) -> None:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            if self._abort.is_set():
                return
            if remaining > _SPIN_THRESHOLD_S:
                # Event.wait doubles as a sleep and a responsive abort check.
                self._abort.wait(min(remaining - _SPIN_THRESHOLD_S, _WAIT_SLICE_S))
                continue
            while time.perf_counter() < deadline:
                pass
            return

    # -- clipboard -------------------------------------------------------------------

    def _paste(self, text: str) -> bool:
        tool = self._clipboard_tool
        if not tool:
            return False
        previous = _clipboard_read(tool)
        if not _clipboard_write(tool, text):
            return False
        try:
            if not self.dry_run:
                ctrl, v = km.KEYS["leftctrl"], km.KEYS["v"]
                self.devices.keyboard.emit([(km.EV_KEY, ctrl, 1), (km.EV_KEY, v, 1)])
                time.sleep(0.012)
                self.devices.keyboard.emit([(km.EV_KEY, v, 0), (km.EV_KEY, ctrl, 0)])
                # The target app reads the clipboard asynchronously; restoring it too
                # early hands the app the *old* contents.
                time.sleep(0.05)
        finally:
            if previous is not None:
                _clipboard_write(tool, previous)
        return True

    def _emit(self, kind: str, payload: dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, payload)
            except Exception:  # noqa: BLE001 - telemetry must never break execution
                pass


# -- clipboard helpers -------------------------------------------------------------------


def _detect_clipboard_tool() -> str | None:
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return "wl"
    if os.environ.get("DISPLAY") and shutil.which("xclip"):
        return "xclip"
    if shutil.which("wl-copy"):
        return "wl"
    if shutil.which("xclip"):
        return "xclip"
    return None


def _clipboard_read(tool: str) -> str | None:
    cmd = ["wl-paste", "--no-newline"] if tool == "wl" else ["xclip", "-o", "-selection", "clipboard"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=1.5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return ""  # empty clipboard reports non-zero on wl-paste
    return out.stdout.decode("utf-8", errors="replace")


def _clipboard_write(tool: str, text: str) -> bool:
    # Text always goes in over stdin, never as an argv element -- otherwise text starting
    # with '-' is parsed as a flag, and long text can blow the argv limit.
    if tool == "wl":
        cmd = ["wl-copy"] if text else ["wl-copy", "--clear"]
    else:
        cmd = ["xclip", "-i", "-selection", "clipboard"]
    try:
        proc = subprocess.run(
            cmd, input=text.encode("utf-8"), capture_output=True, timeout=2.0, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0
