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

Platforms
---------
The executor talks to an `InputSink` (see sink.py) and does not know which platform it is
on. `DeviceSet` implements it over uinput on Linux; `Win32Sink` implements it over
SendInput on Windows. Everything above -- burst scheduling, timing, held-key tracking,
abort handling -- is shared.

Drags and held modifiers
------------------------
Held state composes: `d:shift;g:0;p:l;g:1;e:l;u:shift` holds Shift across a button press,
a move and a release, which is what shift-drag means everywhere. Modifiers are released
in reverse order so they outlive the thing they modify.

Motion while a button is held is interpolated -- see `_move_to`. This is not cosmetic:
applications infer "a drag is happening" from the motion itself, so a teleport reads as
two unrelated clicks.

Text
----
uinput sends scancodes, so typing via keystrokes produces whatever the compositor's
active XKB layout maps those scancodes to. On a non-US layout, punctuation comes out
wrong -- silently, which is the worst kind of wrong. `TextMode.AUTO` therefore routes
anything outside the plain-ASCII-alphanumeric range through the clipboard instead, which
is layout-independent, unicode-safe, and O(1) rather than O(len(text)) in wall time.
Clipboard contents are saved and restored around the paste.

Windows needs none of that: SendInput's `KEYEVENTF_UNICODE` delivers a UTF-16 code unit
directly with no layout involved, so the sink exposes a native `text()` and the executor
prefers it when present.
"""

from __future__ import annotations

import enum
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
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
from .sink import InputSink

# Reverse keymap, so the US-layout ASCII table (which yields scancodes) can be expressed
# as canonical key names for the platform-neutral sink.
_KEYNAME_BY_CODE = {code: name for name, code in km.KEYS.items()}

__all__ = ["Executor", "ExecutionReport", "TextMode", "PointerMode"]

# Below this, a blocking wait is unreliable and we spin instead.
_SPIN_THRESHOLD_S: Final = 0.0015
# Longest single blocking wait, so abort latency stays bounded even inside a long w:.
_WAIT_SLICE_S: Final = 0.02

# Drag interpolation. A pointer that teleports while a button is held is not a drag as
# far as most applications are concerned -- they watch for motion to cross a threshold
# and to size a selection. These only apply when something is actually held.
_DRAG_MIN_STEPS: Final = 6
_DRAG_MAX_STEPS: Final = 24
_DRAG_STEP_MS: Final = 8


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
        devices: InputSink,
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
        self._supervised: set[str] = set()
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
        self._supervised.clear()
        if self.dry_run:
            self._held_keys.clear()
            self._held_buttons.clear()
            return released
        try:
            for name in list(self._held_keys):
                if km.resolve_key(name) is not None:
                    self.devices.key(name, False)
                    released.append(name)
            self._held_keys.clear()

            for name in list(self._held_buttons):
                if name in km.BUTTONS:
                    self.devices.button(name, False)
                    released.append(f"btn:{name}")
            self._held_buttons.clear()
        except InputDeviceError:
            # Best-effort: if the device is already gone there is nothing to release.
            self._held_keys.clear()
            self._held_buttons.clear()
        return released

    def supervise(self, names: Iterable[str]) -> None:
        """Exempt these from the hold watchdog while something is actively driving them.

        The watchdog exists for input nobody is watching -- a burst that ended with `d:`
        and no matching `u:`, or one interrupted between the two. A latched hold reflex is
        the opposite case: its key is re-evaluated twenty times a second and released on
        the guard going false, on leaving the state, on pause and on stop. Letting a 4 s
        timer pull it out from under a guard that still wants it down would break exactly
        the case latches exist for, and silently -- the latch would still believe it was
        engaged, so it would never press again.
        """
        self._supervised.update(names)

    def unsupervise(self, names: Iterable[str]) -> None:
        self._supervised.difference_update(names)

    def enforce_hold_watchdog(self) -> list[str]:
        """Release anything held longer than `max_hold_ms`. Called once per cycle."""
        if self.max_hold_ms <= 0:
            return []
        now = time.monotonic()
        limit = self.max_hold_ms / 1000.0
        stale = [
            k for k, t in self._held_keys.items()
            if now - t > limit and k not in self._supervised
        ]
        stale_btn = [
            b for b, t in self._held_buttons.items()
            if now - t > limit and f"btn:{b}" not in self._supervised
        ]
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

    def run(
        self, burst: Burst, *, label: str = "", wait_s: float | None = None
    ) -> ExecutionReport:
        """Run a burst, optionally giving up if the device is busy.

        `wait_s` bounds how long to wait for the execution lock. It exists for reflexes.
        A reflex that fires while a 500 ms actuator burst is in flight would otherwise
        queue behind it and land half a second after the condition that triggered it --
        and a reaction delivered that late is not merely useless, it is a reaction to a
        situation that has already resolved, applied to whatever is happening now. Timing
        out and journalling the miss is the correct outcome; the next tick is 50 ms away.

        Releases must never pass a timeout. Letting go is the safe direction, and a
        dropped release is a key left down on the user's desktop.
        """
        report = ExecutionReport(dry_run=self.dry_run)
        if not burst.actions:
            return report

        if not self._lock.acquire(timeout=wait_s if wait_s is not None else -1):
            report.ok = False
            report.error = "input busy: another burst was still running"
            self._emit("busy", {"label": label, "source": burst.source, "waited_s": wait_s})
            return report

        try:
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
        finally:
            self._lock.release()

        self._emit("burst", {"label": label, "source": burst.source, **report.as_dict()})
        return report

    def held_names(self) -> set[str]:
        """Everything down right now, as guard-visible names (`w`, `btn:l`).

        Feeds `held()` in a guard expression, which is how a rule avoids re-pressing what
        a latch already holds.
        """
        return set(self._held_keys) | {f"btn:{b}" for b in self._held_buttons}

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
            return self._move_to(action.x, action.y, cursor)
        if isinstance(action, MoveRel):
            cx, cy = self._cursor
            w, h = self.devices.screen
            return self._move_to(
                max(0, min(cx + action.dx, w - 1)),
                max(0, min(cy + action.dy, h - 1)),
                cursor,
                relative=True,
            )
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
        names = []
        for name in action.keys:
            if km.resolve_key(name) is None:
                raise InputDeviceError(f"unknown key {name!r}")
            names.append(km.canonical_key(name))

        self._wait_until(cursor)
        if not self.dry_run:
            # Press in order, modifiers first as written.
            for name in names:
                self.devices.key(name, True)
        hold_end = cursor + action.hold_ms / 1000.0
        self._wait_until(hold_end)
        if not self.dry_run:
            # Release in reverse, so modifiers outlive the key they modified.
            for name in reversed(names):
                self.devices.key(name, False)
        return hold_end

    def _do_text(self, action: TypeText, cursor: float) -> float:
        # Windows SendInput can deliver a UTF-16 code unit directly with no layout
        # involved, so there is nothing for the clipboard workaround to fix there.
        native_text = getattr(self.devices, "text", None)
        if native_text is not None and self.text_mode is not TextMode.CLIPBOARD:
            self._wait_until(cursor)
            if not self.dry_run:
                native_text(action.text)
            return cursor + len(action.text) * 0.001

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
        for ch in action.text:
            mapped = km.ascii_to_keys(ch)
            if mapped is None:
                # Unreachable via AUTO, but a playbook can force KEYS mode.
                continue
            code, needs_shift = mapped
            self._wait_until(t)
            if not self.dry_run:
                name = _KEYNAME_BY_CODE.get(code)
                if name is not None:
                    if needs_shift:
                        self.devices.key("leftshift", True)
                    self.devices.key(name, True)
                    self.devices.key(name, False)
                    if needs_shift:
                        self.devices.key("leftshift", False)
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

    def _move_to(
        self, x: int, y: int, cursor: float, *, relative: bool = False
    ) -> float:
        """Move to (x, y), interpolating when a button is held.

        A drag is not "button down, jump, button up". Applications decide a drag has
        started by watching motion: a file manager needs to cross a drag threshold before
        it begins drag-and-drop, a canvas needs the intermediate points to draw through,
        and a selection rectangle needs them to size itself. Teleporting the pointer with
        a button held produces a click at the origin and a click at the destination --
        which is why this looked correct in a trace and failed against real software.

        So when anything is held, the move is broken into steps spaced a few milliseconds
        apart. When nothing is held there is no reason to pay for that, and the pointer
        jumps as before.
        """
        held = bool(self._held_buttons)
        start = self._cursor

        if not held:
            self._wait_until(cursor)
            self._emit_move(x, y, relative=relative, previous=start)
            return cursor

        distance = max(abs(x - start[0]), abs(y - start[1]))
        if distance <= 1:
            self._wait_until(cursor)
            self._emit_move(x, y, relative=relative, previous=start)
            return cursor

        # One step per ~40 px, bounded. Enough for an application to see continuous
        # motion without turning a long drag into hundreds of events.
        steps = max(_DRAG_MIN_STEPS, min(_DRAG_MAX_STEPS, distance // 40))
        gap = _DRAG_STEP_MS / 1000.0
        previous = start
        for i in range(1, steps + 1):
            self._wait_until(cursor + (i - 1) * gap)
            if self._abort.is_set():
                break
            # Ease slightly: real drags are not perfectly linear, and a first small step
            # is what crosses an application's drag threshold cleanly.
            fraction = i / steps
            nx = int(round(start[0] + (x - start[0]) * fraction))
            ny = int(round(start[1] + (y - start[1]) * fraction))
            self._emit_move(nx, ny, relative=relative, previous=previous)
            previous = (nx, ny)
        return cursor + steps * gap

    def _emit_move(
        self, x: int, y: int, *, relative: bool, previous: tuple[int, int]
    ) -> None:
        if relative:
            self._do_move_rel(x - previous[0], y - previous[1])
            self._cursor = (x, y)
        else:
            self._do_move_abs(x, y)

    def _do_move_abs(self, x: int, y: int) -> None:
        self._cursor = (x, y)
        if self.dry_run:
            return
        if self.pointer_mode is PointerMode.RELATIVE:
            # No absolute channel available: home to the top-left with a deliberately
            # over-large delta, then step out to the target. Pointer acceleration does
            # not apply to the clamp at 0,0, so this lands exactly.
            w, h = self.devices.screen
            self.devices.move_rel(-w * 2, -h * 2)
            if x or y:
                self.devices.move_rel(x, y)
            return
        self.devices.move_abs(x, y)

    def _do_move_rel(self, dx: int, dy: int) -> None:
        cx, cy = self._cursor
        w, h = self.devices.screen
        self._cursor = (max(0, min(cx + dx, w - 1)), max(0, min(cy + dy, h - 1)))
        if self.dry_run or (dx == 0 and dy == 0):
            return
        self.devices.move_rel(dx, dy)

    def _do_scroll(self, action: Scroll, cursor: float) -> float:
        t = cursor
        step = action.step_ms / 1000.0
        direction = 1 if action.amount > 0 else -1
        for _ in range(abs(action.amount)):
            self._wait_until(t)
            if not self.dry_run:
                self.devices.scroll(direction, action.axis)
            t += step
        return t

    # -- primitives ------------------------------------------------------------------

    def _key_event(self, name: str, value: int) -> bool:
        if km.resolve_key(name) is None:
            raise InputDeviceError(f"unknown key {name!r}")
        if not self.dry_run:
            self.devices.key(km.canonical_key(name), bool(value))
        return True

    def _button_event(self, name: str, value: int) -> bool:
        if name not in km.BUTTONS:
            raise InputDeviceError(f"unknown button {name!r}")
        if not self.dry_run:
            self.devices.button(name, bool(value))
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
                self.devices.key("leftctrl", True)
                self.devices.key("v", True)
                time.sleep(0.012)
                self.devices.key("v", False)
                self.devices.key("leftctrl", False)
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
