"""Stopping a run that has gone wrong.

Autonomous input control needs an exit that does not itself require the machine to be
usable. If the actuator is holding Alt and spamming Tab, opening a terminal to type a
stop command is not realistic. So there are four independent stops, in order of how
little they assume:

  1. **Panic file.** A file appears in the runtime dir and every loop notices within one
     cycle. `voltage stop` writes it; so does the MCP `voltage.stop` tool. Works from any
     other machine over SSH, from a file manager, from anything.
  2. **Deadman timer.** If a cycle does not complete within `deadman_s`, the watchdog
     fires on its own thread and releases every held key. This is what covers the case
     where the loop itself has wedged -- a model backend hanging, a capture blocking --
     because nothing in the main path is running to notice.
  3. **Physical-input contention.** If the user touches the real keyboard or mouse, stop.
     This is the one people actually reach for: grabbing the mouse should end it. Needs
     read access to `/dev/input/event*`, which means membership of the `input` group, so
     it is optional and reports clearly when unavailable.
  4. **Budget exhaustion.** Cycle, wall-clock, burst and rejection limits from the
     playbook. Not an emergency stop so much as a guarantee of termination.

Every path funnels into `trip()`, and every path releases held input on the way out.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["KillSwitch", "panic_path", "write_panic", "clear_panic"]


def _runtime_dir() -> Path:
    """Where the panic file lives.

    Must be somewhere a *different* process, possibly a different shell or an ssh
    session, can reliably write -- that is the whole point of a file-based kill switch.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(base) / "voltage-input-mcp"
    else:
        base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/voltage-{os.getuid()}"
        path = Path(base) / "voltage-input-mcp"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        suffix = os.getpid() if sys.platform == "win32" else os.getuid()
        path = Path(tempfile.gettempdir()) / f"voltage-input-mcp-{suffix}"
        path.mkdir(parents=True, exist_ok=True)
    return path


def panic_path() -> Path:
    return _runtime_dir() / "PANIC"


def write_panic(reason: str = "manual stop") -> Path:
    path = panic_path()
    path.write_text(f"{reason}\n{time.time()}\n")
    return path


def clear_panic() -> None:
    try:
        panic_path().unlink()
    except FileNotFoundError:
        pass


@dataclass(slots=True)
class KillSwitch:
    """Aggregates every stop condition for a single run."""

    deadman_s: float = 6.0
    watch_physical_input: bool = True
    poll_interval_s: float = 0.15
    on_trip: Callable[[str], None] | None = None

    _tripped: threading.Event = field(default_factory=threading.Event, init=False)
    _reason: str = field(default="", init=False)
    _deadline: float = field(default=0.0, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _physical_thread: threading.Thread | None = field(default=None, init=False)
    _physical_status: str = field(default="not started", init=False)

    # -- state -----------------------------------------------------------------------

    @property
    def tripped(self) -> bool:
        return self._tripped.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def physical_watch_status(self) -> str:
        return self._physical_status

    def trip(self, reason: str) -> None:
        if self._tripped.is_set():
            return
        self._reason = reason
        self._tripped.set()
        if self.on_trip is not None:
            try:
                self.on_trip(reason)
            except Exception:  # noqa: BLE001 - the stop path must never raise
                pass

    def reset(self) -> None:
        self._tripped.clear()
        self._reason = ""
        self._deadline = 0.0

    # -- deadman ---------------------------------------------------------------------

    def touch(self, budget_s: float | None = None) -> None:
        """Restart the deadman timer. Call once per cycle."""
        self._deadline = time.monotonic() + (budget_s or self.deadman_s)

    def clear_deadline(self) -> None:
        self._deadline = 0.0

    # -- lifecycle -------------------------------------------------------------------

    def start(self) -> None:
        clear_panic()
        self._stop.clear()
        self.reset()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._watch, name="voltage-killswitch", daemon=True
            )
            self._thread.start()
        if self.watch_physical_input:
            self._start_physical_watch()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._physical_thread = None

    def _watch(self) -> None:
        path = panic_path()
        while not self._stop.is_set():
            if path.exists():
                try:
                    detail = path.read_text().splitlines()[0]
                except (OSError, IndexError):
                    detail = "panic file"
                self.trip(f"panic file: {detail}")
                return
            deadline = self._deadline
            if deadline and time.monotonic() > deadline:
                self.trip(
                    f"deadman timer: no cycle progress for {self.deadman_s:.1f}s "
                    f"(a model backend or the capture layer is probably hung)"
                )
                return
            self._stop.wait(self.poll_interval_s)

    # -- physical input contention ---------------------------------------------------

    def _start_physical_watch(self) -> None:
        if self._physical_thread is not None and self._physical_thread.is_alive():
            return
        self._physical_thread = threading.Thread(
            target=self._watch_physical, name="voltage-physical-watch", daemon=True
        )
        self._physical_thread.start()

    def _watch_physical(self) -> None:
        """Trip when a *real* device produces input.

        Our own virtual devices are excluded by name. This is deliberately generous about
        what counts as user intervention: any key, any button, or sustained pointer motion
        means the human wants the machine back.
        """
        try:
            import evdev
        except ImportError:
            self._physical_status = (
                "unavailable: python-evdev is not installed "
                "(pacman -S python-evdev). Grab-the-mouse-to-stop is disabled."
            )
            return

        try:
            paths = evdev.list_devices()
        except OSError as exc:
            self._physical_status = f"unavailable: cannot list input devices ({exc})"
            return

        devices = []
        for path in paths:
            try:
                dev = evdev.InputDevice(path)
            except (OSError, PermissionError):
                continue
            if dev.name.startswith("voltage-"):
                dev.close()
                continue
            caps = dev.capabilities()
            if evdev.ecodes.EV_KEY in caps or evdev.ecodes.EV_REL in caps:
                devices.append(dev)
            else:
                dev.close()

        if not devices:
            self._physical_status = (
                "unavailable: no readable input devices. Add yourself to the 'input' "
                "group (sudo usermod -aG input $USER) and re-login to enable "
                "grab-the-mouse-to-stop."
            )
            return

        self._physical_status = f"watching {len(devices)} device(s)"
        motion = 0
        try:
            from select import select

            fds = {dev.fd: dev for dev in devices}
            while not self._stop.is_set() and not self._tripped.is_set():
                ready, _, _ = select(list(fds), [], [], 0.2)
                for fd in ready:
                    for event in fds[fd].read():
                        if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                            self.trip("physical input: a real key or button was pressed")
                            return
                        if event.type == evdev.ecodes.EV_REL and abs(event.value) > 2:
                            motion += abs(event.value)
                            # A threshold rather than a single event, so a nudge of the
                            # trackpad while reading does not abort a long run.
                            if motion > 90:
                                self.trip("physical input: the pointer was moved by hand")
                                return
        except OSError as exc:
            self._physical_status = f"stopped: {exc}"
        finally:
            for dev in devices:
                try:
                    dev.close()
                except OSError:
                    pass

    # -- reporting -------------------------------------------------------------------

    def status(self) -> dict[str, object]:
        return {
            "tripped": self.tripped,
            "reason": self._reason,
            "deadman_s": self.deadman_s,
            "panic_file": str(panic_path()),
            "physical_watch": self._physical_status,
        }
