"""Windows input injection via `SendInput`, ctypes only.

The Linux and Windows sides solve the same problem from opposite directions, and the
differences are worth knowing because they change what is easy:

  uinput   injects at the kernel evdev layer, *below* the display server. Works in
           games, on the console, everywhere -- but sends scancodes, so what a keystroke
           produces depends on the active keyboard layout, and typing punctuation on a
           non-US layout silently produces the wrong character.

  SendInput injects at the Win32 message layer. It offers `KEYEVENTF_UNICODE`, which
           delivers a UTF-16 code unit directly with no layout involved at all. Typing is
           therefore *more* correct here than on Linux, and needs no clipboard fallback.

Two Windows-specific hazards are handled below:

  * **DPI.** Without declaring per-monitor DPI awareness, Windows lies to the process
    about screen coordinates on any scaled display, and every click lands proportionally
    off. Declared at import.
  * **UIPI.** A non-elevated process cannot send input to an elevated window. This fails
    silently -- SendInput returns success and nothing happens. Detected and reported by
    `probe_win32` rather than left to look like a broken burst.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Final

from ..errors import InputDeviceError

__all__ = ["Win32Sink", "probe_win32", "available"]


def available() -> bool:
    return sys.platform == "win32"


# -- Win32 structures --------------------------------------------------------------------

if sys.platform == "win32":  # pragma: no cover - exercised only on Windows
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
else:  # keep the module importable on Linux so tests and tooling can read it
    _user32 = None  # type: ignore[assignment]
    ULONG_PTR = ctypes.c_ulonglong  # type: ignore[misc]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


INPUT_MOUSE: Final = 0
INPUT_KEYBOARD: Final = 1

MOUSEEVENTF_MOVE: Final = 0x0001
MOUSEEVENTF_LEFTDOWN: Final = 0x0002
MOUSEEVENTF_LEFTUP: Final = 0x0004
MOUSEEVENTF_RIGHTDOWN: Final = 0x0008
MOUSEEVENTF_RIGHTUP: Final = 0x0010
MOUSEEVENTF_MIDDLEDOWN: Final = 0x0020
MOUSEEVENTF_MIDDLEUP: Final = 0x0040
MOUSEEVENTF_XDOWN: Final = 0x0080
MOUSEEVENTF_XUP: Final = 0x0100
MOUSEEVENTF_WHEEL: Final = 0x0800
MOUSEEVENTF_HWHEEL: Final = 0x1000
MOUSEEVENTF_ABSOLUTE: Final = 0x8000
MOUSEEVENTF_VIRTUALDESK: Final = 0x4000

KEYEVENTF_EXTENDEDKEY: Final = 0x0001
KEYEVENTF_KEYUP: Final = 0x0002
KEYEVENTF_UNICODE: Final = 0x0004

XBUTTON1: Final = 0x0001
XBUTTON2: Final = 0x0002

SM_XVIRTUALSCREEN: Final = 76
SM_YVIRTUALSCREEN: Final = 77
SM_CXVIRTUALSCREEN: Final = 78
SM_CYVIRTUALSCREEN: Final = 79

# Virtual-key codes, keyed by the same canonical names keymap.py uses on Linux, so a
# burst string is portable across platforms without translation.
VK: Final[dict[str, int]] = {
    **{c: ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz"},
    **{d: ord(d) for d in "0123456789"},
    "enter": 0x0D, "esc": 0x1B, "tab": 0x09, "space": 0x20, "backspace": 0x08,
    "delete": 0x2E, "insert": 0x2D,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "leftctrl": 0xA2, "rightctrl": 0xA3,
    "leftshift": 0xA0, "rightshift": 0xA1,
    "leftalt": 0xA4, "rightalt": 0xA5,
    "leftmeta": 0x5B, "rightmeta": 0x5C,
    "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
    "sysrq": 0x2C, "pause": 0x13, "menu": 0x5D,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    "minus": 0xBD, "equal": 0xBB, "comma": 0xBC, "dot": 0xBE, "slash": 0xBF,
    "semicolon": 0xBA, "apostrophe": 0xDE, "leftbrace": 0xDB, "rightbrace": 0xDD,
    "backslash": 0xDC, "grave": 0xC0,
    "kpplus": 0x6B, "kpminus": 0x6D, "kpasterisk": 0x6A, "kpslash": 0x6F,
    "kpenter": 0x0D, "kpdot": 0x6E,
    **{f"kp{i}": 0x60 + i for i in range(10)},
    "mute": 0xAD, "volumedown": 0xAE, "volumeup": 0xAF,
    "playpause": 0xB3, "nextsong": 0xB0, "previoussong": 0xB1,
    "back": 0xA6, "forward": 0xA7, "refresh": 0xA8,
}

# Keys that must carry KEYEVENTF_EXTENDEDKEY or Windows treats them as their numpad twin.
_EXTENDED: Final[frozenset[str]] = frozenset({
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "insert", "delete", "rightctrl", "rightalt", "leftmeta", "rightmeta",
    "numlock", "kpslash", "kpenter",
})

_BUTTON_FLAGS: Final[dict[str, tuple[int, int, int]]] = {
    # name -> (down flag, up flag, mouseData)
    "l": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, 0),
    "r": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, 0),
    "m": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, 0),
    "4": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON1),
    "5": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON2),
}


def _declare_dpi_awareness() -> None:
    """Without this, every coordinate is wrong on a scaled display.

    Windows reports virtualised coordinates to DPI-unaware processes, so on a 150%
    display the desktop appears smaller than it is and every click lands proportionally
    short. Must happen before any screen metric is read.
    """
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:  # noqa: BLE001 - pre-8.1 fallback
        try:
            _user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


class Win32Sink:
    """Input sink with the same surface as the Linux `DeviceSet`.

    The executor talks to this interface and does not know which platform it is on.
    """

    def __init__(self, screen: tuple[int, int] | None = None) -> None:
        if sys.platform != "win32":
            raise InputDeviceError("Win32Sink is only usable on Windows")
        _declare_dpi_awareness()
        self._origin = (
            _user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            _user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
        )
        self.screen = screen or (
            _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )
        self._open = False

    # -- lifecycle (no devices to create; kept for interface parity) ------------------

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    # -- emission --------------------------------------------------------------------

    def _send(self, events: list[INPUT]) -> None:
        if not events:
            return
        array = (INPUT * len(events))(*events)
        sent = _user32.SendInput(len(events), array, ctypes.sizeof(INPUT))
        if sent != len(events):
            error = ctypes.get_last_error()
            hint = ""
            if error == 5:  # ERROR_ACCESS_DENIED
                hint = (
                    " -- UIPI blocked it: the foreground window belongs to an elevated "
                    "process and this one is not. Run as administrator to drive it."
                )
            raise InputDeviceError(
                f"SendInput delivered {sent}/{len(events)} events (error {error}){hint}"
            )

    def key(self, name: str, down: bool) -> None:
        code = VK.get(name)
        if code is None:
            raise InputDeviceError(f"unknown key {name!r} on Windows")
        flags = 0 if down else KEYEVENTF_KEYUP
        if name in _EXTENDED:
            flags |= KEYEVENTF_EXTENDEDKEY
        event = INPUT(type=INPUT_KEYBOARD)
        event.ki = KEYBDINPUT(wVk=code, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
        self._send([event])

    def text(self, value: str) -> None:
        """Type via KEYEVENTF_UNICODE -- layout-independent, no clipboard needed.

        Each UTF-16 code unit is sent as its own down/up pair, so characters outside the
        BMP (emoji) work too: Python's UTF-16 encoding splits them into surrogates and
        Windows reassembles them.
        """
        events: list[INPUT] = []
        raw = value.encode("utf-16-le")
        units = [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]
        for unit in units:
            for up in (False, True):
                event = INPUT(type=INPUT_KEYBOARD)
                event.ki = KEYBDINPUT(
                    wVk=0,
                    wScan=unit,
                    dwFlags=KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0),
                    time=0,
                    dwExtraInfo=0,
                )
                events.append(event)
        self._send(events)

    def button(self, name: str, down: bool) -> None:
        flags = _BUTTON_FLAGS.get(name)
        if flags is None:
            raise InputDeviceError(f"unknown button {name!r}")
        down_flag, up_flag, data = flags
        event = INPUT(type=INPUT_MOUSE)
        event.mi = MOUSEINPUT(
            dx=0, dy=0, mouseData=data,
            dwFlags=down_flag if down else up_flag, time=0, dwExtraInfo=0,
        )
        self._send([event])

    def move_abs(self, x: int, y: int) -> None:
        """Absolute move across the whole virtual desktop.

        Windows wants 0..65535 normalised over the virtual desktop when ABSOLUTE and
        VIRTUALDESK are combined, and the origin can be negative on a multi-monitor
        layout where a secondary display sits left of or above the primary.
        """
        width, height = self.screen
        ox, oy = self._origin
        nx = 0 if width <= 1 else round((x - ox) * 65535 / (width - 1))
        ny = 0 if height <= 1 else round((y - oy) * 65535 / (height - 1))
        event = INPUT(type=INPUT_MOUSE)
        event.mi = MOUSEINPUT(
            dx=max(0, min(65535, nx)),
            dy=max(0, min(65535, ny)),
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            time=0,
            dwExtraInfo=0,
        )
        self._send([event])

    def move_rel(self, dx: int, dy: int) -> None:
        event = INPUT(type=INPUT_MOUSE)
        event.mi = MOUSEINPUT(
            dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0
        )
        self._send([event])

    def scroll(self, amount: int, axis: str = "v") -> None:
        # One detent is WHEEL_DELTA (120), matching REL_WHEEL_HI_RES on Linux.
        event = INPUT(type=INPUT_MOUSE)
        event.mi = MOUSEINPUT(
            dx=0, dy=0,
            mouseData=ctypes.c_uint32(120 * amount).value,
            dwFlags=MOUSEEVENTF_WHEEL if axis == "v" else MOUSEEVENTF_HWHEEL,
            time=0, dwExtraInfo=0,
        )
        self._send([event])


def probe_win32() -> dict[str, object]:
    """Readiness check mirroring `probe_uinput`."""
    if sys.platform != "win32":
        return {"ok": False, "reason": "not running on Windows"}
    try:
        _declare_dpi_awareness()
        width = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        height = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"cannot read screen metrics: {exc}"}

    result: dict[str, object] = {
        "backend": "sendinput",
        "ok": True,
        "screen": [width, height],
        "note": (
            "SendInput cannot drive windows owned by an elevated process (UIPI). If a "
            "burst appears to do nothing over an admin window, that is why -- run this "
            "elevated to control it."
        ),
    }
    try:
        elevated = bool(ctypes.WinDLL("shell32").IsUserAnAdmin())
        result["elevated"] = elevated
    except Exception:  # noqa: BLE001
        result["elevated"] = None
    return result
