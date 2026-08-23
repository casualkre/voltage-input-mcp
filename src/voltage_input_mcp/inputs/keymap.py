"""Linux input event codes, key-name aliases, and ASCII -> scancode mapping.

Codes are lifted from `linux/input-event-codes.h`. They are inlined rather than read
from a header because this must work identically whether or not kernel headers are
installed, and they have not changed in twenty years.

A caveat that matters more than it looks
----------------------------------------
uinput injects **scancodes**, not characters. What a scancode produces depends entirely
on the XKB layout the compositor has active. `KEY_Q` is 'q' on a US layout and on a
Turkish-Q layout, but ';' `KEY_SEMICOLON` is 's' on Turkish-Q, and most punctuation
moves. So `_ASCII` below is a **US-layout** table and typing via scancodes is only
correct when the active layout matches.

This is why `TextMode.CLIPBOARD` exists (see executor.py) and why `auto` picks it for
anything non-alphanumeric. Clipboard paste is layout-independent, unicode-safe, and
O(1) in the length of the text instead of O(n) keystrokes.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "EV_SYN", "EV_KEY", "EV_REL", "EV_ABS", "EV_MSC",
    "SYN_REPORT", "REL_X", "REL_Y", "REL_WHEEL", "REL_HWHEEL",
    "REL_WHEEL_HI_RES", "REL_HWHEEL_HI_RES",
    "ABS_X", "ABS_Y", "MSC_SCAN",
    "BUTTONS", "KEYS", "ALIASES", "MODIFIERS",
    "resolve_key", "ascii_to_keys", "all_key_codes", "is_typeable",
]

# -- event types ------------------------------------------------------------------------
EV_SYN: Final = 0x00
EV_KEY: Final = 0x01
EV_REL: Final = 0x02
EV_ABS: Final = 0x03
EV_MSC: Final = 0x04

SYN_REPORT: Final = 0x00

REL_X: Final = 0x00
REL_Y: Final = 0x01
REL_HWHEEL: Final = 0x06
REL_WHEEL: Final = 0x08
REL_WHEEL_HI_RES: Final = 0x0B
REL_HWHEEL_HI_RES: Final = 0x0C

ABS_X: Final = 0x00
ABS_Y: Final = 0x01

MSC_SCAN: Final = 0x04

INPUT_PROP_POINTER: Final = 0x00
INPUT_PROP_DIRECT: Final = 0x01

BUS_USB: Final = 0x03
BUS_VIRTUAL: Final = 0x06

# -- pointer buttons --------------------------------------------------------------------
BTN_LEFT: Final = 0x110
BTN_RIGHT: Final = 0x111
BTN_MIDDLE: Final = 0x112
BTN_SIDE: Final = 0x113
BTN_EXTRA: Final = 0x114

BUTTONS: Final[dict[str, int]] = {
    "l": BTN_LEFT,
    "r": BTN_RIGHT,
    "m": BTN_MIDDLE,
    "4": BTN_SIDE,
    "5": BTN_EXTRA,
}

# -- keyboard keys ----------------------------------------------------------------------
KEYS: Final[dict[str, int]] = {
    "esc": 1,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
    "minus": 12, "equal": 13, "backspace": 14, "tab": 15,
    "q": 16, "w": 17, "e": 18, "r": 19, "t": 20,
    "y": 21, "u": 22, "i": 23, "o": 24, "p": 25,
    "leftbrace": 26, "rightbrace": 27, "enter": 28, "leftctrl": 29,
    "a": 30, "s": 31, "d": 32, "f": 33, "g": 34,
    "h": 35, "j": 36, "k": 37, "l": 38,
    "semicolon": 39, "apostrophe": 40, "grave": 41,
    "leftshift": 42, "backslash": 43,
    "z": 44, "x": 45, "c": 46, "v": 47, "b": 48, "n": 49, "m": 50,
    "comma": 51, "dot": 52, "slash": 53, "rightshift": 54,
    "kpasterisk": 55, "leftalt": 56, "space": 57, "capslock": 58,
    "f1": 59, "f2": 60, "f3": 61, "f4": 62, "f5": 63,
    "f6": 64, "f7": 65, "f8": 66, "f9": 67, "f10": 68,
    "numlock": 69, "scrolllock": 70,
    "kp7": 71, "kp8": 72, "kp9": 73, "kpminus": 74,
    "kp4": 75, "kp5": 76, "kp6": 77, "kpplus": 78,
    "kp1": 79, "kp2": 80, "kp3": 81, "kp0": 82, "kpdot": 83,
    "key102nd": 86, "f11": 87, "f12": 88,
    "kpenter": 96, "rightctrl": 97, "kpslash": 98, "sysrq": 99, "rightalt": 100,
    "home": 102, "up": 103, "pageup": 104, "left": 105, "right": 106,
    "end": 107, "down": 108, "pagedown": 109, "insert": 110, "delete": 111,
    "mute": 113, "volumedown": 114, "volumeup": 115, "power": 116,
    "kpequal": 117, "pause": 119, "kpcomma": 121,
    "leftmeta": 125, "rightmeta": 126, "compose": 127,
    "stop": 128, "again": 129, "undo": 131, "copy": 133, "open": 134,
    "paste": 135, "find": 136, "cut": 137, "help": 138, "menu": 139,
    "nextsong": 163, "playpause": 164, "previoussong": 165, "stopcd": 166,
    "back": 158, "forward": 159, "refresh": 173,
    "f13": 183, "f14": 184, "f15": 185, "f16": 186, "f17": 187, "f18": 188,
    "f19": 189, "f20": 190, "f21": 191, "f22": 192, "f23": 193, "f24": 194,
    "print": 210,
}

# Names an orchestrator or actuator is likely to write, mapped to canonical names.
ALIASES: Final[dict[str, str]] = {
    "escape": "esc",
    "return": "enter",
    "ret": "enter",
    "newline": "enter",
    "ctrl": "leftctrl", "control": "leftctrl", "lctrl": "leftctrl", "rctrl": "rightctrl",
    "alt": "leftalt", "lalt": "leftalt", "ralt": "rightalt", "altgr": "rightalt",
    "shift": "leftshift", "lshift": "leftshift", "rshift": "rightshift",
    "meta": "leftmeta", "super": "leftmeta", "win": "leftmeta", "cmd": "leftmeta",
    "lmeta": "leftmeta", "rmeta": "rightmeta", "lsuper": "leftmeta", "rsuper": "rightmeta",
    "del": "delete", "ins": "insert",
    "pgup": "pageup", "pgdn": "pagedown", "pagedn": "pagedown",
    "bksp": "backspace", "bs": "backspace",
    "spacebar": "space",
    "caps": "capslock", "num": "numlock", "scroll": "scrolllock",
    "printscreen": "sysrq", "prtsc": "sysrq", "prntscrn": "sysrq",
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "period": "dot", "fullstop": "dot",
    "hyphen": "minus", "dash": "minus",
    "quote": "apostrophe", "singlequote": "apostrophe",
    "backtick": "grave", "tilde": "grave",
    "lbracket": "leftbrace", "rbracket": "rightbrace",
    "openbracket": "leftbrace", "closebracket": "rightbrace",
    "plus": "kpplus", "asterisk": "kpasterisk", "star": "kpasterisk",
    "less": "comma", "greater": "dot",
}

MODIFIERS: Final[frozenset[str]] = frozenset(
    {"leftctrl", "rightctrl", "leftshift", "rightshift", "leftalt", "rightalt",
     "leftmeta", "rightmeta"}
)


def resolve_key(name: str) -> int | None:
    """Canonicalise a key name and return its scancode, or None if unknown."""
    key = name.strip().lower().replace("-", "").replace("_", "")
    key = ALIASES.get(key, key)
    return KEYS.get(key)


def canonical_key(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    return ALIASES.get(key, key)


def all_key_codes() -> list[int]:
    """Every scancode the virtual keyboard should advertise."""
    return sorted(set(KEYS.values()))


# -- ASCII typing (US layout) -----------------------------------------------------------

_SHIFTED: Final[dict[str, str]] = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5",
    "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
    "_": "minus", "+": "equal",
    "{": "leftbrace", "}": "rightbrace", "|": "backslash",
    ":": "semicolon", '"': "apostrophe", "~": "grave",
    "<": "comma", ">": "dot", "?": "slash",
}

_UNSHIFTED: Final[dict[str, str]] = {
    "-": "minus", "=": "equal", "[": "leftbrace", "]": "rightbrace",
    "\\": "backslash", ";": "semicolon", "'": "apostrophe", "`": "grave",
    ",": "comma", ".": "dot", "/": "slash",
    " ": "space", "\t": "tab", "\n": "enter", "\r": "enter",
}


def ascii_to_keys(char: str) -> tuple[int, bool] | None:
    """Map one character to (scancode, needs_shift) on a US layout.

    Returns None for anything not typeable as a single US keystroke -- accented
    characters, emoji, CJK. Callers should fall back to clipboard paste for those.
    """
    if not char:
        return None
    if "a" <= char <= "z":
        return KEYS[char], False
    if "A" <= char <= "Z":
        return KEYS[char.lower()], True
    if "0" <= char <= "9":
        return KEYS[char], False
    if char in _UNSHIFTED:
        return KEYS[_UNSHIFTED[char]], False
    if char in _SHIFTED:
        return KEYS[_SHIFTED[char]], True
    return None


def is_typeable(text: str) -> bool:
    """True if every character can be produced by a single US-layout keystroke."""
    return all(ascii_to_keys(c) is not None for c in text)
