"""Virtual input devices via `/dev/uinput`, implemented with ctypes/fcntl only.

Why not python-evdev
--------------------
evdev works, but it is a C extension, and this project targets whatever Python the user
already has (3.14 here, where several extensions still lack wheels). uinput itself is
about 150 lines of ioctl plumbing that has been ABI-stable for two decades, so vendoring
it removes a build dependency and a whole class of "works on my machine". It also gives
direct control over event batching, which matters: writing a click's press+release+SYN
in a *single* `os.write` is both atomic and measurably faster than three syscalls.

Why uinput at all
-----------------
The session here is Wayland/KWin. `xdotool` and friends talk to an X server that Wayland
clients are not connected to -- they can only reach XWayland windows, which is to say
almost nothing on a modern KDE desktop. uinput injects at the *kernel evdev* layer,
below the display server entirely, so libinput picks the events up exactly as it would
from a real USB device. It works on X11, Wayland, the console, and inside games that
read raw input.

Device topology
---------------
Three devices are created rather than one, because libinput classifies a device by the
capabilities it advertises and a single device claiming keyboard + absolute pointer +
relative pointer gets classified unpredictably:

    voltage-keyboard      EV_KEY (all keys), EV_MSC
    voltage-pointer-abs   BTN_*, ABS_X/ABS_Y, wheel, INPUT_PROP_POINTER
    voltage-pointer-rel   BTN_*, REL_X/REL_Y, wheel

libinput merges all pointer devices into one logical cursor, so a button pressed on the
absolute device and released on the relative one behaves correctly. Having both means
absolute positioning for desktop UI and relative deltas for pointer-locked games, and
`calibrate` can determine at runtime which one the compositor actually honours.
"""

from __future__ import annotations

import fcntl
import os
import struct
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from ..errors import InputDeviceError
from . import keymap as km

__all__ = ["UInputDevice", "DeviceSet", "UINPUT_PATH", "probe_uinput"]

UINPUT_PATH: Final = "/dev/uinput"

# -- ioctl encoding (asm-generic/ioctl.h) ------------------------------------------------
_IOC_NRBITS: Final = 8
_IOC_TYPEBITS: Final = 8
_IOC_SIZEBITS: Final = 14
_IOC_NRSHIFT: Final = 0
_IOC_TYPESHIFT: Final = _IOC_NRSHIFT + _IOC_NRBITS      # 8
_IOC_SIZESHIFT: Final = _IOC_TYPESHIFT + _IOC_TYPEBITS  # 16
_IOC_DIRSHIFT: Final = _IOC_SIZESHIFT + _IOC_SIZEBITS   # 30
_IOC_NONE: Final = 0
_IOC_WRITE: Final = 1


def _ioc(direction: int, type_: str, nr: int, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (size << _IOC_SIZESHIFT)
        | (ord(type_) << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
    )


# struct input_event { struct timeval time; __u16 type; __u16 code; __s32 value; }
# On 64-bit Linux timeval is two 64-bit words, so the struct is 24 bytes.
_EVENT_FMT: Final = "llHHi"
_EVENT_SIZE: Final = struct.calcsize(_EVENT_FMT)

# struct input_id { __u16 bustype, vendor, product, version; }
# struct uinput_setup { struct input_id id; char name[80]; __u32 ff_effects_max; }
_SETUP_FMT: Final = "4H80sI"
_SETUP_SIZE: Final = struct.calcsize(_SETUP_FMT)

# struct input_absinfo { __s32 value, minimum, maximum, fuzz, flat, resolution; }
# struct uinput_abs_setup { __u16 code; struct input_absinfo absinfo; }
_ABS_SETUP_FMT: Final = "H2x6i"
_ABS_SETUP_SIZE: Final = struct.calcsize(_ABS_SETUP_FMT)

_UI: Final = "U"
UI_DEV_CREATE: Final = _ioc(_IOC_NONE, _UI, 1, 0)
UI_DEV_DESTROY: Final = _ioc(_IOC_NONE, _UI, 2, 0)
UI_DEV_SETUP: Final = _ioc(_IOC_WRITE, _UI, 3, _SETUP_SIZE)
UI_ABS_SETUP: Final = _ioc(_IOC_WRITE, _UI, 4, _ABS_SETUP_SIZE)
UI_SET_EVBIT: Final = _ioc(_IOC_WRITE, _UI, 100, 4)
UI_SET_KEYBIT: Final = _ioc(_IOC_WRITE, _UI, 101, 4)
UI_SET_RELBIT: Final = _ioc(_IOC_WRITE, _UI, 102, 4)
UI_SET_ABSBIT: Final = _ioc(_IOC_WRITE, _UI, 103, 4)
UI_SET_MSCBIT: Final = _ioc(_IOC_WRITE, _UI, 104, 4)
UI_SET_PROPBIT: Final = _ioc(_IOC_WRITE, _UI, 110, 4)

# Absolute axes are reported in a resolution-independent 0..ABS_MAX range and scaled to
# the desktop at write time, the same convention QEMU's virtio-tablet and VNC use. This
# keeps the device valid across resolution changes and multi-monitor reconfiguration.
ABS_MAX: Final = 65535

# libinput/udev need time to enumerate a freshly created device. Writing events before
# that completes is the single most common cause of "uinput does nothing" -- the events
# are accepted by the kernel and dropped on the floor because nothing is listening yet.
SETTLE_SECONDS: Final = 0.4


@dataclass(slots=True)
class DeviceCaps:
    name: str
    keys: Sequence[int] = ()
    rel_axes: Sequence[int] = ()
    abs_axes: Sequence[int] = ()
    props: Sequence[int] = ()
    misc: Sequence[int] = ()
    vendor: int = 0x1D6B  # Linux Foundation
    product: int = 0x0001
    version: int = 0x0100


class UInputDevice:
    """One virtual evdev device. Not thread-safe; the executor owns it on one thread."""

    __slots__ = ("_fd", "_caps", "_created")

    def __init__(self, caps: DeviceCaps) -> None:
        self._caps = caps
        self._fd: int | None = None
        self._created = False

    # -- lifecycle -------------------------------------------------------------------

    def open(self, *, settle: bool = True) -> None:
        """Create the device. Pass `settle=False` when opening several at once and
        sleeping afterwards -- the settle delay is wall-clock, not per-device."""
        if self._fd is not None:
            return
        try:
            fd = os.open(UINPUT_PATH, os.O_WRONLY | os.O_NONBLOCK)
        except FileNotFoundError as exc:
            raise InputDeviceError(
                f"{UINPUT_PATH} does not exist; the uinput kernel module is not loaded. "
                "Run: sudo modprobe uinput  (and see scripts/setup.sh for making it "
                "persistent)"
            ) from exc
        except PermissionError as exc:
            raise InputDeviceError(
                f"no write permission on {UINPUT_PATH}. Either add yourself to the "
                "'input' group and re-login, or install the udev rule from "
                "scripts/setup.sh which grants an ACL to the active seat user."
            ) from exc

        self._fd = fd
        try:
            self._configure()
            fcntl.ioctl(fd, UI_DEV_CREATE)
            self._created = True
        except OSError as exc:
            self.close()
            raise InputDeviceError(
                f"failed to create virtual device {self._caps.name!r}: {exc}"
            ) from exc

        if settle:
            time.sleep(SETTLE_SECONDS)

    def _configure(self) -> None:
        fd = self._fd
        assert fd is not None
        caps = self._caps

        fcntl.ioctl(fd, UI_SET_EVBIT, km.EV_SYN)

        if caps.keys:
            fcntl.ioctl(fd, UI_SET_EVBIT, km.EV_KEY)
            for code in caps.keys:
                fcntl.ioctl(fd, UI_SET_KEYBIT, code)

        if caps.rel_axes:
            fcntl.ioctl(fd, UI_SET_EVBIT, km.EV_REL)
            for code in caps.rel_axes:
                fcntl.ioctl(fd, UI_SET_RELBIT, code)

        if caps.abs_axes:
            fcntl.ioctl(fd, UI_SET_EVBIT, km.EV_ABS)
            for code in caps.abs_axes:
                fcntl.ioctl(fd, UI_SET_ABSBIT, code)
                # value, min, max, fuzz, flat, resolution
                absinfo = struct.pack(_ABS_SETUP_FMT, code, 0, 0, ABS_MAX, 0, 0, 0)
                fcntl.ioctl(fd, UI_ABS_SETUP, absinfo)

        if caps.misc:
            fcntl.ioctl(fd, UI_SET_EVBIT, km.EV_MSC)
            for code in caps.misc:
                fcntl.ioctl(fd, UI_SET_MSCBIT, code)

        for prop in caps.props:
            fcntl.ioctl(fd, UI_SET_PROPBIT, prop)

        name = caps.name.encode("utf-8")[:79]
        setup = struct.pack(
            _SETUP_FMT, km.BUS_USB, caps.vendor, caps.product, caps.version, name, 0
        )
        fcntl.ioctl(fd, UI_DEV_SETUP, setup)

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            if self._created:
                fcntl.ioctl(self._fd, UI_DEV_DESTROY)
        except OSError:
            pass
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            self._created = False

    def __enter__(self) -> UInputDevice:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._fd is not None and self._created

    @property
    def name(self) -> str:
        return self._caps.name

    # -- writing ---------------------------------------------------------------------

    def emit(self, events: Iterable[tuple[int, int, int]], *, sync: bool = True) -> None:
        """Write a batch of (type, code, value) events in one syscall.

        Timestamps are left zeroed; the kernel fills them in on receipt. Batching is not
        just an optimisation -- events between SYN_REPORTs are one atomic input frame, so
        a press and its coordinates must land together or the compositor may see a click
        at the previous cursor position.
        """
        if self._fd is None:
            raise InputDeviceError(f"device {self._caps.name!r} is not open")

        payload = bytearray()
        for ev_type, code, value in events:
            payload += struct.pack(_EVENT_FMT, 0, 0, ev_type, code, value)
        if sync:
            payload += struct.pack(_EVENT_FMT, 0, 0, km.EV_SYN, km.SYN_REPORT, 0)
        if not payload:
            return

        try:
            os.write(self._fd, bytes(payload))
        except BlockingIOError:
            # The uinput fd is opened non-blocking, but it has no meaningful buffer
            # limit for our volumes. Retry once, then give up loudly rather than
            # silently dropping input in the middle of a burst.
            time.sleep(0.001)
            try:
                os.write(self._fd, bytes(payload))
            except OSError as exc:
                raise InputDeviceError(f"uinput write failed on {self._caps.name!r}: {exc}") from exc
        except OSError as exc:
            raise InputDeviceError(f"uinput write failed on {self._caps.name!r}: {exc}") from exc

    def syn(self) -> None:
        self.emit((), sync=True)


# -- device factories --------------------------------------------------------------------


def keyboard_caps() -> DeviceCaps:
    return DeviceCaps(
        name="voltage-keyboard",
        keys=km.all_key_codes(),
        misc=(km.MSC_SCAN,),
        product=0x0101,
    )


def pointer_abs_caps() -> DeviceCaps:
    return DeviceCaps(
        name="voltage-pointer-abs",
        keys=tuple(km.BUTTONS.values()),
        abs_axes=(km.ABS_X, km.ABS_Y),
        # HI_RES wheel axes are declared alongside the classic ones because modern
        # toolkits read HI_RES for smooth scrolling and older ones only see REL_WHEEL.
        rel_axes=(km.REL_WHEEL, km.REL_HWHEEL, km.REL_WHEEL_HI_RES, km.REL_HWHEEL_HI_RES),
        # Declaring POINTER (and *not* DIRECT) is what stops udev's input_id builtin
        # from tagging an ABS_X/ABS_Y device as a touchscreen.
        props=(km.INPUT_PROP_POINTER,),
        product=0x0102,
    )


def pointer_rel_caps() -> DeviceCaps:
    return DeviceCaps(
        name="voltage-pointer-rel",
        keys=tuple(km.BUTTONS.values()),
        rel_axes=(
            km.REL_X, km.REL_Y, km.REL_WHEEL, km.REL_HWHEEL,
            km.REL_WHEEL_HI_RES, km.REL_HWHEEL_HI_RES,
        ),
        props=(km.INPUT_PROP_POINTER,),
        product=0x0103,
    )


@dataclass(slots=True)
class DeviceSet:
    """The three virtual devices, opened and closed together.

    Devices are created lazily on first `open()` and then kept alive for the process
    lifetime. Creating them costs ~1.2s of settle time, so churning them per run would
    dominate short tasks.
    """

    screen: tuple[int, int] = (1920, 1080)
    keyboard: UInputDevice = field(init=False)
    pointer_abs: UInputDevice = field(init=False)
    pointer_rel: UInputDevice = field(init=False)
    _open: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.keyboard = UInputDevice(keyboard_caps())
        self.pointer_abs = UInputDevice(pointer_abs_caps())
        self.pointer_rel = UInputDevice(pointer_rel_caps())

    def open(self) -> None:
        if self._open:
            return
        # Create all three, then settle once. The delay exists to let udev and libinput
        # enumerate the new devices; that happens concurrently, so it is wall-clock time
        # rather than a per-device cost.
        for dev in (self.keyboard, self.pointer_abs, self.pointer_rel):
            dev.open(settle=False)
        time.sleep(SETTLE_SECONDS)
        self._open = True

    def close(self) -> None:
        for dev in (self.pointer_rel, self.pointer_abs, self.keyboard):
            dev.close()
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def to_abs(self, x: int, y: int) -> tuple[int, int]:
        """Screen pixels -> the device's 0..ABS_MAX axis range."""
        w, h = self.screen
        ax = 0 if w <= 1 else round(max(0, min(x, w - 1)) * ABS_MAX / (w - 1))
        ay = 0 if h <= 1 else round(max(0, min(y, h - 1)) * ABS_MAX / (h - 1))
        return ax, ay

    def __enter__(self) -> DeviceSet:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def probe_uinput() -> dict[str, object]:
    """Non-destructive check of whether input injection can work at all.

    Used by `voltage.doctor` so the orchestrator learns about a permissions problem
    before it writes a playbook, not thirty seconds into a run.
    """
    result: dict[str, object] = {"path": UINPUT_PATH}
    if not os.path.exists(UINPUT_PATH):
        result["ok"] = False
        result["reason"] = "missing"
        result["fix"] = "sudo modprobe uinput"
        return result
    result["writable"] = os.access(UINPUT_PATH, os.W_OK)
    if not result["writable"]:
        result["ok"] = False
        result["reason"] = "permission denied"
        result["fix"] = (
            "sudo usermod -aG input $USER && reboot, or install the udev rule in "
            "scripts/setup.sh"
        )
        return result
    try:
        fd = os.open(UINPUT_PATH, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
        result["ok"] = True
    except OSError as exc:
        result["ok"] = False
        result["reason"] = str(exc)
    return result
