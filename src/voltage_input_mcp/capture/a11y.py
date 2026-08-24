"""Read the desktop's accessibility tree instead of guessing at pixels.

Why this exists
---------------
Every GUI toolkit already publishes, over D-Bus, exactly what a vision model is being
asked to infer: the name of each control, its role, its state, and often its position.
AT-SPI is that publication, it has been standard on Linux desktops for twenty years, and
this project spent its entire life until now ignoring it in favour of asking a 3B model to
look at a screenshot.

The difference is not incremental. Asking the VLM "is there a Save button" costs ~1.4 s,
returns a label from a closed vocabulary, and can confabulate. Asking AT-SPI returns
`push button "Save"` because that is literally the widget's name, in about a millisecond,
and it cannot be wrong about it -- the application is the one saying so.

What it does not do
-------------------
Two honest limits, and neither is small.

**Games are invisible.** Roblox, or anything drawing its own UI into a GL surface,
publishes nothing. This is a desktop-control sensor and it contributes exactly zero to the
game case. That is fine -- games have HUDs with fixed layouts, which is what probes and
glyph reading are for -- but it means this complements the vision layer rather than
replacing it.

**Wayland breaks coordinates.** Measured on this machine: menu items report real extents
like `(41,0,44x30)`, while most other widgets report `(0,0,0x0)`. Wayland deliberately
denies clients a global coordinate space, so a toolkit often cannot say where on screen it
is even though it knows its own name and role perfectly well. So elements arrive with
`grounded=False` when their extents are unusable, and a caller that needs to *click*
something has to fall back to vision for the position while still trusting AT-SPI for the
identity. Under X11 the coordinates are complete.

Cost
----
Every property read is a D-Bus round trip, so a full tree walk is far too slow for the
reflex path -- hundreds of milliseconds on a busy desktop. It is bounded hard (depth,
node count, wall clock), runs on a worker, and is cached with a TTL. The default is off:
a sensor that stalls the loop is worse than no sensor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["A11yElement", "A11ySource", "a11y_available"]

# A tree walk is D-Bus round trips all the way down. These caps are what keep a busy
# desktop -- a browser with fifty tabs publishes a very large tree -- from turning one
# snapshot into a multi-second stall.
_MAX_NODES = 400
_MAX_DEPTH = 12
_MAX_SECONDS = 0.6

# Roles worth reporting. A full tree is mostly structural filler (panels, fillers,
# sections) that no playbook will ever reference, and including it would bury the handful
# of controls that matter.
_USEFUL_ROLES: frozenset[str] = frozenset({
    "push button", "button", "toggle button", "check box", "radio button",
    "menu item", "check menu item", "radio menu item", "menu", "menu bar",
    "text", "entry", "password text", "combo box", "list item", "list box",
    "tab", "page tab", "link", "slider", "spin button", "tree item",
    "dialog", "alert", "frame", "window", "label", "heading", "tool bar",
})


def a11y_available() -> tuple[bool, str]:
    """Whether the accessibility bus can be reached, and why not if it cannot."""
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi  # noqa: F401
    except (ImportError, ValueError) as exc:
        return False, (
            f"python gobject bindings for AT-SPI are missing ({exc}). "
            f"Arch: sudo pacman -S python-gobject at-spi2-core  |  "
            f"Debian/Ubuntu: sudo apt install python3-gi gir1.2-atspi-2.0"
        )
    try:
        from gi.repository import Atspi

        Atspi.init()
        desktop = Atspi.get_desktop(0)
        count = desktop.get_child_count()
    except Exception as exc:  # noqa: BLE001 - any bus failure is the same answer here
        return False, f"the accessibility bus is not reachable: {exc}"
    if count <= 0:
        return False, (
            "the accessibility bus is up but no application is publishing to it. "
            "Qt apps need QT_ACCESSIBILITY=1 and GTK apps GTK_MODULES=gail:atk-bridge; "
            "some desktops only start the bridge when a screen reader is running."
        )
    return True, ""


@dataclass(frozen=True, slots=True)
class A11yElement:
    """One control, as the application itself describes it."""

    name: str
    role: str
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    application: str = ""
    # False when the toolkit could not give usable screen coordinates -- routine under
    # Wayland. The element is still trustworthy for *identity* and useless for *aim*, and
    # conflating those is how a click ends up at (0, 0).
    grounded: bool = False

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "role": self.role, "app": self.application,
            "box": [self.x, self.y, self.w, self.h], "grounded": self.grounded,
        }


@dataclass(slots=True)
class A11ySource:
    """Cached, bounded snapshots of the accessibility tree."""

    ttl_s: float = 0.75
    max_nodes: int = _MAX_NODES
    enabled: bool = True
    _cached: list[A11yElement] = field(default_factory=list, init=False)
    _cached_at: float = field(default=0.0, init=False)
    _failed: str = field(default="", init=False)
    snapshots: int = field(default=0, init=False)
    last_ms: float = field(default=0.0, init=False)
    truncated: bool = field(default=False, init=False)

    def snapshot(self, *, force: bool = False) -> list[A11yElement]:
        """Current controls, from cache unless it has expired.

        Never raises. A desktop where the bus goes away mid-run should degrade to "no
        accessibility elements", not take the loop down with it.
        """
        if not self.enabled:
            return []
        now = time.monotonic()
        if not force and self._cached_at and (now - self._cached_at) < self.ttl_s:
            return self._cached
        try:
            started = time.perf_counter()
            elements = self._walk()
            self.last_ms = (time.perf_counter() - started) * 1000.0
            self.snapshots += 1
            self._cached = elements
            self._cached_at = now
            self._failed = ""
        except Exception as exc:  # noqa: BLE001 - a missing bus must not kill a run
            self._failed = f"{type(exc).__name__}: {exc}"
            self._cached = []
            self._cached_at = now
        return self._cached

    def _walk(self) -> list[A11yElement]:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        Atspi.init()
        desktop = Atspi.get_desktop(0)
        deadline = time.perf_counter() + _MAX_SECONDS
        out: list[A11yElement] = []
        self.truncated = False

        def visit(node, app_name: str, depth: int) -> None:
            if len(out) >= self.max_nodes or time.perf_counter() > deadline:
                self.truncated = True
                return
            if depth > _MAX_DEPTH:
                return
            try:
                role = node.get_role_name()
                name = (node.get_name() or "").strip()
            except Exception:  # noqa: BLE001 - a window closing mid-walk is normal
                return

            if name and role in _USEFUL_ROLES:
                x = y = w = h = 0
                grounded = False
                try:
                    ext = node.get_extents(Atspi.CoordType.SCREEN)
                    x, y, w, h = int(ext.x), int(ext.y), int(ext.width), int(ext.height)
                    # Wayland routinely reports 0x0. Treat that as "identity known,
                    # position unknown" rather than as a control at the origin.
                    grounded = w > 0 and h > 0
                except Exception:  # noqa: BLE001
                    pass
                out.append(A11yElement(
                    name=name[:80], role=role, x=x, y=y, w=w, h=h,
                    application=app_name, grounded=grounded,
                ))

            try:
                children = node.get_child_count()
            except Exception:  # noqa: BLE001
                return
            for i in range(min(children, 64)):
                try:
                    child = node.get_child_at_index(i)
                except Exception:  # noqa: BLE001
                    continue
                if child is not None:
                    visit(child, app_name, depth + 1)

        for i in range(desktop.get_child_count()):
            if len(out) >= self.max_nodes or time.perf_counter() > deadline:
                self.truncated = True
                break
            try:
                app = desktop.get_child_at_index(i)
                app_name = app.get_name() or ""
            except Exception:  # noqa: BLE001
                continue
            if app is not None:
                visit(app, app_name, 0)
        return out

    # -- queries used by guards --------------------------------------------------------

    def find(self, needle: str, *, role: str | None = None) -> A11yElement | None:
        """First control whose name contains `needle`, case-insensitively.

        Prefers a grounded match over an ungrounded one at equal quality, because a
        caller asking for an element usually intends to aim at it, and a match with real
        coordinates is strictly more useful than one without.
        """
        want = needle.strip().lower()
        if not want:
            return None
        best: A11yElement | None = None
        for element in self.snapshot():
            if role and element.role != role:
                continue
            name = element.name.lower()
            if want == name:
                if element.grounded:
                    return element
                best = best or element
            elif want in name and best is None:
                best = element
        return best

    def names(self) -> list[str]:
        return [e.name for e in self.snapshot()]

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "elements": len(self._cached),
            "grounded": sum(1 for e in self._cached if e.grounded),
            "snapshots": self.snapshots,
            "last_ms": round(self.last_ms, 1),
            "truncated": self.truncated,
            "error": self._failed,
        }
