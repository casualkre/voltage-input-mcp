"""The burst DSL: the atomic unit of actuation.

Design rationale
----------------
The whole point of this project is that a *decision* is slow (a local model round trip,
150-600 ms) while an *input* is fast (microseconds). If the actuator emitted one input
per decision, the effective input rate would equal the model rate and nothing would be
gained over the orchestrator driving a computer-use tool directly.

So the actuator does not emit an input. It emits a **burst**: an ordered, timed
programme of inputs that the executor runs at millisecond precision without any further
model involvement. One 300 ms decision can produce 40 inputs spanning 500 ms.

The surface syntax is deliberately terse rather than JSON. Decode time dominates the
actuator's latency, and terseness is the cheapest available speedup:

    JSON  {"actions":[{"type":"move","x":640,"y":360},{"type":"click","button":"left"}]}
    burst m:640,360;c:l

That is ~34 tokens versus ~6. Combined with a GBNF grammar (generated per cycle by
llm/grammar.py) that makes malformed output *structurally impossible*, a 1.7B model
becomes a reliable
emitter -- it cannot produce a token that would break the parse, so there is no retry
loop and no defensive parsing of half-formed JSON.

Grammar
-------
    burst   := action (';' action)*
    action  := verb ':' payload

    k:<chord>     key chord, press-all then release-reverse       k:ctrl+shift+t
    d:<key>       key down (held until a matching u:)             d:shift
    u:<key>       key up                                          u:shift
    t:"<text>"    type literal text                               t:"README.md"
    g:<n>         move to the centre of seen element #n           g:2
    m:<x>,<y>     absolute pointer move, screen pixels            m:640,360
    r:<dx>,<dy>   relative pointer move                           r:+10,-5
    c:<btn>[n]    click n times (default 1)                       c:l  c:l2  c:r
    p:<btn>       button down                                     p:l
    e:<btn>       button up                                       e:l
    s:<±n>        vertical scroll, n detents                      s:-3
    h:<±n>        horizontal scroll                               h:+1
    w:<ms>        wait                                            w:120

Buttons: l (left), r (right), m (middle), 4 (side/back), 5 (extra/forward).

About `g:` -- why not just use `m:`
-----------------------------------
Asking a 1.7B model to transcribe a four-digit coordinate pair it just read out of a
prompt is asking for the one thing small models are worst at: copying numbers accurately.
An off-by-one-digit `m:1240,58` instead of `m:124,58` is a click on the wrong half of the
screen, and it happens often enough to matter.

`g:2` sidesteps it entirely. The actuator picks an *index* into the elements the vision
layer just reported, and the parser resolves that to the element's centre in screen
pixels. The grammar is generated with exactly as many indices as there are elements this
cycle, so an out-of-range reference is not merely rejected -- it is unrepresentable. It
also makes `Policy.require_target_element` free: a `g:`-derived click is by construction
on something that was actually observed.

`m:` remains available for states that genuinely need raw coordinates (games, canvases,
drag paths).

Everything is normalised into `Action` objects here; nothing downstream parses strings.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..errors import BurstParseError

__all__ = [
    "Action",
    "KeyChord",
    "KeyDown",
    "KeyUp",
    "TypeText",
    "MoveAbs",
    "MoveRel",
    "Click",
    "ButtonDown",
    "ButtonUp",
    "Scroll",
    "SetAxis",
    "Wait",
    "Burst",
    "parse_burst",
    "parse_hold",
    "format_burst",
]

Button = Literal["l", "r", "m", "4", "5"]
_BUTTONS: frozenset[str] = frozenset({"l", "r", "m", "4", "5"})

# Conservative caps. These are parser-level sanity limits, not the safety policy --
# the governor (safety/governor.py) applies the real, playbook-scoped limits.
MAX_ACTIONS = 128
MAX_TEXT_LEN = 2048
MAX_WAIT_MS = 10_000
MAX_CLICK_REPEAT = 3
MAX_SCROLL = 25


@dataclass(frozen=True, slots=True)
class Action:
    """Base class. `duration_ms` is the action's own contribution to burst wall time."""

    @property
    def duration_ms(self) -> int:
        return 0

    def render(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class KeyChord(Action):
    keys: tuple[str, ...]
    hold_ms: int = 12

    @property
    def duration_ms(self) -> int:
        return self.hold_ms

    def render(self) -> str:
        return "k:" + "+".join(self.keys)


@dataclass(frozen=True, slots=True)
class KeyDown(Action):
    key: str

    def render(self) -> str:
        return f"d:{self.key}"


@dataclass(frozen=True, slots=True)
class KeyUp(Action):
    key: str

    def render(self) -> str:
        return f"u:{self.key}"


@dataclass(frozen=True, slots=True)
class TypeText(Action):
    text: str
    # Per-keystroke delay when typing via scancodes. Ignored in clipboard mode.
    interval_ms: int = 8

    @property
    def duration_ms(self) -> int:
        return len(self.text) * self.interval_ms

    def render(self) -> str:
        escaped = self.text.replace("\\", "\\\\").replace('"', '\\"')
        return f't:"{escaped}"'


@dataclass(frozen=True, slots=True)
class MoveAbs(Action):
    x: int
    y: int

    def render(self) -> str:
        return f"m:{self.x},{self.y}"


@dataclass(frozen=True, slots=True)
class MoveRel(Action):
    dx: int
    dy: int

    def render(self) -> str:
        return f"r:{self.dx:+d},{self.dy:+d}"


@dataclass(frozen=True, slots=True)
class Click(Action):
    button: Button = "l"
    count: int = 1
    # Gap between press and release, and between repeats of a multi-click.
    press_ms: int = 18
    gap_ms: int = 40

    @property
    def duration_ms(self) -> int:
        return self.count * self.press_ms + max(0, self.count - 1) * self.gap_ms

    def render(self) -> str:
        return f"c:{self.button}" + (str(self.count) if self.count > 1 else "")


@dataclass(frozen=True, slots=True)
class ButtonDown(Action):
    button: Button = "l"

    def render(self) -> str:
        return f"p:{self.button}"


@dataclass(frozen=True, slots=True)
class ButtonUp(Action):
    button: Button = "l"

    def render(self) -> str:
        return f"e:{self.button}"


@dataclass(frozen=True, slots=True)
class Scroll(Action):
    amount: int
    axis: Literal["v", "h"] = "v"
    step_ms: int = 12

    @property
    def duration_ms(self) -> int:
        return abs(self.amount) * self.step_ms

    def render(self) -> str:
        verb = "s" if self.axis == "v" else "h"
        return f"{verb}:{self.amount:+d}"


@dataclass(frozen=True, slots=True)
class SetAxis(Action):
    """Set an analog gamepad axis to a fraction of its travel, and leave it there.

    This is the verb that makes continuous control a value instead of a schedule. `a:ly,-0.7`
    is "forward at seventy percent" -- one event, and it stays set until changed. The
    keyboard equivalent is holding W, which is either full speed or nothing, and every
    latch, dead band and minimum hold time in the reflex layer exists to manage that
    binary-ness. None of that is needed here.
    """

    axis: str
    value: float

    def render(self) -> str:
        return f"a:{self.axis},{self.value:+.2f}"


@dataclass(frozen=True, slots=True)
class Wait(Action):
    ms: int

    @property
    def duration_ms(self) -> int:
        return self.ms

    def render(self) -> str:
        return f"w:{self.ms}"


@dataclass(frozen=True, slots=True)
class Burst:
    """A parsed, validated programme of inputs."""

    actions: tuple[Action, ...] = field(default_factory=tuple)
    source: str = ""

    def __len__(self) -> int:
        return len(self.actions)

    def __bool__(self) -> bool:
        return bool(self.actions)

    def __iter__(self):
        return iter(self.actions)

    @property
    def duration_ms(self) -> int:
        """Estimated wall time. The governor budgets against this before execution."""
        return sum(a.duration_ms for a in self.actions)

    @property
    def input_count(self) -> int:
        """Number of real input events, excluding pure waits. Used for rate limiting."""
        return sum(1 for a in self.actions if not isinstance(a, Wait))

    def render(self) -> str:
        return ";".join(a.render() for a in self.actions)

    def held_keys(self) -> set[str]:
        """Keys left down at the end of the burst -- the executor must release these."""
        held: set[str] = set()
        for a in self.actions:
            if isinstance(a, KeyDown):
                held.add(a.key)
            elif isinstance(a, KeyUp):
                held.discard(a.key)
        return held

    def held_buttons(self) -> set[str]:
        held: set[str] = set()
        for a in self.actions:
            if isinstance(a, ButtonDown):
                held.add(a.button)
            elif isinstance(a, ButtonUp):
                held.discard(a.button)
        return held


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

_KEY_NAME = re.compile(r"^[a-z0-9_]{1,16}$")
_INT = re.compile(r"^[+-]?\d{1,6}$")


def _split_actions(src: str) -> list[str]:
    """Split on ';' while respecting quoted text payloads.

    A naive `src.split(";")` breaks on `t:"a;b"`, which is exactly the kind of input a
    small model produces when the task involves punctuation.
    """
    parts: list[str] = []
    buf: list[str] = []
    in_quotes = False
    escaped = False
    for ch in src:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_quotes:
            buf.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            continue
        if ch == ";" and not in_quotes:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if in_quotes:
        raise BurstParseError("unterminated quoted text payload", source=src)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _unescape(text: str) -> str:
    out: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(ch, ch))
            escaped = False
        elif ch == "\\":
            escaped = True
        else:
            out.append(ch)
    return "".join(out)


def _int_arg(raw: str, what: str, lo: int, hi: int) -> int:
    if not _INT.match(raw):
        raise BurstParseError(f"{what} must be an integer, got {raw!r}")
    value = int(raw)
    if not lo <= value <= hi:
        raise BurstParseError(f"{what} out of range [{lo}, {hi}]: {value}")
    return value


def _key_name(raw: str) -> str:
    """Validate a key name against the actual keymap, not just its shape.

    Resolving here rather than deferring to the governor means a typo in a reflex burst
    is caught by `validate_playbook` instead of surfacing as a mysterious refusal at
    cycle 30 of a live run.
    """
    from ..inputs.keymap import resolve_key

    key = raw.strip().lower()
    if not _KEY_NAME.match(key):
        raise BurstParseError(f"invalid key name {raw!r}")
    if resolve_key(key) is None:
        raise BurstParseError(
            f"unknown key {raw!r}; use names like enter, esc, tab, space, ctrl, shift, "
            f"alt, meta, f1-f12, up/down/left/right, or a single letter or digit"
        )
    return key


def _button(raw: str) -> Button:
    btn = raw.strip().lower()
    if btn not in _BUTTONS:
        raise BurstParseError(f"invalid button {raw!r}; expected one of {sorted(_BUTTONS)}")
    return btn  # type: ignore[return-value]


def _pair(payload: str, what: str, lo: int, hi: int) -> tuple[int, int]:
    if "," not in payload:
        raise BurstParseError(f"{what} needs 'x,y', got {payload!r}")
    a, _, b = payload.partition(",")
    return _int_arg(a.strip(), f"{what} x", lo, hi), _int_arg(b.strip(), f"{what} y", lo, hi)


def parse_burst(
    src: str,
    *,
    screen: tuple[int, int] | None = None,
    elements: Sequence[tuple[int, int]] | None = None,
) -> Burst:
    """Parse a burst string.

    `screen` is (width, height). When given, absolute moves are range-checked against it;
    this catches a model that emitted normalised 0-1000 coordinates instead of pixels.

    `elements` is the list of (centre_x, centre_y) points that `g:<n>` indexes into --
    normally the centres of this cycle's observed elements, in the same order they were
    shown to the actuator. A `g:` with no `elements` supplied is an error rather than a
    silent no-op, because a burst that meant to click something and instead clicked
    nowhere is worse than a refused cycle.
    """
    src = (src or "").strip()
    if not src or src == ".":
        return Burst(actions=(), source=src)

    max_x, max_y = (screen[0] - 1, screen[1] - 1) if screen else (65535, 65535)

    actions: list[Action] = []
    for token in _split_actions(src):
        verb, sep, payload = token.partition(":")
        verb = verb.strip().lower()
        if not sep:
            raise BurstParseError(f"action {token!r} is missing ':'", source=src)
        payload = payload.strip()

        if verb == "k":
            # A chord is a set: `k:shift+shift` is not "shift, harder", it is `k:shift`
            # with two tokens wasted. Small models do produce this -- observed live from a
            # 1.7B as `k:shift+shift+shift+shift` -- and executing it verbatim means a
            # redundant press and a redundant release around the real one. Deduplicating
            # in first-seen order keeps modifier ordering, which the executor relies on to
            # release in reverse.
            seen_keys: dict[str, None] = {}
            for part in payload.split("+"):
                if part.strip():
                    seen_keys.setdefault(_key_name(part), None)
            keys = tuple(seen_keys)
            if not keys:
                raise BurstParseError("empty key chord", source=src)
            if len(keys) > 5:
                raise BurstParseError(f"key chord too long ({len(keys)} keys)", source=src)
            actions.append(KeyChord(keys=keys))

        elif verb == "d":
            actions.append(KeyDown(key=_key_name(payload)))

        elif verb == "u":
            actions.append(KeyUp(key=_key_name(payload)))

        elif verb == "t":
            if len(payload) < 2 or payload[0] != '"' or payload[-1] != '"':
                raise BurstParseError(f'text must be double-quoted, got {payload!r}', source=src)
            text = _unescape(payload[1:-1])
            if len(text) > MAX_TEXT_LEN:
                raise BurstParseError(f"text too long ({len(text)} > {MAX_TEXT_LEN})", source=src)
            actions.append(TypeText(text=text))

        elif verb == "m":
            # Parse with a generous range, then range-check against the screen here, so
            # the diagnostic below is actually reachable. Checking inside _pair with
            # max(max_x, max_y) hid it on every landscape display, which is all of them.
            x, y = _pair(payload, "move", 0, 99_999)
            if screen and (x > max_x or y > max_y):
                raise BurstParseError(
                    f"absolute move ({x},{y}) is outside the {screen[0]}x{screen[1]} screen; "
                    "emit screen pixels, not normalised coordinates",
                    source=src,
                )
            actions.append(MoveAbs(x=x, y=y))

        elif verb == "g":
            index = _int_arg(payload, "element index", 0, 63)
            if not elements:
                raise BurstParseError(
                    f"g:{index} references an observed element, but no elements were "
                    "reported this cycle",
                    source=src,
                )
            if index >= len(elements):
                raise BurstParseError(
                    f"g:{index} is out of range; only {len(elements)} element(s) were "
                    "reported this cycle",
                    source=src,
                )
            ex, ey = elements[index]
            actions.append(MoveAbs(x=int(ex), y=int(ey)))

        elif verb == "r":
            dx, dy = _pair(payload, "relative move", -20000, 20000)
            actions.append(MoveRel(dx=dx, dy=dy))

        elif verb == "c":
            # Payload is <button>[count]: "l", "l2", "r", "4", "42".
            # Buttons are single characters, so a 2-char payload is always button+count.
            if len(payload) == 2 and payload[1].isdigit():
                btn, count = _button(payload[0]), int(payload[1])
            elif len(payload) == 1:
                btn, count = _button(payload), 1
            else:
                raise BurstParseError(
                    f"click payload {payload!r} must be <button>[count], e.g. 'l' or 'l2'",
                    source=src,
                )
            if not 1 <= count <= MAX_CLICK_REPEAT:
                raise BurstParseError(
                    f"click repeat {count} out of range [1, {MAX_CLICK_REPEAT}]", source=src
                )
            actions.append(Click(button=btn, count=count))

        elif verb == "p":
            actions.append(ButtonDown(button=_button(payload)))

        elif verb == "e":
            actions.append(ButtonUp(button=_button(payload)))

        elif verb == "a":
            from ..inputs.keymap import PAD_AXES

            if "," not in payload:
                raise BurstParseError(
                    f"axis needs '<axis>,<value>', got {payload!r}", source=src
                )
            axis, _, raw = payload.partition(",")
            axis = axis.strip().lower()
            if axis not in PAD_AXES:
                raise BurstParseError(
                    f"unknown axis {axis!r}; valid: {', '.join(sorted(PAD_AXES))} "
                    f"(lx/ly left stick, rx/ry right stick, lt/rt triggers, dx/dy d-pad)",
                    source=src,
                )
            try:
                value = float(raw.strip())
            except ValueError:
                raise BurstParseError(
                    f"axis value must be a number between -1 and 1, got {raw!r}",
                    source=src,
                ) from None
            if not -1.0 <= value <= 1.0:
                raise BurstParseError(
                    f"axis value {value} is outside -1.0..1.0; it is a fraction of the "
                    f"stick's travel, not a raw kernel value",
                    source=src,
                )
            actions.append(SetAxis(axis=axis, value=value))

        elif verb in ("s", "h"):
            amount = _int_arg(payload, "scroll", -MAX_SCROLL, MAX_SCROLL)
            if amount == 0:
                continue  # a no-op scroll is not an error, just drop it
            actions.append(Scroll(amount=amount, axis="v" if verb == "s" else "h"))

        elif verb == "w":
            ms = _int_arg(payload, "wait", 0, MAX_WAIT_MS)
            if ms == 0:
                continue
            actions.append(Wait(ms=ms))

        else:
            raise BurstParseError(
                f"unknown verb {verb!r} in {token!r}; valid verbs are "
                f"k d u t g m r c p e s h w a",
                source=src,
            )

        if len(actions) > MAX_ACTIONS:
            raise BurstParseError(f"burst exceeds {MAX_ACTIONS} actions", source=src)

    return Burst(actions=tuple(actions), source=src)


def parse_hold(src: str) -> tuple[Burst, Burst, tuple[str, ...]]:
    """Parse a hold spec into `(press, release, names)`.

    A hold spec is a comma-separated list of things to hold down for as long as a
    condition lasts::

        w                    hold the W key
        shift, w             hold both
        btn:r                hold the right mouse button (mouse-look in most games)
        shift, btn:l         shift-drag, held open across many frames

    Returning bursts rather than raw key codes is the point. It means a latch reaches the
    device through exactly the same path as everything else -- governor review, verb
    allowlist, deny_keys, rate limit, held-key bookkeeping, panic release. A latch that
    tried to hold `leftmeta` is refused for the same reason `k:leftmeta` is, and by the
    same code, rather than by a second policy check written specially for latches and
    subtly different from the first.

    Release order is the reverse of press order, so a modifier outlives what it modified.
    """
    press: list[Action] = []
    release: list[Action] = []
    names: list[str] = []
    seen: set[str] = set()

    for raw in src.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token.startswith("btn:"):
            button = _button(token[4:])
            name = f"btn:{button}"
            if name in seen:
                raise BurstParseError(f"{name} is held twice in {src!r}", source=src)
            press.append(ButtonDown(button=button))
            release.append(ButtonUp(button=button))
        else:
            key = _key_name(token)
            name = key
            if name in seen:
                raise BurstParseError(f"{key} is held twice in {src!r}", source=src)
            press.append(KeyDown(key=key))
            release.append(KeyUp(key=key))
        seen.add(name)
        names.append(name)

    if not press:
        raise BurstParseError(
            "hold spec is empty; write something like 'w' or 'shift, btn:r'", source=src
        )
    if len(press) > 6:
        raise BurstParseError(
            f"a hold spec may cover at most 6 keys or buttons, got {len(press)}", source=src
        )

    release.reverse()
    return (
        Burst(actions=tuple(press), source=format_burst(press)),
        Burst(actions=tuple(release), source=format_burst(release)),
        tuple(names),
    )


def format_burst(actions: list[Action]) -> str:
    return ";".join(a.render() for a in actions)
