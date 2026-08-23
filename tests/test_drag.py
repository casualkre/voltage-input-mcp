"""Multi-input composition: drags, held modifiers, and chorded input.

These are the shapes that decide whether this works against real software. A trace can
look structurally correct and still fail every application, which is exactly what a
teleporting drag did.
"""

from __future__ import annotations

from voltage_input_mcp.inputs.executor import Executor
from voltage_input_mcp.models.burst import parse_burst

SCREEN = (1920, 1080)


class Sink:
    """Records what reaches the platform layer."""

    screen = SCREEN
    is_open = True

    def __init__(self):
        self.events: list[tuple] = []

    def open(self): ...
    def close(self): ...
    def key(self, name, down): self.events.append(("key", name, down))
    def button(self, name, down): self.events.append(("btn", name, down))
    def move_abs(self, x, y): self.events.append(("abs", x, y))
    def move_rel(self, dx, dy): self.events.append(("rel", dx, dy))
    def scroll(self, amount, axis="v"): self.events.append(("scroll", amount, axis))


def emit(src: str, elements=None) -> list[tuple]:
    sink = Sink()
    burst = parse_burst(src, screen=SCREEN, elements=elements)
    Executor(sink, dry_run=False).run(burst)
    return sink.events


def moves(events) -> list[tuple]:
    return [e for e in events if e[0] in ("abs", "rel")]


# -- drags -----------------------------------------------------------------------------


def test_drag_produces_a_motion_stream_not_a_teleport():
    """An application decides a drag happened by watching motion between the buttons."""
    events = emit("m:200,200;p:l;m:900,700;e:l")
    between = [
        e for i, e in enumerate(events)
        if e[0] == "abs"
        and any(x[0] == "btn" and x[2] for x in events[:i])
        and any(x[0] == "btn" and not x[2] for x in events[i:])
    ]
    assert len(between) >= 6, f"only {len(between)} motion points inside the drag"


def test_drag_starts_near_the_origin_and_ends_exactly_on_target():
    """The first small step is what crosses an application's drag threshold."""
    events = emit("m:200,200;p:l;m:900,700;e:l")
    inside = [e for e in events if e[0] == "abs"][1:]
    first, last = inside[0], inside[-1]
    assert abs(first[1] - 200) < 200 and abs(first[2] - 200) < 200
    assert (last[1], last[2]) == (900, 700), "a drag must land exactly on its target"


def test_motion_is_monotonic_towards_the_target():
    events = [e for e in emit("m:100,100;p:l;m:800,100;e:l") if e[0] == "abs"]
    xs = [e[1] for e in events]
    assert xs == sorted(xs), "the pointer must not jitter backwards during a drag"


def test_no_interpolation_when_nothing_is_held():
    """Interpolation costs time; a plain move should stay a single jump."""
    assert len(moves(emit("m:200,200;m:900,700"))) == 2


def test_short_drag_is_not_over_interpolated():
    events = emit("m:200,200;p:l;m:201,201;e:l")
    assert len(moves(events)) <= 3


# -- held modifiers --------------------------------------------------------------------


def test_shift_drag_holds_shift_across_the_whole_gesture():
    events = emit("d:shift;g:0;p:l;g:1;e:l;u:shift", elements=[(200, 200), (900, 700)])
    shift_down = next(i for i, e in enumerate(events) if e[:2] == ("key", "leftshift") and e[2])
    shift_up = next(i for i, e in enumerate(events) if e[:2] == ("key", "leftshift") and not e[2])
    btn_down = next(i for i, e in enumerate(events) if e[0] == "btn" and e[2])
    btn_up = next(i for i, e in enumerate(events) if e[0] == "btn" and not e[2])
    assert shift_down < btn_down < btn_up < shift_up


def test_ctrl_click_multi_select():
    events = emit("d:ctrl;g:0;c:l;g:1;c:l;u:ctrl", elements=[(100, 100), (300, 300)])
    clicks = [e for e in events if e[0] == "btn"]
    assert len(clicks) == 4                       # two full press/release pairs
    assert events[0] == ("key", "leftctrl", True)
    assert events[-1] == ("key", "leftctrl", False)


def test_chord_releases_modifiers_in_reverse_order():
    """Modifiers must outlive the key they modify."""
    events = emit("k:ctrl+shift+t")
    downs = [e[1] for e in events if e[0] == "key" and e[2]]
    ups = [e[1] for e in events if e[0] == "key" and not e[2]]
    assert downs == ["leftctrl", "leftshift", "t"]
    assert ups == ["t", "leftshift", "leftctrl"]


def test_held_keys_survive_to_the_next_burst_and_are_tracked():
    sink = Sink()
    ex = Executor(sink, dry_run=False)
    ex.run(parse_burst("d:shift", screen=SCREEN))
    # Names are canonicalised: `shift` is tracked as `leftshift`, so a later
    # `u:shift` and the panic release both refer to the same physical key.
    assert ex.held()[0] == ["leftshift"]
    released = ex.release_all()
    assert "leftshift" in released
    assert ex.held() == ([], [])


def test_game_style_strafe_while_attacking():
    """Movement held across clicks -- the shape most games need."""
    events = emit("d:a;w:60;c:l;w:60;c:l;u:a")
    assert events[0] == ("key", "a", True)
    assert events[-1] == ("key", "a", False)
    assert len([e for e in events if e[0] == "btn" and e[2]]) == 2


def test_relative_drag_also_interpolates():
    """Camera drags in games are relative; a single big delta is still a teleport."""
    events = emit("m:500,500;p:l;r:+400,+0;e:l")
    assert len([e for e in events if e[0] == "rel"]) >= 6
