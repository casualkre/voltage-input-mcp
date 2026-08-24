"""Analog input: the verb that makes continuous control a value, not a schedule.

Every hard problem in the reflex layer -- latches, hysteresis, chatter, minimum hold
times, interpolated steering bursts -- exists because a key is binary. "Forward at 70%"
cannot be expressed with W, so it has to be forged out of timing. An axis is a magnitude,
and the whole category goes away.
"""

from __future__ import annotations

import pytest

from voltage_input_mcp.errors import BurstParseError
from voltage_input_mcp.inputs import keymap as km
from voltage_input_mcp.inputs.executor import Executor
from voltage_input_mcp.models.burst import SetAxis, parse_burst


class PadSink:
    screen = (1920, 1080)
    is_open = True

    def __init__(self):
        self.axes: list[tuple[str, float]] = []

    def open(self): ...
    def close(self): ...
    def key(self, name, down): ...
    def button(self, name, down): ...
    def move_abs(self, x, y): ...
    def move_rel(self, dx, dy): ...
    def scroll(self, amount, axis="v"): ...
    def axis(self, name, value): self.axes.append((name, value))


class NoPadSink(PadSink):
    """A sink with no gamepad -- Windows has no virtual pad without a driver."""

    axis = None

    def __init__(self):
        super().__init__()
        del self.axes


# -- the verb ----------------------------------------------------------------------------


def test_an_axis_is_a_magnitude_not_a_keypress():
    burst = parse_burst("a:ly,-0.7")
    assert len(burst.actions) == 1
    action = burst.actions[0]
    assert isinstance(action, SetAxis)
    assert action.axis == "ly" and action.value == pytest.approx(-0.7)


def test_axes_round_trip_through_render():
    src = "a:lx,+0.50;a:ly,-1.00;a:rt,+0.25"
    assert parse_burst(parse_burst(src).render()).render() == src


@pytest.mark.parametrize("bad,why", [
    ("a:zz,0.5", "unknown axis"),
    ("a:lx,5", "outside"),
    ("a:lx,-2", "outside"),
    ("a:lx", "needs"),
    ("a:lx,abc", "must be a number"),
])
def test_malformed_axis_requests_are_refused(bad, why):
    with pytest.raises(BurstParseError, match=why):
        parse_burst(bad)


# -- kernel mapping ------------------------------------------------------------------------


def test_sticks_are_signed_and_centred_but_triggers_are_not():
    assert km.pad_axis_value("lx", 0.0) == 0
    assert km.pad_axis_value("lx", 1.0) == km.STICK_MAX
    assert km.pad_axis_value("lx", -1.0) == km.STICK_MIN
    assert km.pad_axis_value("lt", 0.0) == 0
    assert km.pad_axis_value("lt", 1.0) == km.TRIGGER_MAX


def test_a_negative_trigger_clamps_to_zero_rather_than_wrapping():
    """A trigger has no negative half; wrapping would read as a full pull."""
    assert km.pad_axis_value("rt", -0.8) == 0


def test_the_dpad_is_a_three_state_axis():
    assert km.pad_axis_value("dx", 0.9) == 1
    assert km.pad_axis_value("dx", -0.9) == -1
    assert km.pad_axis_value("dx", 0.1) == 0


def test_each_axis_declares_the_range_the_kernel_expects():
    """Declaring one range for every axis makes a pad rest at full deflection."""
    from voltage_input_mcp.inputs.uinput import gamepad_caps

    caps = gamepad_caps()
    assert caps.abs_range(km.ABS_X) == (km.STICK_MIN, km.STICK_MAX)
    assert caps.abs_range(km.ABS_Z) == (km.TRIGGER_MIN, km.TRIGGER_MAX)
    assert caps.abs_range(km.ABS_HAT0X) == (-1, 1)


def test_the_pad_advertises_a_gamepad_button_so_the_kernel_classifies_it():
    """udev tags a device as a joystick only if it has ABS axes *and* a BTN_GAMEPAD key.

    Without that it is an unclassifiable absolute device, invisible to SDL and to games.
    Verified live: the kernel gave it `js0`.
    """
    from voltage_input_mcp.inputs.uinput import gamepad_caps

    assert km.BTN_SOUTH in gamepad_caps().keys
    assert km.ABS_X in gamepad_caps().abs_axes


# -- execution -----------------------------------------------------------------------------


def test_setting_an_axis_reaches_the_device():
    sink = PadSink()
    Executor(sink, dry_run=False).run(parse_burst("a:ly,-0.7;a:rt,0.5"))
    assert sink.axes == [("ly", pytest.approx(-0.7)), ("rt", pytest.approx(0.5))]


def test_dry_run_touches_nothing():
    sink = PadSink()
    Executor(sink, dry_run=True).run(parse_burst("a:ly,-1.0"))
    assert sink.axes == []


def test_release_all_centres_every_deflected_axis():
    """A stick left at 0.8 is the analog stuck key, and worse: it is invisible.

    A held key at least shows up in `held()`. A deflected stick looks like nothing at
    all while the character keeps walking into a wall.
    """
    sink = PadSink()
    ex = Executor(sink, dry_run=False)
    ex.run(parse_burst("a:lx,+0.8;a:ly,-0.4"))
    released = ex.release_all()
    assert set(released) >= {"axis:lx", "axis:ly"}
    assert sink.axes[-2:] == [("lx", 0.0), ("ly", 0.0)]


def test_an_axis_that_was_already_centred_is_not_reported_as_released():
    sink = PadSink()
    ex = Executor(sink, dry_run=False)
    ex.run(parse_burst("a:lx,0.0"))
    assert "axis:lx" not in ex.release_all()


def test_a_sink_without_a_gamepad_degrades_instead_of_crashing():
    """Windows has no virtual pad without a third-party driver."""
    sink = NoPadSink()
    report = Executor(sink, dry_run=False).run(parse_burst("a:ly,-1.0"))
    assert report.ok


def test_the_axis_verb_is_policy_gated_like_every_other():
    """`a` has to be in the verb table or a policy cannot allow or deny it."""
    from voltage_input_mcp.models.playbook import VERB_NAMES, VERBS

    assert "a" in VERBS
    assert "a" in VERB_NAMES
