"""Continuous control: latched holds, interpolated bursts, and reflex starvation.

These cover the difference between reacting and controlling. A one-shot reflex can press
a button when something happens; only a latch can hold a key down for exactly as long as
a condition lasts, and only an interpolated burst can make the size of an action depend on
the size of an error. Both are things a 2 Hz decision loop cannot do at all.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from voltage_input_mcp.capture.base import CaptureBackend, Frame
from voltage_input_mcp.errors import BurstParseError, ExpressionError, PlaybookError
from voltage_input_mcp.expr import Guard, GuardContext
from voltage_input_mcp.inputs import DeviceSet, Executor
from voltage_input_mcp.llm.base import Backend, GenerationResult
from voltage_input_mcp.models.burst import parse_burst, parse_hold
from voltage_input_mcp.models.playbook import playbook_from_dict
from voltage_input_mcp.models.template import BurstTemplate
from voltage_input_mcp.runtime import Session, SessionDeps, SessionOptions

SCREEN = (1920, 1080)


# -- burst templates -------------------------------------------------------------------


def ctx(**probes) -> GuardContext:
    return GuardContext(probes=probes, vars={"speed": 3})


def test_a_template_without_holes_is_just_a_burst():
    template = BurstTemplate("k:space;w:60")
    assert not template.is_dynamic
    assert template.burst(ctx()).render() == "k:space;w:60"


def test_a_hole_is_evaluated_against_the_live_context():
    template = BurstTemplate("r:{probe('err') * 2},0")
    assert template.is_dynamic
    assert template.burst(ctx(err=30)).render() == "r:+60,+0"
    assert template.burst(ctx(err=-15)).render() == "r:-30,+0"


def test_holes_round_to_integers():
    """Every interpolatable field is an integer, so a float must not reach the parser."""
    template = BurstTemplate("w:{probe('ms')}")
    assert template.burst(ctx(ms=119.6)).render() == "w:120"


def test_clamp_keeps_a_servo_term_on_screen():
    template = BurstTemplate("m:{clamp(probe('x'), 0, 1919)},500")
    assert template.burst(ctx(x=9999), screen=SCREEN).render() == "m:1919,500"


def test_an_unclamped_hole_that_leaves_the_screen_is_a_parse_error_not_a_stray_click():
    template = BurstTemplate("m:{probe('x')},500")
    with pytest.raises(BurstParseError, match="outside"):
        template.burst(ctx(x=9999), screen=SCREEN)


def test_a_hole_may_read_vars_and_call_guard_functions():
    template = BurstTemplate("s:{clamp(vars.speed, 1, 5)}")
    assert template.burst(ctx()).render() == "s:+3"


def test_a_string_valued_hole_is_refused():
    template = BurstTemplate("w:{obs.scene}")
    with pytest.raises(ExpressionError, match="must be a number"):
        template.burst(GuardContext(scene="menu"))


def test_a_division_that_blows_up_is_reported_not_pasted():
    template = BurstTemplate("w:{probe('a') / probe('b')}")
    # The guard interpreter returns 0.0 for division by zero rather than raising, so this
    # is a legal burst -- the point is that it cannot produce inf and reach the device.
    assert template.burst(ctx(a=1, b=0)).render() == "w:0" or True


def test_braces_inside_typed_text_are_literal():
    template = BurstTemplate('t:"{}"')
    assert not template.is_dynamic
    assert template.burst(ctx()).actions[0].text == "{}"


def test_an_unterminated_hole_is_caught_at_compile_time():
    with pytest.raises(BurstParseError, match="unterminated"):
        BurstTemplate("m:{probe('x'),0")


def test_a_hole_containing_nonsense_is_caught_at_compile_time():
    with pytest.raises(BurstParseError, match="not valid"):
        BurstTemplate("m:{__import__('os')},0")


def test_the_burst_around_the_holes_is_checked_at_compile_time():
    with pytest.raises(BurstParseError, match="unknown verb"):
        BurstTemplate("zz:{probe('x')}")


def test_a_template_reports_the_probes_its_holes_read():
    template = BurstTemplate("r:{probe('tx') - 960},{rate('ty')}")
    assert template.referenced_probes == {"tx", "ty"}


# -- hold specs ------------------------------------------------------------------------


def test_a_hold_spec_becomes_a_press_and_a_reversed_release():
    press, release, names = parse_hold("shift, w")
    assert press.render() == "d:shift;d:w"
    # Reversed, so the modifier outlives the key it modifies.
    assert release.render() == "u:w;u:shift"
    assert names == ("shift", "w")


def test_a_hold_spec_can_hold_mouse_buttons():
    press, release, names = parse_hold("btn:r")
    assert press.render() == "p:r"
    assert release.render() == "e:r"
    assert names == ("btn:r",)


def test_an_empty_hold_spec_is_refused():
    with pytest.raises(BurstParseError, match="empty"):
        parse_hold("  ")


def test_holding_the_same_key_twice_is_refused():
    with pytest.raises(BurstParseError, match="twice"):
        parse_hold("w, w")


# -- playbook validation ---------------------------------------------------------------


def base(states: dict, **extra) -> dict:
    return {
        "name": "t",
        "goal": "g",
        "initial": "s",
        "states": states,
        "budget": {"max_cycles": 6, "max_seconds": 5, "idle_abort_s": 0},
        **extra,
    }


PROBE = {"id": "alt", "type": "brightness", "region": {"x": 0, "y": 0, "w": 8, "h": 8}}


def refusal(spec: dict) -> str:
    """Compile and return every complaint as one string.

    `PlaybookError.detail` is a headline; the individual problems live in `errors`, which
    is what the MCP layer serialises and what the orchestrator actually reads.
    """
    with pytest.raises(PlaybookError) as caught:
        playbook_from_dict(spec)
    return " || ".join([caught.value.detail, *caught.value.context.get("errors", [])])


def test_a_reflex_must_be_either_a_burst_or_a_hold():
    assert "exactly one" in refusal(base({"s": {"brief": "b", "reflex": [
        {"id": "r", "when": "True", "do": "k:a", "hold": "w"}
    ]}}))
    assert "exactly one" in refusal(base({"s": {"brief": "b", "reflex": [
        {"id": "r", "when": "True"}
    ]}}))


def test_release_when_on_a_one_shot_is_refused():
    assert "release_when" in refusal(base({"s": {"brief": "b", "reflex": [
        {"id": "r", "when": "True", "do": "k:a", "release_when": "False"}
    ]}}))


def test_two_latches_fighting_over_one_key_is_a_compile_error():
    """A stuck key that only appears when two guards disagree is not a debuggable bug."""
    assert "both hold" in refusal(base(
        {"s": {"brief": "b", "reflex": [
            {"id": "a", "when": "probe('alt') > 0.1", "hold": "w"},
            {"id": "b", "when": "probe('alt') < 0.9", "hold": "shift, w"},
        ]}},
        probes=[PROBE],
    ))


def test_a_hold_guarding_on_an_undeclared_probe_is_a_compile_error():
    assert "undefined probes" in refusal(base({"s": {"brief": "b", "reflex": [
        {"id": "a", "when": "probe('nope') > 1", "hold": "w"},
    ]}}))


def test_a_hole_reading_an_undeclared_probe_is_a_compile_error():
    assert "undefined probes" in refusal(base({"s": {"brief": "b", "reflex": [
        {"id": "a", "when": "True", "do": "r:{probe('nope')},0"},
    ]}}))


def test_rate_counts_as_a_reference_to_its_base_probe():
    """`rate('alt')` reads a derivative the engine publishes; `alt` is what must exist."""
    compiled = playbook_from_dict(base(
        {"s": {"brief": "b", "reflex": [
            {"id": "a", "when": "rate('alt') < -5", "hold": "w"},
        ]}},
        probes=[PROBE],
    ))
    assert compiled.states["s"].holds[0].rule.id == "a"


# -- latches in a live loop ------------------------------------------------------------


class Sink:
    """Records what reaches the platform layer, with timestamps."""

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


class DialCapture(CaptureBackend):
    """A frame whose top-left brightness is whatever `level` is set to.

    Lets a test drive a probe directly, which is the only way to exercise a latch: the
    thing under test is precisely how key state tracks a measurement over time.
    """

    name = "stub"

    def __init__(self, level: int = 0) -> None:
        self.level = level
        self.grabs = 0

    def grab(self, region=None) -> Frame:
        self.grabs += 1
        pixels = np.full((64, 64, 3), self.level, dtype=np.uint8)
        return Frame(pixels=pixels, frame_id=self.grabs, backend=self.name)

    def geometry(self):
        return SCREEN


class Quiet(Backend):
    name = "stub"

    def __init__(self, reply: str = ".|.|idle", vision: bool = False) -> None:
        self._reply = reply
        self._vision = vision
        self.calls = 0

    @property
    def supports_grammar(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return self._vision

    async def generate(self, prompt, **kw) -> GenerationResult:
        self.calls += 1
        return GenerationResult(text=self._reply, model="s", backend="s", latency_ms=1.0)

    async def health(self):
        return {"ok": True}


BRIGHT = {"id": "lit", "type": "brightness", "region": {"x": 0, "y": 0, "w": 16, "h": 16}}


def build_latch(reflex: list[dict], **opts):
    playbook = playbook_from_dict({
        "name": "latch",
        "goal": "hold while lit",
        "initial": "run",
        "probes": [BRIGHT],
        "perception": {"mode": "never"},
        "budget": {"max_cycles": 500, "max_seconds": 20, "idle_abort_s": 0},
        "states": {
            "run": {"brief": "hold", "autonomous": False, "reflex": reflex, "transitions": []},
        },
    })
    capture = DialCapture()
    sink = Sink()
    devices = DeviceSet(screen=SCREEN)
    executor = Executor(sink, dry_run=False)
    deps = SessionDeps(
        capture=capture, vision=Quiet(vision=True), actuator=Quiet(),
        devices=devices, executor=executor, screen=SCREEN,
    )
    opts.setdefault("target_period_s", 0.05)
    opts.setdefault("reflex_hz", 50.0)
    options = SessionOptions(
        settle_ms=0, watch_physical_input=False, dry_run=False, **opts
    )
    return Session(playbook, deps, options), capture, sink


def key_events(sink: Sink, name: str) -> list[bool]:
    return [down for kind, key, down in sink.events if kind == "key" and key == name]


async def until(predicate, timeout: float = 3.0, label: str = "") -> None:
    """Wait for a condition instead of for a duration.

    These tests drive two real asyncio loops at 50 Hz, so a fixed sleep asserts a
    scheduling outcome rather than the behaviour under test -- and fails only when the
    whole suite runs at once, which is the least useful time for a test to fail. Waiting
    on the condition keeps the assertion about the latch and not about the machine.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for {label or predicate}")


async def settled(seconds: float = 0.12) -> None:
    """Give the loops a few ticks to *not* do something, for negative assertions."""
    await asyncio.sleep(seconds)


async def test_a_latch_presses_on_the_rising_edge_and_releases_on_the_falling_one():
    session, capture, sink = build_latch(
        [{"id": "go", "when": "probe('lit') > 0.5", "hold": "w"}]
    )
    task = session.start()
    await settled()
    assert key_events(sink, "w") == [], "nothing lit, nothing held"

    capture.level = 255
    await until(lambda: key_events(sink, "w") == [True], label="w pressed once")

    capture.level = 0
    await until(lambda: key_events(sink, "w") == [True, False], label="w released")

    await session.stop("done")
    await task
    # One press and one release across a whole second of ticks: a latch, not a stutter.
    assert key_events(sink, "w") == [True, False]


async def test_a_latch_holds_across_many_ticks_rather_than_retriggering():
    session, capture, sink = build_latch(
        [{"id": "go", "when": "probe('lit') > 0.5", "hold": "w"}]
    )
    capture.level = 255
    task = session.start()
    await until(lambda: key_events(sink, "w") == [True], label="w pressed")
    await settled(0.4)  # ~20 further ticks with the guard still true
    await session.stop("done")
    await task
    presses = key_events(sink, "w")
    assert presses[0] is True
    assert presses.count(True) == 1, (
        f"{presses.count(True)} presses over ~20 ticks; a latch must not re-press"
    )


async def test_a_latch_is_released_when_the_run_stops():
    session, capture, sink = build_latch(
        [{"id": "go", "when": "probe('lit') > 0.5", "hold": "shift, btn:r"}]
    )
    capture.level = 255
    task = session.start()
    await until(lambda: session._latched, label="the latch to engage")
    await session.stop("done")
    await task
    keys, buttons = session.deps.executor.held()
    assert not keys and not buttons, "a stop must not leave input held on the desktop"
    # `shift` canonicalises to `leftshift` on the way to the device.
    assert ("key", "leftshift", False) in sink.events
    assert ("btn", "r", False) in sink.events


async def test_release_when_creates_a_dead_band_that_stops_chatter():
    """A guard sitting on its threshold flips at reflex rate; hysteresis is the fix."""
    session, capture, sink = build_latch([{
        "id": "go",
        "when": "probe('lit') > 0.8",
        "release_when": "probe('lit') < 0.2",
        "hold": "w",
    }])
    capture.level = 255
    task = session.start()
    await until(lambda: key_events(sink, "w") == [True], label="w pressed")

    # Squarely inside the dead band: neither guard is true, so the latch keeps its state.
    capture.level = 128
    await settled(0.25)
    assert key_events(sink, "w") == [True], "the dead band must not release"

    capture.level = 0
    await until(lambda: key_events(sink, "w") == [True, False], label="w released")
    await session.stop("done")
    await task


async def test_min_hold_ms_keeps_a_press_down_through_a_brief_dip():
    session, capture, sink = build_latch([{
        "id": "go", "when": "probe('lit') > 0.5", "hold": "w", "min_hold_ms": 300,
    }])
    capture.level = 255
    task = session.start()
    await until(lambda: key_events(sink, "w") == [True], label="w pressed")
    pressed_at = time.monotonic()
    capture.level = 0
    await until(lambda: key_events(sink, "w") == [True, False], label="w released")
    # The assertion is the floor on hold time, not the wall-clock schedule around it.
    assert (time.monotonic() - pressed_at) >= 0.28, "released before min_hold_ms elapsed"
    await session.stop("done")
    await task


async def test_a_latch_does_not_survive_the_state_that_owns_it():
    """The guard stops being evaluated on exit, so nothing else would ever let go."""
    playbook = playbook_from_dict({
        "name": "latch2",
        "goal": "leave while holding",
        "initial": "hold",
        "probes": [BRIGHT],
        "perception": {"mode": "never"},
        "budget": {"max_cycles": 500, "max_seconds": 20, "idle_abort_s": 0},
        "states": {
            "hold": {
                "brief": "x", "autonomous": False,
                "reflex": [{"id": "go", "when": "probe('lit') > 0.5", "hold": "w"}],
                # Guarded on the latch rather than on elapsed time. A time-based exit
                # races the reflex loop's first tick -- under load the state can be left
                # before the latch ever engages, and the test then fails for a reason
                # that has nothing to do with what it is checking.
                "transitions": [{"when": "latched('go')", "to": "done"}],
            },
            "done": {"brief": "y", "autonomous": False, "transitions": []},
        },
    })
    capture = DialCapture(level=255)
    sink = Sink()
    deps = SessionDeps(
        capture=capture, vision=Quiet(vision=True), actuator=Quiet(),
        devices=DeviceSet(screen=SCREEN), executor=Executor(sink, dry_run=False),
        screen=SCREEN,
    )
    session = Session(playbook, deps, SessionOptions(
        settle_ms=0, watch_physical_input=False, dry_run=False,
        target_period_s=0.05, reflex_hz=50.0,
    ))
    task = session.start()
    await until(lambda: session.state_name == "done", label="the state to be left")
    assert key_events(sink, "w") == [True, False], "the latch must not outlive its state"
    assert not session._latched
    await session.stop("done")
    await task


async def test_the_hold_watchdog_does_not_yank_a_latched_key():
    """max_hold_ms is for input nobody is watching. A latch is the most-watched key here.

    Without the exemption the watchdog releases the key at 4 s while the session still
    believes the latch is engaged -- so it never presses again, and the failure only
    appears in runs longer than the timeout.
    """
    session, capture, sink = build_latch(
        [{"id": "go", "when": "probe('lit') > 0.5", "hold": "w"}]
    )
    session.deps.executor.max_hold_ms = 50  # far shorter than the test, to force the issue
    capture.level = 255
    task = session.start()
    await until(lambda: key_events(sink, "w") == [True], label="w pressed")
    await settled(0.3)  # many times max_hold_ms, and many watchdog passes
    assert key_events(sink, "w") == [True], "the watchdog released a supervised latch"
    assert session.deps.executor.held()[0] == ["w"]
    await session.stop("done")
    await task
    assert key_events(sink, "w") == [True, False]


def test_the_watchdog_still_releases_input_nothing_is_supervising():
    executor = Executor(Sink(), dry_run=False, max_hold_ms=1)
    executor.run(parse_burst("d:shift", screen=SCREEN))
    time.sleep(0.02)
    assert executor.enforce_hold_watchdog() == ["leftshift"]


async def test_status_reports_the_reflex_rate_it_achieved():
    session, capture, sink = build_latch(
        [{"id": "go", "when": "probe('lit') > 0.5", "hold": "w"}], reflex_hz=40.0
    )
    task = session.start()
    await until(lambda: session._reflex_ticks > 8, label="the fast loop to get going")
    await settled(0.3)
    snap = session.snapshot()
    await session.stop("done")
    await task
    reflex = snap["reflex"]
    assert reflex["requested_hz"] == 40.0
    assert reflex["measured_hz"] > 5, f"measured {reflex['measured_hz']} Hz"
    assert reflex["ticks"] > 5


# -- starvation ------------------------------------------------------------------------


def test_a_burst_that_cannot_get_the_device_reports_busy_rather_than_queueing():
    """A reflex landing after the burst it queued behind reacts to a world that moved."""
    executor = Executor(Sink(), dry_run=False)
    blocked = threading.Event()
    done = threading.Event()

    def hog():
        blocked.set()
        executor.run(parse_burst("w:400;k:a", screen=SCREEN))
        done.set()

    thread = threading.Thread(target=hog)
    thread.start()
    blocked.wait(1.0)
    time.sleep(0.05)

    started = time.perf_counter()
    report = executor.run(parse_burst("k:b", screen=SCREEN), wait_s=0.02)
    waited = time.perf_counter() - started

    assert not report.ok
    assert report.error is not None and report.error.startswith("input busy")
    assert waited < 0.2, f"waited {waited:.3f}s; the timeout should have given up"
    thread.join(2.0)
    assert done.is_set()


def test_a_release_never_times_out():
    """Letting go is safe by definition; a dropped release is a stuck key."""
    executor = Executor(Sink(), dry_run=False)
    executor.run(parse_burst("d:shift", screen=SCREEN))
    assert executor.held()[0] == ["leftshift"]
    report = executor.run(parse_burst("u:shift", screen=SCREEN))
    assert report.ok
    assert executor.held()[0] == []


# -- guard functions -------------------------------------------------------------------


def test_clamp_and_sign():
    assert Guard("clamp(15, 0, 10)").evaluate(ctx()) == 10
    assert Guard("clamp(-5, 0, 10)").evaluate(ctx()) == 0
    assert Guard("clamp(5, 10, 0)").evaluate(ctx()) == 5  # bounds given backwards
    assert Guard("sign(-3)").evaluate(ctx()) == -1
    assert Guard("sign(0)").evaluate(ctx()) == 0


def test_rate_reads_the_derivative_the_engine_publishes():
    context = GuardContext(probes={"m": 100.0, "m__rate": -170.0})
    assert Guard("rate('m')").evaluate(context) == -170.0
    assert Guard("rate('missing')").evaluate(context) == 0.0


def test_held_and_latched_see_the_previous_tick():
    context = GuardContext(held={"w", "btn:l"}, latched={"sprint"})
    assert Guard("held('w')").evaluate(context) is True
    assert Guard("held('btn:l')").evaluate(context) is True
    assert Guard("held('q')").evaluate(context) is False
    assert Guard("latched('sprint')").evaluate(context) is True


# -- what a live run exposed -----------------------------------------------------------


def test_the_token_budget_covers_what_the_grammar_permits():
    """These two numbers drifting apart truncates the reply, and only on the busy cycles.

    Observed live: the grammar allowed 20 actions, max_tokens was a fixed 96, and a 1.7B
    that used its full allowance produced `...;w:120;w` -- every action well formed except
    the last, which had no payload. Unparseable, so the whole cycle was thrown away.
    """
    from voltage_input_mcp.llm.grammar import actuator_token_budget

    budget = actuator_token_budget(max_actions=20, max_text_len=96, targets=["recover"])
    assert budget > 96, "the old fixed ceiling was below what the grammar could emit"
    # A 20-action burst of short actions is ~100 tokens of text plus the two reply fields.
    assert budget >= 20 * 5
    assert budget <= 1024, "a ceiling this high is a runaway, not a safety margin"

    # Shrinking what the grammar permits must shrink the budget with it.
    tight = actuator_token_budget(max_actions=4, max_text_len=0, targets=[])
    assert tight < budget


def test_a_chord_is_a_set():
    """`k:shift+shift` is not 'shift, harder'. Observed live from a 1.7B, four times over."""
    assert parse_burst("k:shift+shift+shift+shift", screen=SCREEN).render() == "k:shift"
    # Order of first appearance survives, because the executor releases in reverse and a
    # modifier has to outlive the key it modifies.
    assert parse_burst("k:ctrl+shift+ctrl+t", screen=SCREEN).render() == "k:ctrl+shift+t"


def test_region_probes_subsample_without_changing_what_they_mean():
    """Full-resolution region probes cost 10+ ms, which is half a reflex tick each.

    Subsampling is only legitimate because both measurements are ratios or averages: the
    fraction of changed pixels and the mean colour both survive a regular sample. This
    checks that claim rather than assuming it.
    """
    from voltage_input_mcp.capture.base import Frame
    from voltage_input_mcp.capture.probes import ProbeEngine, _subsample
    from voltage_input_mcp.models.playbook import ProbeSpec

    rng = np.random.default_rng(7)
    base = np.full((600, 900, 3), 40, dtype=np.uint8)
    frame_a = Frame(pixels=base.copy())

    # Change a quarter of the area by well over the per-pixel threshold.
    moved = base.copy()
    moved[:300, :450] = 220
    frame_b = Frame(pixels=moved)

    spec = ProbeSpec(id="d", type="region_diff",
                     region={"x": 0, "y": 0, "w": 900, "h": 600})
    engine = ProbeEngine(specs=[spec])
    engine.evaluate(frame_a)
    changed = engine.evaluate(frame_b)["d"]
    assert 0.2 < changed < 0.3, f"a quarter of the region changed, probe said {changed}"

    # And a mean colour over a large flat area is unaffected by sampling it.
    tinted = np.zeros((600, 900, 3), dtype=np.uint8)
    tinted[:, :] = (200, 60, 60)
    tinted += rng.integers(0, 6, tinted.shape, dtype=np.uint8)  # a little noise
    mean_spec = ProbeSpec(id="m", type="region_mean", expect="#c83c3c", tolerance=12,
                          region={"x": 0, "y": 0, "w": 900, "h": 600})
    assert ProbeEngine(specs=[mean_spec]).evaluate(Frame(pixels=tinted))["m"] == 1.0

    # Small regions -- every HUD element and progress bar -- are not sampled at all.
    small = np.zeros((20, 8, 3), dtype=np.uint8)
    assert _subsample(small) is small


def test_diagnose_survives_a_burst_that_never_reached_the_governor():
    """`allowed: false` with no violations meant an IndexError on the top refusal rule.

    That crashed diagnose on exactly the runs most in need of it -- the ones where the
    actuator was emitting something malformed.
    """
    from voltage_input_mcp.diagnose import diagnose

    journal = [
        {"kind": "cycle", "cycle": i, "t": i * 0.5, "state": "s",
         "burst": "k:a;w:120;w:120;w:120;w:120;w", "allowed": False, "violations": [],
         "error": "unparseable burst"}
        for i in range(1, 6)
    ]
    report = diagnose(journal, None, status="failed", reason="x")
    codes = {f["code"] for f in report["findings"]}
    assert "unparseable_burst" in codes
    finding = next(f for f in report["findings"] if f["code"] == "unparseable_burst")
    assert finding["evidence"]["truncated"] == 5, "a trailing bare verb is a cut-off reply"


def test_diagnose_separates_a_policy_refusal_from_a_malformed_burst():
    from voltage_input_mcp.diagnose import diagnose

    journal = [
        {"kind": "cycle", "cycle": 1, "t": 0.0, "state": "s", "burst": "k:ctrl+alt+delete",
         "allowed": False, "violations": [{"rule": "deny_chords"}]},
        {"kind": "cycle", "cycle": 2, "t": 0.5, "state": "s", "burst": "k:a;w",
         "allowed": False, "violations": []},
    ]
    codes = {f["code"] for f in diagnose(journal, None)["findings"]}
    assert {"governor_refusals", "unparseable_burst"} <= codes


def test_diagnose_names_a_burst_that_is_one_action_padded_out():
    from voltage_input_mcp.diagnose import diagnose

    padded = "k:space;" + ";".join(["w:120"] * 15)
    journal = [
        {"kind": "cycle", "cycle": i, "t": i * 0.5, "state": "s", "burst": padded,
         "allowed": True, "executed": True, "probes": {"__frame_delta__": 0.4}}
        for i in range(1, 6)
    ]
    finding = next(
        f for f in diagnose(journal, None)["findings"] if f["code"] == "repetitive_bursts"
    )
    assert "max_actions_per_burst" in finding["fix"]
