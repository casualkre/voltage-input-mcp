"""End-to-end loop test with stub backends and a stub capture device.

Exercises the parts of the cycle that only appear when it actually runs: perception
gating, reflex precedence, transitions, governor refusal accounting, budget termination,
and the guarantee that held keys are released on the way out.
"""

from __future__ import annotations

import asyncio

import numpy as np

from voltage_input_mcp.capture.base import CaptureBackend, Frame
from voltage_input_mcp.inputs import DeviceSet, Executor
from voltage_input_mcp.llm.base import Backend, GenerationResult
from voltage_input_mcp.models.playbook import playbook_from_dict
from voltage_input_mcp.runtime import Session, SessionDeps, SessionOptions

SCREEN = (1920, 1080)


class StubCapture(CaptureBackend):
    """Returns a frame that changes only when `mutate` is called."""

    name = "stub"

    def __init__(self) -> None:
        self._pixels = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.grabs = 0

    def mutate(self) -> None:
        self._pixels = np.random.default_rng(self.grabs).integers(
            0, 255, (1080, 1920, 3), dtype=np.uint8
        )

    def grab(self, region=None) -> Frame:
        self.grabs += 1
        return Frame(pixels=self._pixels.copy(), frame_id=self.grabs, backend=self.name)

    def geometry(self):
        return SCREEN


class StubBackend(Backend):
    """Replays canned replies and counts calls."""

    name = "stub"

    def __init__(self, replies: list[str], *, vision: bool = False) -> None:
        self._replies = replies
        self._vision = vision
        self.calls = 0

    @property
    def supports_grammar(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return self._vision

    async def generate(self, prompt, **kw) -> GenerationResult:
        text = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return GenerationResult(text=text, model="stub", backend="stub", latency_ms=1.0)

    async def health(self):
        return {"ok": True, "backend": self.name}


VISION_SEES_TARGET = '{"s":"a window","e":[{"l":"target","b":[100,100,200,150],"c":0.9}]}'
VISION_SEES_NOTHING = '{"s":"empty","e":[]}'


def build(playbook: dict, vision_replies, actuator_replies, **opts):
    compiled = playbook_from_dict(playbook)
    capture = StubCapture()
    devices = DeviceSet(screen=SCREEN)
    executor = Executor(devices, dry_run=True)
    deps = SessionDeps(
        capture=capture,
        vision=StubBackend(vision_replies, vision=True),
        actuator=StubBackend(actuator_replies),
        devices=devices,
        executor=executor,
        screen=SCREEN,
    )
    opts.setdefault("target_period_s", 0.01)
    options = SessionOptions(
        settle_ms=0, watch_physical_input=False, dry_run=True, **opts
    )
    return Session(compiled, deps, options), capture, deps


BASIC = {
    "name": "loop_test",
    "goal": "click the target",
    "initial": "find",
    "budget": {"max_cycles": 12, "max_seconds": 20, "idle_abort_s": 0},
    "perception": {"mode": "always"},
    "states": {
        "find": {
            "brief": "Click the target.",
            "watch": ["target"],
            "transitions": [{"when": "sees('target')", "to": "@success"}],
        }
    },
}


async def test_run_reaches_success():
    session, _, _ = build(BASIC, [VISION_SEES_TARGET], ["g:0;c:l|.|clicking"])
    await session.start()
    assert session.status == "succeeded"
    assert session._cycle >= 1


async def test_run_hits_the_cycle_budget_when_the_guard_never_fires():
    session, _, _ = build(BASIC, [VISION_SEES_NOTHING], [".|.|waiting"])
    await session.start()
    assert session.status == "failed"
    assert "max_cycles" in session.reason


async def test_on_change_mode_skips_the_vision_model_on_a_static_screen():
    playbook = {**BASIC, "perception": {"mode": "on_change", "change_threshold": 0.02}}
    playbook["states"] = {
        "find": {
            "brief": "wait",
            "watch": ["target"],
            "transitions": [{"when": "run_cycles() >= 6", "to": "@success"}],
        }
    }
    session, capture, deps = build(playbook, [VISION_SEES_TARGET], [".|.|idle"])
    await session.start()
    # The screen never changes, so vision should run about once, not once per cycle.
    assert capture.grabs >= 1
    assert deps.vision.calls <= 2, f"vision ran {deps.vision.calls} times on a static screen"
    # And the decision loop reuses the reflex loop's frame rather than grabbing its own,
    # so captures are strictly fewer than cycles. Asserting one grab per cycle -- which
    # this test used to do -- would be asserting the duplication away again.
    assert capture.grabs < session._cycle, (
        f"{capture.grabs} captures for {session._cycle} cycles; the loops are not sharing"
    )


async def test_reusing_one_sample_across_cycles_does_not_re_run_vision():
    """The gate has to be answered live, not read out of the sample's probe snapshot.

    One capture is shared by several decision cycles. The probe dict in it was computed
    before the vision call that follows, so a cached `delta_vs_vision` still reads 1.0 on
    every cycle that reuses the sample -- and `on_change` quietly degrades into `always`,
    which is the single thing it exists to prevent. It only shows up when cycles are fast
    enough to reuse a sample, which is why making an unrelated path faster exposed it.
    """
    playbook = {**BASIC, "perception": {"mode": "on_change", "change_threshold": 0.02}}
    playbook["states"] = {
        "find": {
            "brief": "wait",
            "watch": ["target"],
            "transitions": [{"when": "run_cycles() >= 10", "to": "@success"}],
        }
    }
    # Cycles far faster than _SHARE_MAX_AGE_S, so nearly every one reuses the sample.
    session, capture, deps = build(
        playbook, [VISION_SEES_TARGET], [".|.|idle"], target_period_s=0.001
    )
    await session.start()
    assert session._cycle >= 10
    assert deps.vision.calls <= 2, (
        f"vision ran {deps.vision.calls} times across {session._cycle} cycles of an "
        "unchanged screen; the on_change gate is reading a stale baseline"
    )


async def test_on_change_measures_against_the_frame_vision_last_saw():
    """The gate asks "has the screen moved since the VLM looked", not "since last frame".

    With reflexes sampling at 20 Hz and decisions at 2 Hz, a delta measured against the
    previous frame answers a question about the last 50 ms. A screen that drifts steadily
    would then read as unchanged on every individual comparison and vision would never
    run again.
    """
    playbook = {**BASIC, "perception": {"mode": "on_change", "change_threshold": 0.02}}
    playbook["states"] = {
        "find": {
            "brief": "wait",
            "watch": ["target"],
            "transitions": [{"when": "run_cycles() >= 4", "to": "@success"}],
        }
    }
    session, capture, deps = build(
        playbook, [VISION_SEES_TARGET], [".|.|idle"], target_period_s=0.05
    )
    # Mutate on every grab, so each individual frame differs from the last and from the
    # baseline. Vision must keep running.
    original = capture.grab

    def mutating_grab(region=None):
        capture.mutate()
        return original(region)

    capture.grab = mutating_grab  # type: ignore[method-assign]
    await session.start()
    assert deps.vision.calls >= 2, "a screen that keeps changing must keep being looked at"


async def test_reflex_preempts_the_actuator():
    playbook = {
        **BASIC,
        "probes": [
            {"id": "hot", "type": "brightness", "region": {"x": 0, "y": 0, "w": 10, "h": 10}}
        ],
        "states": {
            "find": {
                "brief": "wait",
                "watch": ["target"],
                "reflex": [
                    {"id": "always", "when": "probe('hot') >= 0", "do": "k:q",
                     "cooldown_ms": 0, "exclusive": True}
                ],
                "transitions": [{"when": "run_cycles() >= 3", "to": "@success"}],
            }
        },
    }
    # Reflexes run in their own loop at reflex_hz, so an exclusive one suppresses the
    # *next* decision rather than being checked inside the cycle. Either way the
    # guarantee is the same: a stale decision never overrides a fresh reaction.
    session, _, deps = build(
        playbook, [VISION_SEES_NOTHING], ["k:z|.|should not run"],
        target_period_s=0.15, reflex_hz=50.0,
    )
    await session.start()

    assert session._reflex_fires > 0, "the fast reflex loop never fired"
    # Once the reflex loop is up it suppresses every subsequent decision. A single
    # decision can slip through at startup, before the loop's first tick -- asserting
    # exactly zero would be asserting a scheduler ordering rather than the guarantee.
    assert deps.actuator.calls < session._cycle, (
        f"actuator ran {deps.actuator.calls} times in {session._cycle} cycles; "
        "an exclusive reflex should have suppressed all but the startup one"
    )
    assert session._reflex_fires > deps.actuator.calls, (
        "reflexes should outnumber decisions -- that is the point of the fast loop"
    )


async def test_illegal_transition_proposal_is_ignored():
    """The actuator may only propose targets the state declares."""
    playbook = {
        **BASIC,
        "states": {
            "find": {
                "brief": "wait",
                "watch": ["target"],
                "transitions": [{"when": "run_cycles() >= 4", "to": "@success"}],
            }
        },
    }
    session, _, _ = build(playbook, [VISION_SEES_NOTHING], [".|somewhere_else|bad"])
    await session.start()
    assert session.status == "succeeded"
    assert session.state_name == "find"


async def test_governor_refusals_are_counted_and_end_the_run():
    playbook = {
        **BASIC,
        "budget": {"max_cycles": 30, "max_seconds": 20, "max_rejections": 3,
                   "idle_abort_s": 0},
        "states": {
            "find": {"brief": "x", "watch": ["target"], "transitions": []},
        },
    }
    # ctrl+alt+delete is refused by the default policy on every cycle.
    session, _, _ = build(playbook, [VISION_SEES_NOTHING], ["k:ctrl+alt+delete|.|bad"])
    await session.start()
    assert session.governor.rejections >= 3
    assert "governor refused" in session.reason


async def test_unparseable_burst_does_not_kill_the_run():
    playbook = {
        **BASIC,
        "states": {
            "find": {
                "brief": "x",
                "watch": ["target"],
                "transitions": [{"when": "run_cycles() >= 3", "to": "@success"}],
            }
        },
    }
    session, _, _ = build(playbook, [VISION_SEES_NOTHING], ["this is not a burst|.|junk"])
    await session.start()
    assert session.status == "succeeded"


async def test_stop_releases_held_keys():
    playbook = {
        **BASIC,
        "budget": {"max_cycles": 500, "max_seconds": 30, "idle_abort_s": 0},
        "states": {"find": {"brief": "hold", "watch": ["target"], "transitions": []}},
    }
    session, _, deps = build(playbook, [VISION_SEES_NOTHING], ["d:shift|.|holding"])
    task = session.start()
    await asyncio.sleep(0.15)
    # stop() drains the loop itself, so the task is already finished when it returns.
    await session.stop("test")
    assert task.done()
    assert session.status == "stopped"
    keys, buttons = deps.executor.held()
    assert not keys and not buttons, "held input must be released when a run stops"


async def test_snapshot_is_serialisable_and_informative():
    session, _, _ = build(BASIC, [VISION_SEES_TARGET], ["g:0;c:l|.|ok"])
    await session.start()
    snap = session.snapshot()
    import json

    json.dumps(snap)  # must not raise
    assert snap["status"] == "succeeded"
    assert "timings" in snap and "governor" in snap
    assert snap["journal"]["recent"]
