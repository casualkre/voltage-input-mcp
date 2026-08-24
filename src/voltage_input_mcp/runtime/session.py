"""The run loop.

Where the latency actually goes
-------------------------------
The serial chain per cycle is: capture -> perceive -> decide -> execute. Those steps are
genuinely dependent -- the decision needs the perception, and the perception needs to
reflect the previous burst -- so there is no clever way to overlap the chain itself. The
speed comes from making steps *not happen*:

  * **Bursts.** One decision produces many inputs. This is the main lever and it is
    unbounded: a 40-action burst costs one decision, so effective input rate is set by
    the burst, not the model.
  * **Reflexes.** Probe-driven rules fire with no model in the path at all, between
    decisions, in microseconds.
  * **`perception.mode = on_change`.** Most cycles in a real task look at a screen that
    has not moved. A 40 us frame-diff decides whether to spend 300 ms on the vision model
    or reuse the cached observation. On typical desktop work this skips the VLM on 60-80%
    of cycles.
  * **Prompt-cache locality.** See prompts.py -- the prompt is ordered so that only the
    changing tail needs re-prefilling.

The one honest overlap available is perception with *idle* time: after a burst executes,
the loop usually has to wait to hit its target period anyway, so the next capture and
vision call are started during that wait rather than after it. That is real but modest;
it is not pipelining the dependent chain, and it is not claimed to be.

Ordering within a cycle
-----------------------
Reflexes are checked before transitions, and transitions before the actuator. That order
is deliberate: a reflex is a reaction that must not wait for a decision, a transition is
the orchestrator's control flow and outranks anything the small model wants, and the
actuator only gets to act when neither of the two faster, more trustworthy layers had
something to say.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Final

from ..capture import CaptureBackend, Frame, ProbeEngine, encode_png
from ..errors import Aborted, BurstParseError, ExpressionError, SessionError
from ..expr import GuardContext
from ..inputs import DeviceSet, ExecutionReport, Executor
from ..inputs.executor import PointerMode
from ..llm import Backend, actuator_grammar, observation_grammar
from ..llm.grammar import actuator_token_budget, vision_vocabulary
from ..llm.ollama import BURST_SCHEMA, OBSERVATION_SCHEMA
from ..models.burst import Burst, parse_burst
from ..models.observation import CoordinateMapper, Observation, parse_vision_output
from ..models.playbook import (
    TERMINALS,
    CompiledHold,
    CompiledPlaybook,
    CompiledReflex,
    CompiledState,
    Perception,
)
from ..models.template import BurstTemplate
from ..safety import Governor, KillSwitch, Verdict
from .journal import CycleRecord, Journal
from .prompts import ACTUATOR_SYSTEM, VISION_SYSTEM, actuator_prompt, vision_prompt
from .recall import CacheEntry, PolicyCache, situation_key
from .tuner import Tunable, Tuner

__all__ = ["Session", "SessionOptions", "SessionDeps", "RunStatus"]

RunStatus = str  # "pending" | "running" | "paused" | "succeeded" | "failed" | "stopped" | "error"


@dataclass(slots=True)
class SessionOptions:
    target_period_s: float = 0.5
    dry_run: bool | None = None          # None -> use the playbook's policy
    keep_frames: bool = False
    settle_ms: int = 60                  # pause after a burst before looking again
    vision_max_tokens: int = 192
    # A floor, not a ceiling. The real limit is derived from what the grammar permits --
    # see llm.grammar.actuator_token_budget -- because a ceiling below that does not make
    # the burst shorter, it makes it truncated and unparseable.
    actuator_max_tokens: int = 96
    vision_timeout_s: float = 15.0
    actuator_timeout_s: float = 12.0
    vision_temperature: float = 0.1
    actuator_temperature: float = 0.25
    watch_physical_input: bool = True
    # The reflex loop runs independently of the decision loop, at capture rate. This is
    # the whole point of the small-model design and it was previously wasted: reflexes
    # were evaluated once per decision cycle, so a 0.5 s loop gave 2 Hz reactions no
    # matter how cheap the probes were. Capture is ~2 ms and probes ~40 us, so 20-30 Hz
    # costs almost nothing and is what makes air control or a dodge possible at all.
    reflex_hz: float = 20.0
    reflex_enabled: bool = True
    # Recall the actuator's past decisions for situations that repeat. Costs one
    # nearest-neighbour scan over a small bucket; saves a model round trip on every hit.
    recall_enabled: bool = True


# The reflex loop is the only thing that captures, so it must not be allowed to eat the
# machine. If a tick costs more than this fraction of the period -- which happens on a
# non-streaming capture backend where every grab is a 30-40 ms DBus round trip -- it
# stretches its own period rather than running flat out. The measured rate is reported in
# `snapshot()`, so what actually happened is visible rather than assumed.
_REFLEX_DUTY: Final = 0.5
# How stale a frame the decision loop will accept from the reflex loop before capturing
# its own. Generous, because the vision call it feeds takes an order of magnitude longer
# than this.
_SHARE_MAX_AGE_S: Final = 0.08
# How long a number probe may go without producing a value before a latch depending on it
# is treated as flying blind. Well above any sane `ocr_interval_ms`, low enough that a key
# comes up within a second of the HUD disappearing rather than staying down indefinitely.
_BLIND_PROBE_S: Final = 1.2


@dataclass(slots=True)
class SessionDeps:
    capture: CaptureBackend
    vision: Backend
    actuator: Backend
    devices: DeviceSet
    executor: Executor
    screen: tuple[int, int]


@dataclass(slots=True)
class Sample:
    """One capture plus the probe values measured from it.

    Both loops read this. The reflex loop produces it at reflex rate; the decision loop
    consumes the latest rather than capturing again. That is not only cheaper -- it is
    what keeps the two loops agreeing about the screen. When each evaluated probes
    separately against its own frame, every probe carrying history (`region_diff`, the
    frame delta, number rates) was being advanced by whichever loop happened to run last.
    """

    frame: Frame
    probes: dict[str, float]
    at: float


@dataclass(slots=True)
class Perceived:
    frame: Frame | None = None
    observation: Observation = field(default_factory=Observation)
    probes: dict[str, float] = field(default_factory=dict)
    source: str = "skipped"
    capture_ms: float = 0.0
    vision_ms: float = 0.0
    error: str | None = None


class Session:
    """One playbook run."""

    def __init__(
        self,
        playbook: CompiledPlaybook,
        deps: SessionDeps,
        options: SessionOptions | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        self.id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        self.playbook = playbook
        self.deps = deps
        self.options = options or SessionOptions()

        policy = playbook.spec.policy
        self.dry_run = policy.dry_run if self.options.dry_run is None else self.options.dry_run
        deps.executor.dry_run = self.dry_run
        deps.executor.max_hold_ms = policy.max_hold_ms
        # A playbook may set the pointer mode for its own duration. A game with pointer
        # lock needs relative motion and desktop UI needs absolute, and requiring a config
        # edit and a server restart between the two is friction with no purpose. Saved so
        # the machine goes back to its configured default when the run ends.
        self._pointer_mode_was = deps.executor.pointer_mode
        if policy.pointer_mode is not None:
            deps.executor.pointer_mode = PointerMode(policy.pointer_mode)

        self.governor = Governor(
            policy, screen=deps.screen, max_rejections=playbook.spec.budget.max_rejections
        )
        self.probes = ProbeEngine(specs=list(playbook.spec.probes))
        self.journal = Journal(self.id, keep_frames=self.options.keep_frames)
        self.killswitch = KillSwitch(
            deadman_s=playbook.spec.budget.deadman_s,
            watch_physical_input=self.options.watch_physical_input,
            on_trip=self._on_kill,
        )

        self.status: RunStatus = "pending"
        self.reason: str = ""
        self.state_name: str = playbook.spec.initial
        self.vars: dict[str, Any] = dict(playbook.spec.vars)

        self._started_at = 0.0
        self._state_entered_at = 0.0
        self._cycle = 0
        self._cycles_in_state = 0
        self._bursts = 0
        self._last_burst = ""
        self._last_result = ""
        self._last_observation = Observation()
        self._steer_hint: str | None = None
        self._forced_state: str | None = None
        self._pending: asyncio.Task[Perceived] | None = None
        self._pause = asyncio.Event()
        self._pause.set()
        self._stop_requested = False
        self._task: asyncio.Task[None] | None = None
        # reflex id -> (last fire time, fire count). Reset on every state entry.
        self._rule_fired: dict[str, tuple[float, int]] = {}
        self._grammar_cache: dict[tuple, str] = {}
        self._reflex_task: asyncio.Task[None] | None = None
        self._reflex_fires = 0
        self._reflex_starved = 0
        self._reflex_ticks = 0
        self._reflex_started_at = 0.0
        self._reflex_errors: dict[str, str] = {}
        # When an exclusive reflex fires in the fast loop, the next decision is skipped.
        # `exclusive` meant "suppress the actuator this cycle" when reflexes were checked
        # inside the cycle; with them decoupled it has to mean "the reaction already
        # happened, do not let a 300 ms-stale decision override it".
        self._reflex_suppress_until = 0.0
        # Hold reflexes currently engaged: rule id -> (state it belongs to, engaged at).
        # Keyed by rule rather than by key so a latch releases exactly what it pressed,
        # leaving anything a burst or the orchestrator is holding alone.
        self._latched: dict[str, tuple[str, float]] = {}
        self._latch_events = 0
        self._sample: Sample | None = None
        self._sample_lock = threading.Lock()

        # Episodic optimisation of guard constants. Inert unless the playbook declares
        # both `tunables` and `reward` -- with neither, `tune()` still resolves to the
        # declared defaults, so a guard written with tune() behaves identically whether
        # the search is running or not and nothing has to be rewritten to turn it off.
        self._tuner = Tuner(
            [
                Tunable(name=name, default=t.default, low=t.min, high=t.max)
                for name, t in playbook.spec.tunables.items()
            ],
            explore=bool(playbook.spec.tunables and playbook.spec.reward),
        )
        self._tuner_loaded = self._tuner.load(playbook.spec.name)
        self._tune_params: dict[str, float] = self._tuner.current()
        self._episode_index = 0
        self._episode_start_value = 0.0
        self._episode_started_at = 0.0

        # Recall: the actuator's own past decisions, keyed by situation. Every hit is a
        # model round trip that does not happen.
        self.recall = PolicyCache(enabled=self.options.recall_enabled)
        self._last_recall: tuple = ((), {})
        self._timings: dict[str, list[float]] = {"capture": [], "vision": [], "actuator": [],
                                                 "execute": [], "cycle": [], "reflex": []}

    # -- lifecycle -------------------------------------------------------------------

    def start(self) -> asyncio.Task[None]:
        if self._task is not None:
            raise SessionError(f"session {self.id} is already running")
        self._task = asyncio.create_task(self._run(), name=f"voltage-{self.id}")
        return self._task

    async def _run(self) -> None:
        self.status = "running"
        self._started_at = time.monotonic()
        self.killswitch.start()
        self.journal.event(
            "start",
            run_id=self.id,
            playbook=self.playbook.spec.name,
            goal=self.playbook.spec.goal,
            dry_run=self.dry_run,
            initial=self.state_name,
            warnings=self.playbook.warnings,
        )

        if self.options.reflex_enabled:
            self._reflex_task = asyncio.create_task(
                self._reflex_loop(), name=f"voltage-reflex-{self.id}"
            )

        try:
            # Take one sample before the first episode so its reward baseline is a real
            # reading rather than a zero that makes episode 1 score the whole counter.
            await asyncio.to_thread(self._sample_fresh, 0.0)
            self._begin_episode()
            if self._tuner_loaded:
                self.journal.event(
                    "tuner_loaded", playbook=self.playbook.spec.name,
                    best={k: round(v, 3) for k, v in self._tuner.best.items()},
                )
            await self._enter_state(self.state_name, first=True)
            while True:
                await self._pause.wait()
                stop = self._budget_check()
                if stop:
                    self._finish("failed" if stop.startswith("budget") else "stopped", stop)
                    break
                if self._stop_requested:
                    self._finish("stopped", "stop requested")
                    break
                if self.killswitch.tripped:
                    self._finish("stopped", self.killswitch.reason)
                    break

                cycle_start = time.monotonic()
                if not await self._cycle_once():
                    break

                # Pace to the target period. A cycle that already overran gets no sleep,
                # so the loop degrades to "as fast as it can" rather than falling further
                # behind -- but the scheduled perception is still running in the
                # background either way, so a slow cycle is not wasted.
                remaining = self.options.target_period_s - (time.monotonic() - cycle_start)
                if remaining > 0:
                    await asyncio.sleep(remaining)
        except Aborted as exc:
            self._finish("stopped", str(exc))
        except asyncio.CancelledError:
            self._finish("stopped", "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - a crashed loop must still clean up
            self._finish("error", f"{type(exc).__name__}: {exc}")
        finally:
            await self._cleanup()

    async def _reflex_loop(self) -> None:
        """Capture, probe, hold and react at `reflex_hz`, independent of decisions.

        Deliberately does not touch the vision model or the actuator. It captures a
        frame, evaluates the probes -- which for a number probe means reading the last
        value a background worker produced, not paying for OCR -- then updates every
        latch and fires at most one one-shot reflex. That path is a couple of
        milliseconds end to end, so it can run at 20-30 Hz while the decision loop plods
        along at 2 Hz underneath it.

        This is where the responsiveness the small models were chosen for actually lives.
        A decision loop cannot dodge; a reflex can. A decision loop cannot hold a key
        down for exactly as long as a condition lasts; a latch can.

        The pacing is duty-limited rather than fixed. On a streaming capture backend a
        tick is ~2 ms and the configured rate is met exactly. On a backend where every
        grab is a DBus round trip, insisting on 20 Hz would spend the machine on capture
        and starve the two model servers, so the loop stretches its own period instead
        and reports the rate it actually achieved.
        """
        period = 1.0 / max(1.0, self.options.reflex_hz)
        self._reflex_started_at = time.monotonic()
        while self.status in ("running", "paused"):
            started = time.perf_counter()
            try:
                if self.status == "running" and not self.killswitch.tripped:
                    await self._reflex_tick()
                    self._reflex_ticks += 1
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - a reflex must never kill the run
                self._note_reflex_error("tick", exc)
            elapsed = time.perf_counter() - started
            self._timings["reflex"].append(elapsed * 1000.0)
            await asyncio.sleep(
                max(0.002, period - elapsed, elapsed * (1.0 / _REFLEX_DUTY - 1.0))
            )

    async def _reflex_tick(self) -> None:
        state = self.playbook.states.get(self.state_name)
        if state is None:
            return
        # Capture even with no rules in this state: the decision loop reads these samples,
        # and probe history has to keep advancing or `region_diff` and the number rates
        # measure across whatever gap the last rule-bearing state left behind.
        sample = await asyncio.to_thread(self._sample_fresh, 0.0)
        if not state.reflexes and not state.holds:
            return

        ctx = self._context_from(sample.probes)
        await self._update_latches(state, ctx)

        picked = self._pick_reflex(state, ctx)
        if picked is None:
            return

        try:
            burst = picked.template.burst(ctx, screen=self.deps.screen)
        except (BurstParseError, ExpressionError) as exc:
            self._note_reflex_error(picked.rule.id, exc)
            self.journal.event(
                "reflex_error", rule=picked.rule.id, source=picked.template.source,
                error=str(exc),
            )
            return
        if not burst.actions:
            return

        verdict = self.governor.review(
            burst, observation=self._last_observation,
            allow_verbs=state.allow_verbs, source=f"reflex:{picked.rule.id}",
        )
        if not verdict.allowed:
            return

        # Give up rather than queue. See Executor.run: a reflex that lands after the
        # burst it queued behind is a reaction applied to a situation that has moved on.
        report = await asyncio.to_thread(
            self.deps.executor.run,
            burst,
            label=f"reflex:{picked.rule.id}",
            wait_s=min(0.05, 1.0 / max(1.0, self.options.reflex_hz) * 0.5),
        )
        if report.error and report.error.startswith("input busy"):
            self._reflex_starved += 1
            self.journal.event("reflex_starved", rule=picked.rule.id, burst=burst.render())
            return

        self._rule_fired[picked.rule.id] = (
            time.monotonic(), self._rule_fired.get(picked.rule.id, (0.0, 0))[1] + 1
        )
        self._reflex_fires += 1
        if picked.rule.exclusive:
            self._reflex_suppress_until = time.monotonic() + self.options.target_period_s
        self.journal.event(
            "reflex", rule=picked.rule.id, burst=burst.render(),
            probes={
                k: round(v, 3) for k, v in sample.probes.items() if not k.startswith("__")
            },
        )

    # -- latches ---------------------------------------------------------------------

    async def _update_latches(self, state: CompiledState, ctx: GuardContext) -> None:
        """Bring every hold reflex's key state into line with its guard.

        Runs before one-shot reflexes so that a `do` rule guarded on `held('w')` sees this
        tick's latch state rather than last tick's.
        """
        for hold in state.holds:
            rule = hold.rule
            engaged = rule.id in self._latched

            if not engaged:
                fires = self._rule_fired.get(rule.id, (0.0, 0))[1]
                if rule.max_fires is not None and fires >= rule.max_fires:
                    continue
                if self._test(hold.guard, ctx, f"{state.name}.hold.{rule.id}"):
                    await self._engage(hold, state)
                continue

            _, since = self._latched[rule.id]
            held_ms = (time.monotonic() - since) * 1000.0

            # Backstop 1: a latch cannot outlive its guard's ability to see.
            #
            # A number probe whose region gets covered -- by a modal, a scene change, a
            # menu -- does not fail. It returns its last value, forever. So a guard like
            # `release_when: rate('meters') < -9` stops being a measurement of anything
            # and silently freezes at whatever it last read, and the key stays down. That
            # is exactly how a hold on `w, shift` ended up typing capital Ws into a
            # window that was not even the game.
            #
            # The guard cannot detect this; only the runtime can, by noticing that every
            # number the guard reads has stopped arriving.
            blind = hold.depends_on_numbers and all(
                self.probes.number_age_s(pid) >= _BLIND_PROBE_S
                for pid in hold.depends_on_numbers
            )
            if blind:
                self.journal.event(
                    "hold_blind", rule=rule.id, holding=list(hold.names),
                    probes=sorted(hold.depends_on_numbers),
                    detail="every number this guard reads has gone stale; releasing",
                )
                await self._disengage(hold)
                continue

            # Backstop 2: an absolute ceiling, whatever the guard says.
            #
            # Latched keys are exempt from the hold watchdog because a guard re-checked
            # fifty times a second is stricter supervision than a timer. That is true
            # while the guard is sound, and it is precisely the assumption that fails
            # here -- so the exemption gets a ceiling rather than being unbounded.
            if held_ms >= self.playbook.spec.policy.max_latch_ms:
                self.journal.event(
                    "hold_expired", rule=rule.id, holding=list(hold.names),
                    held_ms=round(held_ms),
                    detail=f"exceeded policy.max_latch_ms "
                           f"({self.playbook.spec.policy.max_latch_ms}); releasing",
                )
                await self._disengage(hold)
                continue

            if held_ms < rule.min_hold_ms:
                continue
            if hold.release is not None:
                # An explicit release condition, so there is a dead band between the two
                # thresholds in which neither fires and the latch simply keeps its state.
                # That band is the whole point -- it is what stops the key chattering.
                let_go = self._test(hold.release, ctx, f"{state.name}.hold.{rule.id}.release")
            else:
                let_go = not self._test(hold.guard, ctx, f"{state.name}.hold.{rule.id}")
            if let_go:
                await self._disengage(hold)

    async def _engage(self, hold: CompiledHold, state: CompiledState) -> None:
        verdict = self.governor.review(
            hold.press, observation=self._last_observation,
            allow_verbs=state.allow_verbs, source=f"hold:{hold.rule.id}",
        )
        if not verdict.allowed:
            self.journal.event(
                "hold_refused", rule=hold.rule.id, burst=hold.press.render(),
                **verdict.as_dict(),
            )
            return
        report = await asyncio.to_thread(
            self.deps.executor.run, hold.press, label=f"hold:{hold.rule.id}", wait_s=0.05
        )
        if not report.ok:
            # Never record a latch as engaged unless the press actually landed, or the
            # release will be skipped as redundant and the key state stops tracking the
            # guard in the one direction that matters.
            if report.error and report.error.startswith("input busy"):
                self._reflex_starved += 1
            return
        self._latched[hold.rule.id] = (state.name, time.monotonic())
        self.deps.executor.supervise(hold.names)
        self._latch_events += 1
        self._rule_fired[hold.rule.id] = (
            time.monotonic(), self._rule_fired.get(hold.rule.id, (0.0, 0))[1] + 1
        )
        self.journal.event("hold_on", rule=hold.rule.id, holding=list(hold.names))

    async def _disengage(self, hold: CompiledHold) -> None:
        self._latched.pop(hold.rule.id, None)
        self.deps.executor.unsupervise(hold.names)
        self._latch_events += 1
        # No governor review and no timeout. Letting go is safe by construction, and a
        # release that gets refused by a rate limit or dropped because the device was
        # busy is a key left down on the user's desktop.
        await asyncio.to_thread(
            self.deps.executor.run, hold.release_burst, label=f"hold:{hold.rule.id}:off"
        )
        self.journal.event("hold_off", rule=hold.rule.id, holding=list(hold.names))

    async def _release_latches(self, *, state: str | None = None, why: str = "") -> None:
        """Let go of every latch, or only those belonging to one state.

        Called on state exit, pause and shutdown. Scoped by state so a hold cannot outlive
        the state whose guard was driving it -- once the guard is no longer being
        evaluated, nothing would ever release the key.
        """
        for rule_id, (owner, _) in list(self._latched.items()):
            if state is not None and owner != state:
                continue
            hold = self._find_hold(owner, rule_id)
            self._latched.pop(rule_id, None)
            if hold is None:
                continue
            self.deps.executor.unsupervise(hold.names)
            await asyncio.to_thread(
                self.deps.executor.run, hold.release_burst, label=f"hold:{rule_id}:off"
            )
            self.journal.event("hold_off", rule=rule_id, holding=list(hold.names), why=why)

    def _find_hold(self, state_name: str, rule_id: str) -> CompiledHold | None:
        state = self.playbook.states.get(state_name)
        if state is None:
            return None
        return next((h for h in state.holds if h.rule.id == rule_id), None)

    def _note_reflex_error(self, where: str, exc: Exception) -> None:
        """Keep the first error per site, so `status` can show it without a log flood."""
        self._reflex_errors.setdefault(where, f"{type(exc).__name__}: {exc}")

    # -- sampling --------------------------------------------------------------------

    def _sample_fresh(self, max_age_s: float) -> Sample:
        """The latest capture and probe values, taking a new one if the last is stale.

        Runs on a worker thread from both loops, hence the lock: two threads finding the
        sample stale at the same moment would otherwise both capture, and the probe
        engine would advance its history twice for one instant in time.
        """
        sample = self._sample
        if sample is not None and (time.monotonic() - sample.at) <= max_age_s:
            return sample
        with self._sample_lock:
            sample = self._sample
            if sample is not None and (time.monotonic() - sample.at) <= max_age_s:
                return sample
            t0 = time.perf_counter()
            frame = self.deps.capture.grab(None)
            self._timings["capture"].append((time.perf_counter() - t0) * 1000.0)
            probes = self.probes.evaluate(frame)
            sample = Sample(frame=frame, probes=probes, at=time.monotonic())
            self._sample = sample
            return sample

    async def _cleanup(self) -> None:
        if self._reflex_task is not None:
            self._reflex_task.cancel()
            self._reflex_task = None
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        # Releasing held input is the one cleanup step that must never be skipped.
        self._latched.clear()
        released = await asyncio.to_thread(self.deps.executor.release_all)
        self.deps.executor.pointer_mode = self._pointer_mode_was
        if self._tuner.explore and self._tuner.episodes:
            try:
                saved = self._tuner.save(self.playbook.spec.name)
                self.journal.event(
                    "tuner_saved", path=str(saved), **self._tuner.summary()
                )
            except OSError as exc:  # noqa: BLE001 - never fail a run over a cache write
                self.journal.event("tuner_save_failed", error=str(exc))
        self.killswitch.stop()
        self.probes.close()
        self.journal.event(
            "end",
            status=self.status,
            reason=self.reason,
            cycles=self._cycle,
            bursts=self._bursts,
            released=released,
            reflex=self.reflex_summary(),
            timings=self.timing_summary(),
        )
        self.journal.close()

    def _on_kill(self, reason: str) -> None:
        # Runs on the killswitch thread. Interrupt any burst in flight immediately;
        # the loop notices on its next check.
        self.deps.executor.request_abort()

    def _finish(self, status: RunStatus, reason: str) -> None:
        if self.status in ("succeeded", "failed", "stopped", "error"):
            return
        self.status = status
        self.reason = reason

    # -- the cycle -------------------------------------------------------------------

    async def _cycle_once(self) -> bool:
        """Run one pass. Returns False when the run should end."""
        cycle_start = time.perf_counter()
        self._cycle += 1
        self._cycles_in_state += 1
        self.killswitch.touch()

        record = CycleRecord(cycle=self._cycle, t=time.time(), state=self.state_name)
        state = self.playbook.state(self.state_name)

        percept = await self._take_perception(state)
        record.perception = percept.source
        record.probes = {k: round(v, 4) for k, v in percept.probes.items()}
        record.scene = percept.observation.scene
        record.elements = [e.as_dict() for e in percept.observation.elements]
        record.flags = sorted(percept.observation.flags)
        if percept.error:
            record.error = percept.error
        if percept.frame is not None and self.options.keep_frames:
            record.frame = self.journal.save_frame(
                self._cycle, percept.frame.to_png((960, 540))
            )

        ctx = self._context(percept)

        # 1. Global outcome guards, checked before anything acts.
        if self.playbook.success and self._test(self.playbook.success, ctx, "success_when"):
            record.note = "success_when matched"
            self.journal.cycle(record)
            self._finish("succeeded", "success_when matched")
            return False
        if self.playbook.failure and self._test(self.playbook.failure, ctx, "failure_when"):
            record.note = "failure_when matched"
            self.journal.cycle(record)
            self._finish("failed", "failure_when matched")
            return False

        # 2. Reflexes and latches -- only when the fast loop is off. With it on, both run
        # there at reflex_hz, and doing them here as well would double every reaction.
        if not self.options.reflex_enabled:
            await self._update_latches(state, ctx)
        reflex = None if self.options.reflex_enabled else self._pick_reflex(state, ctx)
        if reflex is not None:
            rule = reflex.rule
            record.burst_source = f"reflex:{rule.id}"
            try:
                burst = reflex.template.burst(ctx, screen=self.deps.screen)
            except (BurstParseError, ExpressionError) as exc:
                record.error = f"reflex {rule.id} did not render: {exc}"
                burst = None
            if burst is not None:
                record.burst = burst.render()
                report, verdict = await self._guarded_execute(
                    burst, percept.observation, state, source=record.burst_source
                )
                record.allowed = verdict.allowed
                record.violations = [v.as_dict() for v in verdict.violations]
                record.executed = bool(report and report.ok and not self.dry_run)
                self._reflex_fires += 1
                self._rule_fired[rule.id] = (
                    time.monotonic(), self._rule_fired.get(rule.id, (0.0, 0))[1] + 1
                )
                if rule.exclusive:
                    record.note = "reflex fired exclusively; actuator skipped this cycle"
                    self._finalise_cycle(record, cycle_start)
                    return True

        # 3. Transitions: the orchestrator's control flow outranks the actuator.
        moved = await self._maybe_transition(state, ctx, record)
        if moved is not None:
            self._finalise_cycle(record, cycle_start)
            return moved

        # 4. State-level limits.
        limit = self._state_limit(state)
        if limit:
            target = state.spec.on_timeout or "@failure"
            record.note = limit
            record.transition = target
            await self._goto(target, state, record)
            self._finalise_cycle(record, cycle_start)
            return target not in TERMINALS

        # 5. The actuator -- unless an exclusive reflex just reacted. Letting a decision
        # made from a 300 ms-old frame override a reaction made 20 ms ago is exactly
        # backwards.
        if time.monotonic() < self._reflex_suppress_until:
            record.note = (record.note or "exclusive reflex fired; actuator skipped")
            record.burst_source = record.burst_source or "reflex"
        elif state.spec.autonomous:
            await self._actuate(state, percept, record)

        self._finalise_cycle(record, cycle_start)
        return self.status == "running"

    def _finalise_cycle(self, record: CycleRecord, cycle_start: float) -> None:
        elapsed = (time.perf_counter() - cycle_start) * 1000.0
        record.timing["cycle_ms"] = round(elapsed, 1)
        self._timings["cycle"].append(elapsed)
        self.journal.cycle(record)

    # -- perception ------------------------------------------------------------------

    async def _take_perception(self, state: CompiledState) -> Perceived:
        """Use the perception started during the last cycle's idle time, or do it now."""
        if self._pending is not None:
            pending, self._pending = self._pending, None
            try:
                return await pending
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - fall through to a fresh attempt
                pass
        return await self._perceive(state)

    def _schedule_perception(self, state: CompiledState) -> None:
        """Start the next perception so it runs during the inter-cycle wait."""
        if self._pending is None and self.status == "running":
            self._pending = asyncio.create_task(self._perceive(state))

    async def _perceive(self, state: CompiledState) -> Perceived:
        result = Perceived()
        settings = self.playbook.perception_for(state.name)

        t0 = time.perf_counter()
        try:
            # Reuse the reflex loop's capture when it is recent enough. The frame this
            # feeds is about to sit in front of a model for several hundred milliseconds,
            # so paying for a second grab to make it 30 ms fresher buys nothing.
            sample = await asyncio.to_thread(self._sample_fresh, _SHARE_MAX_AGE_S)
        except Exception as exc:  # noqa: BLE001
            result.error = f"capture failed: {exc}"
            result.observation = self._last_observation
            result.source = "cache"
            return result
        result.capture_ms = (time.perf_counter() - t0) * 1000.0
        result.probes = sample.probes

        frame = sample.frame
        if settings.region is not None:
            x, y, w, h = settings.region.as_tuple()
            frame = frame.crop(x - frame.origin[0], y - frame.origin[1], w, h)
        result.frame = frame

        if not self._should_run_vision(settings, sample.frame):
            result.observation = self._last_observation
            result.source = "cache" if self._last_observation.elements else "skipped"
            return result

        t1 = time.perf_counter()
        observation = await self._run_vision(state, frame, settings)
        result.vision_ms = (time.perf_counter() - t1) * 1000.0
        self._timings["vision"].append(result.vision_ms)

        if observation is None:
            result.observation = self._last_observation
            result.source = "cache"
            result.error = "vision backend failed; reusing the last observation"
            return result

        # Only a successful look resets the change baseline. Moving it on a failed call
        # would have the loop conclude the screen is unchanged relative to an observation
        # that was never made, and `on_change` would then skip vision indefinitely.
        self.probes.mark_perceived(sample.frame)
        self._last_observation = observation
        result.observation = observation
        result.source = "vlm"
        return result

    def _should_run_vision(self, settings: Perception, frame: Frame) -> bool:
        if settings.mode == "never":
            return False
        if settings.mode == "always":
            return True
        if not self._last_observation.elements:
            return True  # nothing cached to fall back on
        if self._last_observation.is_stale(settings.max_cache_age_s):
            return True
        if settings.mode == "cadence":
            return self._cycle % settings.cadence == 0
        # on_change. Measured against the frame vision last saw, not against the previous
        # frame: with the reflex loop sampling at 20 Hz, "changed since the last frame"
        # answers a question about the last 50 ms, which is not what this gate is for.
        # Computed live rather than read from the sample's probe dict -- that dict is a
        # snapshot from before the last vision call moved the baseline.
        return self.probes.delta_vs_vision(frame) >= settings.change_threshold

    async def _run_vision(
        self, state: CompiledState, frame: Frame, settings: Perception
    ) -> Observation | None:
        scaled = frame.downscaled(settings.downscale_to)
        # compress_level=1: this PNG is consumed by a model over localhost within
        # milliseconds, so encode speed matters and file size does not.
        png = await asyncio.to_thread(encode_png, scaled, 1)

        grammar = self._cached_grammar(
            ("vision", tuple(state.spec.watch), settings.max_elements, settings.read_text),
            lambda: observation_grammar(
                state.spec.watch,
                max_elements=settings.max_elements,
                read_text=settings.read_text,
            ),
        )

        result = await self.deps.vision.generate(
            vision_prompt(state),
            system=VISION_SYSTEM,
            image_png=png,
            grammar=grammar if self.deps.vision.supports_grammar else None,
            schema=None if self.deps.vision.supports_grammar else OBSERVATION_SCHEMA,
            max_tokens=self.options.vision_max_tokens,
            temperature=self.options.vision_temperature,
            timeout_s=self.options.vision_timeout_s,
        )
        if not result.ok:
            self.journal.event("vision_error", error=result.error)
            return None

        mapper = CoordinateMapper(
            capture_size=(int(scaled.shape[1]), int(scaled.shape[0])),
            screen_size=self.deps.screen,
            region_size=frame.size,
            origin=frame.origin,
        )
        return parse_vision_output(
            result.text,
            mapper,
            # Must be the same list, in the same order, that generated the grammar --
            # the model reports a label by its index into it.
            vocabulary=vision_vocabulary(state.spec.watch),
            frame_id=frame.frame_id,
            latency_ms=result.latency_ms,
            max_elements=settings.max_elements,
        )

    # -- decision --------------------------------------------------------------------

    async def _actuate(
        self, state: CompiledState, percept: Perceived, record: CycleRecord
    ) -> None:
        observation = percept.observation
        centres = [e.center for e in observation.elements]
        targets = self._legal_targets(state)

        # Have we already answered this exact question? The actuator's decision is a
        # function of the state, what is on screen, and the readings -- all of which
        # recur constantly in a game loop. A hit returns the model's own previous answer
        # to a situation this close, in microseconds instead of ~300 ms.
        recall_key = situation_key(
            state.name, [e.label for e in observation.elements], targets
        )
        recall_vector = self._recall_vector(percept.probes or self.probes.last)
        self._last_recall = (recall_key, recall_vector)
        remembered = self.recall.lookup(recall_key, recall_vector)
        if remembered is not None:
            await self._replay_remembered(
                remembered, observation, state, targets, centres, record
            )
            return

        policy = self.playbook.spec.policy
        max_actions = min(policy.max_actions_per_burst, 20)
        max_text_len = min(policy.max_text_len, 96)

        grammar = None
        schema = None
        if self.deps.actuator.supports_grammar:
            grammar = self._cached_grammar(
                ("actuator", state.name, tuple(targets), len(centres),
                 tuple(sorted(state.allow_verbs))),
                lambda: actuator_grammar(
                    allow_verbs=sorted(state.allow_verbs),
                    targets=targets,
                    allow_keys=policy.allow_keys,
                    deny_keys=policy.deny_keys,
                    max_actions=max_actions,
                    max_text_len=max_text_len,
                    n_elements=len(centres),
                ),
            )
        else:
            schema = BURST_SCHEMA

        # Derived from the grammar rather than configured, so the two cannot disagree.
        # A ceiling below what the grammar permits does not shorten the burst -- it cuts
        # it off mid-action, and the whole cycle is then thrown away on a parse error.
        max_tokens = max(
            self.options.actuator_max_tokens,
            actuator_token_budget(
                max_actions=max_actions, max_text_len=max_text_len, targets=targets
            ),
        )

        prompt = actuator_prompt(
            state,
            observation,
            variables=self.vars,
            targets=targets,
            screen=self.deps.screen,
            last_burst=self._last_burst,
            last_result=self._last_result,
            steer_hint=self._steer_hint,
            cycles_in_state=self._cycles_in_state,
            probes=percept.probes or self.probes.last,
            holding=sorted(self._latched),
        )

        t0 = time.perf_counter()
        result = await self.deps.actuator.generate(
            prompt,
            system=ACTUATOR_SYSTEM,
            grammar=grammar,
            schema=schema,
            max_tokens=max_tokens,
            temperature=self.options.actuator_temperature,
            timeout_s=self.options.actuator_timeout_s,
            stop=["\n\n"],
        )
        actuator_ms = (time.perf_counter() - t0) * 1000.0
        self._timings["actuator"].append(actuator_ms)
        record.timing["actuator_ms"] = round(actuator_ms, 1)
        record.timing["capture_ms"] = round(percept.capture_ms, 1)
        record.timing["vision_ms"] = round(percept.vision_ms, 1)

        if not result.ok:
            record.error = result.error
            record.note = "actuator unavailable"
            self._last_result = "backend error"
            return

        burst_str, proposed, note = _split_actuator_reply(result.text)
        record.burst = burst_str
        record.burst_source = "actuator"
        record.proposed_state = proposed
        record.note = note

        try:
            burst = parse_burst(burst_str, screen=self.deps.screen, elements=centres)
        except BurstParseError as exc:
            record.allowed = False
            record.error = f"unparseable burst: {exc.detail}"
            self._last_burst = burst_str
            self._last_result = "rejected: could not be parsed"
            self.governor.rejections += 1
            return

        report, verdict = await self._guarded_execute(
            burst, observation, state, source="actuator"
        )
        record.allowed = verdict.allowed
        record.violations = [v.as_dict() for v in verdict.violations]
        record.executed = bool(report and report.ok and not self.dry_run)
        if report is not None:
            record.timing["execute_ms"] = round(report.duration_ms, 1)

        self._last_burst = burst_str
        if not verdict.allowed:
            self._last_result = f"refused ({verdict.violations[0].rule})"
        elif report is not None and not report.ok:
            self._last_result = f"failed: {report.error}"
        else:
            self._last_result = "ok"
            # Remember it only now: parsed, allowed by the governor, and executed
            # without error. Anything short of that would teach the table to repeat a
            # failure faster than it made it.
            self.recall.store(
                *self._last_recall,
                burst=burst_str,
                proposed_state=proposed if proposed in targets else None,
                note=note,
            )

        # The actuator may only propose transitions the state already declares; the
        # grammar restricts it to `targets`, and this re-checks in case a non-grammar
        # backend produced something else.
        if proposed and proposed != "." and verdict.allowed:
            if proposed in targets:
                record.transition = proposed
                await self._goto(proposed, state, record)
            else:
                record.note = (
                    f"{note} [ignored illegal transition to {proposed!r}]".strip()
                )

    def _recall_vector(self, probes: dict[str, float]) -> dict[str, float]:
        """The continuous part of a situation, as the cache sees it.

        Only the playbook's declared probes and their rates. The engine's internals are
        excluded on purpose: `__static_for__` grows without bound, so including it would
        make every situation drift steadily away from every stored one and the cache
        would never hit twice.
        """
        wanted = self.playbook.probe_ids
        return {
            key: value
            for key, value in probes.items()
            if key in wanted or key.removesuffix("__rate") in wanted
            if not key.startswith("__")
        }

    async def _replay_remembered(
        self,
        entry: CacheEntry,
        observation: Observation,
        state: CompiledState,
        targets: list[str],
        centres: list[tuple[int, int]],
        record: CycleRecord,
    ) -> None:
        """Execute a decision recalled from the cache, with the full safety path intact.

        Deliberately not a shortcut around the governor. A remembered burst is re-parsed
        and re-reviewed against *this* cycle's observation and policy, because what made
        it safe last time was the situation, and the situation is only approximately the
        same. Skipping that would make the cache a hole in the safety model.
        """
        record.burst_source = "recall"
        record.burst = entry.burst
        record.note = entry.note
        try:
            burst = parse_burst(entry.burst, screen=self.deps.screen, elements=centres)
        except BurstParseError as exc:
            # The stored burst referenced an element index this cycle does not have.
            record.error = f"recalled burst no longer parses: {exc.detail}"
            self.recall.penalise(*self._last_recall)
            return

        report, verdict = await self._guarded_execute(
            burst, observation, state, source="recall"
        )
        record.allowed = verdict.allowed
        record.violations = [v.as_dict() for v in verdict.violations]
        record.executed = bool(report and report.ok and not self.dry_run)
        self._last_burst = entry.burst
        self._last_result = "ok" if verdict.allowed else "refused"
        if not verdict.allowed:
            self.recall.penalise(*self._last_recall)
            return

        proposed = entry.proposed_state
        if proposed and proposed in targets:
            record.transition = proposed
            await self._goto(proposed, state, record)

    async def _guarded_execute(
        self,
        burst: Burst,
        observation: Observation,
        state: CompiledState,
        *,
        source: str,
    ) -> tuple[ExecutionReport | None, Verdict]:
        verdict = self.governor.review(
            burst, observation=observation, allow_verbs=state.allow_verbs, source=source
        )
        if not verdict.allowed:
            self.journal.event(
                "refused", source=source, burst=burst.render(), **verdict.as_dict()
            )
            return None, verdict

        if not burst.actions:
            return None, verdict

        self._bursts += 1
        report = await asyncio.to_thread(self.deps.executor.run, burst, label=source)
        self._timings["execute"].append(report.duration_ms)

        # Let the UI catch up with what the burst just did, then start the next capture
        # and vision call so they run during the remaining inter-cycle wait instead of
        # strictly after it.
        if self.options.settle_ms:
            await asyncio.sleep(self.options.settle_ms / 1000.0)
        self._schedule_perception(state)

        await asyncio.to_thread(self.deps.executor.enforce_hold_watchdog)
        return report, verdict

    # -- transitions and state -------------------------------------------------------

    def _legal_targets(self, state: CompiledState) -> list[str]:
        return [tr.to for _, tr in state.transitions if tr.to not in TERMINALS]

    async def _maybe_transition(
        self, state: CompiledState, ctx: GuardContext, record: CycleRecord
    ) -> bool | None:
        """Returns None if no transition fired, else whether the run continues."""
        if self._forced_state is not None:
            target, self._forced_state = self._forced_state, None
            record.transition = target
            record.note = "forced by steer()"
            await self._goto(target, state, record)
            return target not in TERMINALS

        for guard, transition in state.transitions:
            if not self._test(guard, ctx, f"{state.name}.transition"):
                continue
            for key, value in transition.set.items():
                self.vars[key] = value
            for key, amount in transition.inc.items():
                self.vars[key] = _as_number(self.vars.get(key, 0)) + amount
            record.transition = transition.to
            if transition.note:
                record.note = transition.note
            if transition.ends_episode:
                await self._close_episode(note=transition.note or transition.to)
            await self._goto(transition.to, state, record)
            return transition.to not in TERMINALS
        return None

    # -- episodes --------------------------------------------------------------------

    def _begin_episode(self) -> None:
        """Snapshot the reward baseline and adopt the next set of constants to try."""
        self._episode_index += 1
        self._episode_started_at = time.monotonic()
        spec = self.playbook.spec.reward
        self._episode_start_value = (
            float(self.probes.last.get(spec.probe, 0.0)) if spec else 0.0
        )
        self._tune_params = self._tuner.current()

    async def _close_episode(self, *, note: str = "") -> None:
        """Score the episode that just ended and hand the result to the optimiser.

        The settle wait is not padding. Score counters animate -- money rolls up over a
        second or so -- and reading one the instant the episode ends scores the animation
        rather than the outcome, which teaches the optimiser noise.
        """
        spec = self.playbook.spec.reward
        if spec is None:
            return
        if spec.settle_ms:
            await asyncio.sleep(spec.settle_ms / 1000.0)
            # Force a fresh sample so the settled value is actually read rather than
            # taken from whatever the reflex loop last happened to store.
            await asyncio.to_thread(self._sample_fresh, 0.0)

        seconds = max(1e-3, time.monotonic() - self._episode_started_at)
        final = float(self.probes.last.get(spec.probe, 0.0))
        delta = final - self._episode_start_value
        if spec.mode == "final":
            reward = final
        elif spec.mode == "rate":
            reward = delta / seconds
        else:
            reward = delta

        params = dict(self._tune_params)
        # An episode that scored worse than the running best is evidence against the
        # decisions that produced it. One cycle cannot know this; the episode can.
        if self._tuner.best_score is not None and reward < self._tuner.best_score * 0.5:
            self.recall.penalise(*self._last_recall)
        self._tuner.record(reward, seconds=seconds, note=note)
        self.journal.event(
            "episode",
            index=self._episode_index,
            reward=round(reward, 3),
            seconds=round(seconds, 2),
            start=round(self._episode_start_value, 3),
            final=round(final, 3),
            params={k: round(v, 3) for k, v in params.items()},
            best={k: round(v, 3) for k, v in self._tuner.best.items()},
            sigma=self._tuner.summary()["sigma"],
            note=note,
        )
        self._begin_episode()

    async def _goto(self, target: str, current: CompiledState, record: CycleRecord) -> None:
        # Latches go first, before on_exit and before the state changes. A hold is driven
        # by a guard that is about to stop being evaluated, so anything still down would
        # stay down with nothing left to release it.
        await self._release_latches(state=current.name, why=f"leaving {current.name}")

        if current.on_exit:
            await self._run_template(current.on_exit, current, source="on_exit")

        if target == "@success":
            self._finish("succeeded", f"reached @success from {current.name}")
            return
        if target == "@failure":
            self._finish("failed", f"reached @failure from {current.name}")
            return
        if target == "@stop":
            self._finish("stopped", f"reached @stop from {current.name}")
            return

        await self._enter_state(target)

    async def _enter_state(self, name: str, *, first: bool = False) -> None:
        state = self.playbook.state(name)
        self.state_name = name
        self._state_entered_at = time.monotonic()
        self._cycles_in_state = 0
        self._rule_fired: dict[str, tuple[float, int]] = {}
        self.governor.reset_rate()
        self.journal.event("state", name=name, brief=state.spec.brief, first=first)

        if state.on_enter:
            await self._run_template(state.on_enter, state, source="on_enter")

    async def _run_template(
        self, template: BurstTemplate, state: CompiledState, *, source: str
    ) -> None:
        """Render a playbook-authored burst against the live context, then execute it.

        Rendering can fail at runtime in a way the compile-time check cannot catch -- a
        steering expression that overshoots produces an off-screen move. That is
        journalled with the rendered text, which is the thing the author needs to see,
        and the hook is skipped rather than the run being killed over it.
        """
        ctx = self._context_from(self.probes.last)
        try:
            burst = template.burst(ctx, screen=self.deps.screen)
        except (BurstParseError, ExpressionError) as exc:
            self.journal.event(
                "burst_error", source=source, template=template.source, error=str(exc)
            )
            return
        await self._guarded_execute(burst, self._last_observation, state, source=source)

    def _state_limit(self, state: CompiledState) -> str | None:
        if state.spec.max_cycles and self._cycles_in_state >= state.spec.max_cycles:
            return f"state cycle limit reached ({state.spec.max_cycles})"
        if state.spec.timeout_s:
            elapsed = time.monotonic() - self._state_entered_at
            if elapsed >= state.spec.timeout_s:
                return f"state timeout reached ({state.spec.timeout_s:.1f}s)"
        return None

    # -- reflexes --------------------------------------------------------------------

    def _pick_reflex(self, state: CompiledState, ctx: GuardContext) -> CompiledReflex | None:
        now = time.monotonic()
        for reflex in state.reflexes:  # already sorted by priority
            rule = reflex.rule
            last, count = self._rule_fired.get(rule.id, (0.0, 0))
            if rule.max_fires is not None and count >= rule.max_fires:
                continue
            if last and (now - last) * 1000.0 < rule.cooldown_ms:
                continue
            if self._test(reflex.guard, ctx, f"{state.name}.reflex.{rule.id}"):
                return reflex
        return None

    # -- guards ----------------------------------------------------------------------

    def _context(self, percept: Perceived) -> GuardContext:
        return self._context_from(percept.probes, observation=percept.observation)

    def _context_from(
        self, probes: dict[str, float], *, observation: Observation | None = None
    ) -> GuardContext:
        """Build a guard context.

        The reflex loop passes no observation and gets the last one the vision model
        produced, which is correct: a reflex must never wait for fresh vision, and the
        cached labels are the best information available at reflex rate. What it does get
        fresh is the probes, `held` and `latched` -- everything a fast rule should be
        keying off.
        """
        obs = self._last_observation if observation is None else observation
        return GuardContext(
            elements=[
                {"label": e.label, "conf": e.conf, "box": [e.x, e.y, e.w, e.h]}
                for e in obs.elements
            ],
            texts=list(obs.texts),
            flags=set(obs.flags),
            scene=obs.scene,
            probes=dict(probes),
            vars=self.vars,
            held=self.deps.executor.held_names(),
            latched=set(self._latched),
            tunables=self._tune_params,
            state={
                "name": self.state_name,
                "cycles": self._cycles_in_state,
                "elapsed": time.monotonic() - self._state_entered_at,
            },
            run={
                "cycles": self._cycle,
                "elapsed": time.monotonic() - self._started_at,
                "bursts": self._bursts,
                "reflex_fires": self._reflex_fires,
                "reflex_hz": self.measured_reflex_hz(),
                "rejections": self.governor.rejections,
                "last_burst_ok": self._last_result in ("ok", ""),
            },
        )

    def _test(self, guard, ctx: GuardContext, where: str) -> bool:
        try:
            return guard.test(ctx)
        except ExpressionError as exc:
            # A guard that blows up is journalled and treated as False rather than
            # killing the run -- one bad expression should not strand the desktop.
            self.journal.event("guard_error", where=where, guard=guard.source, error=str(exc))
            return False

    # -- budget ----------------------------------------------------------------------

    def _budget_check(self) -> str | None:
        budget = self.playbook.spec.budget
        if self._cycle >= budget.max_cycles:
            return f"budget: reached max_cycles ({budget.max_cycles})"
        if self._bursts >= budget.max_bursts:
            return f"budget: reached max_bursts ({budget.max_bursts})"
        elapsed = time.monotonic() - self._started_at
        if elapsed >= budget.max_seconds:
            return f"budget: reached max_seconds ({budget.max_seconds:.0f}s)"
        if self.governor.budget_exhausted:
            return (
                f"budget: the governor refused {self.governor.rejections} bursts "
                f"(limit {budget.max_rejections}); the playbook's policy and what the "
                f"actuator wants to do disagree"
            )
        if budget.idle_abort_s:
            static_for = self.probes.last.get("__static_for__", 0.0)
            if static_for >= budget.idle_abort_s:
                return f"budget: the screen has not changed for {static_for:.0f}s"
        return None

    # -- control from the orchestrator ------------------------------------------------

    def steer(
        self,
        *,
        hint: str | None = None,
        variables: dict[str, Any] | None = None,
        force_state: str | None = None,
        dry_run: bool | None = None,
    ) -> dict[str, Any]:
        """Adjust a live run without restarting it."""
        changed: dict[str, Any] = {}
        if hint is not None:
            self._steer_hint = hint or None
            changed["hint"] = hint
        if variables:
            self.vars.update(variables)
            changed["variables"] = variables
        if force_state is not None:
            if force_state not in self.playbook.states and force_state not in TERMINALS:
                raise SessionError(
                    f"cannot force unknown state {force_state!r}; "
                    f"have {sorted(self.playbook.states)}"
                )
            self._forced_state = force_state
            changed["force_state"] = force_state
        if dry_run is not None:
            self.dry_run = dry_run
            self.deps.executor.dry_run = dry_run
            changed["dry_run"] = dry_run
        self.journal.event("steer", **changed)
        return changed

    def pause(self) -> None:
        self._pause.clear()
        self.status = "paused"
        # A latch pressed against a guard that is no longer being evaluated stays down
        # for the whole pause. Nothing about "hold W while descending" implies "keep W
        # down while a human inspects the screen".
        released = self.deps.executor.release_all()
        self._latched.clear()
        self.journal.event("pause", released=released)

    def resume(self) -> None:
        self._pause.set()
        if self.status == "paused":
            self.status = "running"
        self.journal.event("resume")

    async def stop(self, reason: str = "stopped by request") -> None:
        self._stop_requested = True
        self._pause.set()
        self.deps.executor.request_abort()
        self.killswitch.trip(reason)
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=8.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self.deps.executor.clear_abort()

    # -- reporting -------------------------------------------------------------------

    def measured_reflex_hz(self) -> float:
        """The rate the reflex loop actually achieved, not the one it was asked for.

        The configured number is an upper bound that a slow capture backend will not
        reach, and reporting the request instead of the result is precisely the kind of
        claim that makes a design look like it is working when it is not.
        """
        if not self.options.reflex_enabled or not self._reflex_started_at:
            return 0.0
        elapsed = time.monotonic() - self._reflex_started_at
        return round(self._reflex_ticks / elapsed, 1) if elapsed > 0.2 else 0.0

    def reflex_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.options.reflex_enabled,
            "requested_hz": self.options.reflex_hz if self.options.reflex_enabled else 0,
            "measured_hz": self.measured_reflex_hz(),
            "ticks": self._reflex_ticks,
            "fires": self._reflex_fires,
            "starved": self._reflex_starved,
            "latch_events": self._latch_events,
            "latched_now": sorted(self._latched),
            "ocr": self.probes.ocr_stats(),
            "errors": dict(self._reflex_errors),
        }

    def tuner_summary(self) -> dict[str, Any]:
        out = self._tuner.summary()
        out["loaded_from_disk"] = self._tuner_loaded
        out["improvement"] = self._tuner.improvement()
        out["live"] = {k: round(v, 3) for k, v in self._tune_params.items()}
        return out

    def timing_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, samples in self._timings.items():
            if not samples:
                continue
            ordered = sorted(samples)
            out[name] = {
                "n": len(samples),
                "mean_ms": round(sum(samples) / len(samples), 1),
                "p50_ms": round(ordered[len(ordered) // 2], 1),
                "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
            }
        return out

    def snapshot(self, journal_tail: int = 8) -> dict[str, Any]:
        elapsed = (time.monotonic() - self._started_at) if self._started_at else 0.0
        return {
            "run_id": self.id,
            "playbook": self.playbook.spec.name,
            "status": self.status,
            "reason": self.reason,
            "dry_run": self.dry_run,
            "state": self.state_name,
            "cycles": self._cycle,
            "cycles_in_state": self._cycles_in_state,
            "bursts": self._bursts,
            "reflex": self.reflex_summary(),
            "tuner": self.tuner_summary(),
            "recall": self.recall.stats(),
            "held": sorted(self.deps.executor.held_names()),
            "elapsed_s": round(elapsed, 1),
            "vars": dict(self.vars),
            "last_burst": self._last_burst,
            "last_result": self._last_result,
            "observation": self._last_observation.as_dict(),
            "probes": {k: round(v, 4) for k, v in self.probes.last.items()},
            "governor": {
                "approved": self.governor.approvals,
                "refused": self.governor.rejections,
                "recent": self.governor.recent_violations(),
            },
            "killswitch": self.killswitch.status(),
            "timings": self.timing_summary(),
            "journal": {
                "path": str(self.journal.path),
                "recent": [
                    {
                        "cycle": c.get("cycle"),
                        "state": c.get("state"),
                        "perception": c.get("perception"),
                        "burst": c.get("burst"),
                        "source": c.get("burst_source"),
                        "allowed": c.get("allowed"),
                        "transition": c.get("transition"),
                        "note": c.get("note"),
                        "error": c.get("error"),
                    }
                    for c in self.journal.cycles(journal_tail)
                ],
            },
        }

    def _cached_grammar(self, key: tuple, build) -> str:
        cached = self._grammar_cache.get(key)
        if cached is None:
            cached = build()
            self._grammar_cache[key] = cached
        return cached


# -- helpers -------------------------------------------------------------------------------


def _split_actuator_reply(text: str) -> tuple[str, str | None, str]:
    """Parse ``<burst>|<target>|<note>``.

    Under a grammar this always has the right shape. Without one (Ollama), the reply is
    JSON instead, so both forms are accepted.
    """
    text = (text or "").strip()
    if not text:
        return "", None, ""

    if text.startswith("{"):
        import json

        try:
            data = json.loads(text)
            return (
                str(data.get("burst") or "."),
                (str(data.get("next")) or None) if data.get("next") not in (None, ".") else None,
                str(data.get("note") or "")[:80],
            )
        except (ValueError, TypeError):
            return text, None, "unparseable JSON reply"

    parts = text.split("|")
    burst = parts[0].strip()
    target = parts[1].strip() if len(parts) > 1 else ""
    note = parts[2].strip()[:80] if len(parts) > 2 else ""
    return burst, (target if target and target != "." else None), note


def _as_number(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0
