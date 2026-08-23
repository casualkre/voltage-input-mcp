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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..capture import CaptureBackend, Frame, ProbeEngine, encode_png
from ..errors import Aborted, BurstParseError, ExpressionError, SessionError
from ..expr import GuardContext
from ..inputs import DeviceSet, ExecutionReport, Executor
from ..llm import Backend, actuator_grammar, observation_grammar
from ..llm.ollama import BURST_SCHEMA, OBSERVATION_SCHEMA
from ..models.burst import Burst, parse_burst
from ..models.observation import CoordinateMapper, Observation, parse_vision_output
from ..models.playbook import TERMINALS, CompiledPlaybook, CompiledState, Perception
from ..safety import Governor, KillSwitch, Verdict
from .journal import CycleRecord, Journal
from .prompts import ACTUATOR_SYSTEM, VISION_SYSTEM, actuator_prompt, vision_prompt

__all__ = ["Session", "SessionOptions", "SessionDeps", "RunStatus"]

RunStatus = str  # "pending" | "running" | "paused" | "succeeded" | "failed" | "stopped" | "error"


@dataclass(slots=True)
class SessionOptions:
    target_period_s: float = 0.5
    dry_run: bool | None = None          # None -> use the playbook's policy
    keep_frames: bool = False
    settle_ms: int = 60                  # pause after a burst before looking again
    vision_max_tokens: int = 192
    actuator_max_tokens: int = 96
    vision_timeout_s: float = 15.0
    actuator_timeout_s: float = 12.0
    vision_temperature: float = 0.1
    actuator_temperature: float = 0.25
    watch_physical_input: bool = True


@dataclass(slots=True)
class SessionDeps:
    capture: CaptureBackend
    vision: Backend
    actuator: Backend
    devices: DeviceSet
    executor: Executor
    screen: tuple[int, int]


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
        self._timings: dict[str, list[float]] = {"capture": [], "vision": [], "actuator": [],
                                                 "execute": [], "cycle": []}

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

        try:
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

    async def _cleanup(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        # Releasing held input is the one cleanup step that must never be skipped.
        released = await asyncio.to_thread(self.deps.executor.release_all)
        self.killswitch.stop()
        self.journal.event(
            "end",
            status=self.status,
            reason=self.reason,
            cycles=self._cycle,
            bursts=self._bursts,
            released=released,
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

        # 2. Reflexes: no model in the path.
        reflex = self._pick_reflex(state, ctx)
        if reflex is not None:
            guard, burst, rule = reflex
            record.burst_source = f"reflex:{rule.id}"
            record.burst = burst.render()
            report, verdict = await self._guarded_execute(
                burst, percept.observation, state, source=record.burst_source
            )
            record.allowed = verdict.allowed
            record.violations = [v.as_dict() for v in verdict.violations]
            record.executed = bool(report and report.ok and not self.dry_run)
            self._rule_fired[rule.id] = (time.monotonic(), self._rule_fired.get(rule.id, (0, 0))[1] + 1)
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

        # 5. The actuator.
        if state.spec.autonomous:
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
            region = settings.region.as_tuple() if settings.region else None
            frame = await asyncio.to_thread(self.deps.capture.grab, region)
        except Exception as exc:  # noqa: BLE001
            result.error = f"capture failed: {exc}"
            result.observation = self._last_observation
            result.source = "cache"
            return result
        result.capture_ms = (time.perf_counter() - t0) * 1000.0
        self._timings["capture"].append(result.capture_ms)
        result.frame = frame

        result.probes = await asyncio.to_thread(self.probes.evaluate, frame)

        if not self._should_run_vision(settings, result.probes):
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

        self._last_observation = observation
        result.observation = observation
        result.source = "vlm"
        return result

    def _should_run_vision(self, settings: Perception, probes: dict[str, float]) -> bool:
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
        # on_change
        return probes.get("__frame_delta__", 1.0) >= settings.change_threshold

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

        grammar = None
        schema = None
        if self.deps.actuator.supports_grammar:
            grammar = self._cached_grammar(
                ("actuator", state.name, tuple(targets), len(centres),
                 tuple(sorted(state.allow_verbs))),
                lambda: actuator_grammar(
                    allow_verbs=sorted(state.allow_verbs),
                    targets=targets,
                    allow_keys=self.playbook.spec.policy.allow_keys,
                    deny_keys=self.playbook.spec.policy.deny_keys,
                    max_actions=min(self.playbook.spec.policy.max_actions_per_burst, 20),
                    max_text_len=min(self.playbook.spec.policy.max_text_len, 96),
                    n_elements=len(centres),
                ),
            )
        else:
            schema = BURST_SCHEMA

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
        )

        t0 = time.perf_counter()
        result = await self.deps.actuator.generate(
            prompt,
            system=ACTUATOR_SYSTEM,
            grammar=grammar,
            schema=schema,
            max_tokens=self.options.actuator_max_tokens,
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
            await self._goto(transition.to, state, record)
            return transition.to not in TERMINALS
        return None

    async def _goto(self, target: str, current: CompiledState, record: CycleRecord) -> None:
        if current.on_exit:
            await self._guarded_execute(
                current.on_exit, self._last_observation, current, source="on_exit"
            )

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
            await self._guarded_execute(
                state.on_enter, self._last_observation, state, source="on_enter"
            )

    def _state_limit(self, state: CompiledState) -> str | None:
        if state.spec.max_cycles and self._cycles_in_state >= state.spec.max_cycles:
            return f"state cycle limit reached ({state.spec.max_cycles})"
        if state.spec.timeout_s:
            elapsed = time.monotonic() - self._state_entered_at
            if elapsed >= state.spec.timeout_s:
                return f"state timeout reached ({state.spec.timeout_s:.1f}s)"
        return None

    # -- reflexes --------------------------------------------------------------------

    def _pick_reflex(self, state: CompiledState, ctx: GuardContext):
        now = time.monotonic()
        for guard, burst, rule in state.reflexes:  # already sorted by priority
            last, count = self._rule_fired.get(rule.id, (0.0, 0))
            if rule.max_fires is not None and count >= rule.max_fires:
                continue
            if last and (now - last) * 1000.0 < rule.cooldown_ms:
                continue
            if self._test(guard, ctx, f"{state.name}.reflex.{rule.id}"):
                return guard, burst, rule
        return None

    # -- guards ----------------------------------------------------------------------

    def _context(self, percept: Perceived) -> GuardContext:
        observation = percept.observation
        return GuardContext(
            elements=[
                {"label": e.label, "conf": e.conf, "box": [e.x, e.y, e.w, e.h]}
                for e in observation.elements
            ],
            texts=list(observation.texts),
            flags=set(observation.flags),
            scene=observation.scene,
            probes=dict(percept.probes),
            vars=self.vars,
            state={
                "name": self.state_name,
                "cycles": self._cycles_in_state,
                "elapsed": time.monotonic() - self._state_entered_at,
            },
            run={
                "cycles": self._cycle,
                "elapsed": time.monotonic() - self._started_at,
                "bursts": self._bursts,
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
        self.journal.event("pause")

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
