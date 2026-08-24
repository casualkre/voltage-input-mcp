"""The MCP surface.

Tool design follows one rule: the orchestrator is remote, slow, and blind. Every tool
either gives it eyes (`capture`, `observe`, `status`, `journal`), lets it verify before
committing (`validate_playbook`, `calibrate`, `doctor`), or gives it hands
(`execute_burst`, `run`, `steer`). Nothing returns a bare success/failure -- if a call
fails, it returns the reason *and* what to do about it, because a round trip to ask is
expensive.

`voltage_reference` exists so the orchestrator can author a correct playbook without
having read this repository. It returns the burst syntax, the playbook schema, the guard
function table, and a worked example. Call it once before writing the first playbook.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.types import ToolAnnotations

from .app import App, get_app
from .briefing import active_build, briefing_text
from .capture import encode_png
from .errors import VoltageError
from .llm import observation_grammar
from .llm.grammar import vision_vocabulary
from .models.burst import parse_burst
from .models.observation import CoordinateMapper, Observation, parse_vision_output
from .models.playbook import VERB_NAMES, Playbook, playbook_from_dict
from .runtime import Session
from .runtime.prompts import VISION_SYSTEM


def _instructions() -> str:
    """Built at startup so it describes the configuration actually running.

    The same Playbook can be sound on one build and wrong on another -- grammars are
    enforced on llama.cpp and absent on Ollama, and the `hyper` profile cannot ground at
    all -- and none of that is visible to a remote orchestrator. Stating it up front is
    cheaper than a tool call it may not think to make.
    """
    base = (
        "Three-layer input engine. You are the orchestrator: you write a Playbook -- a "
        "state machine with probes, guards, limits and a safety policy -- and it is run "
        "by two small local models plus a model-free fast loop.\n"
        "\n"
        "  decision  ~2 Hz. The two models: one reads the screen, one emits a burst.\n"
        "  burst     ms precision. One decision produces many timed inputs.\n"
        "  reflex    ~20 Hz. Guards over probes, evaluated with no model in the path.\n"
        "            Latched `hold` rules keep a key down for exactly as long as a "
        "condition lasts.\n"
        "\n"
        "Driving this from the decision layer alone gets you 2 Hz and no reactions "
        "between decisions. If your playbook declares no probes and no reflexes, you are "
        "using a remote keyboard with extra steps -- read "
        "voltage_reference(section='control') first.\n"
        "\n"
        "Then voltage_reference(section='loop') for the iteration loop and "
        "section='bursts' for chaining inputs well. voltage_doctor confirms the machine "
        "is ready. After a run that did not do what you wanted, call voltage_diagnose "
        "rather than reading the journal by hand.\n"
        "\n"
        "The local models are small and take instructions literally. Give each state a "
        "short imperative brief, a closed list of things to look for, and explicit "
        "transitions. Do not expect them to infer intent.\n"
    )
    briefing = briefing_text()
    return f"{base}\n{briefing}" if briefing else base


server = MCPServer(
    name="voltage-input",
    version="0.1.0",
    instructions=_instructions(),
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
ACTS = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True)


def _app() -> App:
    return get_app()


def _error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, VoltageError):
        return {"ok": False, **exc.to_dict()}
    return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


# ======================================================================================
# Discovery
# ======================================================================================


@server.tool(annotations=READ_ONLY)
async def voltage_doctor() -> dict[str, Any]:
    """Check that everything needed for a run is present and working.

    Reports the session type, input-device permissions, which capture backends work,
    detected screen geometry, GPU memory versus the selected model profile, and whether
    both model backends respond. When something is missing it returns the exact command
    to fix it. Call this before the first run on a machine.
    """
    try:
        return {"ok": True, **await _app().doctor()}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@server.tool(annotations=READ_ONLY)
async def voltage_reference(
    section: Literal[
        "all", "burst", "bursts", "control", "playbook", "guards", "example", "loop"
    ] = "all",
) -> dict[str, Any]:
    """Return everything needed to author and iterate on a run.

    Call this before your first Playbook. Sections:

      loop      **the learning loop** -- how to go from a failed run to a working one,
                and what each failure mode actually means. Read this second.
      control   **continuous control** -- probes, latched holds and interpolated bursts:
                the layer that runs at ~20 Hz between decisions. Read this before
                driving anything with timing in it, including any game.
      bursts    the burst cookbook: how to chain inputs well, timing rules, ready-made
                patterns for desktop and for games, and the antipatterns that waste
                cycles. Read this if bursts are coming out one action at a time.
      burst     the raw burst syntax
      playbook  the state-machine JSON schema
      guards    expression functions for transitions and reflexes
      example   a complete working Playbook
    """
    from .expr import GUARD_FUNCTIONS, GUARD_NAMESPACES
    from .reference import (
        BURST_COOKBOOK,
        BURST_REFERENCE,
        CONTINUOUS_CONTROL,
        EXAMPLE_PLAYBOOK,
        GUARD_REFERENCE,
        LEARNING_LOOP,
    )

    # Always included, regardless of section: the profile can change mid-session, and
    # the copy in the server instructions was fixed at startup.
    out: dict[str, Any] = {"ok": True, "active_build": active_build()}
    if section in ("all", "loop"):
        out["learning_loop"] = LEARNING_LOOP
    if section in ("all", "control"):
        out["continuous_control"] = CONTINUOUS_CONTROL
    if section in ("all", "bursts"):
        out["burst_cookbook"] = BURST_COOKBOOK
    if section in ("all", "burst"):
        out["burst"] = BURST_REFERENCE
        out["verbs"] = VERB_NAMES
    if section in ("all", "playbook"):
        out["playbook_schema"] = Playbook.model_json_schema()
    if section in ("all", "guards"):
        out["guards"] = GUARD_REFERENCE
        out["guard_functions"] = sorted(GUARD_FUNCTIONS)
        out["guard_namespaces"] = sorted(GUARD_NAMESPACES)
    if section in ("all", "example"):
        out["example"] = EXAMPLE_PLAYBOOK
    return out


# ======================================================================================
# Eyes
# ======================================================================================


@server.tool(annotations=READ_ONLY)
async def voltage_capture(
    region: list[int] | None = None,
    max_width: int = 1280,
) -> Image:
    """Take a screenshot and return it to you directly.

    Use this to see the screen yourself -- before writing a Playbook, to pick coordinates
    for probes and click regions, or to work out why a run went wrong. This does not
    involve the local vision model.

    `region` is [x, y, width, height] in desktop pixels; omit for the whole desktop.
    """
    app = _app()
    rect = tuple(region) if region and len(region) == 4 else None
    frame = await asyncio.to_thread(app.capture().grab, rect)  # type: ignore[arg-type]
    scale = min(1.0, max_width / frame.width) if frame.width else 1.0
    target = (int(frame.width * scale), int(frame.height * scale))
    png = await asyncio.to_thread(encode_png, frame.downscaled(target), 6)
    return Image(data=png, format="png")


@server.tool(annotations=READ_ONLY)
async def voltage_observe(
    watch: list[str],
    region: list[int] | None = None,
    max_elements: int = 6,
    read_text: bool = True,
) -> dict[str, Any]:
    """Run one vision pass and return grounded elements in screen coordinates.

    `watch` is the closed vocabulary the vision model may use -- it can only report
    labels from this list, so name the things your Playbook's guards will test for.

    Use this to check that the vision model can actually find what a state depends on
    before committing to it in a Playbook. If an element does not come back here, a
    `sees(...)` guard on it will never fire.
    """
    app = _app()
    try:
        vision, _ = app.backends()
        rect = tuple(region) if region and len(region) == 4 else None
        frame = await asyncio.to_thread(app.capture().grab, rect)  # type: ignore[arg-type]
        scaled = frame.downscaled((896, 504))
        png = await asyncio.to_thread(encode_png, scaled, 1)

        grammar = observation_grammar(
            watch, max_elements=max_elements, read_text=read_text
        )
        result = await vision.generate(
            "LOOK FOR: " + (", ".join(watch) if watch else "(anything notable)"),
            system=VISION_SYSTEM,
            image_png=png,
            grammar=grammar if vision.supports_grammar else None,
            max_tokens=192,
            temperature=0.1,
            timeout_s=25.0,
        )
        if not result.ok:
            return {"ok": False, "error": "vision_backend", "detail": result.error}

        mapper = CoordinateMapper(
            capture_size=(int(scaled.shape[1]), int(scaled.shape[0])),
            screen_size=app.screen(),
            region_size=frame.size,
            origin=frame.origin,
        )
        observation = parse_vision_output(
            result.text,
            mapper,
            vocabulary=vision_vocabulary(watch),
            latency_ms=result.latency_ms,
            max_elements=max_elements,
        )
        return {
            "ok": True,
            "observation": observation.as_dict(),
            "raw": result.text,
            "model": result.as_dict(),
            "screen": list(app.screen()),
            "hint": (
                "Elements are in desktop pixels. In a burst, prefer g:<index> over "
                "m:<x>,<y> -- the index refers to this list."
                if observation.elements
                else "Nothing from `watch` was found. Try broader or more visual labels, "
                     "or check voltage_capture to see what is actually on screen."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ======================================================================================
# Hands
# ======================================================================================


@server.tool(annotations=ACTS)
async def voltage_execute_burst(
    burst: str,
    dry_run: bool = True,
    label: str = "manual",
) -> dict[str, Any]:
    """Execute one input burst yourself, bypassing the local models entirely.

    For moments that need your judgement rather than the actuator's: opening the right
    application, clicking a specific confirmed target, typing something exact. Also the
    fastest way to sanity-check that input injection works at all.

    Syntax: `m:640,360;c:l;w:120;t:"hello";k:enter`. Call voltage_reference for the full
    list. The safety policy still applies. Defaults to dry_run, so pass dry_run=false to
    actually inject.
    """
    app = _app()
    try:
        parsed = parse_burst(burst, screen=app.screen())
    except VoltageError as exc:
        return _error(exc)

    from .models.playbook import Policy
    from .safety import Governor

    governor = Governor(Policy(dry_run=dry_run), screen=app.screen())
    verdict = governor.review(parsed, observation=Observation(), source=label)
    if not verdict.allowed:
        return {
            "ok": False,
            "error": "safety_violation",
            "burst": parsed.render(),
            **verdict.as_dict(),
        }

    executor = app.executor()
    previous, executor.dry_run = executor.dry_run, dry_run
    try:
        report = await asyncio.to_thread(executor.run, parsed, label=label)
    finally:
        executor.dry_run = previous

    return {
        "ok": report.ok,
        "burst": parsed.render(),
        "actions": len(parsed),
        "estimated_ms": parsed.duration_ms,
        "report": report.as_dict(),
    }


@server.tool(annotations=ACTS)
async def voltage_calibrate(dry_run: bool = False) -> dict[str, Any]:
    """Verify that input injection actually reaches the compositor.

    Creates the virtual devices, moves the pointer to three known points, and captures
    after each to confirm the cursor moved. Reports whether absolute positioning works or
    whether the relative fallback is needed -- which cannot be known without trying,
    since it depends on how libinput classified the virtual device.

    Run this once per machine before trusting a real (non-dry-run) Playbook.
    """
    app = _app()
    try:
        screen = app.screen()
        executor = app.executor()
        previous, executor.dry_run = executor.dry_run, dry_run
        results: list[dict[str, Any]] = []
        try:
            if not dry_run:
                await asyncio.to_thread(app.devices().open)
            targets = [
                (screen[0] // 4, screen[1] // 4),
                (screen[0] // 2, screen[1] // 2),
                (screen[0] * 3 // 4, screen[1] * 3 // 4),
            ]
            for x, y in targets:
                burst = parse_burst(f"m:{x},{y};w:120", screen=screen)
                report = await asyncio.to_thread(executor.run, burst, label="calibrate")
                results.append(
                    {
                        "target": [x, y],
                        "commanded": list(executor.cursor),
                        "ok": report.ok,
                        "error": report.error,
                    }
                )
        finally:
            executor.dry_run = previous

        return {
            "ok": all(r["ok"] for r in results),
            "dry_run": dry_run,
            "screen": list(screen),
            "pointer_mode": app.config.pointer_mode,
            "devices": [d.name for d in (
                app.devices().keyboard, app.devices().pointer_abs, app.devices().pointer_rel
            )],
            "moves": results,
            "note": (
                "Watch the real cursor while this runs. If it did not move, the virtual "
                "pointer was classified in a way KWin ignores -- set pointer_mode = "
                "\"relative\" in voltage.toml and run this again."
                if not dry_run
                else "dry_run: nothing was injected. Re-run with dry_run=false to test."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ======================================================================================
# Playbooks
# ======================================================================================


@server.tool(annotations=READ_ONLY)
async def voltage_validate_playbook(playbook: dict[str, Any]) -> dict[str, Any]:
    """Fully check a Playbook without running it.

    Validates the schema, compiles every guard expression, parses every burst, checks
    that transition targets and probe references exist, and reports unreachable states
    and dead transitions. Errors come back as a complete list, not one at a time.

    Always call this before voltage_run. Warnings are worth reading: "tests for X but X
    is not in `watch`" means a transition that can never fire.
    """
    try:
        compiled = playbook_from_dict(playbook)
    except VoltageError as exc:
        return {"ok": False, **exc.to_dict()}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)

    spec = compiled.spec
    return {
        "ok": True,
        "name": spec.name,
        "states": sorted(compiled.states),
        "initial": spec.initial,
        "probes": sorted(compiled.probe_ids),
        "warnings": compiled.warnings,
        "policy": {
            "dry_run": spec.policy.dry_run,
            "allow_verbs": spec.policy.allow_verbs,
            "max_actions_per_burst": spec.policy.max_actions_per_burst,
            "max_inputs_per_second": spec.policy.max_inputs_per_second,
            "require_target_element": spec.policy.require_target_element,
        },
        "budget": spec.budget.model_dump(),
    }


@server.tool(annotations=ACTS)
async def voltage_run(
    playbook: dict[str, Any],
    dry_run: bool | None = None,
    target_period_s: float | None = None,
    reflex_hz: float | None = None,
    keep_frames: bool = False,
) -> dict[str, Any]:
    """Start a Playbook. Returns immediately with a run_id; poll voltage_status.

    `dry_run` overrides the Playbook's policy. Leave it unset for the Playbook's own
    setting, which defaults to true. A dry run does everything except inject input, so
    it is the correct way to check that your states, guards and transitions behave before
    letting it touch the machine.

    `target_period_s` is the *decision* period -- how often the two models are asked.
    0.5 is a good default; lower it for games, raise it for slow UI.

    `reflex_hz` is the rate of the fast loop that evaluates probes, fires reflexes and
    updates latched holds with no model in the path. Default 20. This is the number that
    decides whether the run can react to anything faster than half a second, and it is
    independent of `target_period_s`. A slow capture backend will not reach the rate you
    ask for; `voltage_status` reports what was measured, not what was requested.

    Stop a run with voltage_stop, adjust it live with voltage_steer. The run also stops
    on its own budget, on any physical keyboard or mouse input from the user, and on the
    panic file.
    """
    app = _app()
    try:
        compiled = playbook_from_dict(playbook)
    except VoltageError as exc:
        return {"ok": False, **exc.to_dict()}

    try:
        options = app.session_options(
            target_period_s=target_period_s,
            reflex_hz=reflex_hz,
            keep_frames=keep_frames or None,
        )
        options.dry_run = dry_run
        session = Session(compiled, app.session_deps(), options)
        await app.register(session)
        session.start()
        await asyncio.sleep(0.05)  # let it reach "running" so the first status is useful
        return {
            "ok": True,
            "run_id": session.id,
            "status": session.status,
            "dry_run": session.dry_run,
            "state": session.state_name,
            "warnings": compiled.warnings,
            "advice": _fast_layer_advice(compiled),
            "journal": str(session.journal.path),
            "next": "poll voltage_status(run_id) to watch it, voltage_stop to end it",
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


def _fast_layer_advice(compiled: Any) -> list[str]:
    """Say up front when a playbook is about to run entirely at decision rate.

    Cheaper than letting the run finish and diagnosing it afterwards, and it lands at the
    moment the orchestrator is still holding the playbook in mind.
    """
    notes: list[str] = []
    holds = sum(len(s.holds) for s in compiled.states.values())
    reflexes = sum(len(s.reflexes) for s in compiled.states.values())
    if not compiled.spec.probes:
        notes.append(
            "No probes declared, so nothing can be measured between decisions and every "
            "guard depends on the vision model. For anything with timing in it, add a "
            "probe -- see voltage_reference(section='control')."
        )
    if not reflexes and not holds:
        notes.append(
            "No reflexes and no holds, so every input waits on a model decision (~2 Hz). "
            "Bursts still batch inputs within a decision, but nothing reacts between them."
        )
    return notes


@server.tool(annotations=READ_ONLY)
async def voltage_status(run_id: str | None = None, journal_tail: int = 8) -> dict[str, Any]:
    """Poll a run: current state, variables, last burst, what the vision model sees.

    Includes recent cycles, governor refusals, and per-stage timings so you can tell
    whether a slow loop is capture, vision, decision, or execution.
    """
    try:
        session = _app().get_session(run_id)
        return {"ok": True, **session.snapshot(journal_tail=journal_tail)}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@server.tool(annotations=ACTS)
async def voltage_steer(
    run_id: str | None = None,
    hint: str | None = None,
    variables: dict[str, Any] | None = None,
    force_state: str | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """Correct a live run without restarting it.

    `hint` is injected into the actuator's prompt as a supervisor note and persists until
    changed -- use it when the actuator is doing something legal but wrong.
    `force_state` jumps the machine on the next cycle. `variables` updates run variables.
    `dry_run` can be flipped either way mid-run.
    """
    try:
        session = _app().get_session(run_id)
        changed = session.steer(
            hint=hint, variables=variables, force_state=force_state, dry_run=dry_run
        )
        return {"ok": True, "run_id": session.id, "changed": changed, "state": session.state_name}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@server.tool(annotations=ACTS)
async def voltage_stop(run_id: str | None = None, reason: str = "stopped by orchestrator") -> dict[str, Any]:
    """Stop a run and release every held key and button.

    Safe to call at any time, including while a burst is mid-flight -- the burst is
    interrupted and anything held is released.
    """
    try:
        session = _app().get_session(run_id)
        await session.stop(reason)
        return {
            "ok": True,
            "run_id": session.id,
            "status": session.status,
            "reason": session.reason,
            "cycles": session.snapshot(journal_tail=0)["cycles"],
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@server.tool(annotations=ACTS)
async def voltage_pause(run_id: str | None = None, resume: bool = False) -> dict[str, Any]:
    """Pause or resume a run. Held input is not released, so a paused run can continue."""
    try:
        session = _app().get_session(run_id)
        session.resume() if resume else session.pause()
        return {"ok": True, "run_id": session.id, "status": session.status}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@server.tool(annotations=READ_ONLY)
async def voltage_journal(
    run_id: str | None = None,
    limit: int = 40,
    only_refused: bool = False,
) -> dict[str, Any]:
    """Read a run's cycle-by-cycle record: what was seen, decided, refused, executed.

    `only_refused=true` filters to cycles the governor blocked, which is the fastest way
    to see where a Playbook's policy and the actuator's intentions disagree.
    """
    try:
        session = _app().get_session(run_id)
        cycles = session.journal.cycles(limit if not only_refused else limit * 4)
        if only_refused:
            cycles = [c for c in cycles if not c.get("allowed", True)][-limit:]
        return {
            "ok": True,
            "run_id": session.id,
            "status": session.status,
            "path": str(session.journal.path),
            "stats": session.journal.stats(),
            "cycles": cycles,
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ======================================================================================
# Learning loop
# ======================================================================================


@server.tool(annotations=READ_ONLY)
async def voltage_diagnose(run_id: str | None = None) -> dict[str, Any]:
    """Explain why a run behaved as it did, and what to change.

    Call this instead of reading the journal by hand. It computes what the journal
    implies but does not state -- `watch` labels the vision model never once reported,
    guards that never evaluated true, whether bursts actually moved the screen, whether
    the actuator is chaining or emitting one action at a time -- and returns each with
    the specific edit that fixes it, ordered blocker-first.

    The distinction it exists for: a burst that **never ran** and a burst that **ran and
    did nothing** look identical in a summary and have unrelated causes. The first is
    policy or grammar; the second is window focus, pointer mode, or an application that
    ignores synthetic input.

    Apply the highest-severity finding, re-run, diagnose again. Changing several things
    at once makes the next diagnosis uninterpretable.
    """
    try:
        from .diagnose import diagnose

        session = _app().get_session(run_id)
        snapshot = session.snapshot(journal_tail=0)
        report = diagnose(
            list(session.journal.tail(limit=1000)),
            session.playbook.spec.model_dump(),
            status=session.status,
            reason=session.reason,
            dry_run=session.dry_run,
        )
        report["run_id"] = session.id
        report["state"] = snapshot.get("state")
        report["ok"] = True
        return report
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@server.tool(annotations=READ_ONLY)
async def voltage_lessons(target: str | None = None, limit: int = 30) -> dict[str, Any]:
    """Recall what previous runs learned about driving something.

    Call this **before writing a Playbook** for a target you have driven before. Lessons
    persist across sessions and are keyed by target ("minecraft", "roblox", "dolphin"),
    so a new Playbook can start from what the last one discovered -- which labels the
    vision model actually recognises, where the HUD probes are, what timing the game
    needs -- rather than rediscovering it.

    Omit `target` to see everything recorded so far.
    """
    try:
        from .diagnose import load_lessons

        entries = load_lessons(target)[:limit]
        return {
            "ok": True,
            "target": target,
            "count": len(entries),
            "lessons": entries,
            "hint": (
                "Record new ones with voltage_learn as you discover them -- especially "
                "working `watch` labels, probe coordinates, and timing values, which are "
                "expensive to rediscover."
                if entries else
                "Nothing recorded for this target yet. Use voltage_learn after a run."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@server.tool(annotations=ACTS)
async def voltage_learn(
    target: str,
    note: str,
    kind: Literal["observation", "label", "timing", "policy", "burst"] = "observation",
    playbook: str = "",
) -> dict[str, Any]:
    """Record something worth carrying to the next run against this target.

    Write these as concrete, reusable facts, not narration:

      good  "the health bar is at x=120..300, y=1010; region_mean on red channel works"
      good  "vision reports 'hotbar' reliably but never 'crosshair' -- do not watch it"
      good  "block placement needs w:100 after the right click or it does not register"
      bad   "the run failed"
      bad   "tried again and it worked better"

    `kind` groups them: label (what the vision model does and does not recognise),
    timing (waits that a specific application needs), policy (what the governor blocked
    and whether that was right), burst (a sequence that works), observation (anything
    else).
    """
    try:
        from .diagnose import Lesson, save_lesson

        note = note.strip()
        if len(note) < 8:
            return {
                "ok": False,
                "error": "note_too_short",
                "detail": "A lesson should be a concrete, reusable fact -- see the "
                          "examples in this tool's description.",
            }
        path = save_lesson(Lesson(target=target, note=note, kind=kind, playbook=playbook))
        return {"ok": True, "target": target, "kind": kind, "stored": str(path)}
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ======================================================================================


def main() -> None:
    """stdio entry point -- what `claude mcp add` launches."""
    try:
        server.run(transport="stdio")
    finally:
        _shutdown()


def main_http(host: str = "127.0.0.1", port: int = 8765, *, allow_remote: bool = False) -> None:
    """HTTP entry point, for clients that add MCP servers by URL ("custom connector").

    Binding is restricted to loopback unless `allow_remote` is passed explicitly.

    That restriction is not boilerplate. This server's whole purpose is to move the mouse,
    press keys, and read the screen of the machine it runs on, and MCP has no
    authentication of its own. A non-loopback bind publishes unauthenticated remote
    control of the desktop to the network. If you genuinely need that, put it behind a
    reverse proxy that terminates TLS and authenticates, and understand that anyone who
    reaches the port owns the machine.
    """
    if not allow_remote and host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit(
            f"refusing to bind {host}: this server can control the desktop and MCP has no "
            f"authentication. Use 127.0.0.1, or pass --allow-remote if you have put "
            f"authentication in front of it and accept that anyone who reaches the port "
            f"can drive this machine."
        )

    banner = f"http://{host}:{port}/mcp"
    print(f"voltage-input MCP listening on {banner}", flush=True)
    print("add it as a custom connector with that URL", flush=True)
    if allow_remote:
        print(
            "WARNING: bound beyond loopback with no authentication -- anyone who can "
            "reach this port can control this computer.",
            flush=True,
        )
    try:
        server.run(transport="streamable-http", host=host, port=port)
    finally:
        _shutdown()


def _shutdown() -> None:
    try:
        asyncio.run(get_app().close())
    except Exception:  # noqa: BLE001 - shutdown is best-effort
        pass


if __name__ == "__main__":
    main()
