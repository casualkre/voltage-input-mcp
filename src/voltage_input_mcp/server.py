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
from .capture import encode_png
from .errors import VoltageError
from .llm import observation_grammar
from .models.burst import parse_burst
from .models.observation import CoordinateMapper, Observation, parse_vision_output
from .models.playbook import VERB_NAMES, Playbook, playbook_from_dict
from .runtime import Session
from .runtime.prompts import VISION_SYSTEM

server = MCPServer(
    name="voltage-input",
    version="0.1.0",
    instructions=(
        "Two-layer input engine. You are the orchestrator: you write a Playbook -- a "
        "state machine with guards, limits and a safety policy -- and two small local "
        "models execute it, one reading the screen and one emitting timed input bursts.\n"
        "\n"
        "Call voltage_reference first if you have not written a Playbook before. Call "
        "voltage_doctor to confirm the machine is ready. Playbooks run in dry_run by "
        "default: they parse, check and journal every burst without touching the input "
        "device, so validate a playbook that way before setting dry_run=false.\n"
        "\n"
        "The local models are small and take instructions literally. Give each state a "
        "short imperative brief, a closed list of things to look for, and explicit "
        "transitions. Do not expect them to infer intent."
    ),
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
    section: Literal["all", "burst", "playbook", "guards", "example"] = "all",
) -> dict[str, Any]:
    """Return the Playbook and burst-DSL reference needed to author a run.

    Read this before writing your first Playbook. `section` narrows the output:
    "burst" for the input syntax, "playbook" for the state-machine schema, "guards" for
    the expression functions available in transitions and reflexes, "example" for a
    complete working Playbook.
    """
    from .expr import GUARD_FUNCTIONS, GUARD_NAMESPACES
    from .reference import BURST_REFERENCE, EXAMPLE_PLAYBOOK, GUARD_REFERENCE

    out: dict[str, Any] = {"ok": True}
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
            result.text, mapper, latency_ms=result.latency_ms, max_elements=max_elements
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
    keep_frames: bool = False,
) -> dict[str, Any]:
    """Start a Playbook. Returns immediately with a run_id; poll voltage_status.

    `dry_run` overrides the Playbook's policy. Leave it unset for the Playbook's own
    setting, which defaults to true. A dry run does everything except inject input, so
    it is the correct way to check that your states, guards and transitions behave before
    letting it touch the machine.

    `target_period_s` is the loop period. 0.5 is a good default; lower it for games,
    raise it for slow UI.

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
            target_period_s=target_period_s, keep_frames=keep_frames or None
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
            "journal": str(session.journal.path),
            "next": "poll voltage_status(run_id) to watch it, voltage_stop to end it",
        }
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


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


def main() -> None:
    """stdio entry point."""
    try:
        server.run(transport="stdio")
    finally:
        try:
            asyncio.run(get_app().close())
        except Exception:  # noqa: BLE001 - shutdown is best-effort
            pass


if __name__ == "__main__":
    main()
