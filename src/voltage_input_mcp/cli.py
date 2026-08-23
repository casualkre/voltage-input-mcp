"""Command line interface.

Exists mostly so the system can be debugged without an MCP client in the loop. `voltage
doctor` and `voltage burst` in particular are the fastest way to find out whether input
and capture work on a given machine.

`voltage stop` is the emergency exit and is deliberately the simplest thing here: it
writes a file. It works from another TTY, over SSH, or from a file manager, and needs
nothing about the running process to still be healthy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any  # noqa: F401  -- used by annotations in the command handlers

from . import __version__
from .config import load_config


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


# -- commands ------------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    from .app import get_app

    report = asyncio.run(get_app(load_config(args.config)).doctor())
    if args.json:
        _print(report)
        return 0 if report.get("ready") else 1

    ok = "OK " if report.get("ready") else "NOT READY "
    print(f"{ok}voltage-input-mcp {__version__}")
    session = report["session"]
    print(f"  session      {session['type']} / {session['desktop']}")
    if session.get("warning"):
        print(f"               {session['warning']}")

    inp = report["input"]
    print(f"  uinput       {'ok' if inp.get('ok') else 'FAIL -- ' + str(inp.get('reason'))}")
    if inp.get("fix"):
        print(f"               fix: {inp['fix']}")
    print(f"  clipboard    {inp.get('clipboard_tool') or 'none (see note)'}")

    cap = report["capture"]
    health = cap.get("health", {})
    latency = f"{health['latency_ms']}ms" if health.get("latency_ms") else ""
    available = [k for k, v in cap["available"].items() if v]
    print(
        f"  capture      {cap.get('selected', '-')} "
        f"{'ok' if health.get('ok') else 'FAIL'} {latency}"
        f"  available={available}"
    )
    if health.get("note"):
        print(f"               {health['note']}")
    print(f"  screen       {report.get('screen') or 'not detected yet'}")
    if cap.get("screen_note"):
        print(f"               {cap['screen_note']}")

    models = report["models"]
    print(f"  vram         {models.get('vram_mb')} MB   profile={models['profile']['name']} "
          f"needs~{models['profile']['estimated_vram_mb']} MB")
    for role in ("vision", "actuator"):
        info = models.get(role) or {}
        mark = "ok" if info.get("ok") else "FAIL"
        print(f"  {role:<12} {mark}  {info.get('model') or info.get('url', '')}")
        if info.get("fix"):
            print(f"               fix: {info['fix']}")
    if models.get("warning"):
        print(f"  warning      {models['warning']}")

    for step in report.get("next_steps", []):
        print(f"  ->  {step}")
    return 0 if report.get("ready") else 1


def cmd_stop(args: argparse.Namespace) -> int:
    from .safety import write_panic

    path = write_panic(args.reason)
    print(f"panic file written: {path}")
    print("every running loop will stop within one cycle and release held input")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    from .safety import clear_panic

    clear_panic()
    print("panic file cleared")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    from .app import get_app
    from .capture import encode_png

    app = get_app(load_config(args.config))
    frame = app.capture().grab()
    out = Path(args.output)
    out.write_bytes(encode_png(frame.pixels))
    print(f"{frame.width}x{frame.height} via {frame.backend} -> {out}")
    return 0


def cmd_burst(args: argparse.Namespace) -> int:
    from .app import get_app
    from .models.burst import parse_burst

    app = get_app(load_config(args.config))
    screen = app.screen() if args.live else (1920, 1080)
    try:
        burst = parse_burst(args.burst, screen=screen)
    except Exception as exc:  # noqa: BLE001
        print(f"parse error: {exc}", file=sys.stderr)
        return 2

    print(f"parsed {len(burst)} action(s), ~{burst.duration_ms} ms")
    for action in burst:
        print(f"  {action.render()}")

    if not args.live:
        print("\n(dry run -- pass --live to actually inject)")
        return 0

    executor = app.executor()
    executor.dry_run = False
    app.devices().open()
    report = executor.run(burst, label="cli")
    _print(report.as_dict())
    return 0 if report.ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from .models.playbook import playbook_from_dict

    try:
        data = json.loads(Path(args.playbook).read_text())
    except (OSError, ValueError) as exc:
        print(f"cannot read playbook: {exc}", file=sys.stderr)
        return 2

    try:
        compiled = playbook_from_dict(data)
    except Exception as exc:  # noqa: BLE001
        print("INVALID", file=sys.stderr)
        detail = getattr(exc, "context", {}) or {}
        for err in detail.get("errors", [str(exc)]):
            print(f"  error: {err}", file=sys.stderr)
        for warn in detail.get("warnings", []):
            print(f"  warn:  {warn}", file=sys.stderr)
        return 1

    print(f"OK  {compiled.spec.name}: {len(compiled.states)} states, "
          f"{len(compiled.probe_ids)} probes, initial={compiled.spec.initial}")
    for warn in compiled.warnings:
        print(f"  warn: {warn}")
    return 0


def _results_path() -> Path:
    from .config import state_dir

    return state_dir() / "comparisons.json"


def cmd_fixture(args: argparse.Namespace) -> int:
    """Capture the screen and write a fixture stub for the orchestrator to annotate."""
    from .app import get_app
    from .capture import encode_png

    app = get_app(load_config(args.config))
    frame = app.capture().grab()
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.name}.png"
    meta = out_dir / f"{args.name}.json"

    png.write_bytes(encode_png(frame.pixels))
    if meta.exists() and not args.force:
        print(f"{meta} already exists (use --force to overwrite the labels)")
    else:
        meta.write_text(json.dumps({
            "name": args.name,
            "screen": list(frame.size),
            "watch": ["<label the vision model may use>", "<another>"],
            "expect": {"<label>": [0, 0, 100, 40]},
            "absent": [],
            "_howto": (
                "Open the PNG, or have the orchestrator view it with voltage_capture, "
                "then fill `expect` with label -> [x, y, w, h] in screen pixels and list "
                "any labels that must NOT be detected under `absent`. The orchestrator is "
                "the reference here -- it is the same model that judges the screen at "
                "runtime."
            ),
        }, indent=2))
        print(f"wrote {meta}")
    print(f"wrote {png}  ({frame.width}x{frame.height} via {frame.backend})")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Evaluate the currently-running models and accumulate results across runs."""
    from .app import get_app
    from .evaluate import Fixture, compare_actuator, compare_vision, format_actuator, format_vision

    store = _results_path()
    saved: dict[str, Any] = {}
    if store.exists():
        try:
            saved = json.loads(store.read_text())
        except ValueError:
            saved = {}

    if args.reset:
        store.unlink(missing_ok=True)
        print("cleared accumulated comparison results")
        return 0

    if not args.list:
        app = get_app(load_config(args.config))
        vision, actuator = app.backends()

        async def go() -> dict[str, Any]:
            health = await vision.health()
            label = args.label or str(
                health.get("model") or getattr(vision, "model_label", "unknown")
            )
            label = Path(str(label)).name

            fixtures = []
            fixture_dir = Path(args.fixtures)
            if fixture_dir.is_dir():
                fixtures = [
                    Fixture.load(p) for p in sorted(fixture_dir.glob("*.json"))
                    if p.with_suffix(".png").exists()
                ]

            entry: dict[str, Any] = {"profile": app.config.profile, "engine": app.config.engine}
            if fixtures:
                entry["vision"] = (await compare_vision(
                    {label: vision}, fixtures, rounds=args.rounds
                ))[label]
            else:
                print(
                    f"no fixtures in {fixture_dir}/ -- skipping grounding accuracy.\n"
                    f"Create one with:  voltage fixture desktop --dir {fixture_dir}\n"
                    f"then have the orchestrator fill in the expected boxes.\n"
                )
            entry["actuator"] = (await compare_actuator(
                {label: actuator}, rounds=args.rounds
            ))[label]
            return {label: entry}

        result = asyncio.run(go())
        saved.update(result)
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(saved, indent=2, default=str))
        print(f"results saved to {store}\n")

    if not saved:
        print("no results yet. Start a model with scripts/serve.sh, then run this again.")
        return 1

    if args.json:
        _print(saved)
        return 0

    vision_results = {k: v["vision"] for k, v in saved.items() if "vision" in v}
    actuator_results = {k: v["actuator"] for k, v in saved.items() if "actuator" in v}
    if vision_results:
        print(format_vision(vision_results))
    if actuator_results:
        print(format_actuator(actuator_results))
    print(
        "  To compare model sizes: stop the servers, edit `profile` in voltage.toml or\n"
        "  run scripts/serve.sh with another profile, then run `voltage compare` again.\n"
        "  Results accumulate, so the table grows with each model you try.\n"
    )
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    from .llm import PROFILES, detect_vram_mb, recommend
    from .llm.profiles import DESKTOP_RESERVE_MB, usable_vram_mb

    vram = detect_vram_mb()
    budget = usable_vram_mb(vram) if vram else 0
    if vram:
        print(f"detected VRAM: {vram} MB")
        print(f"usable:        {budget} MB  (reserving {DESKTOP_RESERVE_MB} MB for the desktop)")
        print(f"recommended:   {recommend(vram).name}\n")
    else:
        print("no CUDA GPU detected\n")
    for profile in PROFILES.values():
        fits = "" if not vram else ("  fits" if profile.fits(budget) else "  TOO BIG")
        print(f"{profile.name:<10} ~{profile.vram_mb:>5} MB{fits}   {profile.description}")
        print(f"{'':<10} vision={profile.vision.label}  actuator={profile.actuator.label}")
        if profile.notes:
            print(f"{'':<10} {profile.notes}")
        print()
    return 0


def cmd_serve_models(args: argparse.Namespace) -> int:
    from .llm import get_profile

    profile = get_profile(args.profile)
    print(f"# {profile.name}: {profile.description}")
    print(f"# estimated VRAM: {profile.vram_mb} MB\n")
    print("# vision model")
    print(" ".join(profile.vision.llama_server_args()) + " &\n")
    print("# actuator model")
    print(" ".join(profile.actuator.llama_server_args(cpu_only=profile.actuator_on_cpu)) + " &")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .app import get_app
    from .bench import format_report, run_benchmark

    app = get_app(load_config(args.config))
    vision, actuator = app.backends()

    async def go():
        return await run_benchmark(vision, actuator, rounds=args.rounds)

    report = asyncio.run(go())
    if args.json:
        _print(report)
    else:
        print(format_report(report))
    return 0 if "error" not in report else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import main as serve_stdio
    from .server import main_http

    if args.http:
        main_http(host=args.host, port=args.port, allow_remote=args.allow_remote)
    else:
        serve_stdio()
    return 0


# -- parser --------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voltage",
        description="VoltageInputMcp -- two-layer local input engine",
    )
    parser.add_argument("--version", action="version", version=f"voltage {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="path to voltage.toml")
    # Not required: bare `voltage` opens the interactive console. Scripted use is
    # unaffected, and a non-TTY invocation with no subcommand prints help rather than
    # blocking on input forever.
    sub = parser.add_subparsers(dest="command", required=False)

    p = sub.add_parser("doctor", help="check that everything needed for a run works")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("stop", help="emergency stop: halt every running loop")
    p.add_argument("reason", nargs="?", default="manual stop")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("clear", help="clear the panic file so runs can start again")
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser("capture", help="save a screenshot")
    p.add_argument("-o", "--output", default="screen.png")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("burst", help="parse (and optionally execute) a burst string")
    p.add_argument("burst")
    p.add_argument("--live", action="store_true", help="actually inject the input")
    p.set_defaults(func=cmd_burst)

    p = sub.add_parser("validate", help="validate a playbook JSON file")
    p.add_argument("playbook", type=Path)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("profiles", help="list model profiles and what fits this GPU")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("serve-models", help="print the llama-server commands for a profile")
    p.add_argument("profile", nargs="?", default="lean")
    p.set_defaults(func=cmd_serve_models)

    p = sub.add_parser("fixture", help="capture a screenshot as a grounding-eval fixture")
    p.add_argument("name")
    p.add_argument("--dir", default="fixtures")
    p.add_argument("--force", action="store_true", help="overwrite existing labels")
    p.set_defaults(func=cmd_fixture)

    p = sub.add_parser(
        "compare",
        help="score the running models on grounding + decision quality; accumulates",
    )
    p.add_argument("--label", help="name for this model in the table (default: from /props)")
    p.add_argument("--fixtures", default="fixtures", help="directory of eval fixtures")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--list", action="store_true", help="print the table without running")
    p.add_argument("--reset", action="store_true", help="clear accumulated results")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("bench", help="measure real model latency against running servers")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("serve", help="run the MCP server (stdio, or HTTP for connectors)")
    p.add_argument("--http", action="store_true", help="serve over HTTP for a custom connector")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit binding beyond loopback (unauthenticated desktop control -- read the docs)",
    )
    p.set_defaults(func=cmd_serve)

    return parser


def cmd_console(args: argparse.Namespace) -> int:
    from .tui import run_console

    return run_console()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "command", None) is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return cmd_console(args)
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
