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


def _wrapped(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(str(text).split()), width) or [""]


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
    # `is not None`, not truthiness: a warm streaming grab measures 0.03 ms and rounds
    # near zero, which is the best possible result and must not read as "not measured".
    latency = (
        f"{health['latency_ms']}ms/frame" if health.get("latency_ms") is not None else ""
    )
    if health.get("first_frame_ms"):
        latency += f" (first {health['first_frame_ms']:.0f}ms)"
    available = [k for k, v in cap["available"].items() if v]
    print(
        f"  capture      {cap.get('selected', '-')} "
        f"{'ok' if health.get('ok') else 'FAIL'} {latency}"
        f"  available={available}"
    )
    if health.get("note"):
        print(f"               {health['note']}")
    if cap.get("reflex_hz"):
        sustainable = cap.get("sustainable_hz")
        stream = "streaming" if cap.get("streaming") else "per-call grab"
        print(
            f"  fast loop    {cap['reflex_hz']:g} Hz requested  ({stream}"
            + (f", ~{sustainable:g} Hz sustainable)" if sustainable else ")")
        )
        if cap.get("reflex_note"):
            for line in _wrapped(cap["reflex_note"], 62):
                print(f"               {line}")
    else:
        print("  fast loop    off -- reflexes fold back into the decision cycle")
    print(f"  ocr          {'ok' if cap.get('ocr') else 'FAIL -- number probes disabled'}")
    if cap.get("ocr_detail"):
        print(f"               {cap['ocr_detail']}")
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
    if models.get("mismatch"):
        print(f"  MISMATCH     {models['mismatch']}")

    for step in report.get("next_steps", []):
        print(f"  ->  {step}")
    return 0 if report.get("ready") else 1


def cmd_reflex(args: argparse.Namespace) -> int:
    """Measure what the fast layer actually achieves on this machine.

    `doctor` predicts a sustainable rate from one capture timing. This runs the real loop
    -- capture, probes, guards, a latch engaging and releasing -- against the real screen
    and reports what happened. It is the difference between "20 Hz should work here" and
    "20 Hz worked here", and it is worth having because every number in this project that
    turned out to be wrong was one that had been predicted rather than measured.

    Injects nothing: the run is dry, and the latch's key events are counted, not sent.
    """
    import asyncio

    from .app import get_app
    from .models.playbook import playbook_from_dict
    from .runtime import Session

    app = get_app(load_config(args.config))
    screen = app.screen()
    # A brightness probe over the middle of the screen, and a latch keyed to it. Whether
    # the latch happens to engage depends on what is on screen, which is fine -- the
    # measurement under test is the loop's rate and cost, not the guard.
    playbook = playbook_from_dict({
        "name": "reflex_check",
        "goal": "Measure the fast loop.",
        "initial": "measure",
        "probes": [
            {"id": "mid", "type": "brightness",
             "region": {"x": screen[0] // 4, "y": screen[1] // 4,
                        "w": screen[0] // 2, "h": screen[1] // 2}},
            {"id": "motion", "type": "region_diff",
             "region": {"x": screen[0] // 4, "y": screen[1] // 4,
                        "w": screen[0] // 2, "h": screen[1] // 2}},
        ],
        "perception": {"mode": "never"},
        "policy": {"dry_run": True, "allow_verbs": ["d", "u", "k", "w"],
                   "allow_keys": ["w"]},
        "budget": {"max_cycles": 10000, "max_seconds": max(1.0, args.seconds),
                   "idle_abort_s": 0},
        "states": {
            "measure": {
                "brief": "x",
                "autonomous": False,
                "reflex": [
                    {"id": "latch", "when": "probe('mid') > 0.2",
                     "release_when": "probe('mid') < 0.1", "hold": "w"},
                ],
                "transitions": [],
            }
        },
    })

    async def go() -> dict:
        options = app.session_options(target_period_s=0.5, reflex_hz=args.hz)
        options.dry_run = True
        options.watch_physical_input = False
        session = Session(playbook, app.session_deps(), options)
        await session.start()
        return session.snapshot()

    print(f"running the fast loop for {args.seconds:g}s at {args.hz:g} Hz "
          f"(dry: nothing is injected)...")
    snap = asyncio.run(go())
    reflex = snap["reflex"]
    ticks = snap["timings"].get("reflex", {})
    capture = snap["timings"].get("capture", {})

    print()
    print(f"  requested       {reflex['requested_hz']:g} Hz")
    print(f"  measured        {reflex['measured_hz']:g} Hz over {reflex['ticks']} ticks")
    if ticks:
        print(f"  tick cost       {ticks['p50_ms']:.2f} ms p50, {ticks['p95_ms']:.2f} ms p95")
    if capture:
        print(f"  of which capture {capture['p50_ms']:.2f} ms p50")
    print(f"  latch events    {reflex['latch_events']}"
          f"  (engaged and released as the guard flipped)")
    print(f"  starved         {reflex['starved']}")
    for where, err in (reflex.get("errors") or {}).items():
        print(f"  error [{where}]  {err}")
    print(f"  probes now      "
          f"{ {k: round(v, 3) for k, v in snap['probes'].items() if not k.startswith('__')} }")

    achieved = float(reflex["measured_hz"])
    print()
    if achieved >= args.hz * 0.85:
        print(f"  OK  the fast layer holds {achieved:g} Hz here. Reflex and hold rules "
              f"will react within ~{1000 / max(achieved, 1):.0f} ms.")
        return 0
    print(f"  SLOW  asked for {args.hz:g} Hz and got {achieved:g}. Guard timings in a "
          f"playbook will be coarser than written.")
    if capture and capture.get("p50_ms", 0) > 5:
        print("  The capture backend is the cause -- check `voltage doctor`. 'portal' "
              "streams; 'kwin' and 'grim' do a round trip per frame.")
    else:
        print("  Capture is cheap, so this is CPU contention -- most likely the model "
              "servers. Lower reflex_hz to what the machine sustains.")
    return 1


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
    from .llm import all_profiles, detect_vram_mb, recommend
    from .llm.profiles import DESKTOP_RESERVE_MB, usable_vram_mb

    vram = detect_vram_mb()
    budget = usable_vram_mb(vram) if vram else 0
    if vram:
        print(f"detected VRAM: {vram} MB")
        print(f"usable:        {budget} MB  (reserving {DESKTOP_RESERVE_MB} MB for the desktop)")
        print(f"recommended:   {recommend(vram).name}\n")
    else:
        print("no CUDA GPU detected\n")
    for profile in all_profiles().values():
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


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download a profile's weights. Pure Python, so it works on every platform.

    The shell script this replaces was fine on Linux and unusable on Windows, which meant
    `voltage setup` could only print instructions there instead of doing the work.
    """
    import urllib.error
    import urllib.request

    from .llm import get_profile

    try:
        profile = get_profile(args.profile)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    target = Path(args.dir) if args.dir else Path(__file__).resolve().parents[2] / "models"
    target.mkdir(parents=True, exist_ok=True)

    wanted: list[tuple[str, str]] = []
    for spec in (profile.vision, profile.actuator):
        if spec.hf_repo and spec.hf_file:
            wanted.append((spec.hf_repo, spec.hf_file))
        if spec.mmproj_repo and spec.mmproj_file:
            wanted.append((spec.mmproj_repo, spec.mmproj_file))
    if not wanted:
        print(f"profile {args.profile!r} uses Ollama; nothing to download here")
        print(f"  ollama pull {profile.vision.ollama_tag}")
        print(f"  ollama pull {profile.actuator.ollama_tag}")
        return 0

    print(f"profile {profile.name}  ->  {target}\n")
    for repo, name in wanted:
        dest = target / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  have    {name}  ({dest.stat().st_size / 2**30:.2f} GB)")
            continue

        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "voltage-input-mcp"})
            with urllib.request.urlopen(request, timeout=30) as response, part.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r  fetch   {name}  {pct:3d}%  "
                              f"{done / 2**30:.2f}/{total / 2**30:.2f} GB", end="", flush=True)
            print()
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            print(f"\n  MISSING {name}  (HTTP {exc.code})", file=sys.stderr)
            print("          the repo or quant may have been renamed -- check", file=sys.stderr)
            print(f"          https://huggingface.co/{repo}/tree/main", file=sys.stderr)
            return 1
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            part.unlink(missing_ok=True)
            print(f"\n  FAILED  {name}: {exc}", file=sys.stderr)
            return 1
        part.replace(dest)

    size = sum(f.stat().st_size for f in target.glob("*.gguf"))
    print(f"\n{size / 2**30:.2f} GB in {target}")
    print(f"\nnext:  voltage serve-models {profile.name}     # prints the launch commands")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Go from nothing to working, running each step rather than describing it."""
    import subprocess

    from .connect import claude_code_command, gather
    from .wizard import detect, plan_steps

    root = Path(__file__).resolve().parents[2]

    def run(cmd: list[str]) -> int:
        print(f"\n$ {' '.join(cmd)}\n")
        try:
            return subprocess.call(cmd, cwd=str(root))
        except (OSError, KeyboardInterrupt):
            return 1

    print("checking what you already have...\n")
    env = detect()
    config = load_config(args.config)

    print(f"  system    {env.platform_label}   {env.session}")
    print(f"  gpu       {env.gpu_name or 'none detected'}"
          f"{f'  {env.vram_mb} MB' if env.vram_mb else ''}")
    print(f"  input     {'ok' if env.input_ok else 'NOT READY'}")
    print(f"  capture   {'ok via ' + env.capture_backend if env.capture_ok else 'NOT READY'}")
    print(f"  engine    {config.engine}   profile {config.profile}")

    steps = plan_steps(env, config.engine, config.profile)
    if not steps:
        print("\nEverything is already set up. Try:  voltage doctor")
        return 0

    print(f"\n{len(steps)} step(s) remaining:\n")
    for i, step in enumerate(steps, 1):
        tag = "" if step.automatic else "   (needs you)"
        print(f"  {i}. {step.title}{tag}")
        print(f"     {step.why}")
    if not args.yes:
        print("\nRun them?  [Y/n] ", end="", flush=True)
        try:
            if input().strip().lower().startswith("n"):
                return 0
        except (EOFError, KeyboardInterrupt):
            return 0

    for step in steps:
        print(f"\n{'=' * 72}\n{step.title}\n{'=' * 72}")
        if not step.automatic:
            print(f"\n  {step.why}")
            if step.detail:
                for line in step.detail.splitlines():
                    print(f"    {line}")
            print("\n  Do that, then re-run `voltage setup` to continue.")
            continue
        if step.key == "path":
            from .tui import _install_path_link

            _install_path_link()
        elif step.key == "ollama-pull":
            from .llm import get_profile

            spec = get_profile(config.profile)
            for tag in (spec.vision.ollama_tag, spec.actuator.ollama_tag):
                if tag:
                    run(["ollama", "pull", tag])
        elif step.key == "llama":
            run(["./scripts/build-llama.sh"])
        elif step.key == "weights":
            # Python downloader rather than the shell script, so this step actually runs
            # on Windows instead of printing instructions.
            cmd_fetch(argparse.Namespace(profile=config.profile, dir=None))
        elif step.key == "serve":
            if sys.platform == "win32":
                print("\n  Start each of these in its own terminal:\n")
                cmd_serve_models(argparse.Namespace(profile=config.profile))
            else:
                run(["./scripts/serve.sh", config.profile])
        elif step.key == "mcp":
            info = gather()
            command = claude_code_command(info)
            print("\nRegistering with Claude Code:")
            print(f"\n$ {command}\n")
            subprocess.call(command, shell=True)

    print(f"\n{'=' * 72}")
    env = detect()
    print(f"input {'ok' if env.input_ok else 'NO'}   "
          f"capture {'ok' if env.capture_ok else 'NO'}   "
          f"models {'ok' if env.servers_up else 'NO'}   "
          f"mcp {'ok' if env.mcp_registered else 'NO'}")
    if env.ready:
        print("\nReady. Restart your AI client, then ask it to run voltage_doctor.")
    else:
        print("\nSome steps are outstanding -- re-run `voltage setup` to continue.")
    return 0


def cmd_instructions(args: argparse.Namespace) -> int:
    """Show, set, or clear the standing instructions sent to the orchestrator."""
    from .briefing import instructions_path, load_instructions, save_instructions

    path = instructions_path()
    if args.clear:
        save_instructions("")
        print(f"cleared {path}")
        return 0
    if args.set is not None:
        save_instructions(args.set)
        print(f"wrote {path}")
        return 0

    current = load_instructions()
    print(f"# {path}")
    if current:
        print()
        print(current)
    else:
        print("\n(nothing set -- the orchestrator gets only the build briefing)")
        print("\nSet with:  voltage instructions --set \"...\"")
        print("Or edit interactively:  voltage  ->  i")
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    """Print connection status and per-client setup instructions."""
    from .connect import CLIENTS, claude_code_command, gather, instructions, stdio_json

    info = gather(port=args.port)
    if args.json:
        print(stdio_json(info))
        return 0

    if args.write_desktop:
        from .tui import _write_desktop_config

        _write_desktop_config(info)
        return 0

    if args.client:
        keys = {c.key for c in CLIENTS}
        if args.client not in keys:
            print(f"unknown client {args.client!r}; try: {', '.join(sorted(keys))}",
                  file=sys.stderr)
            return 2
        name = next(c.name for c in CLIENTS if c.key == args.client)
        print(f"{name}\n")
        for i, (text, block) in enumerate(instructions(args.client, info), 1):
            print(f"{i}. {text}")
            if block:
                print()
                for line in block.splitlines():
                    print(f"     {line}")
            print()
        return 0

    print("server binary   " + info.binary + ("" if info.binary_exists else "   (MISSING)"))
    print("http endpoint   " + info.http_url +
          ("   running" if info.http_running else "   not running"))
    if info.engine == "ollama":
        print("models          ollama http://127.0.0.1:11434")
    else:
        print(f"vision model    {info.vision_url}   {'up' if info.vision_up else 'down'}")
        print(f"actuator model  {info.actuator_url}   {'up' if info.actuator_up else 'down'}")
    if not info.claude_cli:
        print("claude code     cli not installed")
    elif info.registered:
        print(f"claude code     {info.registered_scope}  ({info.registered_status})")
        if info.registered_env_ok is False:
            print("                registered WITHOUT the session environment -- it will")
            print("                connect but cannot capture the screen. Re-register.")
    else:
        print("claude code     not registered")
    if info.missing_env:
        print("missing env     " + ", ".join(info.missing_env) +
              "   (screen capture needs these)")
    print()
    print("clients:  " + ", ".join(c.key for c in CLIENTS))
    print("steps:    voltage connect --client claude-desktop")
    print("json:     voltage connect --json")
    print()
    print("claude code, in one line:")
    print("  " + claude_code_command(info))
    return 0


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

    p = sub.add_parser(
        "reflex",
        help="measure what reflex rate this machine actually achieves (injects nothing)",
    )
    p.add_argument("--hz", type=float, default=20.0, help="rate to ask for (default 20)")
    p.add_argument("--seconds", type=float, default=5.0, help="how long to run")
    p.set_defaults(func=cmd_reflex)

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

    p = sub.add_parser("fetch", help="download a profile's model weights")
    p.add_argument("profile", nargs="?", default="lean")
    p.add_argument("--dir", help="where to put them (default: ./models)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("setup", help="go from nothing to working; safe to re-run")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask before running")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser(
        "instructions", help="standing instructions given to the orchestrator"
    )
    p.add_argument("--set", help="replace them with this text")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_instructions)

    p = sub.add_parser("connect", help="show URLs and per-client MCP setup instructions")
    p.add_argument("--client", help="print steps for one client (see the list it prints)")
    p.add_argument("--json", action="store_true", help="print the mcpServers entry only")
    p.add_argument("--write-desktop", action="store_true",
                   help="write the Claude Desktop config (backs up any existing file)")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_connect)

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
