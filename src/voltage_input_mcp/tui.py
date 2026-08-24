"""Interactive console, shown when `voltage` is run with no arguments.

Deliberately built on nothing but ANSI escapes and `input()`. A tool whose job is to
rescue a half-configured machine should not itself depend on curses working, a terminal
being resizable, or a third-party TUI library being installed -- and this is exactly the
tool someone reaches for when the rest of the setup is broken.

Every screen answers the same question in a different way: what is wrong right now, and
what is the single next command that fixes it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Config, default_config_path, load_config

__all__ = ["run_console"]

REPO_ROOT = Path(__file__).resolve().parents[2]

# -- presentation -------------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str:
    return _c("2", t)


def bold(t: str) -> str:
    return _c("1", t)


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


BANNER = r"""
 ██╗   ██╗ ██████╗ ██╗  ████████╗ █████╗  ██████╗ ███████╗
 ██║   ██║██╔═══██╗██║  ╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝
 ██║   ██║██║   ██║██║     ██║   ███████║██║  ███╗█████╗
 ╚██╗ ██╔╝██║   ██║██║     ██║   ██╔══██║██║   ██║██╔══╝
  ╚████╔╝ ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝███████╗
   ╚═══╝   ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""


def clear() -> None:
    if _COLOR:
        print("\033[2J\033[H", end="")


def banner() -> None:
    print(cyan(BANNER))
    print(dim("   two-layer local input engine  ·  you think, two small models act"))
    print()


def rule(label: str = "") -> None:
    width = min(shutil.get_terminal_size((80, 24)).columns, 78)
    if label:
        print(dim("─" * 2 + f" {label} " + "─" * max(0, width - len(label) - 4)))
    else:
        print(dim("─" * width))


def ok_mark(value: bool | None) -> str:
    if value is None:
        return yellow("  ?  ")
    return green(" ok  ") if value else red("FAIL ")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{cyan('>')} {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return answer or default


def confirm(prompt: str, default: bool = False) -> bool:
    answer = ask(f"{prompt} (y/n)", "y" if default else "n").lower()
    return answer.startswith("y")


def pause() -> None:
    try:
        input(dim("\n  press enter to continue "))
    except (EOFError, KeyboardInterrupt):
        pass


def run(cmd: list[str], *, cwd: Path | None = None, quiet: bool = False) -> int:
    """Run a command, streaming output. Returns the exit code."""
    if not quiet:
        print(dim(f"  $ {' '.join(cmd)}\n"))
    try:
        return subprocess.call(cmd, cwd=str(cwd or REPO_ROOT))
    except (OSError, KeyboardInterrupt) as exc:
        print(red(f"  {exc}"))
        return 1


def capture(cmd: list[str], timeout: float = 6.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False, cwd=str(REPO_ROOT)
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return proc.returncode, (proc.stdout + proc.stderr).decode("utf-8", "replace")


# -- state probing ------------------------------------------------------------------------


def venv_bin(name: str) -> str:
    """Locate a venv entry point. Windows puts them in Scripts/ with a .exe suffix."""
    if sys.platform == "win32":
        candidate = REPO_ROOT / ".venv" / "Scripts" / f"{name}.exe"
    else:
        candidate = REPO_ROOT / ".venv" / "bin" / name
    return str(candidate) if candidate.exists() else name


def server_up(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def mcp_registered() -> bool | None:
    if not shutil.which("claude"):
        return None
    code, out = capture(["claude", "mcp", "list"], timeout=15.0)
    if code != 0 and not out:
        return None
    return "voltage-input" in out


def on_path() -> bool:
    return shutil.which("voltage") is not None


def snapshot(config: Config) -> dict[str, object]:
    from .inputs import probe_input

    return {
        "config": config,
        "uinput": bool(probe_input().get("ok")),
        "vision": server_up(int(config.vision_url.rsplit(":", 1)[-1])),
        "actuator": server_up(int(config.actuator_url.rsplit(":", 1)[-1])),
        "ollama": server_up(11434),
        "mcp": mcp_registered(),
        "path": on_path(),
        "llama": (REPO_ROOT / "vendor/llama.cpp/build/bin/llama-server").exists()
        or shutil.which("llama-server") is not None,
        "models": sorted((REPO_ROOT / "models").glob("*.gguf")) if (REPO_ROOT / "models").is_dir() else [],
    }


def status_block(state: dict[str, object]) -> None:
    config: Config = state["config"]  # type: ignore[assignment]
    engine = config.engine
    rule("status")
    device_label = "SendInput" if sys.platform == "win32" else "/dev/uinput"
    print(f"  {ok_mark(state['uinput'])} input device      {dim(device_label)}")
    if engine == "ollama":
        print(f"  {ok_mark(state['ollama'])} ollama            {dim(config.ollama_url)}")
    else:
        print(f"  {ok_mark(state['vision'])} vision model      {dim(config.vision_url)}")
        print(f"  {ok_mark(state['actuator'])} actuator model    {dim(config.actuator_url)}")
    print(f"  {ok_mark(state['mcp'])} mcp registered    {dim('claude mcp list')}")
    path_hint = dim("found on PATH" if state["path"] else "run guided setup to link it")
    print(f"  {ok_mark(state['path'])} voltage on PATH   {path_hint}")
    print()
    from .briefing import load_instructions

    custom = load_instructions()
    if custom:
        print(f"  {green(' ok  ')} instructions      "
              f"{dim(str(len(custom)) + ' chars sent to the orchestrator')}")
    print()
    print(f"  profile {bold(config.profile)}   engine {bold(engine)}   "
          f"dry_run {bold(str(config.dry_run))}   {len(state['models'])} model file(s)")  # type: ignore[arg-type]
    print()


# -- screens ------------------------------------------------------------------------------


def screen_setup(state: dict[str, object]) -> None:
    """Detect what exists, then do only what is left."""
    from .wizard import detect, plan_steps

    config: Config = state["config"]  # type: ignore[assignment]

    clear()
    banner()
    rule("checking what you already have")
    print(dim("  probing...\n"))
    env = detect()

    clear()
    banner()
    rule("what I found")
    print(f"  system      {bold(env.platform_label)}  {dim(env.session)}   python {env.python}")
    if env.gpu_name:
        vram = f"{env.vram_mb} MB" if env.vram_mb else "unknown VRAM"
        print(f"  gpu         {bold(env.gpu_name)}  {dim(vram)}")
    else:
        print(f"  gpu         {yellow('none detected')} {dim('-- models will run on CPU, slowly')}")
    print(f"  {ok_mark(env.input_ok)} input       "
          f"{dim('SendInput' if env.platform == 'win32' else '/dev/uinput')}")
    print(f"  {ok_mark(env.capture_ok)} capture     {dim(env.capture_backend or 'none')}")
    print(f"  {ok_mark(bool(env.llama_server))} llama.cpp   "
          f"{dim(env.llama_server or 'not found')}")
    print(f"  {ok_mark(env.ollama_running)} ollama      "
          f"{dim(f'{len(env.ollama_models)} model(s) pulled' if env.ollama_running else 'not running')}")
    print(f"  {ok_mark(bool(env.model_files))} weights     {dim(f'{len(env.model_files)} gguf file(s)')}")
    print(f"  {ok_mark(env.claude_cli)} claude cli  "
          f"{dim('registered' if env.mcp_registered else 'not registered' if env.claude_cli else 'not installed')}")
    print()

    # -- choose an engine, if the user has not effectively already chosen one ---------
    engine = config.engine
    if not env.has_any_engine:
        rule("pick a backend")
        print("  Neither llama.cpp nor Ollama is installed. Two options:\n")
        print(f"  {bold('1')}  Ollama      {dim('easiest. One installer, then two pulls.')}")
        print(f"      {dim('No GBNF grammars, so bursts are ~2x slower and can occasionally')}")
        print(f"      {dim('come back malformed and cost a cycle.')}")
        print(f"  {bold('2')}  llama.cpp   {dim('faster, and grammar-constrained so malformed')}")
        print(f"      {dim('output is impossible. Needs a build on Linux, or a prebuilt')}")
        print(f"      {dim('release on Windows.')}\n")
        choice = ask("which", "1")
        engine = "ollama" if choice.strip() == "1" else "llamacpp"
    elif env.ollama_running and not env.llama_server and engine == "llamacpp":
        print(yellow("  Ollama is running but llama.cpp is not installed, and the config asks"))
        print(yellow("  for llama.cpp.\n"))
        if confirm("switch to Ollama for now?", default=True):
            engine = "ollama"
    if engine != config.engine:
        config = _write_config(config, engine=engine)
        state["config"] = config
        print(green(f"  engine = {engine}\n"))

    # -- plan -------------------------------------------------------------------------
    steps = plan_steps(env, engine, config.profile)
    if not steps:
        rule("ready")
        print(green("\n  Everything is set up. Nothing to do.\n"))
        print("  Ask Claude to run " + bold("voltage_doctor") +
              dim("   (an MCP tool -- ask Claude, do not type it in a shell)"))
        pause()
        return

    rule(f"{len(steps)} step(s) left")
    for i, step in enumerate(steps, 1):
        mark = dim("[auto]") if step.automatic else yellow("[manual]")
        sudo = red(" needs sudo") if step.needs_sudo else ""
        print(f"  {i}. {bold(step.title)} {mark}{sudo}")
        print(f"     {dim(step.why)}")
    print()
    if not confirm("work through these now?", default=True):
        return

    for step in steps:
        print()
        rule(step.title)
        print(f"  {step.why}\n")
        if step.detail:
            for line in step.detail.splitlines():
                print(f"  {bold(line) if not line.startswith('http') else cyan(line)}")
            print()
        if not step.automatic:
            print(dim("  This one needs you -- do it in another terminal, then continue."))
            pause()
            continue
        if not confirm(f"do it? ({step.title.lower()})", default=True):
            print(dim("  skipped"))
            continue
        _run_step(step, config, env)

    print()
    rule("done")
    env = detect()
    print(f"  {ok_mark(env.input_ok)} input   {ok_mark(env.capture_ok)} capture   "
          f"{ok_mark(env.servers_up)} models   {ok_mark(bool(env.mcp_registered))} mcp")
    if env.ready:
        print(green("\n  Ready. Restart Claude Code, then ask it to run voltage_doctor."))
    else:
        print(yellow("\n  Some steps are still outstanding -- run setup again to pick up"))
        print(yellow("  where it left off."))
    pause()


def _run_step(step, config: Config, env) -> None:
    """Execute one automatic step."""
    profile = config.profile
    if step.key == "path":
        _install_path_link()
    elif step.key == "ollama-pull":
        from .llm import get_profile

        try:
            spec = get_profile(profile)
            tags = [t for t in (spec.vision.ollama_tag, spec.actuator.ollama_tag) if t]
        except KeyError:
            tags = ["qwen2.5vl:3b", "qwen3:1.7b"]
        for tag in tags:
            run(["ollama", "pull", tag])
    elif step.key == "llama":
        run(["./scripts/build-llama.sh"])
    elif step.key == "weights":
        run(["./scripts/fetch-models.sh", profile])
    elif step.key == "serve":
        if config.engine == "ollama":
            print(dim("  ollama serves on demand; nothing to start"))
        else:
            run(["./scripts/serve.sh", profile])
    elif step.key == "mcp":
        connect_mcp(config)


def _install_path_link() -> None:
    """Symlink the venv entry points into ~/.local/bin."""
    target_dir = Path.home() / ".local" / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for name in ("voltage", "voltage-input-mcp"):
        source = REPO_ROOT / ".venv" / "bin" / name
        if not source.exists():
            continue
        link = target_dir / name
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(source)
            made.append(str(link))
        except OSError as exc:
            print(red(f"  could not link {link}: {exc}"))
    for path in made:
        print(green(f"  linked {path}"))
    if made and str(target_dir) not in os.environ.get("PATH", ""):
        print(yellow(f"\n  {target_dir} is not on your PATH. For fish:"))
        print(bold(f"    fish_add_path {target_dir}"))


def _explain_llama_windows() -> None:
    print("  Download a CUDA build of llama.cpp and put llama-server.exe on PATH:\n")
    print(bold("    https://github.com/ggml-org/llama.cpp/releases"))
    print(dim("    pick llama-<version>-bin-win-cuda-x64.zip"))
    print("\n  Then fetch the weights for this profile from HuggingFace into models\\ .")
    print(dim("  `voltage serve-models` prints the exact filenames and launch flags."))


def _explain_serve_windows() -> None:
    print("  `voltage serve-models` prints the two llama-server command lines for the")
    print("  current profile, already tuned. Run each in its own terminal.\n")
    print(dim("  Set GGML_CUDA_ENABLE_UNIFIED_MEMORY=0 first: if it is 1, a VRAM overflow"))
    print(dim("  spills silently to system RAM and everything becomes ~10x slower."))


def _explain_uinput() -> None:
    """Distinguish the two failure modes, which need completely different fixes."""
    from .inputs import probe_input

    report = probe_input()
    print(f"  {red('input device not usable')}: {report.get('reason')}\n")

    if report.get("fix"):
        # ENODEV or a missing node: the module is the problem, not permissions.
        print("  Fix:")
        print(bold(f"    {report['fix']}"))
        print()
        print(dim("  The device node can exist with no driver behind it -- udev creates"))
        print(dim("  it so that opening it autoloads the module. When that does not"))
        print(dim("  happen, every injected event is silently discarded."))
        return

    print("  This is a permissions problem. Either:\n")
    print(bold("    sudo usermod -aG input $USER") + "      then log out and back in")
    print(dim("      (also enables grab-the-mouse-to-stop)"))
    print("\n  or install the udev rule described in scripts/setup.sh, which grants")
    print("  access to whoever is logged in at the seat rather than permanently.")


def connect_mcp(config: Config) -> None:
    """Register with Claude Code, passing the session environment through.

    The `-e` flags are the whole point. MCP clients start servers with a sanitised
    environment, and without a session bus the capture layer cannot reach the compositor
    -- while input injection still works, because uinput is a device file rather than a
    session service. That asymmetry produces a server that appears to work and silently
    cannot see the screen, which is a miserable thing to debug.
    """
    if not shutil.which("claude"):
        print(red("  the `claude` CLI is not on your PATH"))
        print("  install Claude Code, or register the server manually with this command:")
        print(dim(f"    {venv_bin('voltage-input-mcp')}"))
        return

    passthrough = [
        "WAYLAND_DISPLAY", "DISPLAY", "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR", "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP",
    ]
    env_args: list[str] = []
    missing: list[str] = []
    for key in passthrough:
        value = os.environ.get(key)
        if value:
            env_args += ["-e", f"{key}={value}"]
        else:
            missing.append(key)

    if missing:
        print(yellow(f"  not set in this shell, so not forwarded: {', '.join(missing)}"))
        if "DBUS_SESSION_BUS_ADDRESS" in missing:
            print(yellow("  without a session bus the MCP server cannot capture the screen."))
            print(yellow("  run this from inside your desktop session, not over plain ssh.\n"))

    scope = ask("scope (user = every project, local = this one)", "user")
    binary = str(REPO_ROOT / ".venv" / "bin" / "voltage-input-mcp")

    capture(["claude", "mcp", "remove", "voltage-input", "--scope", scope], timeout=15.0)
    code = run(
        ["claude", "mcp", "add", "voltage-input", "--scope", scope, *env_args, "--", binary]
    )
    if code == 0:
        print(green("\n  registered."))
        print(bold("  Restart Claude Code") + " -- MCP servers are loaded at session start.")
        print("\n  Then ask Claude (do not type these in a shell):")
        print(f"    {bold('voltage_doctor')}     confirm everything is green")
        print(f"    {bold('voltage_capture')}    first run shows a screen-share picker")
        print(f"    {bold('voltage_calibrate')}  moves your real cursor; watch it")
    else:
        print(red("\n  registration failed -- see the output above"))


def screen_models(state: dict[str, object]) -> None:
    from .llm import all_profiles, detect_vram_mb, recommend
    from .llm.profiles import DESKTOP_RESERVE_MB, usable_vram_mb

    config: Config = state["config"]  # type: ignore[assignment]
    while True:
        clear()
        banner()
        rule("models")
        vram = detect_vram_mb()
        budget = usable_vram_mb(vram) if vram else 0
        if vram:
            print(f"  GPU {bold(f'{vram} MB')}   usable {bold(f'{budget} MB')} "
                  f"{dim(f'(reserving {DESKTOP_RESERVE_MB} MB for the desktop)')}")
            print(f"  recommended profile: {green(recommend(vram).name)}\n")

        from .llm.profiles import EXPERIMENTAL

        ordered = list(all_profiles().values())
        stable = [p for p in ordered if p.name not in EXPERIMENTAL]
        experimental = [p for p in ordered if p.name in EXPERIMENTAL]

        def show(profile, position: int, current: str, budget_mb: int | None) -> None:
            mark = "*" if profile.name == current else " "
            fit = "" if budget_mb is None else (
                green("  fits") if profile.fits(budget_mb) else red("  too big")
            )
            print(f"  {mark}{position}. {bold(profile.name):<12} ~{profile.vram_mb:>6} MB{fit}")
            print(f"      {dim(profile.description)}")

        budget_mb = budget if vram else None
        index = 0
        for profile in stable:
            index += 1
            show(profile, index, config.profile, budget_mb)
        if experimental:
            print()
            print(f"  {yellow('experimental')} "
                  f"{dim('-- each trades something away; read the warning')}")
            for profile in experimental:
                index += 1
                show(profile, index, config.profile, budget_mb)
        print()
        print(f"  running: vision {ok_mark(state['vision'])} actuator {ok_mark(state['actuator'])}")
        print()
        print("  [1-5] switch profile   [f] fetch weights   [s] start   [x] stop")
        print("  [b] benchmark          [c] compare models  [q] back")

        choice = ask("choice").lower()
        if choice in ("q", ""):
            return
        if choice.isdigit() and 1 <= int(choice) <= index:
            profile = (stable + experimental)[int(choice) - 1]
            if not _accept_profile(profile, budget if vram else None):
                continue
            config = _write_config(config, profile=profile.name)
            state["config"] = config
            print(green(f"  profile set to {profile.name}"))
            if confirm("fetch its weights now?", default=True):
                run(["./scripts/fetch-models.sh", profile.name])
            pause()
        elif choice == "f":
            run(["./scripts/fetch-models.sh", config.profile])
            pause()
        elif choice == "s":
            run(["./scripts/serve.sh", config.profile])
            state.update(snapshot(config))
            pause()
        elif choice == "x":
            run(["./scripts/serve.sh", "--stop"])
            state.update(snapshot(config))
            pause()
        elif choice == "b":
            run([venv_bin("voltage"), "bench"])
            pause()
        elif choice == "c":
            run([venv_bin("voltage"), "compare"])
            pause()


def _accept_profile(profile, budget: int | None) -> bool:
    """Show a profile's warning and VRAM reality before switching to it."""
    from .llm.profiles import EXPERIMENTAL

    if profile.notes:
        print()
        for line in _wrap(profile.notes, 72):
            print(f"  {dim(line)}")

    risky = False
    if profile.warning:
        risky = True
        print()
        print(f"  {yellow('WARNING')}")
        for line in _wrap(profile.warning, 72):
            print(f"  {yellow(line)}")

    if budget is not None and not profile.fits(budget):
        risky = True
        print()
        print(f"  {red('WILL NOT FIT')}  needs ~{profile.vram_mb} MB, you have ~{budget} MB free.")
        print(f"  {red('Layers will spill to system RAM and everything gets ~10x slower.')}")

    if profile.name in EXPERIMENTAL or risky:
        print()
        return confirm(f"use {profile.name} anyway?", default=False)
    return True


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width) or [""]


_EDITABLE: list[tuple[str, str, str]] = [
    ("engine", "llamacpp | ollama", "which backend serves the models"),
    ("profile", "lean | balanced | split | quality | ollama", "model pair"),
    ("dry_run", "true | false", "never inject input unless a run opts in"),
    ("target_period_s", "0.05 - 30", "decision period; how often the models are asked"),
    ("reflex_hz", "0 - 120", "fast loop: probes, holds and reflexes, no model. 0 = off"),
    ("capture_backend", "auto | portal | kwin | grim | x11", "screen capture"),
    ("pointer_mode", "absolute | relative", "switch if calibrate shows no movement"),
    ("text_mode", "auto | keys | clipboard", "how text is typed"),
    ("watch_physical_input", "true | false", "touch the mouse to stop a run"),
    ("keep_frames", "true | false", "save a PNG per cycle for debugging"),
    ("settle_ms", "0 - 2000", "pause after a burst before looking again"),
]


def screen_config(state: dict[str, object]) -> None:
    config: Config = state["config"]  # type: ignore[assignment]
    while True:
        clear()
        banner()
        rule("configuration")
        path = default_config_path()
        print(f"  {dim(str(path))}"
              f"{dim('  (not created yet; built-in defaults apply)') if not path.exists() else ''}\n")
        for i, (key, allowed, why) in enumerate(_EDITABLE, 1):
            value = getattr(config, key)
            print(f"  {i:>2}. {key:<22} {bold(str(value)):<22} {dim(allowed)}")
            print(f"      {dim(why)}")
        print()
        print(f"  [1-{len(_EDITABLE)}] edit   [r] reset to defaults   [q] back")

        choice = ask("choice").lower()
        if choice in ("q", ""):
            return
        if choice == "r":
            if confirm("delete the config file and use built-in defaults?"):
                path.unlink(missing_ok=True)
                state["config"] = config = load_config()
                print(green("  reset"))
                pause()
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(_EDITABLE):
            continue

        key, allowed, _ = _EDITABLE[int(choice) - 1]
        current = getattr(config, key)
        new = ask(f"{key} ({allowed})", str(current))
        if new == str(current):
            continue
        try:
            config = _write_config(config, **{key: _coerce(new, current)})
            state["config"] = config
            print(green(f"  {key} = {getattr(config, key)}"))
        except Exception as exc:  # noqa: BLE001 - show the validation error, keep going
            print(red(f"  rejected: {exc}"))
        pause()


def _coerce(raw: str, current: object) -> object:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _write_config(config: Config, **changes: object) -> Config:
    """Validate through Config, then persist. Never writes something that will not load."""
    from dataclasses import fields, replace

    updated = replace(config, **changes)  # type: ignore[arg-type]
    Config(**{f.name: getattr(updated, f.name)
              for f in fields(Config) if not f.name.startswith("_")})

    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Written by `voltage` interactive config.",
        "# Every key can also be set as VOLTAGE_<KEY> in the environment, which wins.",
        "",
        "[voltage]",
    ]
    for field_ in fields(Config):
        if field_.name.startswith("_"):
            continue
        value = getattr(updated, field_.name)
        if value is None:
            continue
        if isinstance(value, bool):
            lines.append(f"{field_.name} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{field_.name} = {value}")
        elif isinstance(value, tuple):
            lines.append(f"{field_.name} = [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f'{field_.name} = "{value}"')
    path.write_text("\n".join(lines) + "\n")
    return updated


def screen_custom_profiles(state: dict[str, object]) -> None:
    """Create, inspect and delete user-defined model profiles."""
    from .llm import all_profiles, delete_custom, load_custom, save_custom
    from .llm.custom import profiles_path
    from .llm.profiles import Profile

    while True:
        clear()
        banner()
        rule("custom profiles")
        path = profiles_path()
        print(f"  {dim(str(path))}\n")
        try:
            custom = load_custom()
            error = None
        except Exception as exc:  # noqa: BLE001
            custom, error = {}, str(exc)
        if error:
            print(red(f"  profiles.toml could not be read: {error}\n"))
        if custom:
            for name, profile in custom.items():
                shadow = dim("  (shadows a built-in)") if name in (
                    "lean", "balanced", "split", "quality", "ollama") else ""
                print(f"  {bold(name)}{shadow}")
                print(f"    vision   {profile.vision.label}")
                print(f"    actuator {profile.actuator.label}   ~{profile.vram_mb} MB")
        else:
            print(dim("  none yet.\n"))
            print("  A custom profile lets you use your own models -- a different quant,")
            print("  a newer VL model, or a fine-tune. It is merged over the built-ins by")
            print("  name, so naming one `lean` retunes the built-in without forking.\n")
            print(dim("  The vision slot is the constrained one: it must emit grounded"))
            print(dim("  bounding boxes. Qwen2.5-VL, Qwen3-VL, InternVL, MiniCPM-V and"))
            print(dim("  UI-TARS all do. A general captioner will not. The actuator slot"))
            print(dim("  is forgiving -- under a grammar, almost any 1B+ instruct model works."))
        print()
        print("  [n] new   [c] copy a built-in as a starting point   [e] open in $EDITOR")
        print("  [d] delete   [q] back")

        choice = ask("choice").lower()
        if choice in ("q", ""):
            return
        if choice == "n":
            _new_custom_profile()
        elif choice == "c":
            name = ask("built-in to copy", "lean")
            try:
                source = all_profiles()[name]
            except KeyError:
                print(red(f"  no profile {name!r}"))
                pause()
                continue
            new_name = ask("new name", f"my_{name}")
            copied = Profile(
                name=new_name, description=f"copy of {name}",
                vision=source.vision, actuator=source.actuator,
                min_vram_mb=source.min_vram_mb, notes=source.notes,
                actuator_on_cpu=source.actuator_on_cpu,
            )
            save_custom(copied)
            print(green(f"  wrote {new_name} to {path}"))
            print(dim("  use [e] to edit the model files and ports"))
            pause()
        elif choice == "e":
            editor = os.environ.get("EDITOR") or ("notepad" if sys.platform == "win32" else "nano")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("# Custom profiles. See [c] to copy a built-in.\n")
            run([editor, str(path)], quiet=True)
        elif choice == "d":
            name = ask("delete which")
            if name and confirm(f"delete {name}?"):
                delete_custom(name)
                print(green("  deleted"))
                pause()


def _new_custom_profile() -> None:
    from .llm import save_custom
    from .llm.profiles import ModelSpec, Profile

    print()
    rule("new profile")
    name = ask("profile name (letters, digits, underscore)")
    if not name or not name.replace("_", "").isalnum():
        print(red("  invalid name"))
        pause()
        return

    backend = ask("models come from: (h)uggingface gguf or (o)llama", "h").lower()
    specs = {}
    for role, port in (("vision", 8080), ("actuator", 8081)):
        print(f"\n  {bold(role)} model")
        if role == "vision":
            print(dim("    must be able to emit bounding boxes on request"))
        if backend.startswith("o"):
            tag = ask("  ollama tag", "qwen2.5vl:3b" if role == "vision" else "qwen3:1.7b")
            specs[role] = ModelSpec(
                role=role, label=tag, hf_repo="", hf_file="", ollama_tag=tag,
                params_b=float(ask("  size in billions of params", "3")),
                weights_mb=int(ask("  approx weights MB", "2000")),
                n_ctx=int(ask("  context", "2048")), port=port,
            )
        else:
            repo = ask("  huggingface repo (owner/name)")
            file = ask("  gguf filename")
            mmproj = ask("  mmproj filename (vision only, blank if none)", "") if role == "vision" else ""
            specs[role] = ModelSpec(
                role=role, label=ask("  display label", file or repo),
                hf_repo=repo, hf_file=file,
                mmproj_repo=repo if mmproj else None, mmproj_file=mmproj or None,
                params_b=float(ask("  size in billions of params", "3")),
                weights_mb=int(ask("  approx weights MB", "2000")),
                n_ctx=int(ask("  context", "2048")), port=port,
            )

    profile = Profile(
        name=name, description=ask("description", "custom profile"),
        vision=specs["vision"], actuator=specs["actuator"], min_vram_mb=0,
    )
    path = save_custom(profile)
    print(green(f"\n  saved to {path}"))
    print(f"  estimated VRAM: {bold(str(profile.vram_mb))} MB")
    print(dim("  select it from the models screen, then fetch its weights"))
    pause()


def screen_connect(state: dict[str, object]) -> None:
    """What is set up, what the URLs are, and exactly how to wire this into a client."""
    from .connect import CLIENTS, desktop_config_path, gather, instructions, stdio_json

    while True:
        clear()
        banner()
        info = gather()

        rule("what is set up")
        print(f"  {ok_mark(info.binary_exists)} server binary")
        print(f"      {dim(info.binary)}")
        if not info.binary_exists:
            print(f"      {red('missing -- run setup, or pip install -e . in the repo')}")

        print(f"  {ok_mark(info.http_running)} http endpoint  "
              f"{dim('running' if info.http_running else 'not running')}")
        print(f"      {cyan(info.http_url) if info.http_running else dim(info.http_url)}")
        if not info.http_running:
            print(f"      {dim('start with:')} {bold('voltage serve --http')}")

        engine_ok = info.vision_up and info.actuator_up
        print(f"  {ok_mark(engine_ok)} models         {dim(info.engine + ' / ' + info.profile)}")
        if info.engine == "ollama":
            print(f"      {dim('http://127.0.0.1:11434')}")
        else:
            print(f"      {dim('vision   ' + info.vision_url)}  "
                  f"{green('up') if info.vision_up else red('down')}")
            print(f"      {dim('actuator ' + info.actuator_url)}  "
                  f"{green('up') if info.actuator_up else red('down')}")

        print(f"  {ok_mark(info.registered if info.claude_cli else None)} claude code    ", end="")
        if not info.claude_cli:
            print(dim("cli not installed"))
        elif info.registered:
            print(dim(f"{info.registered_scope}  ({info.registered_status})"))
            if info.registered_env_ok is False:
                print(f"      {yellow('registered without the session environment --')}")
                print(f"      {yellow('it will connect but cannot capture the screen.')}")
                print(f"      {dim('re-register with [a] to fix.')}")
        else:
            print(dim("not registered"))

        print(f"  {ok_mark(info.desktop_has_entry if info.desktop_config else None)} "
              f"claude desktop ", end="")
        if not info.desktop_config:
            print(dim(f"config not found ({desktop_config_path().name})"))
        else:
            print(dim("configured" if info.desktop_has_entry else "config exists, no entry"))

        if info.missing_env:
            print()
            print(f"  {yellow('note')} not set in this shell, so cannot be forwarded:")
            print(f"       {yellow(', '.join(info.missing_env))}")
            print(dim("       Screen capture needs these. Input injection does not, so a"))
            print(dim("       server started without them appears to work and cannot see"))
            print(dim("       the screen. Run this from inside your desktop session."))

        print()
        rule("set up as a custom MCP server")
        for i, client in enumerate(CLIENTS, 1):
            tag = cyan("[http]") if client.transport == "http" else dim("[stdio]")
            print(f"  {bold(str(i))}  {client.name:<32} {tag}")
        print()
        print("  [1-7] step-by-step for that client   [j] print the JSON")
        print("  [a] register with Claude Code now    [w] write Claude Desktop config")
        print("  [q] back")

        choice = ask("choice").lower()
        if choice in ("q", ""):
            return
        if choice == "j":
            clear()
            banner()
            rule("mcpServers entry")
            print()
            print(stdio_json(info))
            print()
            print(dim("  Paths and environment are filled in from this machine -- copy it"))
            print(dim("  verbatim, do not substitute placeholders."))
            pause()
        elif choice == "a":
            connect_mcp(state["config"])  # type: ignore[arg-type]
            pause()
        elif choice == "w":
            _write_desktop_config(info)
            pause()
        elif choice.isdigit() and 1 <= int(choice) <= len(CLIENTS):
            client = CLIENTS[int(choice) - 1]
            clear()
            banner()
            rule(client.name)
            print(f"  transport: {bold(client.transport)}    config: {dim(client.config_hint)}")
            if client.note:
                print()
                for line in _wrap(client.note, 72):
                    print(f"  {dim(line)}")
            print()
            for n, (text, block) in enumerate(instructions(client.key, info), 1):
                for j, line in enumerate(_wrap(text, 70)):
                    print(f"  {bold(str(n) + '.') if j == 0 else '  '} {line}")
                if block:
                    print()
                    for line in block.splitlines():
                        print(f"       {cyan(line)}")
                print()
            pause()


def _write_desktop_config(info) -> None:
    """Merge the entry into Claude Desktop's config, preserving anything already there."""
    import json

    from .connect import desktop_config_path

    path = desktop_config_path()
    print()
    print(f"  target: {dim(str(path))}")

    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except ValueError:
            print(red("  the existing file is not valid JSON -- refusing to overwrite it."))
            print(dim("  Fix or move it, then try again."))
            return
        backup = path.with_suffix(".json.bak")
        try:
            backup.write_text(path.read_text())
            print(dim(f"  backed up to {backup}"))
        except OSError as exc:
            print(red(f"  could not write a backup ({exc}); not touching the file."))
            return

    servers = data.setdefault("mcpServers", {})
    if "voltage-input" in servers and not confirm("entry exists -- replace it?", default=True):
        return

    entry: dict[str, object] = {"command": info.binary, "args": []}
    if info.env:
        entry["env"] = dict(info.env)
    servers["voltage-input"] = entry

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as exc:
        print(red(f"  could not write: {exc}"))
        return
    print(green("  written."))
    print(f"  {bold('Fully quit Claude Desktop and reopen it')} -- reloading is not enough.")


def screen_instructions(state: dict[str, object]) -> None:
    """Standing instructions the orchestrator receives with every session."""
    from .briefing import (
        MAX_INSTRUCTIONS,
        TEMPLATES,
        instructions_path,
        load_instructions,
        save_instructions,
    )

    while True:
        clear()
        banner()
        rule("your instructions")
        path = instructions_path()
        current = load_instructions()

        print(f"  {dim(str(path))}\n")
        print("  Anything you write here is given to the orchestrating model at the start")
        print("  of every session, marked as coming from you. Use it for things it cannot")
        print("  work out on its own:\n")
        print(dim("    - applications or windows that are off limits"))
        print(dim("    - quirks of a specific game (timing, which keys, where the HUD is)"))
        print(dim("    - how you want it to behave by default, e.g. always ask first"))
        print()
        print(f"  {yellow('This does not weaken safety.')} The governor checks every burst in")
        print(dim("  code -- no instruction here can permit something a policy forbids."))
        print()

        if current:
            rule("current")
            for line in current.splitlines()[:18]:
                print(f"  {line}")
            extra = len(current.splitlines()) - 18
            if extra > 0:
                print(dim(f"  ... {extra} more line(s)"))
            print()
            print(dim(f"  {len(current)} / {MAX_INSTRUCTIONS} characters"))
        else:
            print(dim("  Nothing set. The orchestrator gets only the build briefing."))
        print()
        print("  [e] edit in $EDITOR   [t] start from a template   [c] clear   [q] back")

        choice = ask("choice").lower()
        if choice in ("q", ""):
            return
        if choice == "e":
            editor = os.environ.get("EDITOR") or (
                "notepad" if sys.platform == "win32" else "nano"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                save_instructions(TEMPLATES["minimal"])
            run([editor, str(path)], quiet=True)
            if len(load_instructions()) >= MAX_INSTRUCTIONS:
                print(yellow(f"  Truncated at {MAX_INSTRUCTIONS} characters -- this text "
                             f"sits in the model's context for the whole session, so it "
                             f"is capped."))
                pause()
        elif choice == "t":
            print()
            for i, name in enumerate(TEMPLATES, 1):
                print(f"  {i}. {name}")
            pick = ask("template", "1")
            names = list(TEMPLATES)
            if pick.isdigit() and 1 <= int(pick) <= len(names):
                name = names[int(pick) - 1]
                if current and not confirm(f"replace what is there with '{name}'?"):
                    continue
                save_instructions(TEMPLATES[name])
                print(green(f"  wrote the '{name}' template -- edit it with [e]"))
                pause()
        elif choice == "c":
            if current and confirm("clear your instructions?"):
                save_instructions("")
                print(green("  cleared"))
                pause()


def screen_test(state: dict[str, object]) -> None:
    while True:
        clear()
        banner()
        rule("diagnostics")
        print("  [d] doctor        full readiness report")
        print("  [r] reflex        measure the fast loop's real rate on this machine")
        print("  [c] capture       save a screenshot to screen.png")
        print("  [b] burst         parse a burst string (dry run)")
        print(f"  [k] calibrate     {yellow('moves your real cursor')} -- watch the screen")
        print("  [p] profiles      what fits this GPU")
        print("  [j] journal dir   where run records are written")
        print("  [q] back\n")

        choice = ask("choice").lower()
        if choice in ("q", ""):
            return
        if choice == "d":
            run([venv_bin("voltage"), "doctor"])
        elif choice == "r":
            print(dim("\n  Runs the real reflex loop for five seconds against your"))
            print(dim("  screen. Nothing is injected -- the run is dry.\n"))
            run([venv_bin("voltage"), "reflex", "--seconds", "5"])
        elif choice == "c":
            run([venv_bin("voltage"), "capture", "-o", "screen.png"])
        elif choice == "b":
            burst = ask("burst", 'm:640,360;c:l;w:120;t:"hello"')
            if burst:
                run([venv_bin("voltage"), "burst", burst])
        elif choice == "k":
            print(yellow("\n  This moves the real pointer to three points on screen."))
            print("  Watch it. If the cursor does not move, absolute positioning is not")
            print("  being honoured and pointer_mode should be set to \"relative\".\n")
            if confirm("run it for real?", default=False):
                run([venv_bin("voltage"), "burst",
                     f"m:{200},{200};w:300;m:{600},{400};w:300;m:{300},{600}", "--live"])
            else:
                print(dim("  skipped"))
        elif choice == "p":
            run([venv_bin("voltage"), "profiles"])
        elif choice == "j":
            from .runtime import default_journal_dir

            print(f"  {default_journal_dir()}")
        pause()


def screen_help() -> None:
    clear()
    banner()
    rule("how the pieces fit")
    print(f"""
  There are two different surfaces, and mixing them up is the usual first
  stumble:

  {bold('shell commands')}   typed in your terminal, with a {bold('space')}
                   {dim('voltage doctor, voltage bench, voltage stop')}

  {bold('MCP tools')}        asked of Claude, with an {bold('underscore')}
                   {dim('voltage_doctor, voltage_capture, voltage_run')}
                   These are not shell commands. Typing voltage_doctor in a
                   terminal will always say "unknown command" -- it is a tool
                   name in Claude's namespace, not a program on disk.

  {bold('The division of labour')}

    You (Claude)   writes a Playbook: a state machine saying what to look
                   for, what is allowed, and when to move on. Thinks once.
    Vision model   "of these specific things, which are on screen, where?"
    Actuator       "given that, which inputs?" -- emits a timed burst
    Governor       refuses anything the Playbook did not permit

  {bold('Emergency stop')}, works even when the desktop is unusable:

      {bold('voltage stop')}

  It writes a file. Every running loop notices within one cycle and releases
  every held key. Works over ssh, from another TTY, from a file manager.
""")
    pause()


# -- main loop -----------------------------------------------------------------------------


def run_console() -> int:
    if not sys.stdin.isatty():
        print("voltage: not a terminal; run `voltage --help` for the command list",
              file=sys.stderr)
        return 2

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001 - a broken config must not lock you out
        print(red(f"config error: {exc}"))
        print("starting with built-in defaults; use the config screen to fix it.\n")
        config = Config()

    state = snapshot(config)

    while True:
        clear()
        banner()
        status_block(state)
        rule("menu")
        ready = state["uinput"] and state["mcp"] and (
            state["ollama"] if state["config"].engine == "ollama"  # type: ignore[union-attr]
            else state["vision"] and state["actuator"]
        )
        print(f"  {bold('1')}  setup        {dim('fix whatever is not ready')}"
              f"{'' if ready else yellow('   <- start here')}")
        print(f"  {bold('2')}  models       {dim('switch profile, fetch, serve, benchmark')}")
        print(f"  {bold('3')}  config       {dim('edit settings')}")
        print(f"  {bold('p')}  profiles     {dim('add your own models')}")
        print(f"  {bold('i')}  instructions {dim('what the orchestrator is told about your setup')}")
        print(f"  {bold('4')}  connect      {dim('urls, status, and per-client setup steps')}")
        print(f"  {bold('5')}  diagnostics  {dim('doctor, capture, calibrate')}")
        print(f"  {bold('6')}  help         {dim('shell commands vs MCP tools')}")
        print(f"  {bold('r')}  refresh      {bold('q')}  quit")
        print()

        choice = ask("choice").lower()
        if choice in ("q", "quit", "exit"):
            print(dim("\n  bye\n"))
            return 0
        if choice == "1":
            screen_setup(state)
        elif choice == "2":
            screen_models(state)
        elif choice == "3":
            screen_config(state)
        elif choice == "p":
            screen_custom_profiles(state)
        elif choice == "i":
            screen_instructions(state)
        elif choice == "4":
            screen_connect(state)
        elif choice == "5":
            screen_test(state)
        elif choice == "6":
            screen_help()
        state.update(snapshot(state["config"]))  # type: ignore[arg-type]
