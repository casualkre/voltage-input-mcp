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
from collections.abc import Callable
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
    from .inputs import probe_uinput

    return {
        "config": config,
        "uinput": bool(probe_uinput().get("ok")),
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
    print(f"  {ok_mark(state['uinput'])} input device      {dim('/dev/uinput')}")
    if engine == "ollama":
        print(f"  {ok_mark(state['ollama'])} ollama            {dim(config.ollama_url)}")
    else:
        print(f"  {ok_mark(state['vision'])} vision model      {dim(config.vision_url)}")
        print(f"  {ok_mark(state['actuator'])} actuator model    {dim(config.actuator_url)}")
    print(f"  {ok_mark(state['mcp'])} mcp registered    {dim('claude mcp list')}")
    print(f"  {ok_mark(state['path'])} voltage on PATH   {dim('~/.local/bin/voltage')}")
    print()
    print(f"  profile {bold(config.profile)}   engine {bold(engine)}   "
          f"dry_run {bold(str(config.dry_run))}   {len(state['models'])} model file(s)")  # type: ignore[arg-type]
    print()


# -- screens ------------------------------------------------------------------------------


def screen_setup(state: dict[str, object]) -> None:
    """Guided fix-what-is-broken flow."""
    clear()
    banner()
    rule("guided setup")
    print("  Fixes whatever is not ready, in dependency order.\n")

    config: Config = state["config"]  # type: ignore[assignment]
    steps: list[tuple[str, bool, Callable[[], None]]] = []

    if not state["path"]:
        steps.append(("put `voltage` on your PATH", True, _install_path_link))
    if not state["uinput"]:
        steps.append(("fix /dev/uinput permissions", False, _explain_uinput))
    if config.engine == "llamacpp":
        if not state["llama"]:
            steps.append(("build llama.cpp (~20 min)", True,
                          lambda: run(["./scripts/build-llama.sh"])))
        if not state["models"]:
            steps.append((f"download {config.profile} weights", True,
                          lambda: run(["./scripts/fetch-models.sh", config.profile])))
        if not (state["vision"] and state["actuator"]):
            steps.append(("start the model servers", True,
                          lambda: run(["./scripts/serve.sh", config.profile])))
    elif not state["ollama"]:
        steps.append(("start ollama", False, lambda: print(
            "  systemctl --user start ollama   (or: systemctl start ollama)")))
    if not state["mcp"]:
        steps.append(("connect to Claude Code", True, lambda: connect_mcp(config)))

    if not steps:
        print(green("  Everything is ready. Nothing to do.\n"))
        print("  Next: ask Claude to run " + bold("voltage_doctor") + dim("  (an MCP tool, not a shell command)"))
        pause()
        return

    for i, (label, _auto, _fn) in enumerate(steps, 1):
        print(f"  {i}. {label}")
    print()

    for label, automatic, action in steps:
        if not confirm(f"{label}?", default=automatic):
            print(dim("  skipped\n"))
            continue
        action()
        print()
    pause()


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


def _explain_uinput() -> None:
    """Distinguish the two failure modes, which need completely different fixes."""
    from .inputs import probe_uinput

    report = probe_uinput()
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
    from .llm import PROFILES, detect_vram_mb, recommend
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

        for i, profile in enumerate(PROFILES.values(), 1):
            mark = "*" if profile.name == config.profile else " "
            fit = "" if not vram else (
                green("  fits") if profile.fits(budget) else red("  too big")
            )
            print(f"  {mark}{i}. {bold(profile.name):<18} ~{profile.vram_mb:>5} MB{fit}")
            print(f"      {dim(profile.description)}")
        print()
        print(f"  running: vision {ok_mark(state['vision'])} actuator {ok_mark(state['actuator'])}")
        print()
        print("  [1-5] switch profile   [f] fetch weights   [s] start   [x] stop")
        print("  [b] benchmark          [c] compare models  [q] back")

        choice = ask("choice").lower()
        if choice in ("q", ""):
            return
        if choice.isdigit() and 1 <= int(choice) <= len(PROFILES):
            name = list(PROFILES)[int(choice) - 1]
            config = _write_config(config, profile=name)
            state["config"] = config
            print(green(f"  profile set to {name}"))
            if confirm("fetch its weights now?", default=True):
                run(["./scripts/fetch-models.sh", name])
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


_EDITABLE: list[tuple[str, str, str]] = [
    ("engine", "llamacpp | ollama", "which backend serves the models"),
    ("profile", "lean | balanced | split | quality | ollama", "model pair"),
    ("dry_run", "true | false", "never inject input unless a run opts in"),
    ("target_period_s", "0.05 - 30", "loop period; lower for games"),
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
        print("  [1-10] edit   [r] reset to defaults   [q] back")

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


def screen_test(state: dict[str, object]) -> None:
    while True:
        clear()
        banner()
        rule("diagnostics")
        print("  [d] doctor        full readiness report")
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
        print(f"  {bold('4')}  connect      {dim('register with Claude Code')}")
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
        elif choice == "4":
            clear()
            banner()
            rule("connect to Claude Code")
            connect_mcp(state["config"])  # type: ignore[arg-type]
            pause()
        elif choice == "5":
            screen_test(state)
        elif choice == "6":
            screen_help()
        state.update(snapshot(state["config"]))  # type: ignore[arg-type]
