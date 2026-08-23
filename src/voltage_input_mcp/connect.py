"""Everything needed to register this server with an MCP client.

Two transports, and which one a client wants is not obvious from the outside:

    stdio    the client launches the server as a subprocess and talks over pipes.
             What Claude Code, Claude Desktop, Cursor, Windsurf, Zed and Continue use.
    http     the server runs on its own and the client connects to a URL. What
             "custom connector" means in claude.ai and similar hosted clients.

The recurring failure with the stdio form is the environment. Clients start servers with
a sanitised environment -- PATH, HOME, and little else. Input injection survives that
(uinput and SendInput are OS facilities, not session services) but screen capture does
not, because reaching the compositor needs `DBUS_SESSION_BUS_ADDRESS` and
`WAYLAND_DISPLAY`. The result is a server that looks like it works and silently cannot
see the screen. Every config this module emits therefore carries the environment
explicitly, filled in from the live session rather than left as a placeholder.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ConnectionInfo", "gather", "PASSTHROUGH_ENV",
    "stdio_json", "claude_code_command", "desktop_config_path", "CLIENTS",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Variables the server needs that a sanitised launch environment will not provide.
PASSTHROUGH_ENV: tuple[str, ...] = (
    "WAYLAND_DISPLAY",
    "DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
)


@dataclass(slots=True)
class ConnectionInfo:
    """What exists right now, resolved to concrete paths and URLs."""

    binary: str = ""
    binary_exists: bool = False
    on_path: bool = False

    http_port: int = 8765
    http_running: bool = False
    http_url: str = ""

    vision_url: str = ""
    actuator_url: str = ""
    vision_up: bool = False
    actuator_up: bool = False
    engine: str = ""
    profile: str = ""

    env: dict[str, str] = field(default_factory=dict)
    missing_env: list[str] = field(default_factory=list)

    claude_cli: bool = False
    registered: bool = False
    registered_scope: str = ""
    registered_status: str = ""
    registered_env_ok: bool | None = None
    desktop_config: Path | None = None
    desktop_has_entry: bool = False


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def desktop_config_path() -> Path:
    """Claude Desktop's config file for this platform."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Claude" / "claude_desktop_config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path(
        os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    ) / "Claude" / "claude_desktop_config.json"


def _server_binary() -> tuple[str, bool]:
    """Absolute path to the stdio entry point.

    Deliberately absolute rather than the bare name: a client launching this has no
    reason to share your PATH, and "command not found" from inside an MCP client is
    considerably harder to diagnose than a wrong path in a config file.
    """
    name = "voltage-input-mcp.exe" if sys.platform == "win32" else "voltage-input-mcp"
    venv = REPO_ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / name
    if venv.exists():
        return str(venv), True
    found = shutil.which("voltage-input-mcp")
    if found:
        return found, True
    return str(venv), False


def gather(port: int = 8765) -> ConnectionInfo:
    """Resolve the current connection picture."""
    from .config import load_config

    info = ConnectionInfo()
    info.binary, info.binary_exists = _server_binary()
    info.on_path = shutil.which("voltage-input-mcp") is not None

    info.http_port = port
    info.http_running = _port_open(port)
    info.http_url = f"http://127.0.0.1:{port}/mcp"

    try:
        config = load_config()
        info.engine = config.engine
        info.profile = config.profile
        info.vision_url = config.vision_url
        info.actuator_url = config.actuator_url
    except Exception:  # noqa: BLE001
        info.engine, info.profile = "?", "?"
        info.vision_url, info.actuator_url = "http://127.0.0.1:8080", "http://127.0.0.1:8081"

    if info.engine == "ollama":
        info.vision_up = info.actuator_up = _port_open(11434)
    else:
        try:
            info.vision_up = _port_open(int(info.vision_url.rsplit(":", 1)[-1]))
            info.actuator_up = _port_open(int(info.actuator_url.rsplit(":", 1)[-1]))
        except ValueError:
            pass

    for key in PASSTHROUGH_ENV:
        value = os.environ.get(key)
        if value:
            info.env[key] = value
        elif sys.platform != "win32":
            # Windows needs none of these; on Linux/macOS a missing one is meaningful.
            info.missing_env.append(key)

    info.claude_cli = shutil.which("claude") is not None
    if info.claude_cli:
        import subprocess

        # `claude mcp get` reports scope, live connection status, and the environment the
        # server was registered with -- which is the thing that actually goes wrong.
        # (`claude mcp list` has no --scope flag; asking for one just errors.)
        try:
            proc = subprocess.run(
                ["claude", "mcp", "get", "voltage-input"],
                capture_output=True, timeout=20.0, check=False,
            )
            out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        except (OSError, subprocess.TimeoutExpired):
            out = ""

        if "voltage-input:" in out:
            info.registered = True
            for line in out.splitlines():
                stripped = line.strip()
                if stripped.startswith("Scope:"):
                    info.registered_scope = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Status:"):
                    info.registered_status = (
                        stripped.split(":", 1)[1].strip().replace("\u2714", "").strip()
                    )
            # Was it registered WITH the session environment? A server registered from a
            # shell that lacked DBUS_SESSION_BUS_ADDRESS is connected and blind, and
            # nothing about its status says so.
            needed = [k for k in ("DBUS_SESSION_BUS_ADDRESS", "WAYLAND_DISPLAY", "DISPLAY")
                      if k in info.env]
            if needed:
                info.registered_env_ok = all(f"{k}=" in out for k in needed[:1]) or all(
                    k in out for k in needed[:1]
                )

    path = desktop_config_path()
    if path.exists():
        info.desktop_config = path
        try:
            data = json.loads(path.read_text())
            info.desktop_has_entry = "voltage-input" in (data.get("mcpServers") or {})
        except (OSError, ValueError):
            pass

    return info


# --------------------------------------------------------------------------------------
# Config generation
# --------------------------------------------------------------------------------------


def stdio_json(info: ConnectionInfo, *, indent: int = 2) -> str:
    """The `mcpServers` fragment nearly every desktop client accepts."""
    entry: dict[str, object] = {"command": info.binary, "args": []}
    if info.env:
        entry["env"] = dict(info.env)
    return json.dumps({"mcpServers": {"voltage-input": entry}}, indent=indent)


def claude_code_command(info: ConnectionInfo, scope: str = "user") -> str:
    """A single shell line for `claude mcp add`, environment included."""
    parts = ["claude", "mcp", "add", "voltage-input", "--scope", scope]
    for key, value in info.env.items():
        parts += ["-e", f"{key}={value}"]
    parts += ["--", info.binary]

    quoted = []
    for part in parts:
        if " " in part or '"' in part:
            quoted.append('"' + part.replace('"', '\\"') + '"')
        else:
            quoted.append(part)
    return " ".join(quoted)


@dataclass(frozen=True, slots=True)
class Client:
    key: str
    name: str
    transport: str
    config_hint: str
    note: str = ""


CLIENTS: tuple[Client, ...] = (
    Client(
        "claude-code", "Claude Code (CLI)", "stdio",
        "registered with `claude mcp add`; no file to edit",
        "Restart Claude Code after adding -- servers are loaded at session start.",
    ),
    Client(
        "claude-desktop", "Claude Desktop", "stdio",
        "claude_desktop_config.json",
        "Fully quit and reopen the app; reloading the window is not enough.",
    ),
    Client(
        "claude-ai", "claude.ai custom connector", "http",
        "add the URL in the connector settings",
        "Needs `voltage serve --http` running. Loopback only -- a hosted client can "
        "only reach it if it runs on this machine.",
    ),
    Client(
        "cursor", "Cursor", "stdio",
        ".cursor/mcp.json in the project, or ~/.cursor/mcp.json globally",
    ),
    Client(
        "windsurf", "Windsurf", "stdio",
        "~/.codeium/windsurf/mcp_config.json",
    ),
    Client(
        "zed", "Zed", "stdio",
        "settings.json, under context_servers",
        "Zed nests the command differently: see its docs for the exact shape.",
    ),
    Client(
        "generic", "Any other MCP client", "stdio",
        "whatever it uses for mcpServers",
    ),
)


def instructions(client_key: str, info: ConnectionInfo) -> list[tuple[str, str]]:
    """Numbered steps for one client, as (text, copyable_block) pairs.

    An empty block means the step is prose only.
    """
    steps: list[tuple[str, str]] = []

    if client_key == "claude-ai":
        steps.append((
            "Start the HTTP server. Leave it running -- the client connects to it.",
            "voltage serve --http",
        ))
        steps.append((
            "In the client's connector settings, add a custom connector with this URL:",
            info.http_url,
        ))
        steps.append((
            "It binds to loopback only. That is deliberate: this server moves your mouse "
            "and reads your screen, and MCP has no authentication of its own. A hosted "
            "client can only reach it if it runs on this machine.",
            "",
        ))
        return steps

    if client_key == "claude-code":
        steps.append(("Register the server -- one command, copy it whole:",
                      claude_code_command(info)))
        steps.append(("Restart Claude Code. Servers load at session start.", ""))
        steps.append(("Confirm it worked:", "claude mcp get voltage-input"))
        steps.append(("Then ask Claude (do not type this in a shell): voltage_doctor", ""))
        return steps

    if client_key == "claude-desktop":
        steps.append(("Write the config for me -- backs up anything already there:",
                      "voltage connect --write-desktop"))
        steps.append(("Fully quit Claude Desktop and reopen it. Reloading is not enough.",
                      ""))
        steps.append(("Prefer to do it by hand? Open this file:",
                      str(desktop_config_path())))
        steps.append(("...and merge this into it:", stdio_json(info)))
        return steps

    hint = next((c.config_hint for c in CLIENTS if c.key == client_key), "its MCP config")
    steps.append(("Print the entry (and copy it):", "voltage connect --json"))
    steps.append((f"Paste it into {hint}:", stdio_json(info)))
    steps.append(("Restart the client.", ""))
    return steps
