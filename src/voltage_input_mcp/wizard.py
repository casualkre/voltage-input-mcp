"""Setup wizard: detect what exists, then continue from there.

The failure mode this avoids is a setup script that assumes a starting point. People
arrive here with Ollama already installed, or a llama.cpp build from another project, or
a GPU with 24 GB, or none at all, or on Windows. A linear "run these seven commands"
script is wrong for all of them and actively misleading for most.

So this detects first and *then* chooses a route. Nothing is run without saying what it
will do, nothing already done is done again, and every step that needs a decision explains
the trade-off in one line rather than asking the user to already know it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Environment", "detect", "Step", "plan_steps"]

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class Environment:
    """Everything the wizard needs to know, gathered once."""

    platform: str = ""
    platform_label: str = ""
    python: str = ""
    session: str = ""

    gpu_name: str | None = None
    vram_mb: int | None = None
    gpu_kind: str = "none"          # nvidia | amd | apple | none

    llama_server: str | None = None
    llama_has_fa_all_quants: bool | None = None
    ollama: bool = False
    ollama_running: bool = False
    ollama_models: list[str] = field(default_factory=list)

    model_files: list[str] = field(default_factory=list)
    input_ok: bool = False
    input_reason: str = ""
    input_fix: str = ""
    capture_ok: bool = False
    capture_backend: str = ""
    capture_note: str = ""

    claude_cli: bool = False
    mcp_registered: bool | None = None
    on_path: bool = False
    venv: bool = False

    servers_up: bool = False

    @property
    def has_any_engine(self) -> bool:
        return bool(self.llama_server) or self.ollama

    @property
    def ready(self) -> bool:
        return self.input_ok and self.capture_ok and self.servers_up and bool(self.mcp_registered)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _port_open(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _run(cmd: list[str], timeout: float = 6.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return proc.returncode, (proc.stdout + proc.stderr).decode("utf-8", "replace")


def _detect_gpu() -> tuple[str, str | None, int | None]:
    """Identify the accelerator without assuming NVIDIA."""
    if _which("nvidia-smi"):
        code, out = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        )
        if code == 0 and out.strip():
            first = out.strip().splitlines()[0]
            name, _, total = first.partition(",")
            try:
                return "nvidia", name.strip(), int(total.strip())
            except ValueError:
                return "nvidia", name.strip(), None

    if sys.platform == "darwin":
        code, out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if "Apple" in out:
            return "apple", out.strip(), None

    if sys.platform.startswith("linux") and Path("/sys/class/drm").exists():
        code, out = _run(["lspci"], timeout=4.0)
        if "Advanced Micro Devices" in out and ("VGA" in out or "Display" in out):
            return "amd", "AMD GPU", None

    return "none", None, None


def _find_llama_server() -> str | None:
    local = REPO_ROOT / "vendor" / "llama.cpp" / "build" / "bin" / (
        "llama-server.exe" if sys.platform == "win32" else "llama-server"
    )
    if local.exists():
        return str(local)
    return _which("llama-server")


def _llama_supports_fa_all_quants() -> bool | None:
    """Whether the build has the quantised-KV flash-attention kernels.

    Serving uses `--cache-type-k q8_0 -fa on`; a build without GGML_CUDA_FA_ALL_QUANTS
    silently falls back to a slow path. The CMake cache is only present for a local
    build, so a llama-server from PATH returns None (unknown) rather than a guess.
    """
    cache = REPO_ROOT / "vendor" / "llama.cpp" / "build" / "CMakeCache.txt"
    if not cache.exists():
        return None
    try:
        return "GGML_CUDA_FA_ALL_QUANTS:BOOL=ON" in cache.read_text(errors="replace")
    except OSError:
        return None


def detect() -> Environment:
    """Gather the whole picture. Cheap enough to re-run after every step."""
    from .config import load_config
    from .inputs import probe_input

    env = Environment()
    env.platform = sys.platform
    env.platform_label = {
        "linux": "Linux", "win32": "Windows", "darwin": "macOS",
    }.get(sys.platform, sys.platform)
    env.python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if sys.platform == "win32":
        env.session = "Windows desktop"
    else:
        env.session = (
            f"{os.environ.get('XDG_SESSION_TYPE', 'unknown')} / "
            f"{os.environ.get('XDG_CURRENT_DESKTOP', 'unknown')}"
        )

    env.gpu_kind, env.gpu_name, env.vram_mb = _detect_gpu()

    env.llama_server = _find_llama_server()
    env.llama_has_fa_all_quants = _llama_supports_fa_all_quants()
    env.ollama = _which("ollama") is not None
    env.ollama_running = _port_open(11434)
    if env.ollama_running:
        code, out = _run(["ollama", "list"], timeout=8.0)
        if code == 0:
            env.ollama_models = [
                line.split()[0] for line in out.splitlines()[1:] if line.strip()
            ]

    models_dir = REPO_ROOT / "models"
    if models_dir.is_dir():
        env.model_files = sorted(p.name for p in models_dir.glob("*.gguf"))

    probe = probe_input()
    env.input_ok = bool(probe.get("ok"))
    env.input_reason = str(probe.get("reason", ""))
    env.input_fix = str(probe.get("fix", ""))

    try:
        from .capture import detect_backends

        detected = detect_backends()
        env.capture_ok = any(detected.values())
        env.capture_backend = next((k for k, v in detected.items() if v), "")
        if env.capture_backend == "portal":
            env.capture_note = "first capture shows a one-time screen-share picker"
        elif not env.capture_ok:
            env.capture_note = f"none available (probed {detected})"
    except Exception as exc:  # noqa: BLE001
        env.capture_ok = False
        env.capture_note = str(exc)[:120]

    env.claude_cli = _which("claude") is not None
    if env.claude_cli:
        code, out = _run(["claude", "mcp", "list"], timeout=15.0)
        env.mcp_registered = "voltage-input" in out if (code == 0 or out) else None

    env.on_path = _which("voltage") is not None
    env.venv = (REPO_ROOT / ".venv").exists()

    try:
        config = load_config()
        if config.engine == "ollama":
            env.servers_up = env.ollama_running
        else:
            env.servers_up = _port_open(
                int(config.vision_url.rsplit(":", 1)[-1])
            ) and _port_open(int(config.actuator_url.rsplit(":", 1)[-1]))
    except Exception:  # noqa: BLE001
        env.servers_up = _port_open(8080) and _port_open(8081)

    return env


# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Step:
    """One thing the wizard can do, with enough context to decide whether to."""

    key: str
    title: str
    why: str
    detail: str = ""
    automatic: bool = True          # can the wizard just do it?
    needs_sudo: bool = False
    done: bool = False


def plan_steps(env: Environment, engine: str, profile: str) -> list[Step]:
    """Work out what is left to do, given what already exists."""
    steps: list[Step] = []

    if not env.on_path:
        steps.append(Step(
            "path", "Put `voltage` on your PATH",
            "so you can type `voltage` from anywhere instead of a long venv path",
        ))

    if not env.input_ok:
        steps.append(Step(
            "input", "Enable input injection",
            "nothing can be typed or clicked until this works",
            detail=f"{env.input_reason}\n{env.input_fix}" if env.input_reason else "",
            automatic=False,
            needs_sudo=sys.platform != "win32",
        ))

    if not env.capture_ok:
        steps.append(Step(
            "capture", "Enable screen capture",
            "the vision model has nothing to look at without it",
            detail=env.capture_note,
            automatic=False,
        ))

    if engine == "ollama":
        if not env.ollama:
            steps.append(Step(
                "ollama-install", "Install Ollama",
                "the zero-build way to run both models",
                detail="https://ollama.com/download",
                automatic=False,
            ))
        elif not env.ollama_running:
            steps.append(Step(
                "ollama-start", "Start the Ollama service",
                "it is installed but not listening on 11434",
                detail=(
                    "ollama serve" if sys.platform == "win32"
                    else "systemctl --user start ollama   (or: sudo systemctl start ollama)"
                ),
                automatic=False,
            ))
        else:
            from .llm import get_profile

            try:
                spec = get_profile(profile)
                wanted = [spec.vision.ollama_tag, spec.actuator.ollama_tag]
            except KeyError:
                wanted = ["qwen2.5vl:3b", "qwen3:1.7b"]
            missing = [
                tag for tag in wanted
                if tag and not any(m.startswith(tag.split(":")[0]) for m in env.ollama_models)
            ]
            if missing:
                steps.append(Step(
                    "ollama-pull", f"Pull {len(missing)} model(s)",
                    "roughly 3.5 GB; one-time",
                    detail="  ".join(f"ollama pull {t}" for t in missing),
                ))
    else:
        if not env.llama_server:
            steps.append(Step(
                "llama", "Get llama.cpp",
                "faster than Ollama here, and the only backend that supports GBNF "
                "grammars -- which is what keeps the small models reliable",
                detail=(
                    "Download a CUDA build and put llama-server.exe on your PATH:\n"
                    "  https://github.com/ggml-org/llama.cpp/releases\n"
                    "  (pick llama-<version>-bin-win-cuda-x64.zip)"
                    if sys.platform == "win32" else "./scripts/build-llama.sh"
                ),
                automatic=sys.platform != "win32",
            ))
        if not env.model_files:
            steps.append(Step(
                "weights", f"Download the {profile} weights",
                "roughly 3 GB; one-time",
                detail=f"voltage fetch {profile}",
                automatic=True,   # pure-Python downloader; works on every platform
            ))
        if not env.servers_up:
            steps.append(Step(
                "serve", "Start the model servers",
                "two llama-server instances, one per role",
                detail=(
                    "voltage serve-models   # prints both commands, run each in a terminal"
                    if sys.platform == "win32" else "./scripts/serve.sh " + profile
                ),
                automatic=True,
            ))

    if not env.claude_cli:
        steps.append(Step(
            "claude", "Install Claude Code (optional)",
            "only needed to register automatically; you can add the server by URL instead",
            detail="https://claude.com/claude-code",
            automatic=False,
        ))
    elif not env.mcp_registered:
        steps.append(Step(
            "mcp", "Connect to Claude Code",
            "registers this server so Claude can drive it",
        ))

    return steps
