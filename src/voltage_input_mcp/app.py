"""Shared application state: devices, capture, backends, live sessions.

Everything expensive is created lazily and then kept. Creating the uinput devices costs
~0.4 s of settle time, opening a PipeWire stream costs a portal round trip, and loading
models costs seconds -- none of which should be paid per tool call.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .capture import CaptureBackend, create_backend, detect_backends
from .config import Config, load_config, state_dir
from .errors import SessionError, VoltageError
from .inputs import Executor, InputSink, PointerMode, TextMode, create_sink, probe_input
from .llm import Backend, build_backends, detect_vram_mb, get_profile, recommend
from .runtime import Session, SessionDeps, SessionOptions

__all__ = ["App", "get_app"]


class App:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self._capture: CaptureBackend | None = None
        self._devices: InputSink | None = None
        self._executor: Executor | None = None
        self._vision: Backend | None = None
        self._actuator: Backend | None = None
        self._screen: tuple[int, int] | None = self.config.screen
        self._screen_warning: str | None = None
        self.sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    # -- lazily built components -----------------------------------------------------

    @property
    def profile(self):
        return get_profile(self.config.profile)

    def capture(self) -> CaptureBackend:
        if self._capture is None:
            self._capture = create_backend(
                self.config.capture_backend,
                state_dir=state_dir(),
                cursor=self.config.capture_cursor,
            )
        return self._capture

    def screen(self) -> tuple[int, int]:
        """Desktop geometry.

        Tries, in order: explicit config, a capture, the kernel's DRM mode list, then a
        1920x1080 default. The DRM fallback matters more than it looks: MCP clients
        commonly launch servers with a sanitised environment that omits
        `DBUS_SESSION_BUS_ADDRESS` and `WAYLAND_DISPLAY`, which makes capture impossible.
        Input injection does not need capture -- it only needs the screen size for range
        checks -- so a capture failure must not take the input path down with it.
        """
        if self._screen is not None:
            return self._screen
        try:
            self._screen = self.capture().geometry()
            return self._screen
        except Exception as exc:  # noqa: BLE001
            self._screen_warning = f"capture unavailable ({exc}); geometry not measured"

        drm = _screen_from_drm()
        if drm is not None:
            self._screen = drm
            self._screen_warning = (
                f"{self._screen_warning}. Using the kernel's DRM mode list: "
                f"{drm[0]}x{drm[1]}"
            )
            return drm

        self._screen = (1920, 1080)
        self._screen_warning = (
            f"{self._screen_warning}, and no DRM mode was readable. Falling back to "
            f"1920x1080 -- set `screen = [w, h]` in voltage.toml if that is wrong."
        )
        return self._screen

    def devices(self) -> InputSink:
        if self._devices is None:
            self._devices = create_sink(self.screen())
        return self._devices

    def executor(self) -> Executor:
        if self._executor is None:
            self._executor = Executor(
                self.devices(),
                dry_run=self.config.dry_run,
                text_mode=TextMode(self.config.text_mode),
                pointer_mode=PointerMode(self.config.pointer_mode),
            )
        return self._executor

    def backends(self) -> tuple[Backend, Backend]:
        if self._vision is None or self._actuator is None:
            self._vision, self._actuator = build_backends(
                self.profile,
                engine=self.config.engine,
                vision_url=self.config.vision_url,
                actuator_url=self.config.actuator_url,
                ollama_url=self.config.ollama_url,
            )
        return self._vision, self._actuator

    def session_deps(self) -> SessionDeps:
        vision, actuator = self.backends()
        return SessionDeps(
            capture=self.capture(),
            vision=vision,
            actuator=actuator,
            devices=self.devices(),
            executor=self.executor(),
            screen=self.screen(),
        )

    def session_options(self, **overrides: Any) -> SessionOptions:
        base = SessionOptions(
            target_period_s=self.config.target_period_s,
            dry_run=None,
            keep_frames=self.config.keep_frames,
            settle_ms=self.config.settle_ms,
            watch_physical_input=self.config.watch_physical_input,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(base, key):
                setattr(base, key, value)
        return base

    # -- sessions --------------------------------------------------------------------

    def active(self) -> list[Session]:
        return [s for s in self.sessions.values() if s.status in ("running", "paused")]

    def get_session(self, run_id: str | None) -> Session:
        if run_id:
            try:
                return self.sessions[run_id]
            except KeyError:
                raise SessionError(
                    f"no run {run_id!r}; known runs: {sorted(self.sessions)}"
                ) from None
        live = self.active()
        if len(live) == 1:
            return live[0]
        if not live:
            if self.sessions:
                # Fall back to the most recent finished run so `status` still works.
                return max(self.sessions.values(), key=lambda s: s.id)
            raise SessionError("no runs have been started")
        raise SessionError(
            f"{len(live)} runs are active; pass run_id "
            f"(one of {[s.id for s in live]})"
        )

    async def register(self, session: Session) -> None:
        async with self._lock:
            live = self.active()
            if len(live) >= self.config.max_concurrent_runs:
                raise SessionError(
                    f"{len(live)} run(s) already active and max_concurrent_runs is "
                    f"{self.config.max_concurrent_runs}. Stop {live[0].id} first."
                )
            self.sessions[session.id] = session

    async def stop_all(self, reason: str = "server shutdown") -> list[str]:
        stopped: list[str] = []
        for session in self.active():
            try:
                await session.stop(reason)
                stopped.append(session.id)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
        return stopped

    # -- diagnostics -----------------------------------------------------------------

    async def doctor(self) -> dict[str, Any]:
        """Everything an orchestrator needs to know before writing a playbook."""
        import os
        import shutil

        missing_env = [k for k in _ENV_KEYS_FOR_DISPLAY if not os.environ.get(k)]
        report: dict[str, Any] = {
            "config": self.config.as_dict(),
            "session": {
                "type": os.environ.get("XDG_SESSION_TYPE", "unknown"),
                "desktop": os.environ.get("XDG_CURRENT_DESKTOP", "unknown"),
                "wayland": bool(os.environ.get("WAYLAND_DISPLAY")),
                "missing_env": missing_env,
            },
        }
        if "DBUS_SESSION_BUS_ADDRESS" in missing_env or (
            "WAYLAND_DISPLAY" in missing_env and "DISPLAY" in missing_env
        ):
            # This is the most likely real-world failure: MCP clients launch servers with
            # a sanitised environment, and without these variables there is no way to
            # reach the compositor. Input injection still works (uinput is a device file,
            # not a session service) but capture and clipboard do not.
            report["session"]["warning"] = (
                f"missing {missing_env} in this process's environment. Screen capture "
                f"and clipboard typing need them; uinput injection does not. If this "
                f"server was launched by an MCP client, pass them through explicitly -- "
                f"see 'Launching from an MCP client' in README.md."
            )

        report["input"] = probe_input()
        report["input"]["clipboard_tool"] = (
            "wl-copy" if shutil.which("wl-copy")
            else "xclip" if shutil.which("xclip")
            else None
        )
        if not report["input"]["clipboard_tool"]:
            report["input"]["clipboard_note"] = (
                "no clipboard tool found; typing non-ASCII text will fall back to "
                "scancodes and depend on the active keyboard layout. "
                "Install wl-clipboard."
            )

        backends = detect_backends()
        report["capture"] = {"available": backends}
        try:
            backend = self.capture()
            report["capture"]["selected"] = backend.name
            health = await asyncio.to_thread(backend.health)
            report["capture"]["health"] = health
            # `size` is absent when a streaming backend reported readiness without
            # actually opening a session -- see PortalBackend.health.
            if health.get("ok") and health.get("size"):
                self._screen = self._screen or tuple(health["size"])
        except VoltageError as exc:
            report["capture"]["error"] = exc.to_dict()

        report["screen"] = list(self._screen) if self._screen else None
        if report["screen"] is None:
            report["capture"]["screen_note"] = (
                "geometry is detected from the first capture, which has not happened yet"
            )

        from .llm.profiles import DESKTOP_RESERVE_MB, usable_vram_mb

        vram = detect_vram_mb()
        budget = usable_vram_mb(vram) if vram else None
        profile = self.profile
        report["models"] = {
            "vram_mb": vram,
            "usable_vram_mb": budget,
            "desktop_reserve_mb": DESKTOP_RESERVE_MB,
            "profile": profile.as_dict(),
            "profile_fits": profile.fits(budget) if budget else None,
            "recommended": recommend(vram).name if vram else None,
        }
        if budget and not profile.fits(budget):
            report["models"]["warning"] = (
                f"profile {profile.name!r} needs about {profile.vram_mb} MB, but of this "
                f"GPU's {vram} MB only ~{budget} MB is free once the desktop has its "
                f"share. Expect layers to spill to system RAM and cycle times to roughly "
                f"triple. Try profile {recommend(vram).name!r}."
            )

        try:
            vision, actuator = self.backends()
            report["models"]["vision"] = await vision.health()
            report["models"]["actuator"] = await actuator.health()
        except Exception as exc:  # noqa: BLE001
            report["models"]["error"] = str(exc)

        report["runs"] = {
            "active": [s.id for s in self.active()],
            "known": list(self.sessions),
        }

        # Surface a configured-vs-running profile disagreement here too: doctor is
        # where people look, and the mismatch is otherwise silent.
        try:
            from .briefing import active_build

            build = active_build()
            report["active_build"] = build
            if build.get("mismatch"):
                report["models"]["mismatch"] = build["mismatch"]
        except Exception:  # noqa: BLE001
            pass

        report["ready"] = bool(
            report["input"].get("ok")
            and report["capture"].get("health", {}).get("ok")
            and report["models"].get("vision", {}).get("ok")
            and report["models"].get("actuator", {}).get("ok")
        )
        if not report["ready"]:
            report["next_steps"] = _next_steps(report)
        return report

    async def close(self) -> None:
        await self.stop_all()
        if self._devices is not None:
            self._devices.close()
        for backend in (self._vision, self._actuator):
            if backend is not None:
                await backend.close()
        if self._capture is not None:
            self._capture.stop()


def _screen_from_drm() -> tuple[int, int] | None:
    """Read the current mode from the kernel, with no display server involved.

    `/sys/class/drm/*/modes` lists modes for connected connectors, preferred first, and
    is readable even when there is no session bus to talk to.
    """
    import glob

    best: tuple[int, int] | None = None
    for path in sorted(glob.glob("/sys/class/drm/*/modes")):
        try:
            with open(path, encoding="ascii") as handle:
                first = handle.readline().strip()
        except OSError:
            continue
        if "x" not in first:
            continue
        width, _, height = first.partition("x")
        # Trailing refresh markers like "1920x1080i" or "1920x1080@60" appear on some
        # connectors; keep only the leading digits.
        digits_w = "".join(c for c in width if c.isdigit())
        digits_h = "".join(c for c in height if c.isdigit())
        if not digits_w or not digits_h:
            continue
        candidate = (int(digits_w), int(digits_h))
        if best is None or candidate[0] * candidate[1] > best[0] * best[1]:
            best = candidate
    return best


_ENV_KEYS_FOR_DISPLAY = (
    "WAYLAND_DISPLAY", "DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR",
)


def _next_steps(report: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if not report["input"].get("ok"):
        steps.append(report["input"].get("fix") or "fix /dev/uinput permissions")
    capture = report.get("capture", {})
    if not capture.get("health", {}).get("ok"):
        steps.append(
            "no working capture backend: "
            + str(capture.get("error", {}).get("detail") or capture.get("available"))
        )
    models = report.get("models", {})
    for role in ("vision", "actuator"):
        info = models.get(role) or {}
        if not info.get("ok"):
            steps.append(
                f"{role} model not ready: {info.get('error', 'unreachable')}"
                + (f" -- {info['fix']}" if info.get("fix") else "")
            )
    return steps


_APP: App | None = None


def get_app(config: Config | None = None) -> App:
    global _APP
    if _APP is None:
        _APP = App(config)
    return _APP
