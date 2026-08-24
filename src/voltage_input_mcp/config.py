"""Configuration: TOML file, environment overrides, sane defaults.

Precedence is environment > file > default. The file is optional; the defaults are
chosen to be correct on a KDE Wayland desktop with a small GPU, because that is the
configuration this was built and verified against.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError

__all__ = ["Config", "load_config", "default_config_path"]


def default_config_path() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "voltage-input-mcp" / "voltage.toml"


def state_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    path = Path(base) / "voltage-input-mcp"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(slots=True)
class Config:
    # -- models ----------------------------------------------------------------------
    profile: str = "lean"
    engine: str = "llamacpp"                      # "llamacpp" | "ollama"
    vision_url: str = "http://127.0.0.1:8080"
    actuator_url: str = "http://127.0.0.1:8081"
    ollama_url: str = "http://127.0.0.1:11434"

    # -- capture ---------------------------------------------------------------------
    capture_backend: str = "auto"                 # auto | portal | kwin | grim | x11 | windows
    capture_cursor: bool = True

    # -- input -----------------------------------------------------------------------
    text_mode: str = "auto"                       # auto | keys | clipboard
    pointer_mode: str = "absolute"                # absolute | relative
    screen: tuple[int, int] | None = None         # None -> detect from a capture

    # -- run defaults ----------------------------------------------------------------
    dry_run: bool = True
    target_period_s: float = 0.5
    # The fast loop: probes, reflexes and latched holds, with no model in the path. This
    # is a separate number from `target_period_s` on purpose -- it is the one that decides
    # whether a run can react to anything faster than a decision. 0 disables it and folds
    # reflexes back into the decision cycle, which is only sensible for pure desktop work.
    reflex_hz: float = 20.0
    settle_ms: int = 60
    keep_frames: bool = False
    watch_physical_input: bool = True
    max_concurrent_runs: int = 1

    _source: str = field(default="defaults", compare=False)

    # -- validation ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.engine not in ("llamacpp", "ollama"):
            raise ConfigError(f"engine must be 'llamacpp' or 'ollama', got {self.engine!r}")
        if self.capture_backend not in ("auto", "kwin", "portal", "grim", "x11", "windows"):
            raise ConfigError(f"unknown capture_backend {self.capture_backend!r}")
        if self.text_mode not in ("auto", "keys", "clipboard"):
            raise ConfigError(f"unknown text_mode {self.text_mode!r}")
        if self.pointer_mode not in ("absolute", "relative"):
            raise ConfigError(f"unknown pointer_mode {self.pointer_mode!r}")
        if not 0.05 <= self.target_period_s <= 30.0:
            raise ConfigError(
                f"target_period_s must be between 0.05 and 30, got {self.target_period_s}"
            )
        if not 0.0 <= self.reflex_hz <= 120.0:
            raise ConfigError(
                f"reflex_hz must be between 0 (off) and 120, got {self.reflex_hz}"
            )
        if self.screen is not None:
            self.screen = (int(self.screen[0]), int(self.screen[1]))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_source", None)
        data["config_source"] = self._source
        return data


_ENV_PREFIX = "VOLTAGE_"


def _coerce(name: str, raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if name == "screen":
        try:
            w, h = raw.lower().replace(" ", "").split("x")
            return (int(w), int(h))
        except ValueError as exc:
            raise ConfigError(f"VOLTAGE_SCREEN must look like 1920x1080, got {raw!r}") from exc
    return raw


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML plus VOLTAGE_* environment overrides."""
    values: dict[str, Any] = {}
    source = "defaults"

    candidate = path or default_config_path()
    if candidate.exists():
        try:
            with candidate.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read config at {candidate}: {exc}") from exc
        # Accept both a flat table and a [voltage] section.
        values.update(data.get("voltage", data))
        source = str(candidate)

    known = {f.name for f in fields(Config) if not f.name.startswith("_")}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(
            f"unknown config keys in {source}: {sorted(unknown)}. Known keys: {sorted(known)}"
        )

    if isinstance(values.get("screen"), list):
        values["screen"] = tuple(values["screen"])

    defaults = Config()
    for key in known:
        env_name = f"{_ENV_PREFIX}{key.upper()}"
        raw = os.environ.get(env_name)
        if raw is not None:
            values[key] = _coerce(key, raw, getattr(defaults, key))
            source = f"{source} + env"

    config = Config(**values)
    object.__setattr__(config, "_source", source)
    return config
