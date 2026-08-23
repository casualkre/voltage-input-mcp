"""User-defined model profiles.

The built-in profiles cover the machines this was developed against. They will not cover
yours: a different GPU, a quantisation you prefer, a model that did not exist when this
was written, or a fine-tune of your own. Custom profiles are stored as TOML next to the
config and merged over the built-ins by name, so a custom profile can also *shadow* a
built-in one to retune it without editing the package.

The vision model is the constrained slot. It must emit grounded bounding boxes on demand
-- Qwen2.5-VL, Qwen3-VL, InternVL, MiniCPM-V and UI-TARS all do; a general captioner will
describe your screen beautifully and put the boxes in the wrong place. The actuator slot
is far more forgiving: under a GBNF grammar it is choosing among a small set of legal
continuations, so almost any competent instruct model of 1B or more will do.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from .profiles import EXPERIMENTAL, ModelSpec, Profile
from .profiles import PROFILES as BUILTIN

__all__ = [
    "profiles_path", "load_custom", "save_custom", "delete_custom",
    "all_profiles", "profile_to_toml",
]


def profiles_path() -> Path:
    from ..config import default_config_path

    return default_config_path().parent / "profiles.toml"


def _spec_from_dict(role: str, data: dict[str, Any], default_port: int) -> ModelSpec:
    required = "ollama_tag" if data.get("ollama_tag") and not data.get("hf_repo") else "hf_repo"
    if required not in data and "ollama_tag" not in data:
        raise ConfigError(
            f"{role}: needs either hf_repo + hf_file (for llama.cpp) or ollama_tag"
        )
    if data.get("hf_repo") and not data.get("hf_file"):
        raise ConfigError(f"{role}: hf_repo given without hf_file")

    return ModelSpec(
        role="vision" if role == "vision" else "actuator",
        label=str(data.get("label") or data.get("hf_file") or data.get("ollama_tag") or role),
        hf_repo=str(data.get("hf_repo", "")),
        hf_file=str(data.get("hf_file", "")),
        mmproj_repo=data.get("mmproj_repo") or (data.get("hf_repo") or None),
        mmproj_file=data.get("mmproj_file"),
        ollama_tag=data.get("ollama_tag"),
        params_b=float(data.get("params_b", 3.0)),
        quant=str(data.get("quant", "Q4_K_M")),
        weights_mb=int(data.get("weights_mb", 2000)),
        n_ctx=int(data.get("n_ctx", 2048)),
        port=int(data.get("port", default_port)),
        gpu_layers=int(data.get("gpu_layers", 99)),
        batch_size=int(data.get("batch_size", 1024 if role == "vision" else 512)),
        ubatch_size=int(data.get("ubatch_size", 768 if role == "vision" else 256)),
        threads=int(data.get("threads", 4 if role == "vision" else 8)),
        extra_args=tuple(data.get("extra_args", ())),
    )


def load_custom(path: Path | None = None) -> dict[str, Profile]:
    """Read user profiles. A malformed file is reported, never silently ignored."""
    target = path or profiles_path()
    if not target.exists():
        return {}
    try:
        with target.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read custom profiles at {target}: {exc}") from exc

    out: dict[str, Profile] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            continue
        try:
            vision = _spec_from_dict("vision", body.get("vision") or {}, 8080)
            actuator = _spec_from_dict("actuator", body.get("actuator") or {}, 8081)
        except ConfigError as exc:
            raise ConfigError(f"custom profile {name!r}: {exc.detail}") from exc
        out[name] = Profile(
            name=name,
            description=str(body.get("description", "custom profile")),
            vision=vision,
            actuator=actuator,
            min_vram_mb=int(body.get("min_vram_mb", 0)),
            notes=str(body.get("notes", "")),
            actuator_on_cpu=bool(body.get("actuator_on_cpu", False)),
        )
    return out


def all_profiles() -> dict[str, Profile]:
    """Every profile: stable, experimental, and custom, in that precedence order.

    Custom wins on a name collision, deliberately: shadowing `lean` is the natural way to
    retune it for your own hardware without forking the package.

    Experimental profiles are included here so `get_profile` and the serve scripts can
    use them by name, but `recommend()` skips them -- each trades something away that
    should be an explicit choice rather than a default.
    """
    merged = {**BUILTIN, **EXPERIMENTAL}
    try:
        merged.update(load_custom())
    except ConfigError:
        # A broken profiles.toml must not make the tool unusable; the profiles screen
        # surfaces the error when the user goes looking.
        pass
    return merged


def profile_to_toml(profile: Profile) -> str:
    """Serialise a profile, so a built-in can be copied and edited as a starting point."""
    lines = [f"[{profile.name}]", f'description = "{profile.description}"']
    if profile.notes:
        lines.append(f'notes = "{profile.notes}"')
    if profile.actuator_on_cpu:
        lines.append("actuator_on_cpu = true")
    for role, spec in (("vision", profile.vision), ("actuator", profile.actuator)):
        lines.append("")
        lines.append(f"[{profile.name}.{role}]")
        lines.append(f'label = "{spec.label}"')
        if spec.hf_repo:
            lines.append(f'hf_repo = "{spec.hf_repo}"')
            lines.append(f'hf_file = "{spec.hf_file}"')
        if spec.mmproj_file:
            lines.append(f'mmproj_file = "{spec.mmproj_file}"')
        if spec.ollama_tag:
            lines.append(f'ollama_tag = "{spec.ollama_tag}"')
        lines.append(f"params_b = {spec.params_b}")
        lines.append(f"weights_mb = {spec.weights_mb}")
        lines.append(f"n_ctx = {spec.n_ctx}")
        lines.append(f"port = {spec.port}")
    return "\n".join(lines) + "\n"


def save_custom(profile: Profile, path: Path | None = None) -> Path:
    """Append or replace a profile in the user's profiles.toml."""
    target = path or profiles_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if target.exists():
        try:
            with target.open("rb") as handle:
                existing = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            existing = {}
    existing.pop(profile.name, None)

    # Rebuild the file: keep every other profile verbatim by re-serialising from the
    # loaded objects, then append the new one.
    chunks = ["# Custom model profiles for VoltageInputMcp.",
              "# Merged over the built-ins by name; a matching name shadows the built-in.",
              ""]
    for name in existing:
        try:
            for other in load_custom(target).values():
                if other.name == name:
                    chunks.append(profile_to_toml(other))
        except ConfigError:
            pass
    chunks.append(profile_to_toml(profile))
    target.write_text("\n".join(chunks))
    return target


def delete_custom(name: str, path: Path | None = None) -> bool:
    target = path or profiles_path()
    if not target.exists():
        return False
    remaining = {k: v for k, v in load_custom(target).items() if k != name}
    chunks = ["# Custom model profiles for VoltageInputMcp.", ""]
    chunks.extend(profile_to_toml(p) for p in remaining.values())
    target.write_text("\n".join(chunks))
    return True
