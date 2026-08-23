"""Local model backends and the grammar generator."""

from __future__ import annotations

from .base import Backend, GenerationResult
from .grammar import actuator_grammar, burst_grammar, observation_grammar
from .llamacpp import LlamaCppBackend
from .ollama import BURST_SCHEMA, OBSERVATION_SCHEMA, OllamaBackend
from .profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    ModelSpec,
    Profile,
    detect_vram_mb,
    get_profile,
    recommend,
)

__all__ = [
    "Backend", "GenerationResult", "LlamaCppBackend", "OllamaBackend",
    "BURST_SCHEMA", "OBSERVATION_SCHEMA",
    "burst_grammar", "actuator_grammar", "observation_grammar",
    "Profile", "ModelSpec", "PROFILES", "DEFAULT_PROFILE", "get_profile", "recommend",
    "detect_vram_mb", "build_backends",
]


def build_backends(
    profile: Profile,
    *,
    engine: str = "llamacpp",
    vision_url: str | None = None,
    actuator_url: str | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
) -> tuple[Backend, Backend]:
    """Construct the (vision, actuator) backend pair for a profile."""
    if engine == "ollama":
        return (
            OllamaBackend(
                profile.vision.ollama_tag or "qwen2.5vl:3b",
                base_url=ollama_url, vision=True, num_ctx=profile.vision.n_ctx,
            ),
            OllamaBackend(
                profile.actuator.ollama_tag or "qwen3:1.7b",
                base_url=ollama_url, num_ctx=profile.actuator.n_ctx,
            ),
        )
    if engine != "llamacpp":
        raise ValueError(f"unknown engine {engine!r}; expected 'llamacpp' or 'ollama'")

    return (
        LlamaCppBackend(
            vision_url or f"http://127.0.0.1:{profile.vision.port}",
            model_label=profile.vision.label,
            vision=True,
        ),
        LlamaCppBackend(
            actuator_url or f"http://127.0.0.1:{profile.actuator.port}",
            model_label=profile.actuator.label,
        ),
    )
