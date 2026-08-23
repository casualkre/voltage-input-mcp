"""Local model backend interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Backend", "GenerationResult"]


@dataclass(slots=True)
class GenerationResult:
    text: str = ""
    model: str = ""
    backend: str = ""
    latency_ms: float = 0.0
    prefill_ms: float | None = None
    decode_ms: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    stop_reason: str = ""
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def decode_tps(self) -> float:
        if not self.decode_ms or not self.tokens_out:
            return 0.0
        return self.tokens_out / (self.decode_ms / 1000.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 1),
            "prefill_ms": round(self.prefill_ms, 1) if self.prefill_ms else None,
            "decode_ms": round(self.decode_ms, 1) if self.decode_ms else None,
            "tokens": {"in": self.tokens_in, "out": self.tokens_out},
            "decode_tps": round(self.decode_tps, 1),
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


class Backend(abc.ABC):
    """A local inference server.

    `grammar` is GBNF and is the preferred constraint mechanism; `schema` is a JSON
    Schema for backends that only support structured JSON output. A backend that
    supports neither must still accept both arguments and ignore them -- the caller
    always parses defensively, constraints just make failure vanishingly rare.
    """

    name: str = "base"

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        grammar: str | None = None,
        schema: dict[str, Any] | None = None,
        image_png: bytes | None = None,
        system: str | None = None,
        max_tokens: int = 96,
        temperature: float = 0.2,
        stop: list[str] | None = None,
        timeout_s: float = 20.0,
    ) -> GenerationResult: ...

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]: ...

    async def warmup(self) -> None:
        """Pre-load weights and prime the KV cache so the first real cycle is not slow."""
        try:
            await self.generate("ok", max_tokens=1, temperature=0.0, timeout_s=120.0)
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass

    async def close(self) -> None: ...

    @property
    def supports_grammar(self) -> bool:
        return False

    @property
    def supports_vision(self) -> bool:
        return False
