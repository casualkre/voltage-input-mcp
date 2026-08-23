"""Ollama backend -- the compatibility path.

Ollama is already installed on most machines and needs no build step, so it is the
zero-setup option. It is genuinely slower for this workload, and it is worth being
precise about why rather than hand-waving:

  * No GBNF. Ollama constrains output to a JSON schema. The burst DSL is not JSON, so on
    this backend the actuator is asked for `{"burst": "...", "next": "...", ...}` and the
    burst *string inside it* is unconstrained -- the model can emit a malformed burst and
    we have to reject it and lose the cycle. On llama.cpp that is structurally impossible.
  * JSON wrapping roughly triples output tokens, and decode is the actuator's critical
    path.
  * Less direct control over prompt-cache reuse and slot assignment.

Expect roughly 1.5-2.5x the cycle time of the llama.cpp path, plus an occasional dropped
cycle from an unparseable burst. Fine for getting started, worth migrating off.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from ..errors import BackendError
from .base import Backend, GenerationResult

__all__ = ["OllamaBackend", "BURST_SCHEMA", "OBSERVATION_SCHEMA"]

# Used in place of a grammar. Shape is enforced; the burst string inside is not.
BURST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "burst": {"type": "string", "maxLength": 400},
        "next": {"type": "string", "maxLength": 40},
        "note": {"type": "string", "maxLength": 60},
    },
    "required": ["burst", "next"],
}

OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "s": {"type": "string", "maxLength": 80},
        "e": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "l": {"type": "string"},
                    "b": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "c": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["l", "b"],
            },
        },
        "t": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 60}},
        "f": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
    },
    "required": ["s", "e"],
}


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        vision: bool = False,
        num_ctx: int = 4096,
    ) -> None:
        self.model = model
        self.model_label = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._vision = vision
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(60.0, connect=3.0)
        )

    @property
    def supports_grammar(self) -> bool:
        return False

    @property
    def supports_vision(self) -> bool:
        return self._vision

    async def close(self) -> None:
        await self._client.aclose()

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
    ) -> GenerationResult:
        started = time.perf_counter()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "top_k": 40,
                "top_p": 0.9,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop
        if image_png is not None:
            payload["images"] = [base64.b64encode(image_png).decode("ascii")]
        # `grammar` is accepted and ignored -- Ollama has no GBNF support. The caller
        # always supplies a schema alongside for exactly this reason.
        if schema is not None:
            payload["format"] = schema

        try:
            response = await self._client.post("/api/generate", json=payload, timeout=timeout_s)
        except httpx.TimeoutException:
            return self._failure(f"ollama timed out after {timeout_s:.1f}s", started)
        except httpx.HTTPError as exc:
            return self._failure(
                f"cannot reach ollama at {self.base_url}: {exc}. Is the daemon running "
                f"(systemctl --user status ollama)?",
                started,
            )

        if response.status_code >= 400:
            body = response.text[:300]
            hint = ""
            if response.status_code == 404:
                hint = f"  Try: ollama pull {self.model}"
            return self._failure(f"ollama returned {response.status_code}: {body}{hint}", started)

        try:
            data = response.json()
        except ValueError as exc:
            return self._failure(f"ollama returned non-JSON: {exc}", started)

        elapsed = (time.perf_counter() - started) * 1000.0
        return GenerationResult(
            text=(data.get("response") or "").strip(),
            model=self.model,
            backend=self.name,
            latency_ms=elapsed,
            prefill_ms=_ns_to_ms(data.get("prompt_eval_duration")),
            decode_ms=_ns_to_ms(data.get("eval_duration")),
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
            stop_reason=str(data.get("done_reason") or ""),
        )

    def _failure(self, message: str, started: float) -> GenerationResult:
        return GenerationResult(
            model=self.model,
            backend=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=message,
        )

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/api/tags", timeout=3.0)
        except httpx.HTTPError as exc:
            return {
                "backend": self.name, "url": self.base_url, "ok": False,
                "error": f"unreachable: {exc}",
                "fix": "systemctl --user start ollama",
            }
        if response.status_code != 200:
            return {"backend": self.name, "ok": False, "status": response.status_code}
        try:
            names = [m.get("name", "") for m in (response.json().get("models") or [])]
        except ValueError:
            names = []
        present = any(n == self.model or n.startswith(f"{self.model}:") for n in names)
        info: dict[str, Any] = {
            "backend": self.name, "url": self.base_url, "ok": present,
            "model": self.model, "installed_models": names, "vision": self._vision,
        }
        if not present:
            info["error"] = f"model {self.model!r} is not pulled"
            info["fix"] = f"ollama pull {self.model}"
        return info

    async def ensure(self) -> None:
        info = await self.health()
        if not info.get("ok"):
            raise BackendError(
                f"ollama backend for {self.model!r} is not ready: {info.get('error')}",
                fix=info.get("fix"),
            )


def _ns_to_ms(value: Any) -> float | None:
    try:
        return float(value) / 1e6
    except (TypeError, ValueError):
        return None
