"""llama.cpp server backend -- the fast path.

Preferred over Ollama for three concrete reasons, all of which matter at this loop rate:

  1. **GBNF grammars.** Ollama constrains to a JSON schema; llama.cpp constrains to an
     arbitrary grammar. The burst DSL is not JSON, and expressing it as JSON would cost
     roughly 5x the output tokens -- which is the actuator's critical path.
  2. **Prompt caching with `cache_prompt`.** The static parts of both prompts (system
     text, grammar preamble, state brief) are identical across cycles. llama.cpp reuses
     the KV cache for the common prefix, so per-cycle prefill drops to just the changed
     tail. On a 6 GB laptop GPU this is the difference between a 0.9 s and a 0.25 s
     cycle.
  3. **Explicit slot control**, so vision and actuator never evict each other.

Text generation uses the native `/completion` endpoint. Vision goes through
`/v1/chat/completions`, because that is where llama.cpp's multimodal (mtmd) path lives;
`grammar` is accepted there as a non-standard extension, with `response_format` as a
fallback for builds that do not honour it.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from ..errors import BackendError
from .base import Backend, GenerationResult

__all__ = ["LlamaCppBackend"]


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        model_label: str = "",
        vision: bool = False,
        api_key: str | None = None,
        connect_timeout_s: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_label = model_label or base_url
        self._vision = vision
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=connect_timeout_s),
        )
        self._props: dict[str, Any] | None = None

    @property
    def supports_grammar(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return self._vision

    async def close(self) -> None:
        await self._client.aclose()

    # -- generation ------------------------------------------------------------------

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
        try:
            if image_png is not None:
                payload, endpoint = self._vision_payload(
                    prompt, system, image_png, grammar, schema, max_tokens, temperature, stop
                )
            else:
                payload, endpoint = self._text_payload(
                    prompt, system, grammar, schema, max_tokens, temperature, stop
                )
            response = await self._client.post(endpoint, json=payload, timeout=timeout_s)
        except httpx.TimeoutException:
            return self._failure(
                f"llama.cpp at {self.base_url} timed out after {timeout_s:.1f}s", started
            )
        except httpx.HTTPError as exc:
            return self._failure(
                f"cannot reach llama.cpp at {self.base_url}: {exc}. Is llama-server "
                f"running? Start it with scripts/serve.sh",
                started,
            )

        if response.status_code >= 400:
            detail = response.text[:400]
            if grammar and "grammar" in detail.lower():
                detail += (
                    "  (this build may not accept `grammar` on /v1/chat/completions; "
                    "the JSON-schema fallback will be used on retry)"
                )
            return self._failure(f"llama.cpp returned {response.status_code}: {detail}", started)

        try:
            data = response.json()
        except ValueError as exc:
            return self._failure(f"llama.cpp returned non-JSON: {exc}", started)

        return self._parse(data, started, endpoint)

    def _text_payload(
        self,
        prompt: str,
        system: str | None,
        grammar: str | None,
        schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> tuple[dict[str, Any], str]:
        full = f"{system}\n\n{prompt}" if system else prompt
        payload: dict[str, Any] = {
            "prompt": full,
            "n_predict": max_tokens,
            "temperature": temperature,
            # The single highest-leverage flag in this file: reuse the KV cache for the
            # unchanged prefix instead of re-prefilling the whole prompt every cycle.
            "cache_prompt": True,
            "stream": False,
            # Low top_k narrows the search without needing temperature 0, which under a
            # grammar tends to produce degenerate minimal outputs (empty bursts).
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        }
        if stop:
            payload["stop"] = stop
        if grammar:
            payload["grammar"] = grammar
        elif schema:
            payload["json_schema"] = schema
        return payload, "/completion"

    def _vision_payload(
        self,
        prompt: str,
        system: str | None,
        image_png: bytes,
        grammar: str | None,
        schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> tuple[dict[str, Any], str]:
        encoded = base64.b64encode(image_png).decode("ascii")
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "cache_prompt": True,
            "top_k": 40,
        }
        if stop:
            payload["stop"] = stop
        if grammar:
            payload["grammar"] = grammar
        elif schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "observation", "schema": schema, "strict": True},
            }
        return payload, "/v1/chat/completions"

    def _parse(self, data: dict[str, Any], started: float, endpoint: str) -> GenerationResult:
        elapsed = (time.perf_counter() - started) * 1000.0

        if endpoint.startswith("/v1/"):
            choices = data.get("choices") or [{}]
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
            stop_reason = choices[0].get("finish_reason") or ""
            usage = data.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens") or 0)
            tokens_out = int(usage.get("completion_tokens") or 0)
        else:
            text = data.get("content") or ""
            stop_reason = (
                "length" if data.get("stopped_limit") else
                "stop" if data.get("stopped_word") or data.get("stopped_eos") else ""
            )
            tokens_in = int(data.get("tokens_evaluated") or 0)
            tokens_out = int(data.get("tokens_predicted") or 0)

        timings = data.get("timings") or {}
        prefill_ms = timings.get("prompt_ms")
        decode_ms = timings.get("predicted_ms")
        if tokens_in == 0 and timings.get("prompt_n"):
            tokens_in = int(timings["prompt_n"])
        if tokens_out == 0 and timings.get("predicted_n"):
            tokens_out = int(timings["predicted_n"])

        return GenerationResult(
            text=text.strip(),
            model=self.model_label,
            backend=self.name,
            latency_ms=elapsed,
            prefill_ms=float(prefill_ms) if prefill_ms is not None else None,
            decode_ms=float(decode_ms) if decode_ms is not None else None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            stop_reason=stop_reason,
            raw=data if len(str(data)) < 4000 else {},
        )

    def _failure(self, message: str, started: float) -> GenerationResult:
        return GenerationResult(
            model=self.model_label,
            backend=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=message,
        )

    # -- introspection ---------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/health", timeout=3.0)
        except httpx.HTTPError as exc:
            return {
                "backend": self.name, "url": self.base_url, "ok": False,
                "error": f"unreachable: {exc}",
                "fix": "start llama-server (see scripts/serve.sh)",
            }
        ok = response.status_code == 200
        info: dict[str, Any] = {
            "backend": self.name, "url": self.base_url, "ok": ok,
            "status": response.status_code, "vision": self._vision,
        }
        props = await self.props()
        if props:
            info["model"] = props.get("model_path") or props.get("model") or self.model_label
            info["n_ctx"] = props.get("n_ctx") or (props.get("default_generation_settings") or {}).get("n_ctx")
        return info

    async def props(self) -> dict[str, Any]:
        if self._props is not None:
            return self._props
        try:
            response = await self._client.get("/props", timeout=3.0)
            if response.status_code == 200:
                self._props = response.json()
                return self._props
        except (httpx.HTTPError, ValueError):
            pass
        return {}

    async def ensure(self) -> None:
        """Raise a helpful BackendError if the server is not usable."""
        info = await self.health()
        if not info.get("ok"):
            raise BackendError(
                f"llama.cpp backend {self.model_label!r} is not available: "
                f"{info.get('error') or info.get('status')}",
                url=self.base_url,
            )
