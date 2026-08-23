"""Latency benchmark against running model servers.

Tuning without measurement is superstition. This drives both backends with prompts of the
exact shape the run loop uses -- same grammar, same image size, same token counts -- and
reports where the time actually goes, plus the cycle time those numbers imply.

The most useful number it produces is the **cached-prefix** actuator latency. The first
call to a fresh server pays full prefill; every subsequent call in a real run reuses the
cached static prefix. Benchmarking only cold calls overstates cycle time by 2-3x, so this
deliberately reports both and projects from the warm one.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np

from .capture.base import encode_png
from .llm import actuator_grammar, observation_grammar
from .llm.base import Backend
from .runtime.prompts import ACTUATOR_SYSTEM, VISION_SYSTEM

__all__ = ["run_benchmark"]

WATCH = ["address bar", "file list", "save button", "dialog"]

ACTUATOR_PROMPT = """\
TASK: Click the Save button, then type the filename and press Enter.
HINT: The dialog already has focus.
CAN GO TO: confirm, retry
SCREEN: 1920x1080

SEEN:
  0 save button at 1204,712 (96x32)
  1 file list at 460,400 (800x420)
  2 dialog at 560,300 (800x460)
TEXT: Save As | Documents
VARS: attempts=1

BURST:"""


def _synthetic_screenshot(width: int, height: int) -> bytes:
    """A structured image, not noise.

    Uniform noise is pathological for a vision tower and a flat colour is trivially
    compressible; neither reflects a real desktop. This draws rectangles and text-like
    bands so the encode size and attention pattern are in a realistic range.
    """
    rng = np.random.default_rng(7)
    img = np.full((height, width, 3), 240, dtype=np.uint8)
    img[: height // 12] = (60, 64, 72)                       # title bar
    img[height // 12 : height // 6] = (250, 250, 252)        # toolbar
    for i in range(8, height - 8, max(12, height // 22)):    # text-like rows
        band = rng.integers(30, 90)
        img[i : i + 3, width // 12 : width // 12 * 7] = band
    img[height // 3 : height // 3 + 40, width // 2 : width // 2 + 180] = (40, 110, 220)
    return encode_png(img, 1)


async def _time_call(backend: Backend, /, **kwargs: Any) -> tuple[float, Any]:
    started = time.perf_counter()
    result = await backend.generate(**kwargs)
    return (time.perf_counter() - started) * 1000.0, result


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)
    return {
        "min_ms": round(ordered[0], 1),
        "median_ms": round(statistics.median(ordered), 1),
        "max_ms": round(ordered[-1], 1),
    }


async def run_benchmark(
    vision: Backend,
    actuator: Backend,
    *,
    rounds: int = 5,
    sizes: tuple[tuple[int, int], ...] = ((896, 504), (700, 392), (448, 252)),
) -> dict[str, Any]:
    report: dict[str, Any] = {"rounds": rounds, "vision": {}, "actuator": {}}

    # -- actuator ---------------------------------------------------------------------
    grammar = actuator_grammar(
        allow_verbs=["g", "c", "k", "t", "w"],
        targets=["confirm", "retry"],
        n_elements=3,
        deny_keys=["delete", "leftmeta"],
    )
    report["actuator"]["grammar_bytes"] = len(grammar)

    cold_ms, cold = await _time_call(
        actuator, prompt=ACTUATOR_PROMPT, system=ACTUATOR_SYSTEM,
        grammar=grammar if actuator.supports_grammar else None,
        max_tokens=96, temperature=0.25, timeout_s=120.0,
    )
    if not cold.ok:
        return {"error": f"actuator backend unavailable: {cold.error}"}
    report["actuator"]["cold_ms"] = round(cold_ms, 1)
    report["actuator"]["sample_output"] = cold.text[:80]

    warm: list[float] = []
    tokens_out: list[int] = []
    for i in range(rounds):
        # Vary only the tail, exactly as a real cycle does, so the static prefix stays
        # cached and this measures the path the loop actually takes.
        prompt = ACTUATOR_PROMPT.replace("attempts=1", f"attempts={i + 2}")
        elapsed, result = await _time_call(
            actuator, prompt=prompt, system=ACTUATOR_SYSTEM,
            grammar=grammar if actuator.supports_grammar else None,
            max_tokens=96, temperature=0.25, timeout_s=60.0,
        )
        if result.ok:
            warm.append(elapsed)
            tokens_out.append(result.tokens_out)
            if result.prefill_ms is not None:
                report["actuator"].setdefault("prefill_ms", []).append(round(result.prefill_ms, 1))
            if result.decode_ms is not None:
                report["actuator"].setdefault("decode_ms", []).append(round(result.decode_ms, 1))
    report["actuator"]["warm"] = _stats(warm)
    report["actuator"]["tokens_out_median"] = (
        int(statistics.median(tokens_out)) if tokens_out else 0
    )
    if warm and cold_ms > 0:
        report["actuator"]["prompt_cache_speedup"] = round(cold_ms / statistics.median(warm), 2)

    # -- vision -----------------------------------------------------------------------
    obs_grammar = observation_grammar(WATCH, max_elements=5, read_text=True)
    report["vision"]["grammar_bytes"] = len(obs_grammar)

    for width, height in sizes:
        tokens = (width // 28) * (height // 28)
        png = _synthetic_screenshot(width, height)
        samples: list[float] = []
        prefills: list[float] = []
        for _ in range(max(2, rounds // 2)):
            elapsed, result = await _time_call(
                vision, prompt="LOOK FOR: " + ", ".join(WATCH), system=VISION_SYSTEM,
                image_png=png, grammar=obs_grammar if vision.supports_grammar else None,
                max_tokens=192, temperature=0.1, timeout_s=180.0,
            )
            if not result.ok:
                report["vision"][f"{width}x{height}"] = {"error": result.error}
                break
            samples.append(elapsed)
            if result.prefill_ms is not None:
                prefills.append(result.prefill_ms)
        else:
            report["vision"][f"{width}x{height}"] = {
                "visual_tokens": tokens,
                "png_kb": round(len(png) / 1024, 1),
                **_stats(samples),
                "prefill_ms_median": round(statistics.median(prefills), 1) if prefills else None,
                "ms_per_visual_token": (
                    round(statistics.median(samples) / tokens, 3) if tokens else None
                ),
            }

    # -- projection -------------------------------------------------------------------
    actuator_ms = report["actuator"]["warm"].get("median_ms", 0.0)
    default_vision = report["vision"].get("896x504", {})
    vision_ms = default_vision.get("median_ms", 0.0)
    if actuator_ms:
        # Capture and probes are ~2 ms and ~0.04 ms; the governor is ~0.05 ms. Rounded to
        # 5 ms of fixed overhead, which is generous.
        overhead = 5.0
        report["projection"] = {
            "cycle_with_vision_ms": round(vision_ms + actuator_ms + overhead, 1),
            "cycle_cached_vision_ms": round(actuator_ms + overhead, 1),
            "note": (
                "perception.mode='on_change' skips the vision call whenever the screen "
                "has not moved, so a real run's average sits between these two numbers "
                "-- closer to the lower one for ordinary desktop work."
            ),
        }
        if vision_ms:
            report["projection"]["max_hz_with_vision"] = round(
                1000.0 / (vision_ms + actuator_ms + overhead), 2
            )
            report["projection"]["max_hz_cached"] = round(1000.0 / (actuator_ms + overhead), 2)

    return report


def format_report(report: dict[str, Any]) -> str:
    if "error" in report:
        return f"benchmark failed: {report['error']}"

    lines: list[str] = ["", "ACTUATOR (decode-bound; this is the per-cycle floor)"]
    act = report["actuator"]
    lines.append(f"  cold (uncached prefix)   {act.get('cold_ms')} ms")
    warm = act.get("warm", {})
    lines.append(
        f"  warm (cached prefix)     {warm.get('median_ms')} ms "
        f"(min {warm.get('min_ms')}, max {warm.get('max_ms')})"
    )
    if act.get("prompt_cache_speedup"):
        lines.append(f"  prompt-cache speedup     {act['prompt_cache_speedup']}x")
    lines.append(
        f"  output tokens (median)   {act.get('tokens_out_median')}  "
        f"grammar {act.get('grammar_bytes')} bytes"
    )
    if act.get("sample_output"):
        lines.append(f"  sample                   {act['sample_output']!r}")

    lines += ["", "VISION (decode-bound; scales with OUTPUT tokens, not image size)"]
    lines.append(f"  {'size':<12}{'tokens':>7}{'median':>10}{'prefill':>10}{'ms/token':>10}")
    for key, data in report["vision"].items():
        if not isinstance(data, dict) or "visual_tokens" not in data:
            continue
        lines.append(
            f"  {key:<12}{data['visual_tokens']:>7}{data.get('median_ms', 0):>9.0f}ms"
            f"{data.get('prefill_ms_median') or 0:>9.0f}ms"
            f"{data.get('ms_per_visual_token') or 0:>10.3f}"
        )

    if proj := report.get("projection"):
        lines += ["", "PROJECTED CYCLE TIME"]
        lines.append(
            f"  vision every cycle       {proj['cycle_with_vision_ms']} ms "
            f"({proj.get('max_hz_with_vision', '?')} Hz)"
        )
        lines.append(
            f"  vision cached            {proj['cycle_cached_vision_ms']} ms "
            f"({proj.get('max_hz_cached', '?')} Hz)"
        )
        lines += ["", f"  {proj['note']}"]

    lines += [
        "",
        "TUNING",
        "  - Prefill is small and roughly flat across image sizes; decode dominates.",
        "    If a SMALLER image is slower here, that is expected, not a bug: a blurrier",
        "    screenshot makes the model less certain and it emits more tokens.",
        "  - The real vision knob is perception.max_elements. Each reported element is",
        "    ~21 output tokens, so roughly 500 ms. Set it to the number of things your",
        "    guards actually test for.",
        "  - The real actuator knob is output length. The diagnostic note costs real",
        "    time: 48 chars measured 412 ms/cycle against 184 ms at 12.",
        "  - A prompt-cache speedup below ~1.5x means the cache is not being reused.",
        "    Check --cache-reuse and that nothing dynamic leaked into the prompt prefix.",
        "",
    ]
    return "\n".join(lines)
