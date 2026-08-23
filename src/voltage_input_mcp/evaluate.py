"""Model comparison harness.

Comparing model sizes on this workload needs care, because the obvious experiment gives
the wrong answer. "Which model writes a better burst" is not the question -- the grammar
already guarantees every burst is *valid*, so a bigger model cannot win on syntax. The
questions that actually decide whether a configuration is usable are:

  1. **Grounding accuracy.** Does the vision model put the box on the right thing? A model
     that is 200 ms faster and 40 px off is useless: the click misses.
  2. **Decision quality under constraint.** Given the same observation, does the actuator
     pick the *right* legal action, and does it chain a useful burst rather than emitting
     one timid action per cycle?
  3. **Latency**, which is only interesting once 1 and 2 are acceptable.

Ground truth
------------
Synthetic UI screenshots are a trap here: a drawn rectangle does not read as a button to a
model trained on real interfaces, so scoring against it measures the wrong thing. Instead
fixtures are **real screenshots** with labels supplied by the orchestrating model -- which
is exactly the reference this system uses at runtime anyway. `voltage fixture` captures a
screen and writes a stub for the orchestrator to annotate.

Without fixtures the harness still runs and reports latency plus side-by-side output for a
human to judge; it just cannot score grounding.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import actuator_grammar, observation_grammar
from .llm.base import Backend
from .llm.grammar import vision_vocabulary
from .models.observation import CoordinateMapper, parse_vision_output
from .runtime.prompts import ACTUATOR_SYSTEM, VISION_SYSTEM

__all__ = ["Fixture", "Scenario", "compare_vision", "compare_actuator", "SCENARIOS"]


@dataclass(slots=True)
class Fixture:
    """A real screenshot with orchestrator-supplied ground truth."""

    name: str
    png_path: Path
    watch: list[str]
    # label -> [x, y, w, h] in screen pixels. Omit a label to assert it is NOT present.
    expect: dict[str, list[int]] = field(default_factory=dict)
    absent: list[str] = field(default_factory=list)
    screen: tuple[int, int] = (1920, 1080)

    @classmethod
    def load(cls, path: Path) -> Fixture:
        data = json.loads(path.read_text())
        return cls(
            name=data.get("name", path.stem),
            png_path=path.with_suffix(".png"),
            watch=data["watch"],
            expect={k: list(v) for k, v in (data.get("expect") or {}).items()},
            absent=list(data.get("absent") or []),
            screen=tuple(data.get("screen", (1920, 1080))),  # type: ignore[arg-type]
        )

    def centre(self, label: str) -> tuple[int, int] | None:
        box = self.expect.get(label)
        if not box:
            return None
        return box[0] + box[2] // 2, box[1] + box[3] // 2


@dataclass(slots=True)
class Scenario:
    """A fixed decision problem for the actuator, with a machine-checkable expectation."""

    name: str
    prompt: str
    targets: list[str]
    n_elements: int
    # A burst is "good" if it satisfies this predicate. Deliberately loose: several
    # different bursts are correct, so this checks intent, not an exact string.
    expect_contains: list[str] = field(default_factory=list)
    expect_absent: list[str] = field(default_factory=list)
    expect_min_actions: int = 1


SCENARIOS: list[Scenario] = [
    Scenario(
        name="click_the_named_button",
        prompt=(
            "TASK: Click the Save button.\n"
            "CAN GO TO: confirm\n"
            "SCREEN: 1920x1080\n\n"
            "SEEN:\n"
            "  0 file list at 460,400 (800x420)\n"
            "  1 save button at 1204,712 (96x32)\n"
            "  2 cancel button at 1080,712 (96x32)\n\n"
            "BURST:"
        ),
        targets=["confirm"],
        n_elements=3,
        # Must reference element 1 and click. Referencing 0 or 2 is a grounding failure.
        expect_contains=["g:1", "c:l"],
        expect_absent=["g:0", "g:2"],
    ),
    Scenario(
        name="chain_a_sequence",
        prompt=(
            "TASK: Click the filename field, type report.txt, then press Enter.\n"
            "HINT: Chain the whole sequence into one burst.\n"
            "CAN GO TO: saved\n"
            "SCREEN: 1920x1080\n\n"
            "SEEN:\n"
            "  0 filename field at 900,600 (400x30)\n"
            "  1 save button at 1204,712 (96x32)\n\n"
            "BURST:"
        ),
        targets=["saved"],
        n_elements=2,
        # The whole point of bursts: a good answer does all three in one go.
        expect_contains=["g:0", "c:l", "t:", "k:enter"],
        expect_min_actions=4,
    ),
    Scenario(
        name="wait_when_nothing_is_visible",
        prompt=(
            "TASK: Click the Export button.\n"
            "CAN GO TO: exported\n"
            "SCREEN: 1920x1080\n\n"
            "SEEN: (nothing from the LOOK FOR list is visible)\n\n"
            "BURST:"
        ),
        targets=["exported"],
        n_elements=0,
        # Correct behaviour is to wait, not to guess coordinates and click blindly.
        expect_contains=["."],
        expect_absent=["c:l", "m:"],
        expect_min_actions=0,
    ),
    Scenario(
        name="respect_the_state_machine",
        prompt=(
            "TASK: The dialog shows an error. Go to the recover state.\n"
            "CAN GO TO: recover\n"
            "SCREEN: 1920x1080\n\n"
            "SEEN:\n"
            "  0 error message at 900,500 (400x40)\n"
            "  1 ok button at 960,600 (80x30)\n\n"
            "BURST:"
        ),
        targets=["recover"],
        n_elements=2,
    ),
]


# --------------------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------------------


async def compare_vision(
    backends: dict[str, Backend],
    fixtures: list[Fixture],
    *,
    rounds: int = 3,
    downscale: tuple[int, int] = (896, 504),
    tolerance_px: int = 40,
) -> dict[str, Any]:
    """Score each vision backend's grounding against fixture ground truth.

    The metric is centre distance in screen pixels, not IoU. A click lands at the centre,
    so that is what determines whether the automation works; a loose box with the right
    centre is fine, and a tight box in the wrong place is not.
    """
    import numpy as np
    from PIL import Image

    results: dict[str, Any] = {}

    for model_name, backend in backends.items():
        per_fixture: list[dict[str, Any]] = []
        latencies: list[float] = []

        for fixture in fixtures:
            if not fixture.png_path.exists():
                per_fixture.append({"fixture": fixture.name, "error": "png missing"})
                continue

            with Image.open(fixture.png_path) as img:
                pixels = np.asarray(img.convert("RGB"), dtype=np.uint8)

            from .capture.base import downscale as ds
            from .capture.base import encode_png

            scaled = ds(pixels, downscale)
            png = encode_png(scaled, 1)
            grammar = observation_grammar(fixture.watch, max_elements=6, read_text=True)
            mapper = CoordinateMapper(
                capture_size=(int(scaled.shape[1]), int(scaled.shape[0])),
                screen_size=fixture.screen,
                region_size=(int(pixels.shape[1]), int(pixels.shape[0])),
            )

            found: list[dict[str, Any]] = []
            for _ in range(rounds):
                started = time.perf_counter()
                result = await backend.generate(
                    prompt="LOOK FOR: " + ", ".join(fixture.watch),
                    system=VISION_SYSTEM,
                    image_png=png,
                    grammar=grammar if backend.supports_grammar else None,
                    max_tokens=192,
                    temperature=0.1,
                    timeout_s=180.0,
                )
                latencies.append((time.perf_counter() - started) * 1000.0)
                if not result.ok:
                    per_fixture.append({"fixture": fixture.name, "error": result.error})
                    break
                obs = parse_vision_output(
                    result.text,
                    mapper,
                    vocabulary=vision_vocabulary(fixture.watch),
                    max_elements=6,
                )
                found.append(
                    {
                        "labels": obs.labels,
                        "centres": {e.label: list(e.center) for e in obs.elements},
                    }
                )
            else:
                per_fixture.append(_score_fixture(fixture, found, tolerance_px))

        results[model_name] = {
            "fixtures": per_fixture,
            "latency_ms_median": (
                round(statistics.median(latencies), 1) if latencies else None
            ),
            **_aggregate(per_fixture),
        }

    return results


def _score_fixture(
    fixture: Fixture, runs: list[dict[str, Any]], tolerance_px: int
) -> dict[str, Any]:
    hits: dict[str, list[float]] = {}
    misses: list[str] = []
    for label in fixture.expect:
        truth = fixture.centre(label)
        if truth is None:
            continue
        distances = []
        for run in runs:
            got = run["centres"].get(label)
            if got is None:
                continue
            distances.append(((got[0] - truth[0]) ** 2 + (got[1] - truth[1]) ** 2) ** 0.5)
        if distances:
            hits[label] = distances
        else:
            misses.append(label)

    # A false positive on an `absent` label is worse than a miss: it makes a guard fire
    # when it should not, which sends the state machine down the wrong branch.
    false_positives = [
        label for label in fixture.absent if any(label in run["labels"] for run in runs)
    ]

    detected = len(hits)
    expected = len([label for label in fixture.expect if fixture.centre(label)])
    all_dist = [d for ds_ in hits.values() for d in ds_]
    within = [d for d in all_dist if d <= tolerance_px]

    return {
        "fixture": fixture.name,
        "detected": f"{detected}/{expected}",
        "missed": misses,
        "false_positives": false_positives,
        "median_centre_error_px": (
            round(statistics.median(all_dist), 1) if all_dist else None
        ),
        "within_tolerance": f"{len(within)}/{len(all_dist)}" if all_dist else "0/0",
        "per_label_error_px": {
            k: round(statistics.median(v), 1) for k, v in hits.items()
        },
    }


def _aggregate(per_fixture: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        f["median_centre_error_px"] for f in per_fixture
        if isinstance(f.get("median_centre_error_px"), (int, float))
    ]
    detected = sum(
        int(f["detected"].split("/")[0]) for f in per_fixture if "detected" in f
    )
    expected = sum(
        int(f["detected"].split("/")[1]) for f in per_fixture if "detected" in f
    )
    fps = sum(len(f.get("false_positives") or []) for f in per_fixture)
    return {
        "overall_detected": f"{detected}/{expected}" if expected else "n/a",
        "overall_median_error_px": round(statistics.median(errors), 1) if errors else None,
        "false_positives": fps,
    }


# --------------------------------------------------------------------------------------
# Actuator
# --------------------------------------------------------------------------------------


async def compare_actuator(
    backends: dict[str, Backend],
    *,
    scenarios: list[Scenario] | None = None,
    rounds: int = 3,
) -> dict[str, Any]:
    """Run each actuator backend through fixed decision scenarios."""
    from .models.burst import parse_burst
    from .runtime.session import _split_actuator_reply

    scenarios = scenarios or SCENARIOS
    results: dict[str, Any] = {}

    for model_name, backend in backends.items():
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []

        for scenario in scenarios:
            grammar = actuator_grammar(
                allow_verbs=["g", "c", "k", "t", "w", "m"],
                targets=scenario.targets,
                n_elements=scenario.n_elements,
            )
            outputs: list[str] = []
            passes = 0
            parse_failures = 0

            for _ in range(rounds):
                started = time.perf_counter()
                result = await backend.generate(
                    prompt=scenario.prompt,
                    system=ACTUATOR_SYSTEM,
                    grammar=grammar if backend.supports_grammar else None,
                    max_tokens=96,
                    temperature=0.25,
                    timeout_s=60.0,
                )
                latencies.append((time.perf_counter() - started) * 1000.0)
                if not result.ok:
                    rows.append({"scenario": scenario.name, "error": result.error})
                    break

                burst_str, target, _note = _split_actuator_reply(result.text)
                outputs.append(f"{burst_str}|{target or '.'}")

                centres = [(100 * (i + 1), 200 * (i + 1)) for i in range(scenario.n_elements)]
                try:
                    burst = parse_burst(burst_str, screen=(1920, 1080), elements=centres)
                except Exception:  # noqa: BLE001
                    parse_failures += 1
                    continue

                if _scenario_ok(scenario, burst_str, len(burst)):
                    passes += 1
            else:
                rows.append(
                    {
                        "scenario": scenario.name,
                        "pass_rate": f"{passes}/{rounds}",
                        "parse_failures": parse_failures,
                        "outputs": outputs,
                    }
                )

        total_pass = sum(
            int(r["pass_rate"].split("/")[0]) for r in rows if "pass_rate" in r
        )
        total_runs = sum(
            int(r["pass_rate"].split("/")[1]) for r in rows if "pass_rate" in r
        )
        results[model_name] = {
            "scenarios": rows,
            "overall_pass": f"{total_pass}/{total_runs}" if total_runs else "n/a",
            "parse_failures": sum(r.get("parse_failures", 0) for r in rows),
            "latency_ms_median": round(statistics.median(latencies), 1) if latencies else None,
        }

    return results


def _scenario_ok(scenario: Scenario, burst_str: str, n_actions: int) -> bool:
    if n_actions < scenario.expect_min_actions:
        return False
    if scenario.expect_contains and scenario.expect_contains == ["."]:
        return burst_str.strip() in (".", "")
    if any(needle not in burst_str for needle in scenario.expect_contains):
        return False
    return all(needle not in burst_str for needle in scenario.expect_absent)


# --------------------------------------------------------------------------------------


def format_vision(results: dict[str, Any]) -> str:
    lines = ["", "VISION -- grounding accuracy (centre distance in screen pixels)", ""]
    lines.append(f"  {'model':<28}{'detected':>10}{'err px':>9}{'false+':>8}{'latency':>10}")
    for name, data in results.items():
        lines.append(
            f"  {name:<28}{data.get('overall_detected', '?'):>10}"
            f"{str(data.get('overall_median_error_px', '?')):>9}"
            f"{data.get('false_positives', 0):>8}"
            f"{str(data.get('latency_ms_median', '?')) + 'ms':>10}"
        )
    lines += [
        "",
        "  A click lands at the centre, so centre error is what decides whether the",
        "  automation works. Under ~40 px is comfortably inside a normal button; over",
        "  ~80 px will miss small targets regardless of how fast the model is.",
        "",
    ]
    for name, data in results.items():
        for fixture in data.get("fixtures", []):
            if fixture.get("error"):
                lines.append(f"  {name} / {fixture.get('fixture')}: ERROR {fixture['error']}")
            elif fixture.get("missed") or fixture.get("false_positives"):
                lines.append(
                    f"  {name} / {fixture['fixture']}: missed={fixture['missed']} "
                    f"false+={fixture['false_positives']}"
                )
    return "\n".join(lines)


def format_actuator(results: dict[str, Any]) -> str:
    lines = ["", "ACTUATOR -- decision quality under grammar constraint", ""]
    lines.append(f"  {'model':<28}{'pass':>10}{'parse fail':>12}{'latency':>10}")
    for name, data in results.items():
        lines.append(
            f"  {name:<28}{data.get('overall_pass', '?'):>10}"
            f"{data.get('parse_failures', 0):>12}"
            f"{str(data.get('latency_ms_median', '?')) + 'ms':>10}"
        )
    lines += [
        "",
        "  parse failures should be 0 on llama.cpp -- the grammar makes malformed output",
        "  unrepresentable. A non-zero count means the backend ignored the grammar",
        "  (Ollama does) or the grammar was not sent.",
        "",
        "  Per-scenario output:",
    ]
    for name, data in results.items():
        for row in data.get("scenarios", []):
            if row.get("error"):
                lines.append(f"    {name} / {row['scenario']}: ERROR {row['error']}")
                continue
            lines.append(f"    {name} / {row['scenario']}: {row['pass_rate']}")
            for out in row.get("outputs", [])[:2]:
                lines.append(f"        {out}")
    lines.append("")
    return "\n".join(lines)
