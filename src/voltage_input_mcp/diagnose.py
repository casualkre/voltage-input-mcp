"""Turn a run's journal into a diagnosis, and remember what was learned.

The journal already records everything. That is not the same as being useful: the
orchestrator is remote and did not watch the screen, so handing it 200 cycle records and
expecting a correct conclusion is asking it to re-derive what this process already knows.

So this computes the derived facts the journal implies but does not state -- which guards
never once evaluated true, which `watch` labels the vision model never reported, whether
bursts actually changed the screen, whether the actuator is chaining or emitting one
timid action at a time -- and maps each to the specific edit that fixes it.

The distinction that matters is between *a burst that never ran* and *a burst that ran and
did nothing*. They look identical in a summary and have completely unrelated causes: the
first is policy or grammar, the second is focus, pointer mode, or an application that
ignores synthetic input. Separating them is most of the value here.

`Lesson` is the other half. A diagnosis helps within a run; a lesson survives it. Lessons
are stored per target (an application, a game) so that the next playbook for Minecraft
starts from what the last one discovered rather than rediscovering it.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Finding", "diagnose", "Lesson", "load_lessons", "save_lesson", "lessons_path"]

SEVERITY = ("blocker", "problem", "hint")


@dataclass(slots=True)
class Finding:
    """One diagnosis: what happened, why it matters, and the edit that fixes it."""

    severity: str
    code: str
    what: str
    why: str
    fix: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cycles(journal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in journal if r.get("kind") == "cycle"]


def diagnose(
    journal: list[dict[str, Any]],
    playbook: dict[str, Any] | None = None,
    *,
    status: str = "",
    reason: str = "",
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """Analyse a run. Returns findings ordered blocker-first."""
    cycles = _cycles(journal)
    findings: list[Finding] = []

    if not cycles:
        return {
            "cycles": 0,
            "findings": [
                Finding(
                    "blocker", "no_cycles",
                    "The run produced no cycles.",
                    "It ended before the loop started, so nothing about the playbook was "
                    "exercised.",
                    "Check voltage_status for the reason, and voltage_doctor for a "
                    "backend or capture failure.",
                ).as_dict()
            ],
            "summary": "no cycles ran",
        }

    n = len(cycles)
    states = Counter(c.get("state", "") for c in cycles)
    executed = [c for c in cycles if c.get("executed")]
    refused = [c for c in cycles if c.get("allowed") is False]
    errored = [c for c in cycles if c.get("error")]
    transitions = [c for c in cycles if c.get("transition")]
    perception = Counter(c.get("perception", "") for c in cycles)

    # -- 1. dry run left on ------------------------------------------------------------
    if dry_run:
        findings.append(Finding(
            "blocker", "dry_run",
            "dry_run was on, so nothing was ever injected.",
            "Every burst was parsed, safety-checked and journalled, and then discarded. "
            "A run can look completely healthy in this mode and change nothing on screen.",
            "This is the correct way to validate a playbook. When the journal looks "
            "right, re-run with dry_run=false.",
            {"bursts_that_would_have_run": len([c for c in cycles if c.get("burst")])},
        ))

    # -- 2. labels the vision model never reported -------------------------------------
    seen_labels: set[str] = set()
    for cycle in cycles:
        for element in cycle.get("elements") or []:
            if isinstance(element, dict) and element.get("label"):
                seen_labels.add(element["label"])

    watched: dict[str, set[str]] = {}
    if playbook:
        for name, state in (playbook.get("states") or {}).items():
            watched[name] = set(state.get("watch") or [])

    never_seen = {lbl for group in watched.values() for lbl in group} - seen_labels
    if never_seen and perception.get("vlm", 0) > 0:
        findings.append(Finding(
            "blocker" if len(never_seen) >= len(seen_labels) else "problem",
            "label_never_seen",
            f"The vision model never reported: {sorted(never_seen)}.",
            "Any guard testing sees() on those labels can never fire, so the state "
            "machine is stuck by construction. The vision grammar restricts labels to "
            "the `watch` list, so a label the model cannot recognise is simply absent.",
            "Run voltage_observe with those labels against the real screen. If they do "
            "not come back, rename them to something more visually literal -- 'blue "
            "Save button' beats 'submit control', 'health bar' beats 'vitality "
            "indicator'. Prefer what the thing *looks like* over what it is called.",
            {"never_seen": sorted(never_seen), "actually_seen": sorted(seen_labels)},
        ))

    # -- 3. bursts ran but the screen did not move -------------------------------------
    # The important distinction: a burst that never ran is a policy problem; a burst that
    # ran and changed nothing is a focus/pointer/input-path problem. Opposite fixes.
    deltas = [
        float((c.get("probes") or {}).get("__frame_delta__", 0.0))
        for c in cycles if c.get("executed")
    ]
    if executed and deltas and max(deltas) < 0.01:
        findings.append(Finding(
            "blocker", "input_not_landing",
            f"{len(executed)} bursts executed, but the screen never changed.",
            "The events reached the OS -- the executor reported success -- and nothing "
            "reacted. That is not a playbook problem. Usual causes: the target window "
            "does not have focus; the pointer is being positioned by a channel the "
            "compositor ignores; or the application reads raw input in a way synthetic "
            "events do not reach.",
            "Run voltage_calibrate and watch the real cursor. If it does not move, set "
            "pointer_mode to 'relative' in voltage.toml. If it moves but clicks do "
            "nothing, click the target window first to focus it. For a fullscreen game, "
            "try borderless windowed mode.",
            {"bursts_executed": len(executed), "max_frame_delta": round(max(deltas), 4)},
        ))

    # -- 4. governor refusals ----------------------------------------------------------
    if refused:
        rules = Counter(
            v.get("rule", "?") for c in refused for v in (c.get("violations") or [])
        )
        top, count = rules.most_common(1)[0]
        findings.append(Finding(
            "problem" if len(refused) < n * 0.3 else "blocker",
            "governor_refusals",
            f"{len(refused)} of {n} bursts were refused; most often by `{top}`.",
            "The policy and what the actuator wants to do disagree. Refusal is "
            "whole-burst, so each one costs an entire cycle.",
            _refusal_fix(top),
            {"refused": len(refused), "by_rule": dict(rules)},
        ))

    # -- 5. the actuator is not chaining ----------------------------------------------
    burst_lengths = [
        len([p for p in str(c.get("burst", "")).split(";") if p.strip()])
        for c in cycles if c.get("burst") and c.get("burst") != "."
    ]
    if burst_lengths and sum(burst_lengths) / len(burst_lengths) < 1.6:
        findings.append(Finding(
            "problem", "timid_bursts",
            f"Bursts averaged {sum(burst_lengths) / len(burst_lengths):.1f} actions.",
            "One action per burst throws away the entire point of the design: a burst "
            "costs one decision no matter how many inputs it contains, so a one-action "
            "burst pays full model latency per input.",
            "Say so in the state's `brief` or `hint`: 'Chain the whole sequence into one "
            "burst.' Give an explicit example. Small models copy the shape of an "
            "example far more reliably than they follow an abstract instruction.",
            {"mean_actions": round(sum(burst_lengths) / len(burst_lengths), 2)},
        ))

    # -- 6. a state that never left ----------------------------------------------------
    if playbook:
        stuck = [
            name for name, count in states.items()
            if count >= max(8, n * 0.5)
            and not any(c.get("transition") for c in cycles if c.get("state") == name)
        ]
        for name in stuck:
            state = (playbook.get("states") or {}).get(name, {})
            guards = [t.get("when") for t in state.get("transitions") or []]
            findings.append(Finding(
                "blocker", "state_never_left",
                f"State '{name}' ran {states[name]} cycles and never transitioned.",
                "Every one of its guards evaluated false on every cycle.",
                "Check each guard against what the journal actually saw. If a guard "
                "tests sees('x'), confirm the vision model reports 'x' at all. Add an "
                "escape -- {'when': 'cycles() > 10', 'to': '@failure'} -- so a stuck "
                "state fails fast instead of burning the budget.",
                {"state": name, "cycles": states[name], "guards": guards},
            ))

    # -- 7. nothing on screen ever changed ---------------------------------------------
    static = [
        float((c.get("probes") or {}).get("__static_for__", 0.0)) for c in cycles
    ]
    if static and max(static) > 10 and not executed:
        findings.append(Finding(
            "problem", "screen_static_no_input",
            f"The screen was unchanged for {max(static):.0f}s and no burst executed.",
            "The loop looked at a still screen and never acted -- usually the actuator "
            "correctly waiting because it could not see what the brief asked for.",
            "This is the actuator behaving well on a bad brief. Check finding "
            "`label_never_seen`, or whether the intended window is actually visible.",
            {"static_seconds": round(max(static), 1)},
        ))

    # -- 8. errors ---------------------------------------------------------------------
    if errored:
        sample = next((c.get("error") for c in errored if c.get("error")), "")
        findings.append(Finding(
            "problem", "cycle_errors",
            f"{len(errored)} cycles reported an error.",
            "Backend timeouts, unparseable output, or capture failures. These cost a "
            "cycle each.",
            f"First error: {sample}. If it mentions a backend, check voltage_doctor.",
            {"count": len(errored), "first": sample},
        ))

    # -- 9. perception spend -----------------------------------------------------------
    vlm_share = perception.get("vlm", 0) / n if n else 0
    if vlm_share > 0.8 and n > 10:
        findings.append(Finding(
            "hint", "vision_every_cycle",
            f"The vision model ran on {vlm_share:.0%} of cycles.",
            "Each reported element costs ~21 output tokens, roughly 500 ms. Running "
            "vision every cycle is the single biggest avoidable cost.",
            "Set perception.mode to 'on_change' so a frame diff gates it, and lower "
            "max_elements to the number your guards actually test for.",
            {"vlm_cycles": perception.get("vlm", 0), "total": n},
        ))

    order = {s: i for i, s in enumerate(SEVERITY)}
    findings.sort(key=lambda f: order.get(f.severity, 9))

    return {
        "cycles": n,
        "status": status,
        "reason": reason,
        "executed": len(executed),
        "refused": len(refused),
        "transitions": len(transitions),
        "states_visited": dict(states),
        "perception": dict(perception),
        "findings": [f.as_dict() for f in findings],
        "summary": (
            findings[0].what if findings
            else f"{n} cycles, {len(executed)} bursts executed, no problems detected."
        ),
    }


def _refusal_fix(rule: str) -> str:
    return {
        "deny_labels": "The actuator tried to click an element whose label matches a "
                       "denied pattern. That is the guard working. If the click is "
                       "genuinely required, narrow deny_labels for this playbook -- do "
                       "not remove it wholesale.",
        "allow_verbs": "The actuator used a verb this state forbids. Either add the verb "
                       "to allow_verbs, or reword the brief so it stops trying.",
        "allow_keys": "A key outside the allowlist. For a game, list every key the game "
                      "uses; a key not in allow_keys is absent from the grammar, so the "
                      "actuator cannot even name it.",
        "max_inputs_per_second": "The burst exceeded the rate cap. Raise "
                                 "max_inputs_per_second, or add w: waits inside the burst.",
        "require_target_element": "A click did not land on any observed element. Either "
                                  "the vision model is not finding the target, or the "
                                  "actuator is using m: instead of g:. Restricting "
                                  "allow_verbs to exclude 'm' forces g:.",
        "click_position_unknown": "A click with no preceding move cannot be fenced. Tell "
                                  "the actuator to always move before clicking, or drop "
                                  "the click region policy.",
        "deny_text": "The actuator tried to type something matching a denied pattern.",
    }.get(rule, f"See the journal for what rule `{rule}` blocked and why.")


# --------------------------------------------------------------------------------------
# Lessons
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Lesson:
    """Something learned about driving one target, worth carrying to the next run."""

    target: str
    note: str
    kind: str = "observation"   # observation | label | timing | policy | burst
    playbook: str = ""
    created: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def lessons_path() -> Path:
    from .config import state_dir

    return state_dir() / "lessons.json"


def _slug(target: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", target.strip().lower()).strip("_") or "general"


def load_lessons(target: str | None = None) -> list[dict[str, Any]]:
    """Lessons for one target, or all of them. Newest first."""
    path = lessons_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    entries = data if isinstance(data, list) else []
    if target:
        key = _slug(target)
        entries = [e for e in entries if _slug(str(e.get("target", ""))) == key]
    return sorted(entries, key=lambda e: -float(e.get("created", 0)))


def save_lesson(lesson: Lesson, *, limit: int = 400) -> Path:
    """Append a lesson, de-duplicating identical notes for the same target."""
    path = lessons_path()
    entries = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            entries = loaded if isinstance(loaded, list) else []
        except (OSError, ValueError):
            entries = []

    key = (_slug(lesson.target), lesson.note.strip().lower())
    entries = [
        e for e in entries
        if (_slug(str(e.get("target", ""))), str(e.get("note", "")).strip().lower()) != key
    ]
    entries.append(lesson.as_dict())
    entries = sorted(entries, key=lambda e: -float(e.get("created", 0)))[:limit]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))
    return path
