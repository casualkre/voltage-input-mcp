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
    # A cycle can be `allowed: false` with no violations: the burst never reached the
    # governor because it did not parse. Those two have opposite fixes -- one is a policy
    # disagreement, the other is the actuator emitting something malformed -- so they are
    # separated here rather than being counted together and reported as whichever came
    # first. Reading the top rule off an empty Counter used to raise IndexError, which
    # meant diagnose crashed on precisely the runs most in need of it.
    rules = Counter(
        v.get("rule", "?") for c in refused for v in (c.get("violations") or [])
    )
    if rules:
        top, _ = rules.most_common(1)[0]
        by_rule = len([c for c in refused if c.get("violations")])
        findings.append(Finding(
            "problem" if by_rule < n * 0.3 else "blocker",
            "governor_refusals",
            f"{by_rule} of {n} bursts were refused; most often by `{top}`.",
            "The policy and what the actuator wants to do disagree. Refusal is "
            "whole-burst, so each one costs an entire cycle.",
            _refusal_fix(top),
            {"refused": by_rule, "by_rule": dict(rules)},
        ))

    unparseable = [c for c in refused if not c.get("violations")]
    if unparseable:
        sample = next((str(c.get("burst", "")) for c in unparseable if c.get("burst")), "")
        truncated = [c for c in unparseable if _looks_truncated(str(c.get("burst", "")))]
        findings.append(Finding(
            "blocker" if len(unparseable) > n * 0.3 else "problem",
            "unparseable_burst",
            f"{len(unparseable)} of {n} bursts could not be parsed at all.",
            "These never reached the governor -- the actuator produced something that is "
            "not a burst. Under llama.cpp a GBNF grammar makes that structurally "
            + ("impossible mid-string, so the usual cause is the reply being cut off at "
               "max_tokens with an action half-written."
               if truncated else
               "impossible, so this points at a backend running without one -- Ollama can "
               "only constrain to a JSON schema, leaving the burst string itself free."),
            (
                "The reply hit the token limit. That almost always means the actuator is "
                "repeating itself -- a run of identical actions padding out the burst. "
                "Shorten what it is asked for: lower policy.max_actions_per_burst so the "
                "grammar itself stops it, and put a concrete short burst in the state's "
                "`hint` for it to copy."
                if truncated else
                "Check voltage_doctor: on llama.cpp this should not happen. If you are on "
                "Ollama, expect it occasionally and keep max_actions_per_burst low."
            ),
            {"count": len(unparseable), "truncated": len(truncated), "first": sample[:120]},
        ))

    # -- 5. the actuator is not chaining, or is chaining the same thing over and over ---
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

    repetitive = [
        c for c in cycles
        if c.get("burst") and _repetition_ratio(str(c["burst"])) >= 0.6
    ]
    if repetitive and len(repetitive) >= max(2, n * 0.25):
        sample = str(repetitive[0].get("burst", ""))
        findings.append(Finding(
            "problem", "repetitive_bursts",
            f"{len(repetitive)} of {n} bursts were mostly one action repeated.",
            "A small model with a grammar that permits a long burst and a brief that does "
            "not tell it when to stop will pad: the same action over and over until it "
            "runs out of tokens. Every one of those tokens is decode time -- at ~22 ms a "
            "token, a padded burst can cost more than a second of pure waste per cycle.",
            "Cap it structurally rather than asking nicely. Lower "
            "policy.max_actions_per_burst to the length the task actually needs; the "
            "grammar is generated from it, so a longer burst becomes unrepresentable. "
            "Then put one concrete example burst in the state's `hint` -- small models "
            "copy a shape far more reliably than they follow an instruction.",
            {"count": len(repetitive), "example": sample[:120]},
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

    findings.extend(_diagnose_fast_layer(journal, playbook, cycles))

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
        "reflex": _reflex_stats(journal),
        "findings": [f.as_dict() for f in findings],
        "summary": (
            findings[0].what if findings
            else f"{n} cycles, {len(executed)} bursts executed, no problems detected."
        ),
    }


def _reflex_stats(journal: list[dict[str, Any]]) -> dict[str, Any]:
    """What the fast layer did, from its own events plus the end-of-run summary."""
    kinds = Counter(str(r.get("kind", "")) for r in journal)
    end = next((r for r in reversed(journal) if r.get("kind") == "end"), {})
    summary = end.get("reflex") or {}
    return {
        "fires": kinds.get("reflex", 0),
        "starved": kinds.get("reflex_starved", 0),
        "render_errors": kinds.get("reflex_error", 0) + kinds.get("burst_error", 0),
        "hold_on": kinds.get("hold_on", 0),
        "hold_off": kinds.get("hold_off", 0),
        "hold_refused": kinds.get("hold_refused", 0),
        "measured_hz": summary.get("measured_hz"),
        "requested_hz": summary.get("requested_hz"),
        "ticks": summary.get("ticks"),
        "ocr": summary.get("ocr") or {},
    }


def _diagnose_fast_layer(
    journal: list[dict[str, Any]],
    playbook: dict[str, Any] | None,
    cycles: list[dict[str, Any]],
) -> list[Finding]:
    """Findings about the reflex loop -- the part that runs between decisions.

    This is where a run silently degrades into the thing it was built not to be. Every
    check here distinguishes "the fast layer did its job" from "the fast layer was
    configured but never actually did anything", because those look identical in a
    summary and the second one means the whole run happened at decision rate.
    """
    findings: list[Finding] = []
    stats = _reflex_stats(journal)
    elapsed = float(cycles[-1].get("t", 0.0)) - float(cycles[0].get("t", 0.0)) if cycles else 0.0

    declared_reflexes = 0
    declared_holds = 0
    declared_probes = 0
    number_probes: list[str] = []
    if playbook:
        for spec in playbook.get("probes") or []:
            declared_probes += 1
            if isinstance(spec, dict) and spec.get("type") == "number":
                number_probes.append(str(spec.get("id", "?")))
        for state in (playbook.get("states") or {}).values():
            for rule in state.get("reflex") or []:
                if rule.get("hold"):
                    declared_holds += 1
                else:
                    declared_reflexes += 1

    # -- the fast layer is not being used at all ---------------------------------------
    if playbook and not declared_reflexes and not declared_holds:
        findings.append(Finding(
            "problem", "no_fast_layer",
            "This playbook declares no reflexes and no holds.",
            "Every input therefore waited on a model decision, so the effective input "
            "rate was the decision rate -- around 2 Hz. Bursts help within a decision, "
            "but nothing reacted between them. For anything with timing in it, that is "
            "the difference between controlling and hoping.",
            "Add a probe over the number or colour that matters, then a reflex keyed off "
            "it. Use `hold` for anything continuous ('hold w while the target is ahead') "
            "and `do` for discrete reactions. See voltage_reference(section='control').",
            {"probes": declared_probes},
        ))

    # -- declared but never triggered --------------------------------------------------
    if (declared_reflexes or declared_holds) and not stats["fires"] and not stats["hold_on"]:
        probe_seen: dict[str, float] = {}
        for cycle in cycles:
            for key, value in (cycle.get("probes") or {}).items():
                if not key.startswith("__"):
                    probe_seen[key] = max(probe_seen.get(key, 0.0), float(value))
        findings.append(Finding(
            "blocker", "reflex_never_fired",
            f"{declared_reflexes + declared_holds} reflex rules were declared and not one "
            f"ever triggered.",
            "Their guards were false on every tick, so the fast layer contributed "
            "nothing and the run happened entirely at decision rate.",
            "Compare each guard's threshold against the highest value that probe actually "
            "reached during the run (below). A guard on `probe('x') > 0.5` against a probe "
            "that never exceeded 0.02 is testing the wrong region or the wrong colour; "
            "confirm the region with voltage_capture first.",
            {"peak_probe_values": {k: round(v, 3) for k, v in sorted(probe_seen.items())}},
        ))

    # -- reactions arriving late, or not at all ----------------------------------------
    if stats["starved"]:
        share = stats["starved"] / max(1, stats["starved"] + stats["fires"])
        findings.append(Finding(
            "problem" if share < 0.4 else "blocker",
            "reflex_starved",
            f"{stats['starved']} reflexes were dropped because the input device was busy.",
            "A reflex fired while an actuator burst was still executing. Rather than "
            "queue -- and land as a reaction to a situation that had already resolved -- "
            "it was dropped. The fast layer is being crowded out by the slow one.",
            "Shorten the actuator's bursts: lower policy.max_burst_ms and "
            "max_actions_per_burst, and cut long w: waits out of the brief's example. A "
            "burst that runs for 500 ms is 500 ms in which nothing can react.",
            {"dropped": stats["starved"], "fired": stats["fires"]},
        ))

    # -- the loop could not keep up ----------------------------------------------------
    measured, requested = stats.get("measured_hz"), stats.get("requested_hz")
    if measured and requested and measured < requested * 0.6:
        findings.append(Finding(
            "problem", "reflex_rate_low",
            f"The reflex loop asked for {requested:g} Hz and achieved {measured:g} Hz.",
            "Almost always the capture backend: a streaming source costs ~2 ms a frame, "
            "while one that does a fresh grab per call costs 15-40 ms and cannot be run "
            "at 20 Hz without eating the machine the models need.",
            "Check `voltage_doctor` for which capture backend is active. 'portal' streams "
            "and will hold the rate; 'kwin' and 'grim' do a round trip per frame. If you "
            "are stuck on a non-streaming backend, set reflex_hz to what it can sustain "
            "so the loop stops trying and the timing in your guards stays honest.",
            {"requested_hz": requested, "measured_hz": measured, "ticks": stats.get("ticks")},
        ))

    # -- a latch flipping on its own threshold -----------------------------------------
    flips = stats["hold_on"] + stats["hold_off"]
    if declared_holds and elapsed > 1.0 and flips / elapsed > 4.0:
        by_rule = Counter(
            str(r.get("rule", "?")) for r in journal
            if r.get("kind") in ("hold_on", "hold_off")
        )
        findings.append(Finding(
            "problem", "latch_chatter",
            f"Hold reflexes engaged and released {flips} times in {elapsed:.0f}s.",
            "A latch whose guard sits on its threshold flips at reflex rate. Downstream "
            "that is not a held key at all -- it is a machine-gun of taps, which a game "
            "reads as a character twitching in place rather than moving.",
            "Give the latch hysteresis: set `release_when` to a threshold wider than "
            "`when`, so there is a dead band in which neither fires and the latch keeps "
            "its state. `when: probe('x') < 90` with `release_when: probe('x') > 110`. "
            "`min_hold_ms` is the blunter alternative.",
            {"flips_per_second": round(flips / elapsed, 1), "by_rule": dict(by_rule)},
        ))

    # -- an expression that cannot be turned into a burst ------------------------------
    if stats["render_errors"]:
        sample = next(
            (str(r.get("error", "")) for r in journal
             if r.get("kind") in ("reflex_error", "burst_error")), ""
        )
        findings.append(Finding(
            "problem", "burst_render_error",
            f"{stats['render_errors']} bursts could not be built from their expressions.",
            "An {expression} hole produced a value the burst DSL will not accept -- "
            "usually a coordinate off the edge of the screen from an unclamped steering "
            "term. The fire was skipped rather than sending the pointer somewhere "
            "arbitrary.",
            "Wrap the term: `m:{clamp(probe('tx'), 0, 1919)},540`. For a relative move, "
            "clamp the step instead: `r:{clamp(probe('err') * 0.4, -300, 300)},0`.",
            {"count": stats["render_errors"], "first": sample},
        ))

    # -- the learning loop, if there is one --------------------------------------------
    end = next((r for r in reversed(journal) if r.get("kind") == "end"), {})
    episodes = [r for r in journal if r.get("kind") == "episode"]
    tunables = (playbook or {}).get("tunables") or {}
    reward = (playbook or {}).get("reward")

    if tunables and not reward:
        findings.append(Finding(
            "problem", "tunables_without_reward",
            f"{len(tunables)} tunables are declared and there is no `reward`.",
            "Without a reward there is nothing to optimise against, so the search never "
            "runs and every tunable stays pinned at its default. The playbook looks like "
            "it is learning and is not.",
            "Add a `reward` block naming a number probe -- usually a score or money "
            "counter -- and mark the transition that ends a run `ends_episode: true`.",
            {"tunables": sorted(tunables)},
        ))
    elif tunables and reward and len(episodes) < 6:
        findings.append(Finding(
            "hint", "too_few_episodes",
            f"Only {len(episodes)} scored episodes; the search needs more to say anything.",
            "A (1+1) search compares a proposal against the incumbent over repeated "
            "samples, and episode rewards here are noisy enough that a handful of them "
            "is indistinguishable from luck.",
            "Run longer. Twenty-plus episodes is where the early-versus-late comparison "
            "starts to mean something; the run budget is usually what is cutting it "
            "short.",
            {"episodes": len(episodes)},
        ))
    elif episodes:
        rewards = [float(e.get("reward", 0.0)) for e in episodes]
        third = max(1, len(rewards) // 3)
        early = sum(rewards[:third]) / third
        late = sum(rewards[-third:]) / third
        if late <= early:
            findings.append(Finding(
                "problem", "tuning_not_improving",
                f"Episode reward has not improved: {early:.0f} early, {late:.0f} late.",
                "Either the reward is not measuring what you care about, the tunables are "
                "not the constants that matter, or their bounds exclude the good region.",
                "Check the reward probe reads a real number first -- a number probe that "
                "cannot read returns 0 and makes every episode score identically. Then "
                "widen the bounds on the tunables you most suspect, and cut the ones you "
                "do not: every extra dimension costs episodes.",
                {"early_mean": round(early, 2), "late_mean": round(late, 2),
                 "episodes": len(episodes)},
            ))

    recall = end.get("recall") or {}
    if recall.get("enabled") and recall.get("hit_rate") is not None:
        rate = float(recall["hit_rate"])
        if rate < 0.1 and (recall.get("hits", 0) + recall.get("misses", 0)) > 30:
            findings.append(Finding(
                "hint", "recall_never_hits",
                f"Recall answered {rate:.0%} of decisions; it is not paying for itself.",
                "Situations are not repeating closely enough to be recognised. Usually "
                "the probe vector contains something that drifts continuously -- a timer, "
                "a cumulative score -- so no two moments ever look alike.",
                "Drop continuously-growing probes from the playbook, or bucket them "
                "(scale a score by 0.001 so only large moves register). Recall keys on "
                "the declared probes, so what you declare decides what counts as the "
                "same situation.",
                {**recall},
            ))

    # -- a number probe that never produced a number -----------------------------------
    ocr = stats.get("ocr") or {}
    if number_probes and ocr and not ocr.get("reads") and ocr.get("misses"):
        findings.append(Finding(
            "blocker", "number_probe_never_read",
            f"Number probes {number_probes} never produced a value; "
            f"{ocr.get('misses')} reads failed.",
            "A number probe that cannot read returns 0, and a guard cannot tell that "
            "apart from a HUD that genuinely says 0. Every threshold on it has been "
            "silently comparing against zero for the whole run.",
            "Run voltage_doctor and check the `ocr` line -- on most distributions the "
            "language data is a separate package from tesseract itself. If OCR is "
            "healthy, the region is wrong: voltage_capture, find the digits, and set the "
            "region tightly around them. Set `invert: true` for dark text on light.",
            {"probes": number_probes, **ocr},
        ))

    return findings


def _looks_truncated(burst: str) -> bool:
    """A burst whose last action is cut off mid-write.

    `k:ctrl+t;w:120;w` is what a reply that hit max_tokens looks like: every action well
    formed except the final one, which has no payload. Distinguishing it from genuinely
    malformed output matters because the fix is completely different -- one is a token
    budget, the other is a missing grammar.
    """
    tail = burst.rsplit(";", 1)[-1].strip()
    return bool(tail) and (":" not in tail or not tail.split(":", 1)[1].strip())


def _repetition_ratio(burst: str) -> float:
    """Fraction of a burst taken up by its single most common action.

    1.0 means every action is identical. Anything above ~0.6 in a burst of real length is
    padding rather than a plan -- a genuine sequence varies.
    """
    actions = [p.strip() for p in burst.split(";") if p.strip()]
    if len(actions) < 4:
        return 0.0
    return Counter(actions).most_common(1)[0][1] / len(actions)


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
