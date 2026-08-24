"""Authoring reference returned by the `voltage_reference` tool.

Kept as data rather than prose in a docstring so the MCP layer can serve slices of it,
and so it stays in one place when the DSL changes.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BURST_REFERENCE", "GUARD_REFERENCE", "EXAMPLE_PLAYBOOK",
    "BURST_COOKBOOK", "LEARNING_LOOP", "CONTINUOUS_CONTROL",
]


CONTINUOUS_CONTROL: dict[str, Any] = {
    "read_this_if": (
        "You are about to drive anything with timing in it -- a game, a drag, a slider, "
        "an animation, a progress bar you have to catch. Also read it if a run 'worked' "
        "but felt like a series of guesses rather than control."
    ),
    "the_problem": (
        "A decision costs 150-600 ms. If every input waits on one, the effective input "
        "rate is 2 Hz no matter how fast the inputs themselves are. Bursts fix this "
        "*within* a decision -- one decision, forty inputs -- but between decisions "
        "nothing reacts. In a game that is the difference between flying a character and "
        "jumping off and hoping. The fast layer is what fills the gap, and a playbook "
        "that declares no probes and no reflexes is not using it at all."
    ),
    "three_layers": {
        "decision (~2 Hz)": "The actuator model. Chooses *what to do next* from a "
                            "screenshot and a brief. Emits a burst.",
        "burst (ms precision)": "A timed programme executed with no model in the loop. "
                                "Scheduled against a monotonic cursor, so a 40-action "
                                "burst lands within a few ms of nominal.",
        "reflex (~20 Hz)": "Guards over probes, evaluated in its own loop between "
                           "decisions. No model, no vision, microseconds per rule. This "
                           "is the only layer that can react to something that happens "
                           "in less than half a second.",
    },
    "probes_first": (
        "A reflex is only as good as what it can measure, and the vision model is far too "
        "slow to be in that path. Declare a probe for every quantity a fast rule needs. "
        "`number` probes read a HUD figure through OCR on a background worker -- a guard "
        "reads the last value in microseconds and never waits. Each also publishes a "
        "derivative, so `rate('meters')` is descent speed with no differencing in the "
        "playbook."
    ),
    "do_versus_hold": {
        "do": "A one-shot. Guard goes true, burst runs, cooldown starts. For discrete "
              "reactions: press the button, dodge, dismiss the dialog.",
        "hold": "A latch. Guard goes true and the keys go *down*, and stay down across "
                "frames and across decisions until the guard goes false. For anything "
                "continuous: running, mouse-look, holding a drag open, charging an "
                "attack.",
        "why_it_matters": "'hold w while the target is ahead' written as a `do` is a "
                          "stutter of taps at reflex rate, which reads on screen as a "
                          "character twitching in place rather than moving. It is the "
                          "single most common way a playbook looks right and behaves "
                          "wrong.",
    },
    "hysteresis": (
        "A latch whose guard sits on its threshold flips at reflex rate. Give it a "
        "`release_when` wider than `when` so a dead band exists in which neither is true "
        "and the latch keeps its state -- a Schmitt trigger. `when: probe('m') < 90` with "
        "`release_when: probe('m') > 110`. `min_hold_ms` is the blunter alternative. "
        "voltage_diagnose reports this as `latch_chatter`."
    ),
    "expressions_in_bursts": (
        "A playbook-authored burst may contain {expression} holes, evaluated against the "
        "live context when it fires. This is what makes an action proportional to an "
        "error rather than a constant: `r:{clamp((probe('tx') - 960) * 0.4, -300, 300)},0` "
        "is a steering controller in one line, running at reflex rate. Holes may appear in "
        "reflex `do`, `on_enter` and `on_exit` -- not in what the actuator emits, which "
        "stays literal and grammar-bounded. Always clamp a term that becomes a coordinate; "
        "an unclamped servo overshoots off-screen and the fire is dropped."
    ),
    "worked_example": {
        "task": "Air-control a falling character: brake before impact, steer toward a "
                "landing target, and keep running while airborne.",
        "probes": [
            {"id": "meters", "type": "number", "region": {"x": 1650, "y": 60, "w": 150, "h": 40},
             "ocr_interval_ms": 100},
            {"id": "mph", "type": "number", "region": {"x": 1650, "y": 110, "w": 150, "h": 40},
             "ocr_interval_ms": 100},
        ],
        "reflex": [
            {"_comment": "A latch: hold forward for the whole descent, not a tap per tick.",
             "id": "glide", "when": "probe('meters') > 40",
             "release_when": "probe('meters') < 15", "hold": "w"},
            {"_comment": "A one-shot: brake once, low and falling fast. Exclusive, so a "
                         "stale decision cannot override it.",
             "id": "brake", "when": "probe('meters') < 90 and rate('meters') < -50",
             "do": "k:space", "cooldown_ms": 250, "priority": 10},
            {"_comment": "Proportional steer. The hole makes the correction match the "
                         "error; clamp keeps it on screen.",
             "id": "steer", "when": "abs(probe('mph')) > 5",
             "do": "r:{clamp((960 - probe('mph') * 3), -250, 250)},0", "cooldown_ms": 60},
        ],
        "notes": "The decision layer never sees any of this. It is choosing where to go "
                 "next while the fast layer flies. That division is the whole design.",
    },
    "checklist": [
        "Every quantity a fast rule needs is a probe, not a question for the vision model.",
        "Anything continuous is a `hold`, not a repeated `do`.",
        "Every `hold` has either a `release_when` wider than its `when`, or a min_hold_ms.",
        "Every hole that becomes a coordinate is wrapped in clamp().",
        "policy.max_burst_ms is short. A burst that runs 500 ms is 500 ms in which no "
        "reflex can reach the device -- voltage_diagnose reports that as reflex_starved.",
        "After the run, check voltage_diagnose's `reflex` block: measured_hz should be "
        "close to requested, fires should be non-zero, starved should be zero.",
    ],
}


BURST_REFERENCE: dict[str, Any] = {
    "what": (
        "A burst is one timed programme of input events, executed with no model in the "
        "loop. This is where the speed comes from: one decision produces many inputs, so "
        "the input rate is set by the burst rather than by model latency. Chain the whole "
        "obvious sequence into one burst instead of emitting one action per cycle."
    ),
    "syntax": "actions separated by ';' -- for example: g:0;c:l;w:150;t:\"README.md\";k:enter",
    "actions": {
        "g:N": "move the pointer to the centre of observed element N (preferred)",
        "m:X,Y": "move the pointer to desktop pixel X,Y",
        "r:+DX,-DY": "move the pointer by a relative delta",
        "c:l": "left click. c:r right, c:m middle, c:l2 double-click, c:l3 triple",
        "p:l / e:l": "press / release a mouse button (for drags)",
        "k:ctrl+shift+t": "press a key chord; released in reverse order",
        "d:shift / u:shift": "hold / release a key across actions",
        "t:\"text\"": "type literal text",
        "s:-3 / s:+3": "scroll down / up by 3 detents. h: for horizontal",
        "w:120": "wait 120 ms",
        ".": "an empty burst -- do nothing this cycle and look again",
    },
    "notes": [
        "Prefer g:N over m:X,Y. The element index comes from the SEEN list, and a small "
        "model transcribes an index far more reliably than a four-digit coordinate pair.",
        "Put w: after anything that opens a menu, dialog or new window. Without it the "
        "next action lands before the UI exists.",
        "A burst that ends with a key still held (d: without u:) keeps it held into the "
        "next cycle. That is intentional and useful for games; the executor force-releases "
        "anything held past policy.max_hold_ms.",
    ],
    "actuator_reply_format": (
        "The actuator replies with three '|'-separated fields: <burst>|<next state>|<note>. "
        "Use '.' for the burst to wait, '.' for the state to stay put. Under llama.cpp a "
        "GBNF grammar makes any other shape unrepresentable."
    ),
}


GUARD_REFERENCE: dict[str, Any] = {
    "what": (
        "Guard expressions decide transitions and reflexes. They are evaluated by the "
        "runtime, not by a model, so they are the part of a Playbook that behaves "
        "deterministically. Python-like syntax, restricted to comparisons, boolean logic, "
        "arithmetic and the functions below."
    ),
    "functions": {
        "sees(label, min_conf=0.4)": "the vision model reported this label this cycle",
        "count(label)": "how many elements carry this label",
        "conf(label)": "highest confidence for this label, 0.0 if absent",
        "text(pattern)": "case-insensitive regex over text the vision model read",
        "flag(name)": "a vision flag is set: dialog, loading, error, empty, fullscreen, occluded",
        "probe(id, default=0.0)": "value of a declared probe. 0.0-1.0 for pixel, "
                                  "brightness, diff and template probes; the real "
                                  "measured figure for a `number` probe",
        "rate(id)": "rate of change per second of a number probe -- descent speed, income "
                    "per second, how fast a bar is filling",
        "var(name, default=None)": "a run variable",
        "held(name)": "a key or button is down right now; 'btn:l' for mouse buttons",
        "latched(rule_id)": "a hold reflex is currently engaged",
        "elapsed()": "seconds since entering the current state",
        "cycles()": "loop cycles spent in the current state",
        "run_elapsed() / run_cycles()": "totals for the whole run",
        "changed(threshold=0.02)": "the screen moved materially since the last frame",
        "stalled(seconds=3.0)": "the screen has not changed for this long",
        "last_burst_ok()": "the previous burst executed without being refused",
        "rejections()": "how many bursts the governor has refused this run",
        "clamp(v, lo, hi)": "constrain a value -- use it on any term that becomes a "
                            "coordinate, or a servo will eventually send the pointer "
                            "off-screen",
        "sign(v)": "-1, 0 or 1. Turns an error into a direction",
        "abs min max len int float round bool any all": "plain helpers",
    },
    "namespaces": {
        "vars.<name>": "a run variable; missing names are None, not an error",
        "obs.scene / obs.labels / obs.text / obs.flags / obs.n": "this cycle's observation",
        "probes.<id>": "raw probe value",
        "state.cycles / state.elapsed / state.name": "current state counters",
        "run.cycles / run.elapsed / run.bursts / run.rejections": "run counters",
        "run.reflex_hz / run.reflex_fires": "what the fast layer is actually achieving",
    },
    "examples": [
        "sees('address bar') and vars.attempts < 3",
        "probe('health') < 0.25",
        "probe('meters') < 90 and rate('meters') < -50",
        "stalled(4.0) or cycles() > 12",
        "text('permission denied') or flag('error')",
        "not sees('loading indicator') and sees('file list')",
        "not held('w') and probe('clear ahead') > 0.7",
    ],
    "gotchas": [
        "A guard that tests sees('X') where 'X' is not in that state's `watch` list can "
        "never be true -- the vision grammar cannot emit a label outside `watch`. "
        "voltage_validate_playbook warns about this.",
        "Ordered comparisons against an unset variable are False rather than an error, so "
        "vars.attempts < 3 is safe before `attempts` is first assigned.",
        "Transitions are checked in order and the first true one wins. Put specific "
        "conditions before general ones.",
        "A reflex guard runs at ~20 Hz and sees the *cached* observation -- sees() in a "
        "reflex is up to a decision old. Probes are fresh; vision is not. Key fast rules "
        "off probes.",
        "A number probe that cannot read returns 0.0, which is indistinguishable from a "
        "HUD reading zero. Check voltage_doctor's `ocr` line before trusting a threshold.",
    ],
}


EXAMPLE_PLAYBOOK: dict[str, Any] = {
    "name": "open_downloads_in_dolphin",
    "goal": "Open the file manager and navigate to the Downloads folder, without "
            "deleting anything or confirming any dialog.",
    "initial": "launch",
    "vars": {"attempts": 0},
    "policy": {
        "dry_run": True,
        "allow_verbs": ["g", "c", "k", "t", "w", "m"],
        "require_target_element": False,
        "max_actions_per_burst": 12,
        "max_inputs_per_second": 40,
        "deny_labels": ["delete", "trash", "remove", "confirm", "empty trash", "move to trash"],
        "deny_text": ["rm\\s+-[rf]", "\\bsudo\\b"],
    },
    "budget": {
        "max_cycles": 60,
        "max_seconds": 90,
        "max_rejections": 6,
        "idle_abort_s": 20,
    },
    "perception": {
        "mode": "on_change",
        "max_elements": 5,
        "downscale_to": [896, 504],
    },
    "states": {
        "launch": {
            "brief": "Open the application launcher and start the file manager.",
            "watch": ["application launcher", "search field", "file manager icon"],
            "on_enter": "k:meta;w:400",
            "transitions": [
                {"when": "sees('search field')", "to": "type_name"},
                {"when": "cycles() > 6", "to": "@failure", "note": "launcher never opened"},
            ],
        },
        "type_name": {
            "brief": "Type 'dolphin' into the search field, then press Enter.",
            "watch": ["search field", "file manager icon", "file list"],
            "hint": "The search field already has focus. Do not click it first.",
            "transitions": [
                {"when": "sees('file list')", "to": "navigate", "inc": {"attempts": 1}},
                {"when": "cycles() > 8", "to": "@failure"},
            ],
        },
        "navigate": {
            "brief": "Focus the location bar with ctrl+l, type the Downloads path, "
                     "press Enter.",
            "watch": ["location bar", "file list", "error message"],
            "on_enter": "k:ctrl+l;w:200",
            "transitions": [
                {"when": "text('Downloads')", "to": "@success"},
                {"when": "sees('error message')", "to": "@failure"},
                {"when": "cycles() > 10", "to": "@failure", "note": "navigation stalled"},
            ],
        },
    },
    "success_when": "text('Downloads') and not flag('loading')",
    "notes": "Written for KDE. On another desktop, change the launcher key in "
             "launch.on_enter and the labels in each `watch` list.",
}


BURST_COOKBOOK: dict[str, Any] = {
    "principle": (
        "A burst costs exactly one decision no matter how many inputs it contains. A "
        "one-action burst therefore pays full model latency per input and throws the "
        "whole design away. Chain the entire sequence you are already confident about; "
        "stop only where you genuinely need to look again."
    ),
    "the_three_rules": [
        "Chain everything that does not depend on seeing the result.",
        "Put w: after anything that opens, closes, loads or animates. The next action "
        "lands before the UI exists otherwise -- the most common cause of a run that "
        "looks like model confusion and is actually just impatience.",
        "Stop the burst at the point where you would need to look at the screen to know "
        "what to do next. That is where the next cycle begins.",
    ],
    "timing": {
        "menu or dropdown opens": "w:150 to w:300",
        "window or dialog opens": "w:400 to w:800",
        "application launches": "w:1000+, or better, a state with a `watch` for it",
        "typing then Enter": "w:80 after the text, so the field registers it",
        "between repeated game inputs": "w:30 to w:80; below ~16 ms many games drop them",
        "after a click that loads": "do not guess -- end the burst and let a guard decide",
    },
    "desktop_patterns": {
        "click a seen element": "g:0;c:l",
        "open a menu and pick an item": "g:0;c:l;w:200   (then a new cycle to see the menu)",
        "fill a field": 'g:0;c:l;w:80;t:"value";k:tab',
        "save with a name": 'k:ctrl+s;w:500;t:"report.txt";w:80;k:enter',
        "select all and replace": 'k:ctrl+a;w:50;t:"new text"',
        "drag between two seen elements": "g:0;p:l;w:60;g:1;w:60;e:l",
        "scroll to find something": "s:-3;w:120   (then look again)",
        "dismiss a dialog": "k:esc;w:200",
    },
    "game_patterns": {
        "principle": (
            "Games need held keys and rhythm, not discrete presses. Use d:/u: to hold "
            "across time inside one burst, and reflexes for anything that must react "
            "faster than a decision."
        ),
        "walk forward briefly": "d:w;w:400;u:w",
        "strafe while attacking": "d:a;w:100;c:l;w:150;c:l;w:100;u:a",
        "sprint jump": "d:shift;d:w;w:200;k:space;w:300;u:w;u:shift",
        "mine or hold-attack (Minecraft)": "p:l;w:900;e:l",
        "place a block while looking": "r:+0,-60;w:80;c:r;w:100;r:+0,+60",
        "hotbar then use": "k:3;w:60;c:r",
        "turn the camera": "r:+220,0;w:100   (relative; pointer_mode must be relative)",
        "dodge roll": "d:shift;k:s;w:120;u:shift",
        "combo with timing": "c:l;w:220;c:l;w:220;c:l",
    },
    "antipatterns": [
        "One action per burst. Costs full latency per input.",
        "A click with no preceding g: or m:. The pointer is wherever it was left.",
        "Chaining past a point where the screen must change first -- the rest of the "
        "burst lands in the wrong context.",
        "m:X,Y when g:N is available. Transcribing four-digit coordinates is the thing "
        "small models are worst at; picking an index is what they are best at.",
        "Long t:\"...\" on Linux with a non-US keyboard layout. Scancodes are "
        "layout-dependent; the clipboard path handles it, but keep text short anyway.",
        "Holding a key at the end of a burst without meaning to. It stays held into the "
        "next cycle (deliberately, for games) until max_hold_ms force-releases it.",
    ],
    "worked_example": {
        "task": "Rename a file in a file manager",
        "wrong": [
            "cycle 1: g:0   (just moves)",
            "cycle 2: c:l   (just clicks)",
            "cycle 3: k:f2  (just renames)",
            "-> 3 decisions, ~1.2 s of model time for 3 inputs",
        ],
        "right": [
            'cycle 1: g:0;c:l;w:120;k:f2;w:250;k:ctrl+a;t:"newname.txt";k:enter',
            "-> 1 decision, 7 inputs, ~600 ms of which most is the burst itself",
        ],
    },
}


LEARNING_LOOP: dict[str, Any] = {
    "why": (
        "The first playbook for an unfamiliar target is almost never right, and the "
        "failures are informative in specific ways. This is the loop that converts a "
        "failed run into a better playbook rather than a retry."
    ),
    "loop": [
        {
            "step": 1,
            "name": "Look before writing",
            "do": "voltage_capture, then voltage_observe with candidate labels.",
            "why": "The single most common failure is a `watch` label the vision model "
                   "cannot recognise. Every guard testing it is then dead, and the run "
                   "stalls with no obvious cause. Two minutes here saves a whole run.",
        },
        {
            "step": 2,
            "name": "Write small",
            "do": "Three or four states, each with one imperative brief and an escape "
                  "transition (cycles() > N to @failure).",
            "why": "A stuck state with no escape burns the entire budget before telling "
                   "you anything. Escapes make failures fast and legible.",
        },
        {
            "step": 3,
            "name": "Validate, then dry run",
            "do": "voltage_validate_playbook, then voltage_run with dry_run=true.",
            "why": "Validation catches dead guards and unreachable states statically. "
                   "The dry run exercises the real models against the real screen and "
                   "journals every burst without touching anything.",
        },
        {
            "step": 4,
            "name": "Diagnose",
            "do": "voltage_diagnose(run_id).",
            "why": "Do not read the raw journal first. Diagnose computes what the "
                   "journal implies -- labels never seen, guards that never fired, "
                   "bursts that ran and changed nothing -- and names the edit for each.",
        },
        {
            "step": 5,
            "name": "Change one thing",
            "do": "Apply the highest-severity finding, then re-run.",
            "why": "Changing several things at once makes the next diagnosis "
                   "uninterpretable. Blockers first: they mask everything below them.",
        },
        {
            "step": 6,
            "name": "Record what you learned",
            "do": "voltage_learn(target='minecraft', note='...').",
            "why": "Lessons persist across sessions and are keyed by target, so the next "
                   "playbook for the same game starts from what this one discovered. "
                   "Call voltage_lessons before writing a playbook for a familiar target.",
        },
        {
            "step": 7,
            "name": "Go live, still watching",
            "do": "dry_run=false, then poll voltage_status. Use voltage_steer to correct "
                  "a live run instead of restarting it.",
            "why": "steer injects a supervisor note into the actuator's prompt, forces a "
                   "state, or updates variables -- without losing the run's progress.",
        },
    ],
    "when_things_go_wrong": {
        "nothing happens on screen": "Distinguish 'burst never ran' from 'burst ran and "
            "did nothing'. voltage_diagnose separates them. The first is policy or "
            "grammar; the second is focus, pointer_mode, or an app ignoring synthetic "
            "input -- completely different fixes.",
        "a guard never fires": "The label is probably not in the vision model's "
            "vocabulary. voltage_observe it directly. Rename to what it looks like.",
        "it clicks the wrong thing": "Restrict allow_verbs to exclude 'm' so the actuator "
            "must use g:<index>, and set require_target_element so a click that lands on "
            "nothing is refused rather than executed.",
        "it is too slow": "max_elements is the dominant cost at ~500 ms per reported "
            "element. Then perception.mode='on_change'. Do not shrink downscale_to -- it "
            "measures slower, because a blurrier image makes the model emit more tokens.",
        "it does one action at a time": "Put an explicit burst example in the state's "
            "hint. Small models copy a shape far more reliably than they follow an "
            "abstract instruction.",
        "the governor keeps refusing": "Read the rule name. It is usually correct and "
            "the brief is asking for something the policy forbids.",
        "it works then drifts": "Add probes and reflexes for the recurring corrections. "
            "Anything you find yourself steering repeatedly belongs in the playbook.",
    },
    "games_specifically": [
        "Use allow_keys as an allowlist of exactly the game's keys. Anything omitted is "
        "absent from the grammar, so the actuator physically cannot press Escape and "
        "open the pause menu.",
        "Set perception.mode='cadence' with cadence 3-5. Vision answers slow questions "
        "('is there a boss'); probes and reflexes handle the fast loop.",
        "Read the HUD with probes, not the vision model. A health bar is a region colour "
        "check costing microseconds; asking a VLM costs half a second.",
        "pointer_mode='relative' for anything with mouse-look. Absolute positioning "
        "fights the game's own camera control.",
        "Expect to iterate on probe coordinates. Use voltage_capture, find the pixel, "
        "and record it as a lesson so the next playbook has it.",
    ],
}
