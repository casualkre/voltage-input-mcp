"""Authoring reference returned by the `voltage_reference` tool.

Kept as data rather than prose in a docstring so the MCP layer can serve slices of it,
and so it stays in one place when the DSL changes.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BURST_REFERENCE", "GUARD_REFERENCE", "EXAMPLE_PLAYBOOK"]


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
        "probe(id, default=0.0)": "value of a declared probe, always 0.0-1.0",
        "var(name, default=None)": "a run variable",
        "elapsed()": "seconds since entering the current state",
        "cycles()": "loop cycles spent in the current state",
        "run_elapsed() / run_cycles()": "totals for the whole run",
        "changed(threshold=0.02)": "the screen moved materially since the last frame",
        "stalled(seconds=3.0)": "the screen has not changed for this long",
        "last_burst_ok()": "the previous burst executed without being refused",
        "rejections()": "how many bursts the governor has refused this run",
        "abs min max len int float round bool any all": "plain helpers",
    },
    "namespaces": {
        "vars.<name>": "a run variable; missing names are None, not an error",
        "obs.scene / obs.labels / obs.text / obs.flags / obs.n": "this cycle's observation",
        "probes.<id>": "raw probe value",
        "state.cycles / state.elapsed / state.name": "current state counters",
        "run.cycles / run.elapsed / run.bursts / run.rejections": "run counters",
    },
    "examples": [
        "sees('address bar') and vars.attempts < 3",
        "probe('health') < 0.25",
        "stalled(4.0) or cycles() > 12",
        "text('permission denied') or flag('error')",
        "not sees('loading indicator') and sees('file list')",
    ],
    "gotchas": [
        "A guard that tests sees('X') where 'X' is not in that state's `watch` list can "
        "never be true -- the vision grammar cannot emit a label outside `watch`. "
        "voltage_validate_playbook warns about this.",
        "Ordered comparisons against an unset variable are False rather than an error, so "
        "vars.attempts < 3 is safe before `attempts` is first assigned.",
        "Transitions are checked in order and the first true one wins. Put specific "
        "conditions before general ones.",
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
