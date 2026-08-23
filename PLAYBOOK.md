# Writing a Playbook

A Playbook is what you hand to the local models. It is not a prompt and not a goal — it is
a **state machine** whose transitions are evaluated by the runtime, not by a model.

The division is the whole point:

- **You** decide what the states are, what to look for in each, what is permitted, and
  exactly when to move on. That is the thinking, and it happens once.
- **The vision model** answers one closed question per cycle: *of these specific things,
  which are on screen and where?*
- **The actuator** answers one closed question per cycle: *given that, which inputs?*

Neither of them plans. Write accordingly.

---

## Minimum viable Playbook

```json
{
  "name": "my_task",
  "goal": "Human-readable. Never shown to the small models.",
  "initial": "start",
  "states": {
    "start": {
      "brief": "Click the Save button.",
      "watch": ["save button", "dialog"],
      "transitions": [
        { "when": "not sees('dialog')", "to": "@success" },
        { "when": "cycles() > 8", "to": "@failure" }
      ]
    }
  }
}
```

`brief` and `watch` are the two fields that matter most. Everything else has a sensible
default.

---

## The workflow that works

1. **`voltage_reference`** — the DSL, schema, and guard functions. Once per session.
2. **`voltage_doctor`** — is the machine ready.
3. **`voltage_capture`** — look at the screen yourself. Do not write a Playbook blind.
4. **`voltage_observe(watch=[...])`** — check the vision model can actually find your
   labels. **This is the step people skip and then debug for an hour.** If an element does
   not come back here, a `sees()` guard on it can never fire.
5. **`voltage_validate_playbook`** — compiles guards, parses bursts, checks the graph.
   Reports every error at once, plus warnings for transitions that can never fire.
6. **`voltage_run(dry_run=true)`** — everything except injection.
7. **`voltage_journal`** — read what it *would* have done.
8. **`voltage_run(dry_run=false)`** — only now.

---

## Writing `brief` and `watch`

`brief` is the entire instruction the actuator receives, alongside the observation. It has
no memory of the goal and cannot infer what the state is *for*.

| | |
|---|---|
| ✅ | `"Type the filename into the focused field, then press Enter."` |
| ❌ | `"Handle the save dialog appropriately."` |
| ✅ | `"Click the Downloads entry in the sidebar."` |
| ❌ | `"Navigate to where the user's downloads live."` |

Imperative. Concrete nouns drawn from `watch`. One action's worth of instruction.

`watch` is a **closed vocabulary** — the vision grammar permits these labels and a small
generic set (`dialog`, `button`, `text field`, `menu`, `list item`, `error message`, …)
and *nothing else*. Rules:

- Use visually descriptive names: `"blue Save button"` beats `"submit control"`.
- Name what you will test for. A label nobody guards on is wasted decode time.
- Keep it to 3–6 entries. Every label is tokens in the prompt and options in the grammar.
- If `voltage_observe` cannot find it, rename it before writing the state.

`hint` is optional and goes in alongside `brief`. Use it for the thing that is true but
not visible: `"The field already has focus — do not click it first."`

---

## Transitions

Evaluated **in order**; the first true one wins. Put specific before general.

```json
"transitions": [
  { "when": "sees('error message')", "to": "recover" },
  { "when": "text('Downloads')",      "to": "@success" },
  { "when": "cycles() > 10",          "to": "@failure", "note": "stalled" }
]
```

Targets are a state name or `@success` / `@failure` / `@stop`.

`set` and `inc` update run variables on the way out:

```json
{ "when": "sees('retry button')", "to": "start", "inc": { "attempts": 1 } }
```

**Always give every state an escape.** A `cycles() > N` or `stalled(N)` transition to
`@failure` is not pessimism, it is the difference between a run that ends and a run that
burns its whole budget staring at a screen that will never change.

The actuator may *propose* a transition, but only to a target this state already declares
— the grammar contains exactly those names. It cannot invent control flow.

---

## Guards

Python-like, restricted to comparisons, boolean logic, arithmetic, and these functions:

```
sees(label, min_conf=0.4)     the vision model reported this label
count(label) / conf(label)    how many / highest confidence
text(pattern)                 case-insensitive regex over text read on screen
flag(name)                    dialog | loading | error | empty | fullscreen | occluded
probe(id, default=0.0)        a declared probe, always 0.0–1.0
var(name, default=None)       a run variable
elapsed() / cycles()          seconds / loop cycles in the current state
run_elapsed() / run_cycles()  totals for the run
changed(threshold=0.02)       the screen moved materially
stalled(seconds=3.0)          it has not moved for this long
last_burst_ok() rejections()  did the last burst run; how many were refused
```

Namespaces: `vars.x`, `obs.scene`, `obs.labels`, `obs.text`, `obs.flags`, `obs.n`,
`probes.id`, `state.cycles`, `run.bursts`.

```
sees('address bar') and vars.attempts < 3
probe('health') < 0.25
stalled(4.0) or cycles() > 12
text('permission denied') or flag('error')
```

Missing variables compare False rather than raising, so `vars.attempts < 3` is safe before
`attempts` is ever set.

---

## Probes and reflexes — where the speed lives

A probe is a model-free screen measurement costing microseconds. Anything reducible to
"is this pixel red" or "has this region changed" belongs here, not in a question to the
VLM.

```json
"probes": [
  { "id": "health", "type": "region_mean",
    "region": {"x": 120, "y": 1010, "w": 8, "h": 20},
    "expect": "#c0392b", "tolerance": 60, "channel": "r" },
  { "id": "motion", "type": "region_diff",
    "region": {"x": 460, "y": 200, "w": 1000, "h": 600} }
]
```

Types: `pixel`, `region_mean`, `brightness`, `region_diff`, `template`. All return 0.0–1.0.
Use `voltage_capture` to find real coordinates for your display.

A reflex is a rule that fires on probes with **no model in the path at all**, between
actuator decisions:

```json
"reflex": [
  { "id": "heal", "when": "probe('health') < 0.25", "do": "k:q;w:80",
    "cooldown_ms": 2500, "priority": 10, "exclusive": true }
]
```

`exclusive: true` suppresses the actuator that cycle. Higher `priority` wins when several
fire. This is how a 2 Hz decision loop produces sub-100 ms reactions.

---

## Bursts

Write bursts in `on_enter`, `on_exit`, and reflex `do`. The actuator writes its own.

```
g:2;c:l;w:150;t:"README.md";k:enter
```

| | |
|---|---|
| `g:N` | move to the centre of observed element N — **prefer this** |
| `m:X,Y` | move to a desktop pixel |
| `c:l` | click (`c:r`, `c:m`, `c:l2` double) |
| `k:ctrl+s` | key chord |
| `t:"text"` | type |
| `d:shift` / `u:shift` | hold / release |
| `s:-3` | scroll |
| `w:120` | wait 120 ms |
| `.` | do nothing this cycle |

**Chain the whole obvious sequence into one burst.** That is the entire performance
argument: one decision, many inputs. A Playbook whose bursts are one action long has
thrown the advantage away.

**Put `w:` after anything that opens a menu, dialog or window.** Without it the next action
lands before the UI exists — the most common cause of a run that looks like the model is
confused when it is actually just early.

---

## Policy

Restrictive by default; opt into capability.

```json
"policy": {
  "dry_run": true,
  "allow_verbs": ["g", "c", "k", "t", "w"],
  "allow_keys": ["w", "a", "s", "d", "space", "q"],
  "deny_labels": ["delete", "trash", "confirm", "purchase", "allow"],
  "click_allow_regions": [{"x": 0, "y": 0, "w": 1920, "h": 900}],
  "require_target_element": true,
  "max_actions_per_burst": 16,
  "max_inputs_per_second": 60
}
```

Notes that matter:

- **`allow_keys` beats `deny_keys`.** An allowlist is removed from the *grammar*, so a
  disallowed key cannot be generated at all. For a game, list the game's keys and nothing
  else — the actuator then physically cannot press Escape.
- **`deny_labels` is the important one for desktop work.** It refuses a click on anything
  *called* Delete/Confirm/Purchase/Allow, wherever it appears. Region fencing cannot do
  this, because the danger of a confirmation dialog is that it shows up where you did not
  predict. Matching is word-boundary, so "Send to Trash" is refused and "Buyer name" is not.
- **`require_target_element: true`** forces every click onto something the vision model
  actually reported. Excellent for desktop UI, wrong for games and canvases.
- **Refusal is whole-burst.** One bad action refuses all of it.

Budgets guarantee termination:

```json
"budget": { "max_cycles": 240, "max_seconds": 180, "max_rejections": 12,
            "idle_abort_s": 25, "deadman_s": 6 }
```

`max_rejections` is a signal, not just a limit — hitting it means your policy and what the
actuator wants to do disagree, and the Playbook needs rethinking rather than a bigger cap.

---

## Perception tuning

```json
"perception": { "mode": "on_change", "change_threshold": 0.015,
                "max_elements": 5, "downscale_to": [896, 504] }
```

| mode | use when |
|---|---|
| `on_change` | default. Skips the VLM when the screen has not moved. |
| `cadence` | fast loops where vision answers slow questions — set `cadence: 4` |
| `always` | the screen changes constantly and every frame matters |
| `never` | probes and reflexes only; pure timing sequences |

`downscale_to` is the dominant cost knob: 896×504 is ~400 image tokens; halving the
dimensions roughly quarters prefill. Drop to `[640, 360]` for games. Per-state overrides
are allowed — spend vision where it matters.

---

## Debugging

| Symptom | Cause | Fix |
|---|---|---|
| `sees()` never fires | label not in the model's vocabulary | `voltage_observe` and rename |
| validation warns "not in watch" | guard tests a label the grammar can't emit | add it to `watch` |
| clicks land slightly off | coordinate space | check `voltage_observe` centres against `voltage_capture` |
| clicks land in the wrong place entirely | actuator used `m:` and mistyped | restrict `allow_verbs` to force `g:` |
| burst runs before the UI appears | no settle | add `w:` after the opening action |
| many governor refusals | policy vs. intent mismatch | `voltage_journal(only_refused=true)` |
| run ends "screen has not changed" | stuck | add a `stalled()` transition |
| typed text comes out wrong | scancodes vs. keyboard layout | install `wl-clipboard`; `text_mode="auto"` |
| cursor does not move at all | pointer classification | `voltage_calibrate`, then `pointer_mode="relative"` |

Live correction without restarting:

```
voltage_steer(hint="The dialog opens behind the main window — alt+tab to it first")
voltage_steer(force_state="recover")
voltage_steer(dry_run=false)
```

---

## Checklist

- [ ] Every state has an escape transition (`cycles()`, `stalled()`, or `timeout_s`)
- [ ] Every `sees()` label appears in that state's `watch`
- [ ] `voltage_observe` actually finds those labels
- [ ] `on_enter` bursts include `w:` after anything that opens UI
- [ ] `deny_labels` covers the destructive controls this task could encounter
- [ ] `allow_keys` is set if this is a game
- [ ] There is a path to `@success`
- [ ] Validated, then dry-run, then journal read — before `dry_run=false`
