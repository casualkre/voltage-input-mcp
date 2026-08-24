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

Types: `pixel`, `region_mean`, `brightness`, `region_diff`, `template`, `number`. The first
five return 0.0–1.0. Use `voltage_capture` to find real coordinates for your display.

A `number` probe OCRs a HUD figure into a real value:

```json
{ "id": "meters", "type": "number",
  "region": {"x": 1642, "y": 96, "w": 190, "h": 44}, "ocr_interval_ms": 90 }
```

Each number probe also publishes a derivative, so `rate('meters')` is descent speed with
no differencing in the Playbook.

### Teach it the digits — don't leave it on OCR

By default a number probe uses tesseract: 80–200 ms, on a background worker, and
**approximate**. Measured against a real game HUD it read `414` as `4114` and `0` as
`636`. A reflex guarding on `probe('meters') < 90` cannot survive input that wrong, and no
amount of tuning the guard fixes a lying sensor.

A HUD is one font, one size, one place, ten glyphs. So teach it once:

```bash
voltage learn-digits meters --region 16 636 150 50 --seconds 20
```

Make the number **change** while that runs — a score climbing, an altitude counting down —
so every digit shows up. It clusters the glyph shapes, uses OCR once as a *teacher* (voted
across frames, so a single misread cannot decide a label), and saves the set.

After that the probe reads by correlation: **~1 ms, exact, and inline** rather than on a
background worker, so the value is from *this* frame. Name the glyph set after the probe
and it is picked up automatically, or set `"glyphs": "meters"` explicitly.

> When no glyph clears threshold the probe keeps its last value and stops advancing
> `<id>__age` — so a latch guarding on it goes blind and releases, rather than the probe
> inventing a number. That is the intended behaviour when a modal covers the HUD.

### Reflexes: `do` and `hold`

A reflex fires on probes with **no model in the path at all**, in its own loop at
`reflex_hz` (default 20) between actuator decisions. There are two kinds, and picking the
wrong one is the most common way a Playbook looks right and behaves wrong.

**`do` is a one-shot.** Guard goes true, burst runs, cooldown starts.

```json
{ "id": "heal", "when": "probe('health') < 0.25", "do": "k:q;w:80",
  "cooldown_ms": 2500, "priority": 10, "exclusive": true }
```

`exclusive: true` suppresses the next decision, so a burst chosen from a 300 ms-old frame
cannot override a reaction made 20 ms ago. Higher `priority` wins when several match.

**`hold` is a latch.** The keys go down on the rising edge and stay down — across frames,
across decisions — until the guard goes false.

```json
{ "id": "glide", "when": "probe('meters') > 50",
  "release_when": "probe('meters') < 25", "hold": "w, shift" }
```

Use it for anything continuous: running, mouse-look, holding a drag open, charging an
attack. Written as a repeated `do`, "hold W while the target is ahead" becomes a stutter of
taps at 20 Hz, which on screen is a character twitching in place rather than moving.

`release_when` is a *separate, wider* falling threshold. The gap between the two is a dead
band in which neither fires and the latch keeps its state — a Schmitt trigger. Without it a
guard sitting on its threshold flips at reflex rate; `voltage_diagnose` reports that as
`latch_chatter`. `min_hold_ms` is the blunter alternative.

Latches are released on state exit, on pause and on stop, and are exempt from the
`max_hold_ms` watchdog — a guard re-checked twenty times a second is stricter supervision
than a timer.

### Expressions inside bursts

A Playbook-authored burst may contain `{expression}` holes, evaluated when it fires:

```json
{ "id": "steer", "when": "probe('mph') > 25",
  "do": "r:{clamp((probe('mph') - 60) * 4, -220, 220)},0", "cooldown_ms": 70 }
```

That is a proportional controller in one line, running at reflex rate. Holes work in reflex
`do`, `on_enter` and `on_exit` — not in what the actuator emits, which stays literal and
grammar-bounded, because that is what makes a bad burst unrepresentable rather than
unlikely.

**Always `clamp()` a term that becomes a coordinate.** An unclamped servo overshoots off
the edge of the screen and the fire is dropped rather than the pointer being sent
somewhere arbitrary.

See `examples/air_control.json` for all of this in one place, and
`voltage_reference(section='control')` for the full treatment.

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

`max_elements` is the dominant cost knob, at roughly 500 ms per reported element. Set it
to the number your guards actually test for and no higher. Per-state overrides are
allowed — spend vision where it matters.

`downscale_to` is **not** a useful speed knob, despite looking like the obvious one. Both
models are decode-bound at ~22 ms/token and prefill is ~28 ms across the whole size range,
so a smaller image saves nothing — and it costs, because a blurrier image makes the model
less certain and it emits *more* tokens. Measured: 448×252 came out 2.5× slower than
896×504. Use the largest size that fits your VRAM.

On a 16:9 display the sizes that land exactly on the model's 28-pixel token grid are the
multiples of 448×252 — so `[896, 504]` and `[448, 252]`. Others come out a few pixels off
it and the model resizes internally; grounding is unaffected either way.

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
| mouse-look does nothing in a game | absolute motion fights pointer lock | set `policy.pointer_mode: "relative"` |
| everything happens at ~2 Hz | no probes, no reflexes | see `no_fast_layer` in `voltage_diagnose` |
| a reflex never fires | threshold or region wrong | `reflex_never_fired` lists each probe's peak value |
| reactions arrive late | actuator bursts hog the device | lower `max_burst_ms`; see `reflex_starved` |
| a held key machine-guns | guard sitting on its threshold | add `release_when`; see `latch_chatter` |
| a number probe always reads 0 | OCR unavailable, or wrong region | `voltage doctor` → `ocr` line, then `voltage_capture` |
| the fast loop runs slower than asked | non-streaming capture backend | `voltage doctor`; `portal` streams, `kwin` does not |

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

If anything in this task has timing in it:

- [ ] `voltage reflex` shows the machine holding the rate you are about to assume
- [ ] Every quantity a fast rule needs is a probe, not a question for the vision model
- [ ] Anything continuous is a `hold`, not a repeated `do`
- [ ] Every `hold` has a `release_when` wider than its `when`, or a `min_hold_ms`
- [ ] Every `{...}` hole that becomes a coordinate is wrapped in `clamp()`
- [ ] `max_burst_ms` is short — a 500 ms burst is 500 ms in which nothing can react
- [ ] After the dry run, `voltage_diagnose`'s `reflex` block shows `fires > 0`,
      `starved == 0`, and `measured_hz` close to `requested_hz`
