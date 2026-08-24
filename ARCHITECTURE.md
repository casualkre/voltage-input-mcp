# Architecture

## The problem, precisely

A computer-use tool call costs roughly 600–1500 ms end to end: capture, upload, model
inference, decision down, one input. Any task whose inputs must be *causally ordered and
temporally close* — a game, a drag, a modal sequence, a timeline scrub — is not slow under
that model, it is impossible. The third input has to land before the first one's effect
has decayed.

The naive fix is a faster model. That is the wrong axis. A 3B vision model still costs
~300 ms, which is 40× too slow for a 8 ms input gap. You cannot close a 100× gap by
shrinking the model; you close it by **taking the model out of the inner loop**.

## The decomposition

Three tiers, each an order of magnitude faster than the one above, each strictly less
capable:

| Tier | Latency | What it decides | Runs |
|---|---|---|---|
| Orchestrator (Claude) | ~1 s | The whole plan, as a state machine | Once, up front |
| Local models | ~100–400 ms | Which burst, given a closed question | Per cycle |
| Executor + reflexes | ~1 µs–1 ms | Nothing — it executes | Per input |

The orchestrator does not participate in the loop. It writes a **Playbook**, starts the
run, then polls. Intelligence is spent once, in advance, on the structure — not per input.

## Two loops, not one

The reflex loop and the decision loop run concurrently at different rates. That is the
whole point: a decision loop cannot dodge, and it cannot hold a key down for exactly as
long as a condition lasts. Only something running an order of magnitude faster can.

```
  REFLEX LOOP  ~20 Hz                      DECISION LOOP  ~2 Hz
  no model in the path                     two small models
  ───────────────────────────              ─────────────────────────────────
  capture (portal stream, ~2 ms)  ────────▶ reads the latest sample
        │                          shared   (no second capture)
        ▼                           frame          │
  probes (numpy, ~40 µs)          ────────▶        ▼
  number probes read a cached     probes    did the screen change since
  value; OCR runs on a worker               vision last looked?
        │                                          │ yes
        ▼                                          ▼
  latches: press / release to               vision model, GBNF-constrained
  match each `hold` guard                   (~500 ms per reported element)
        │                                          │
        ▼                                          ▼
  one-shot reflexes, by priority        1. success_when / failure_when
        │                               2. transitions
        │                               3. actuator (~150–200 ms, GBNF)
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
          safety governor (~50 µs) — refuse the whole burst, or pass
                       ▼
          executor: N inputs, ms-precise → /dev/uinput → compositor
```

Ordering inside the decision cycle is deliberate: outcome guards first, then transitions —
the orchestrator's control flow, which outranks anything the small model wants — and the
actuator only when neither had something to say. An `exclusive` reflex that fired since the
last decision suppresses the next one, because letting a burst chosen from a 300 ms-old
frame override a reaction made 20 ms ago is exactly backwards.

Capture happens once, in the reflex loop. Sharing it is cheaper, but the reason it is
*correct* is that `ProbeEngine` carries history — frame deltas, region diffs, number rates
— and two loops advancing that history independently means every one of those measurements
is taken across whatever interval happened to elapse between two unrelated callers.

The executor serialises on one lock, so a reflex firing during a long actuator burst would
queue behind it. It does not: reflexes pass a short acquisition timeout and are dropped,
counted and diagnosed instead. A reaction delivered 400 ms late is not a late reaction, it
is a reaction to a situation that has already resolved, applied to whatever is happening
now.

## Where the latency actually goes

Measured against the `lean` profile on a 6 GB RTX 3050 laptop. These are the numbers the
design is built around.

| Step | Cost | Notes |
|---|---|---|
| capture (portal stream), warm | <0.05 ms | reading the newest frame from a slot |
| capture (portal), first frame | ~140 ms | portal negotiation, paid once per session |
| capture (per-call backend) | 15–60 ms | why streaming is preferred, and why `kwin` caps the reflex rate |
| probe pass, 2 region probes | ~1.1 ms | subsampled; full-resolution it was 16 ms |
| **whole reflex tick** | **~2.6 ms** | capture + probes + guards + latch update, live |
| downscale 1080p → 896×504 | ~11 ms | PIL `BOX`; the numpy block mean it replaced cost ~103 ms |
| vision prefill | ~28 ms | flat across image sizes — decode dominates |
| decode, either model | ~22 ms/token | **this is the bottleneck** |
| vision, empty result | ~0.97 s | the floor: nothing found still costs a round trip |
| vision, live at 644×364 | ~1.4 s p50 | ~21 tokens per reported element on top |
| actuator, 20-action grammar | ~450 ms p50 | dominated by how long a burst it writes |
| governor | ~50 µs | pure Python over a parsed burst |
| burst execution | as specified | 40 inputs over 500 ms costs 500 ms |

Two independent rates fall out of this. The **reflex loop** is bounded by the tick — 2.6 ms
means it could run at several hundred Hz and is held to 20 by configuration, not by
capability. The **decision loop** is bounded by the models: ~450 ms when vision is skipped,
~1.9 s when it is not. Those are not close to each other, which is the entire reason the
two are separate loops rather than one.

The design's job on the decision side is maximising the fraction of cycles that skip
vision, and minimising the element count on the ones that do not. Its job on the reflex
side is keeping the tick small enough that the rate is a choice.

### The levers, in order of impact

**0. Fewer output tokens.** Measured, not assumed: both models are decode-bound at
~22 ms/token, and prefill is ~28 ms regardless of image size. So the levers are
`max_elements` (~500 ms per element), the actuator's diagnostic `note` (55% of its
latency at the original 48-char limit), and the compact `[idx,x1,y1,x2,y2]` element
encoding (27–29% fewer tokens than the object form). The original design assumed vision
was prefill-bound and tuned micro-batches accordingly; that was wrong, and `voltage bench`
is what showed it.

The corollary bit later, and is worth stating separately: **the token ceiling must be at
least what the grammar permits.** These are two numbers describing the same limit, and
when they disagreed — a grammar allowing 20 actions, a fixed 96-token ceiling — the
sampler stopped part-way through an action and the reply did not parse, discarding the
whole cycle. It failed only on the cycles where the model had the most to say, which is
the worst possible distribution. `actuator_token_budget` derives one from the other.

Capping tokens is also not a way to make a grammar-constrained model faster. Output length
is set by the grammar; a lower ceiling does not shorten the burst, it breaks it. The way
to get a shorter burst is `max_actions_per_burst`, which regenerates the grammar.

**1. Bursts.** The unit of actuation is a programme, not an input:

```
d:shift;r:+40,0;k:space;w:120;u:shift;g:2;c:l;w:80;t:"go"
```

Nine inputs, one decision. A 40-action burst still costs one decision. This is unbounded
leverage and it is why the whole approach works at all — **input rate is decoupled from
model rate.**

**2. Reflexes and latches.** Probe-driven rules with no model in the path, evaluated in
their own loop at ~20 Hz. A one-shot reacts:

```json
{"id": "dodge", "when": "probe('danger_flash') > 0.72",
 "do": "d:shift;k:s;w:120;u:shift", "cooldown_ms": 600, "priority": 5}
```

A latch *controls* — the keys go down on the rising edge and stay down until the guard
goes false:

```json
{"id": "glide", "when": "probe('meters') > 50",
 "release_when": "probe('meters') < 25", "hold": "w, shift"}
```

The distinction is not cosmetic. Continuous behaviour expressed as a repeated one-shot is a
stutter of taps at reflex rate, which downstream is not a held key at all. `release_when`
is a separate, wider falling threshold, so a dead band exists in which neither guard fires
and the latch keeps its state — without it a guard sitting on its threshold flips 20 times
a second.

A Playbook-authored burst may also carry `{expression}` holes, evaluated at fire time
against the live context:

```json
{"id": "steer", "when": "probe('mph') > 25",
 "do": "r:{clamp((probe('mph') - 60) * 4, -220, 220)},0", "cooldown_ms": 70}
```

That makes the size of an action depend on the size of the error — a proportional
controller running at reflex rate. The actuator's own bursts stay literal: the GBNF grammar
is what makes a malformed or unsafe burst unrepresentable, and a model that could emit
expressions would be a model that could emit expressions the grammar cannot bound. The
orchestrator is trusted and its Playbook is validated up front; the 1.7B is neither.

**3. Gating perception.** Most cycles look at a screen that has not moved. A 40 µs
frame-diff decides whether to spend 300 ms on the VLM. On ordinary desktop work this skips
vision on the majority of cycles; `perception.mode` picks the policy (`on_change`,
`cadence`, `always`, `never`).

**4. Prompt-cache locality.** llama.cpp reuses the KV cache for the longest common prefix.
Prompts are therefore ordered strictly by rate of change:

```
[ static system + burst syntax ]   never changes      ─┐
[ state brief, hint, targets   ]   changes on transition│ cached
[ SEEN / VARS / LAST           ]   changes every cycle ─┘ re-prefilled
```

For the vision model this puts the **image last**, after the instruction text — the
opposite of the usual chat convention, because the image is the one thing that changes
every cycle and nothing after it can be cached.

## Constraint, not instruction

The stated problem is that small models are "stupid and need clear instructions".
Instructions are advisory. The architecture's answer is that the small models are not
instructed to behave — they are **constrained** so that misbehaving is not expressible.

Under llama.cpp, both models decode against a GBNF grammar regenerated every cycle from
the current state. The grammar masks logits so only tokens continuing a valid parse have
non-zero probability. Concretely:

```
elidx  ::= "0" | "1" | "2"                    ← exactly this cycle's element count
target ::= "." | "type_path" | "fail"         ← exactly this state's transitions
keyname ::= "a" | "b" | ... | "enter" | ...   ← policy allowlist minus denylist
coord  ::= "0" | [1-9] [0-9]{0,2} | "1000"    ← vision: normalised space only
labelidx ::= "0" | "1" | "2"                  ← index into this state's `watch` list
```

So the actuator **cannot** emit a malformed burst, name a denied key, reference an element
that was not observed, or propose an undeclared transition. The vision model **cannot**
invent an element name. There is no retry loop and no defensive JSON parsing, because
malformed output is not improbable — it is unrepresentable.

That `coord` rule deserves a note: VLM grounding classically returns boxes either
normalised to 0–1000 or in the pixel space of the *resized* image the vision tower saw,
which is neither the file you sent nor the screen. Getting it wrong yields clicks with a
constant offset, which reads as a model-quality problem and is not. Locking the grammar to
0–1000 eliminates the ambiguity at the source; `CoordinateMapper` still handles the other
case defensively.

### The layers of defence

Each layer catches what the one above cannot:

| Layer | Mechanism | Catches |
|---|---|---|
| Grammar | logit masking | malformed syntax, denied keys, phantom targets |
| Parser | `parse_burst` | out-of-range coordinates, unknown keys, bad indices |
| Governor | `review()` | dangerous *semantics* — what a legal burst would do |
| Executor | held-key tracking | stuck modifiers on abort |
| Killswitch | 4 independent stops | everything else |

The governor is the one that is never advisory. It sees every burst — actuator, reflex,
`on_enter`, and ones the orchestrator wrote by hand — and refuses **whole bursts**, never
partially. Half-executing an intended sequence leaves the desktop in a state nobody
planned for, which is worse than doing nothing.

Its most important rule is `deny_labels`: refuse a click on any element *labelled* Delete /
Confirm / Purchase / Allow, wherever it appears. Region fencing cannot do this — the whole
danger of a confirmation dialog is that it appears somewhere you did not predict. Matching
is word-boundary, so "Send to Trash" is refused and "Buyer name" is not.

## Platform reality

This was built against KDE Plasma 6 on Wayland, and several design choices are downstream
of what actually works there rather than what the documentation suggests.

### Input: `/dev/uinput`, nothing else

`xdotool` and every X11 automation tool talks to an X server that Wayland clients are not
connected to. Under XWayland they reach XWayland windows, which on a modern KDE desktop is
close to nothing. uinput injects at the **kernel evdev layer**, below the display server
entirely, so libinput receives the events exactly as it would from a USB device. It works
on X11, Wayland, the console, and inside games reading raw input.

Implemented in ~150 lines of `ctypes`/`fcntl` rather than via python-evdev: it removes a
C-extension dependency (which matters on Python 3.14, where wheels are still patchy) and
gives direct control over event batching. Writing a click's press+release+SYN in a single
`os.write` is both atomic and faster than three syscalls — and atomicity matters, since
events between `SYN_REPORT`s form one input frame.

**Three devices, not one.** libinput classifies a device by the capabilities it advertises,
and one device claiming keyboard + absolute pointer + relative pointer gets classified
unpredictably. So: `voltage-keyboard`, `voltage-pointer-abs`, `voltage-pointer-rel`.
libinput merges pointers into one logical cursor, so a button pressed on one and released
on the other behaves correctly. The absolute device declares `INPUT_PROP_POINTER` and
deliberately does **not** advertise `BTN_TOUCH` — udev's `input_id` builtin tags
`ABS_X`/`ABS_Y` + `BTN_TOUCH` as a touchscreen, which is not what we want.

Two details that are silent failures if missed: a freshly created uinput device needs
~400 ms for udev and libinput to enumerate it (events written before that are accepted by
the kernel and dropped), and scroll must emit **both** `REL_WHEEL` and `REL_WHEEL_HI_RES`
or it works in some toolkits and not others.

### Text: scancodes are layout-dependent

uinput sends scancodes. What character results depends on the compositor's active XKB
layout. On a Turkish-Q layout, `KEY_SEMICOLON` is `s`, not `;`. This fails *silently*,
which is the worst kind. So `text_mode = "auto"` routes anything beyond short plain ASCII
through the clipboard instead — layout-independent, unicode-safe, and O(1) rather than
O(len) in wall time. Clipboard contents are saved and restored around the paste.

### Capture: the portal, and why not the obvious alternatives

Four backends, probed in order:

- **`portal`** — xdg-desktop-portal ScreenCast → PipeWire → GStreamer appsink. Works on
  KDE and GNOME. One permission dialog, persisted via a restore token, then a continuous
  stream where `grab()` is a slot read. **This is the one that works.**
- **`kwin`** — `org.kde.KWin.ScreenShot2` over a pipe fd. Raw pixels, no dialog, ~15–40 ms.
  It would be ideal, except **KWin authorises this interface against an allowlist of
  executables** (Spectacle and friends). Any other process gets `NoAuthorized`. Its
  `available()` therefore performs a real 1×1 capture rather than trusting that the
  interface is reachable — because it always is, and that means nothing.
- **`grim`** — wlroots only. Present on many systems, does not work under KWin, which has
  no wlr-screencopy.
- **`x11`** — XWayland only. Last, because it *succeeds* while returning almost nothing,
  and silent wrongness ranks below loud failure.

The portal handshake needs a running GLib main loop, since each method returns a Request
object path and the real answer arrives later as a signal. That loop drives the **global
default** MainContext: the DBus connection binds signal delivery to the thread-default
context in effect when it was created, so a private context would leave every reply queued
with nothing iterating it, and every portal call would time out waiting for an answer that
had already arrived.

KWin captures also need care around deadlock: a 1080p RGBA frame is ~8 MB and a pipe holds
64 KB, so KWin blocks writing long before it can reply. The read runs on a separate thread
started *before* the DBus call.

## Model selection is a packing problem

Two models must be resident at once. On 6 GB, minus ~900 MB the desktop is already holding,
that is the binding constraint and it rules out most obvious choices.

**Vision: Qwen2.5-VL-3B.** GUI grounding is a specific trained skill, not a side effect of
captioning. Qwen2.5-VL emits bounding boxes for on-screen elements in a normalised space —
the only question this layer is ever asked. A general captioner of the same size describes
a screenshot beautifully and puts the boxes in the wrong place. The 7B is better; it does
not fit alongside an actuator.

**Actuator: Qwen3-1.7B.** Under a grammar and a state brief, the actuator selects among a
small set of legal continuations rather than reasoning freely — close to the easiest thing
a small instruct model can be asked. Its decode speed sets the floor on cycle time, so
smaller is genuinely better up to the point of competence.

Both run under llama.cpp with `q8_0` K/V cache (roughly halves cache memory at no
meaningful quality cost for prompts this short) and `--cache-reuse 256`. Two separate
servers, so a vision call can never evict the actuator's cached prefix.

## Stopping

Autonomous input control needs an exit that does not assume the machine is still usable.
If the actuator is holding Alt and spamming Tab, "open a terminal and type a command" is
not a plan. Four independent stops, ordered by how little they assume:

1. **Panic file** — `voltage stop` writes it; every loop notices within one cycle. Works
   over SSH, from another TTY, from a file manager.
2. **Deadman timer** — fires on its own thread if a cycle does not complete. This covers
   the loop itself wedging, where nothing in the main path is running to notice.
3. **Physical input contention** — touch the real mouse or keyboard and it stops. The one
   people actually reach for. Needs `input` group membership; degrades to unavailable and
   says so in `voltage doctor`.
4. **Budgets** — cycle, wall-clock, burst and rejection limits. Termination guarantee
   rather than emergency stop.

Every path funnels through `trip()` and every path releases held input. A run interrupted
between `d:shift` and `u:shift` must not leave Shift stuck down — that is the single most
important cleanup path in the codebase, and it runs on abort, on crash, and on timeout.

## Module map

```
models/burst.py         the burst DSL: parse, validate, resolve g: references
models/observation.py   vision output → screen coordinates (CoordinateMapper)
models/template.py      bursts with {expression} holes — proportional control
models/playbook.py      the Playbook schema + compilation to executable form
expr.py                 sandboxed guard expressions (allowlisted AST walk)

capture/portal.py       ScreenCast → PipeWire → GStreamer (primary)
capture/kwin.py         KWin ScreenShot2 (authorised binaries only)
capture/probes.py       numpy pixel / region / diff / template / OCR number probes

inputs/uinput.py        ctypes virtual devices
inputs/executor.py      burst scheduling, drift-free timing, clipboard text
inputs/keymap.py        scancodes, aliases, US-layout ASCII table

llm/grammar.py          per-cycle GBNF generation — the reliability mechanism
llm/llamacpp.py         fast path: grammars, prompt cache, slots
llm/profiles.py         VRAM-budgeted model profiles

safety/governor.py      every burst passes through here
safety/killswitch.py    the four stops

runtime/session.py      both loops — decisions, and the reflex/latch loop under it
runtime/prompts.py      cache-ordered prompt construction
runtime/journal.py      the record the orchestrator reads back

diagnose.py             journal → named failure modes and the edit that fixes each
server.py               MCP tools
reference.py            self-describing DSL docs for the orchestrator
```
