# VoltageInputMcp

An MCP server that lets a frontier model drive a computer at input speed instead of
tool-call speed.

## The problem

Computer-use tools round-trip to a remote model for every action. Screenshot up, decision
down, one click. That is fine for filling in a form and useless for anything that needs a
*sequence* of inputs delivered quickly — playing a game, working a modal dialog, driving a
timeline, any UI where the third input depends on the first two having already landed.
The bottleneck is not the model's intelligence. It is that intelligence is 800 ms away and
inputs need to be 8 ms apart.

## The shape of the answer

Separate deciding from doing, and put the doing on the same machine as the keyboard.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Layer 1  —  the orchestrator (Claude, or any MCP client)       │
  │  Writes a Playbook: states, what to look for, what is allowed,  │
  │  when to move on. Thinks once, up front. Watches and corrects.  │
  └───────────────────────────┬─────────────────────────────────────┘
                              │  MCP
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  Layer 2  —  two small local models, on your GPU                │
  │                                                                 │
  │   vision (Qwen2.5-VL-3B)     "of these specific things,         │
  │                               which are on screen, and where?"  │
  │   actuator (Qwen3-1.7B)      "given that, which inputs?"        │
  │                                                                 │
  │  Neither plans. Both answer one closed question per cycle.      │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  safety governor  →  /dev/uinput  →  the actual desktop         │
  └─────────────────────────────────────────────────────────────────┘
```

The orchestrator is the brain. The small models are the arms. The arms are not smart and
are never asked to be.

## Where the speed actually comes from

Not from the small models being fast — a 3B VLM still costs ~300 ms. It comes from four
things, in descending order of impact:

**Bursts.** The actuator does not emit an input. It emits a *burst*: a timed programme of
inputs run by a dedicated executor with no model in the loop.

```
g:0;c:l;w:150;t:"README.md";k:enter;w:80;k:ctrl+s
```

That is one decision and seven inputs spanning ~400 ms, scheduled to the millisecond. A
40-action burst still costs one decision. **Input rate is set by the burst, not the model.**

**Reflexes.** Rules that fire off cheap screen probes — one pixel, one region average — in
microseconds, between decisions, with no model at all.

```json
{"id": "heal", "when": "probe('health') < 0.25", "do": "k:q;w:60", "cooldown_ms": 800}
```

**Skipping perception.** Most cycles look at a screen that has not changed. A 40 µs
frame-diff decides whether to spend 300 ms on the vision model or reuse the last
observation. On ordinary desktop work this skips the VLM on most cycles.

**Prompt-cache locality.** Prompts are ordered static-first so llama.cpp reuses the KV
cache and only re-prefills the changed tail.

## Why the small models are reliable despite being small

Because they are not asked to be reliable — they are *constrained*.

Under llama.cpp, both models generate against a **GBNF grammar** that is regenerated every
cycle from the current state. The grammar is not advice. It masks the logits so that only
tokens continuing a valid parse are reachable. Concretely, the actuator **cannot**:

- emit a malformed burst
- name a key the policy denies — the key is not in the grammar
- reference an element that was not observed — the index range is built from this cycle's
  element count
- propose a state transition the Playbook did not declare

And the vision model **cannot** invent a UI element name: its label vocabulary is the
`watch` list you wrote, plus a small generic set. So a `sees("address bar")` guard compares
against a closed vocabulary rather than whatever noun a 3B model felt like producing.

There is no retry loop and no defensive JSON parsing, because malformed output is not
improbable — it is unrepresentable.

## The Playbook

You do not give the small models a goal. You give them a state machine. Transitions are
guard expressions evaluated by the runtime, **not** by a model.

```json
{
  "name": "open_downloads",
  "goal": "Open the file manager at ~/Downloads. Delete nothing, confirm nothing.",
  "initial": "launch",
  "policy": {
    "dry_run": true,
    "allow_verbs": ["g", "c", "k", "t", "w"],
    "deny_labels": ["delete", "trash", "confirm", "empty trash"]
  },
  "budget": { "max_cycles": 60, "max_seconds": 90 },
  "states": {
    "launch": {
      "brief": "Open the application launcher and start the file manager.",
      "watch": ["application launcher", "search field", "file manager icon"],
      "on_enter": "k:meta;w:400",
      "transitions": [
        { "when": "sees('search field')", "to": "type_name" },
        { "when": "cycles() > 6", "to": "@failure", "note": "launcher never opened" }
      ]
    },
    "navigate": {
      "brief": "Focus the location bar with ctrl+l, type the path, press Enter.",
      "watch": ["location bar", "file list", "error message"],
      "on_enter": "k:ctrl+l;w:200",
      "transitions": [
        { "when": "text('Downloads')", "to": "@success" },
        { "when": "sees('error message')", "to": "@failure" }
      ]
    }
  },
  "success_when": "text('Downloads') and not flag('loading')"
}
```

`voltage_reference` returns the full DSL, the JSON schema, and the guard function table, so
an orchestrator can author one without reading this repo.

## Performance tuning

The two models are bottlenecked on different things and are tuned separately.

**Vision is prefill-bound.** Qwen2.5-VL emits one visual token per 28×28 pixel block, so
input size maps to cost exactly:

| `downscale_to` | visual tokens | relative prefill |
|---|---|---|
| 1344×756 | 1296 | 2.25× |
| 896×504 *(default)* | 576 | 1.0× |
| 784×448 | 448 | 0.78× |
| 700×392 | 350 | **0.61×** |
| 448×252 | 144 | 0.25× |

This is the largest single lever in the system. Dropping to 700×392 removes 39% of the
prefill for a 0.5% aspect distortion, which is harmless because `CoordinateMapper` scales
x and y independently. Values are auto-snapped to multiples of 28 — feeding a non-multiple
makes the model resize internally, wasting work *and* shifting grounded boxes.

**The actuator is decode-bound**, emitting 20–40 grammar-constrained tokens against a
mostly-cached prompt. It's given more CPU threads than the vision model despite being
fully GPU-offloaded, because **GBNF evaluation runs on the CPU once per sampled token** —
which also means restricting `allow_keys` is a latency optimization, not just a safety one.

Two settings that fail *silently* if wrong:

- **`GGML_CUDA_FA_ALL_QUANTS=ON` at build time.** We serve with `q8_0` KV cache *and*
  flash attention. Without this flag llama.cpp doesn't compile FA kernels for that KV
  combination and falls back to a slow path — no error, just mysteriously bad numbers.
  `scripts/build-llama.sh` sets it.
- **`GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` at runtime.** If it's `1`, VRAM overflow silently
  spills over PCIe instead of failing. Everything works and is ~10× slower. `serve.sh`
  pins it off.

Measure rather than guess:

```bash
.venv/bin/voltage bench
```

It drives both backends with the exact prompt shapes the loop uses and reports cold vs.
prompt-cached latency, ms-per-visual-token at three input sizes, and the cycle time those
imply. A prompt-cache speedup below ~1.5× means something dynamic leaked into the prompt
prefix.

## Safety

The thing generating inputs is a 1.7B model. The governor is the layer that is not
advisory: every burst passes through it, including reflex bursts and ones you wrote
yourself.

- **`dry_run` is the default.** A new Playbook parses, checks and journals every burst
  while touching nothing.
- **Whole-burst refusal.** Half-executing an intended sequence is worse than not executing
  it.
- **`deny_labels`** refuses a click on anything *called* Delete / Confirm / Purchase /
  Allow, wherever it appears — this is what catches the dialog that pops up somewhere
  unexpected.
- **Region fencing**, key allowlists, denied chords (`ctrl+alt+delete`, `alt+f4`), denied
  text patterns (`rm -rf`, `sudo`), burst-size and inputs-per-second caps.
- **Four independent stops**: `voltage stop` (writes a file — works over SSH), a deadman
  timer that fires on its own thread if the loop wedges, **physical input contention**
  (touch the real mouse and it stops), and Playbook budgets.
- **Held keys are always released** — on abort, on crash, on timeout. A run interrupted
  between `d:shift` and `u:shift` must not leave Shift stuck down.

## Install

```bash
cd voltage-input-mcp && ./scripts/setup.sh
```

That checks `/dev/uinput` access, installs system dependencies, creates the venv, and
prints what is missing. Then:

```bash
./scripts/fetch-models.sh lean && ./scripts/serve.sh lean
```

```bash
.venv/bin/voltage doctor
```

## Launching from an MCP client

MCP clients start servers with a **sanitized environment** — `PATH`, `HOME` and little
else. That is a sensible default and it breaks screen capture, because reaching the
compositor needs `DBUS_SESSION_BUS_ADDRESS` and `WAYLAND_DISPLAY`. Input injection still
works without them (uinput is a device file, not a session service), so the failure looks
confusingly partial: bursts execute, screenshots do not.

Pass them through explicitly:

```bash
claude mcp add voltage-input \
  -e WAYLAND_DISPLAY="$WAYLAND_DISPLAY" \
  -e DISPLAY="$DISPLAY" \
  -e DBUS_SESSION_BUS_ADDRESS="$DBUS_SESSION_BUS_ADDRESS" \
  -e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
  -- /absolute/path/to/voltage-input-mcp/.venv/bin/voltage-input-mcp
```

`voltage_doctor` reports exactly which of these are missing, so if capture is failing that
is the first place to look.

## Requirements

- Linux with `/dev/uinput` (X11, Wayland, or console — it injects below the display server)
- Python 3.11+
- A GPU with ~5 GB free for the `lean` profile; `voltage profiles` shows what fits yours
- llama.cpp for the fast path, or Ollama for a slower zero-build path

Verified against KDE Plasma 6 on Wayland (KWin), CUDA, Python 3.14.

## MCP tools

| Tool | Purpose |
|---|---|
| `voltage_reference` | The Playbook + burst DSL reference. Call this first. |
| `voltage_doctor` | Is this machine ready, and if not, the exact fix |
| `voltage_capture` | A screenshot, returned to you |
| `voltage_observe` | One vision pass — check a `watch` list works before relying on it |
| `voltage_validate_playbook` | Full static check: guards, bursts, graph, dead transitions |
| `voltage_run` | Start a run; returns a `run_id` |
| `voltage_status` | State, vars, last burst, what was seen, per-stage timings |
| `voltage_steer` | Correct a live run — hint, variables, forced state, dry_run |
| `voltage_stop` / `voltage_pause` | Stop or pause; stop always releases held input |
| `voltage_journal` | Cycle-by-cycle record; `only_refused` to see policy conflicts |
| `voltage_execute_burst` | Drive the input yourself, bypassing the local models |
| `voltage_calibrate` | Verify injection reaches the compositor |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — how the loop works, why each choice was made,
  where the time goes
- [PLAYBOOK.md](PLAYBOOK.md) — the authoring guide

## Status

Built and verified as far as it can be without weights on disk. 149 tests cover the burst
DSL, the guard sandbox, the safety governor, playbook compilation, GBNF generation, the
uinput wire encoding, and the run loop itself (driven with stub models — including a check
that `on_change` perception really does skip the vision model on a static screen).

The MCP server was driven end-to-end over stdio by a real client: 13 tools, correct
schemas, `execute_burst` accepted a valid burst and refused `sudo rm -rf /` with both
matching rules.

What has **not** run is a live model: that needs llama.cpp built and weights fetched,
which `scripts/` sets up. Two things were also deliberately not triggered during the
build — the portal permission dialog, and any real input injection — since both act on
your desktop.

Order of operations from here:

```bash
./scripts/setup.sh          # reports what needs sudo, doesn't run it
./scripts/build-llama.sh    # ~15 min with CUDA
./scripts/fetch-models.sh lean
./scripts/serve.sh lean
.venv/bin/voltage doctor    # should now say READY
```

Then in an MCP client: `voltage_calibrate` (watch the cursor actually move),
`voltage_observe` (check the vision model finds your labels), then a `dry_run` Playbook
and read `voltage_journal` before ever setting `dry_run=false`.

## Authorship

Written end to end by **Claude Opus 5** (Anthropic) in a single session — architecture,
implementation, tests, and documentation. A human specified the idea, set the constraints
(KDE Wayland, 6 GB VRAM, "faster than computer-use"), and reviewed the result, but did not
write the code.

The platform findings baked into this repo came from probing the machine during the build
rather than from assumption — that KWin refuses `ScreenShot2` to non-allowlisted
executables, that `grim` can't work under KWin, that MCP clients sanitize away the session
bus. Each is documented at the point in the code where it forced a decision.

`LICENSE` names no individual as copyright holder, and the reasoning is written out there.

## License

MIT. See [LICENSE](LICENSE).
