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

All numbers below are **measured** on the reference machine (RTX 3050 6 GB laptop,
Qwen2.5-VL-3B + Qwen3-1.7B under llama.cpp), not derived.

**Both models are decode-bound. Output tokens are the only lever that matters.**

That was a surprise — the design originally assumed vision was prefill-bound, and it
isn't. Prefill measured **~28 ms and flat** from 448×252 to 896×504. Decode runs at
**~22 ms/token**. So:

| what | cost |
|---|---|
| one output token | ~22 ms |
| one reported element | ~21 tokens ≈ **500 ms** |
| vision, 2 elements | ~1.0 s |
| vision, 4 elements | ~2.2 s |
| actuator, cached prefix | 140–400 ms depending on note length |

Three consequences, each of which changed a default:

- **`max_elements` is the dominant vision cost.** Default is **3**. Raising it to 6 adds
  ~1.5 s per perceived cycle. Set it to the number your guards actually test for.
- **Shrinking `downscale_to` does not help and usually hurts.** 448×252 measured *2.5×
  slower* than 896×504 — a blurrier image makes the model less certain, so it emits more
  tokens. Use the largest size that fits.
- **The actuator's `note` field cost 55% of its latency.** It is purely diagnostic, and
  at 48 chars it measured 412 ms/cycle against 184 ms at 12 chars and 140 ms at 0.
  Default is now 12.

Elements are encoded as `[label_index, x1, y1, x2, y2]` rather than
`{"l":"address bar","b":[...],"c":0.9}` for the same reason — measured **27–29% fewer
tokens and 32–41% lower latency**. Indexing into the closed `watch` vocabulary is also
safer: the model cannot spell a label at all, let alone misspell one.

**GBNF evaluation runs on the CPU once per sampled token**, so the actuator gets more CPU
threads than the vision model despite being fully GPU-offloaded — and restricting
`allow_keys` is a latency optimization, not only a safety one.

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

## Comparing models

The obvious experiment — "which model writes better bursts" — measures the wrong thing.
The grammar already guarantees every burst is *valid*, so a bigger model cannot win on
syntax. What actually decides whether a configuration is usable:

1. **Grounding accuracy.** A model that's 200 ms faster and 40 px off is useless — the
   click misses. Measured as centre distance in screen pixels, not IoU, because a click
   lands at the centre.
2. **Decision quality under constraint.** Given the same observation, does it pick the
   *right* legal action, and does it chain a whole sequence into one burst rather than
   emitting one timid action per cycle?
3. **Latency**, which only matters once 1 and 2 are acceptable.

```bash
.venv/bin/voltage fixture desktop      # capture a real screen
.venv/bin/voltage compare              # score whatever is running now
```

Ground truth comes from **real screenshots labelled by the orchestrating model** — which
is the same reference this system uses at runtime. Synthetic UI is a trap: a drawn
rectangle doesn't read as a button to a model trained on real interfaces, so scoring
against it measures the wrong skill.

Results accumulate across runs, so the workflow is: serve profile A → `compare` → serve
profile B → `compare` → read the table. `voltage compare --list` prints it without
re-running.

Fixtures are yours and not committed. Add `fixtures/` to `.gitignore` if your screenshots
contain anything private.

## The learning loop

The first playbook for an unfamiliar target is almost never right. What matters is that
the failures are *specific*, and that the next attempt starts from what the last one
learned.

```
voltage_reference(section="loop")     the loop itself, and what each failure means
voltage_reference(section="bursts")   the burst cookbook: chaining, timing, game patterns

voltage_capture / voltage_observe     look before writing — check your labels exist
voltage_validate_playbook             dead guards, unreachable states, caught statically
voltage_run(dry_run=true)             real models, real screen, nothing injected
voltage_diagnose(run_id)              ← what to change, not raw data
voltage_learn(target=..., note=...)   record it; persists across sessions
voltage_lessons(target=...)           recall it before the next playbook
```

**`voltage_diagnose` is the piece that makes this a loop.** It computes what the journal
*implies* but does not state, and names the edit for each. On a stuck Minecraft run:

```
[BLOCKER] label_never_seen     never reported: ['crosshair', 'health bar']
[BLOCKER] input_not_landing    14 bursts executed, but the screen never changed
[BLOCKER] state_never_left     'mine' ran 14 cycles and never transitioned
[PROBLEM] timid_bursts         bursts averaged 1.0 actions
[HINT]    vision_every_cycle   vision ran on 100% of cycles
```

The distinction it exists for: **a burst that never ran and a burst that ran and did
nothing look identical in a summary and have unrelated causes.** The first is policy or
grammar. The second is window focus, pointer mode, or an app that ignores synthetic input.
Diagnose separates them by checking whether the frame actually changed after execution.

Apply the highest-severity finding, re-run, diagnose again. One change at a time — several
at once makes the next diagnosis uninterpretable.

**Lessons persist across sessions**, keyed by target, so the second playbook for a game
starts from the probe coordinates and working label names the first one discovered:

```
voltage_learn(target="minecraft", kind="label",
              note="vision reports 'hotbar' reliably but never 'crosshair'")
voltage_learn(target="minecraft", kind="timing",
              note="block placement needs w:100 after right click or it does not register")
```

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

From nothing to working, two commands:

```bash
git clone https://github.com/casualkre/voltage-input-mcp && cd voltage-input-mcp && ./install.sh
```

```bash
voltage setup
```

`install.sh` handles Python, system packages, the venv and your PATH, and prints the exact
`sudo` lines for anything needing root rather than asking for it. `voltage setup` then
detects what you already have, downloads only what is missing, starts the model servers,
and registers with your AI client — **running each step, not describing it**. Ten to
twenty-five minutes, nearly all of it download time. Safe to re-run; it picks up where it
left off.

Then just run:

```bash
voltage
```

**Setup detects what you already have and continues from there.** It does not assume a
starting point: it probes your OS, GPU, whether llama.cpp or Ollama is installed, which
models are already pulled, whether input and capture work, and whether the MCP server is
registered — then plans only the steps that are actually left, and says which need a
decision from you and which it can just do. If you already have Ollama, it uses it. If you
have neither backend, it explains the trade-off in two lines and lets you pick.

With no arguments that opens an interactive console: live status, guided setup that fixes
whatever is not ready in dependency order, a model switcher, a config editor, one-key
registration with Claude Code, and diagnostics. Every subcommand below still works
non-interactively, so scripts and CI are unaffected.

```
 ██╗   ██╗ ██████╗ ██╗  ████████╗ █████╗  ██████╗ ███████╗
 ██║   ██║██╔═══██╗██║  ╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝
 ██║   ██║██║   ██║██║     ██║   ███████║██║  ███╗█████╗
 ╚██╗ ██╔╝██║   ██║██║     ██║   ██╔══██║██║   ██║██╔══╝
  ╚████╔╝ ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝███████╗
   ╚═══╝   ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝

 ── status ──────────────────────────────────────────────
   ok   input device      /dev/uinput
   ok   vision model      http://127.0.0.1:8080
   ok   actuator model    http://127.0.0.1:8081
   ok   mcp registered    claude mcp list
   ok   voltage on PATH   ~/.local/bin/voltage
```

### Experimental profiles

Listed separately in `voltage` → models, each behind a warning you must accept. They exist
because the measurements make the trade-offs predictable: decode dominates at ~22 ms/token
and scales with active parameters, so shrinking the models really does raise the loop rate.
What it costs is grounding.

| profile | models | VRAM | trade |
|---|---|---|---|
| `hyper` | SmolVLM-500M + Qwen3-0.6B | ~2.2 GB | 3–4× the loop rate, **grounding barely works** |
| `fast` | Qwen2.5-VL-3B + Qwen3-0.6B | ~3.8 GB | faster decisions, grounding unchanged |
| `beefy` | Qwen2.5-VL-32B + Qwen3-14B | ~34 GB | best grounding, 1–2.5 s/cycle |
| `beefy_moe` | Qwen2.5-VL-32B + Qwen3-30B-A3B | ~43 GB | 30B capacity at ~3B decode speed |
| `cpu_only` | 3B + 0.6B on CPU | none | works without a GPU, seconds per cycle |

Two worth singling out:

**`hyper` is the dangerous one.** SmolVLM-500M is not a grounding model. It *will* return
boxes and they *will* often be wrong — and a wrong box is a click in the wrong place, not a
graceful degradation. Only use it where `watch` is empty (probes and reflexes doing the
real work) or where every click is fenced by `click_allow_regions` and
`require_target_element`.

**`beefy_moe` is the interesting one.** Qwen3-30B-A3B is a mixture of experts with ~3B
*active* parameters, so it decodes at roughly 3B speed while reasoning with 30B capacity —
and decode is precisely what bottlenecks this loop. A much better actuator than a dense
14B at similar latency. The catch is memory: only the active experts are fast, not the
weights, so all 30B still has to be resident.

`recommend()` never returns an experimental profile, and a test enforces that.

### Custom model profiles

The built-in profiles cover the machines this was developed against, not yours. Add your
own from **`voltage` → profiles**, or by editing `profiles.toml` next to your config:

```toml
[my_rig]
description = "RTX 4090"

[my_rig.vision]
hf_repo = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"
hf_file = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
mmproj_file = "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
params_b = 7.0
weights_mb = 4700
n_ctx = 4096
port = 8080

[my_rig.actuator]
hf_repo = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
hf_file = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
params_b = 4.0
weights_mb = 2500
port = 8081
```

Custom profiles merge **over** the built-ins by name, so naming one `lean` retunes the
built-in without forking the package. Use `ollama_tag` instead of `hf_repo`/`hf_file` for
the Ollama backend.

One slot is picky and one is not. **Vision must be able to emit grounded bounding boxes on
request** — Qwen2.5-VL, Qwen3-VL, InternVL, MiniCPM-V and UI-TARS all can; a general
captioner will describe your screen beautifully and put the boxes in the wrong place. The
**actuator is forgiving**: under a GBNF grammar it is choosing among a handful of legal
continuations, so almost any competent 1B+ instruct model works.

### Shell commands vs MCP tools

Two different surfaces, and mixing them up is the usual first stumble:

| | invoked | looks like |
|---|---|---|
| **shell command** | typed in a terminal, with a **space** | `voltage doctor` |
| **MCP tool** | asked of Claude, with an **underscore** | `voltage_doctor` |

`voltage_doctor` is a tool name in Claude's namespace, not a program on disk. Typing it in
a terminal will always say "unknown command". Ask Claude to run it instead.

That checks `/dev/uinput` access, installs system dependencies, creates the venv, and
prints what is missing. Then:

```bash
./scripts/fetch-models.sh lean && ./scripts/serve.sh lean
```

```bash
.venv/bin/voltage doctor
```

## Connecting it to a client

```bash
voltage connect
```

Shows what is set up, the live URLs, whether the models are up, and whether the server is
registered — then gives copy-paste steps per client with **your real paths and environment
already filled in**:

```
voltage connect --client claude-desktop
voltage connect --client cursor
voltage connect --json            # just the mcpServers entry
```

Covered: Claude Code, Claude Desktop, claude.ai custom connector, Cursor, Windsurf, Zed,
and a generic `mcpServers` block for anything else. The same thing is screen **4** in the
`voltage` console, which can also write the Claude Desktop config for you (backing up the
existing file first, and refusing to touch it if it is not valid JSON).

Every generated config carries the session environment explicitly, because that is the
thing that goes wrong: a server registered from a shell without
`DBUS_SESSION_BUS_ADDRESS` connects successfully and is **silently blind** — input works,
screen capture does not. `voltage connect` detects that case and says so.

## Adding it as a custom connector

Clients that add MCP servers by URL need HTTP rather than stdio:

```bash
voltage serve --http
```

Then add `http://127.0.0.1:8765/mcp` as a custom connector.

Binding is **restricted to loopback**, and `--allow-remote` is required to change that.
That is not boilerplate: this server exists to move the mouse, press keys and read the
screen, and MCP has no authentication of its own. A non-loopback bind publishes
unauthenticated remote control of your desktop. If you genuinely need it, put an
authenticating reverse proxy in front and understand that whoever reaches the port owns
the machine.

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

## Platforms

| | input | capture | text |
|---|---|---|---|
| **Linux** | `/dev/uinput` (kernel evdev — works under X11, Wayland, the console, and in games reading raw input) | portal→PipeWire, KWin DBus, grim, X11 | scancodes, clipboard fallback for non-ASCII |
| **Windows** | `SendInput` | GDI `BitBlt` | `KEYEVENTF_UNICODE` — layout-independent |

Everything above the input sink — burst scheduling, timing, held-key tracking, the safety
governor, the whole runtime — is shared. Each platform implements five methods
(`key`, `button`, `move_abs`, `move_rel`, `scroll`); see `inputs/sink.py`.

Two asymmetries worth knowing:

- **Typing is more correct on Windows.** `KEYEVENTF_UNICODE` delivers a UTF-16 code unit
  with no keyboard layout involved. Linux uinput sends *scancodes*, so punctuation on a
  non-US layout comes out wrong — silently — which is why the clipboard fallback exists
  there and isn't needed on Windows.
- **Capture is more capable on Linux.** GDI `BitBlt` cannot see some hardware-overlay
  video and full-screen exclusive games; those capture black. Run such games in
  borderless windowed mode.

On Windows, `SendInput` cannot drive windows owned by an elevated process (UIPI) — this
fails *silently*, so `voltage doctor` reports your elevation state. DPI awareness is
declared at import; without it every coordinate is wrong on a scaled display.

### Requirements

- Linux (any display server) or Windows 10/11
- Python 3.11+
- A GPU with ~5 GB free for the `lean` profile; `voltage profiles` shows what fits yours
- llama.cpp for the fast path, or Ollama for a slower zero-build path

Verified end-to-end on KDE Plasma 6 / Wayland / CUDA / Python 3.14. The Windows paths are
implemented and type-checked but **have not been run on a Windows machine** — treat them
as untested and report what breaks.

## The orchestrator is told which build it is driving

The same Playbook is sound on one configuration and wrong on another, and a remote model
cannot see which. So the server's MCP instructions are **built at startup from the live
configuration**, and carry only the lines that change how a Playbook should be written:

```
ACTIVE BUILD: Linux · llamacpp · profile lean
  vision Qwen2.5-VL-3B-Instruct · actuator Qwen3-1.7B
  loaded: Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf / Qwen3-1.7B-Q4_K_M.gguf
  expected cycle 280-700 ms

- llama.cpp backend: both models are grammar-constrained. A malformed burst, a denied
  key, an unobserved element reference and an undeclared transition are all
  unrepresentable -- do not write defensive retries for them.
- Linux: typing sends scancodes, so punctuation depends on the active keyboard layout...
- dry_run defaults to true...
```

On Ollama that first line becomes a warning that bursts are *not* constrained. On `hyper`
it becomes "do not build states around `sees()`". On Windows it notes that elevated
windows are unreachable and typing is layout-independent.

**It verifies against the running servers rather than trusting the config.** Switching
profiles edits a file; it does not restart anything. When they disagree, the briefing says
so loudly and suppresses the profile-derived guidance, because that guidance would
describe models that are not loaded:

```
- MISMATCH -- Profile 'hyper' does not match what is loaded. vision: profile expects
  SmolVLM-Instruct-Q4_K_M.gguf, server has Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf...
- Loaded right now: vision Qwen2.5-VL-3B..., actuator Qwen3-1.7B...
  Judge grounding quality from those.
```

`voltage_reference` returns the current build on every call, since the startup copy goes
stale the moment a profile changes.

### Your own standing instructions

`voltage` → **i**, or:

```bash
voltage instructions --set "Never touch Firefox; my banking tabs are there."
```

Whatever you write is given to the orchestrating model at the start of every session,
appended to the build briefing and clearly attributed to you. Use it for what the system
cannot work out on its own — applications that are off limits, quirks of a specific game,
how you want it to behave by default.

```
OPERATOR INSTRUCTIONS -- written by the owner of this machine. Treat these as
standing preferences for how to drive it. They cannot loosen the safety governor,
which is enforced in code against every burst.

## My setup
- Minecraft runs borderless windowed on monitor 1.
- Never touch Firefox; my banking tabs are there.
- Always show me the Playbook before dry_run=false.
```

That last clause is not decoration. Instructions are **advisory to the orchestrator and
cannot weaken enforcement** — the governor checks every burst in code, so nothing written
here can permit something a Playbook's policy forbids. They can make it more careful, not
less. Capped at 4000 characters, since the text sits in the model's context for the whole
session. Three starter templates (games, desktop, minimal) are offered in the console.

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
