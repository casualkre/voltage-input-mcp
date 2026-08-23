# Contributing

## Before you touch anything

```bash
./scripts/setup.sh
.venv/bin/python -m pytest -q
```

149 tests, ~4 seconds, no GPU or models required. They cover the burst DSL, the guard
sandbox, the safety governor, playbook compilation, GBNF generation, the uinput wire
encoding, and the run loop driven with stub models. If they pass, the parts that can be
verified without weights on disk are fine.

## What is worth being careful about

**`safety/governor.py`** is the only layer that is not advisory. Every burst passes
through it — from the actuator, from reflexes, from `on_enter`, and from the orchestrator
directly. Two rules to keep:

- Refusal is **whole-burst**. Never execute part of a burst. Half an intended sequence
  leaves the desktop in a state nobody planned for.
- New policy fields need a test that proves the *unsafe* case is refused, not only that
  the safe case passes.

**`llm/grammar.py`** is where reliability comes from. The grammar is regenerated per cycle
so that bad output is unrepresentable rather than merely unlikely. If you add a burst verb,
add its rule here too — otherwise the parser accepts something the model can never emit,
which is dead code, or worse, the grammar permits something the governor has to catch
later.

`test_grammar_and_encoding.py::test_no_dangling_rule_references` will catch an undefined
rule reference for every combination of allowed verbs. Keep it passing; llama.cpp rejects
a grammar with a dangling reference at load time, which surfaces as a confusing runtime
failure.

**`inputs/uinput.py`** encodes kernel ABI. The struct sizes are baked into the ioctl
numbers, so a mismatch fails *silently* — the ioctl is rejected and the device is
misconfigured rather than erroring. `test_struct_layouts_match_the_kernel` pins them.

**Held-key release** is the most important cleanup path in the project. A run interrupted
between `d:shift` and `u:shift` leaves the compositor believing Shift is physically down,
and the user's keyboard behaves bizarrely until they press and release it by hand. Any new
exit path must go through `Executor.release_all`.

## Adding a capture backend

Implement `CaptureBackend.grab()` returning an RGB `Frame`, add it to `AUTO_ORDER`, and
make `available()` **actually attempt a capture**. Do not check whether an interface exists
— that is how the KWin backend shipped broken. `org.kde.KWin.ScreenShot2` is present and
reachable on every KDE session and refuses every caller not on its executable allowlist.
Reachability told us nothing.

## Adding a model profile

Fill in `weights_mb` honestly and check `Profile.vram_mb` against a real card. The VRAM
estimate errs high on purpose: a profile that does not fit does not fail, it silently
spills over PCIe and runs about ten times slower. Prefer refusing a config to shipping one
that thrashes.

Tune the two roles separately. Vision is prefill-bound and wants a micro-batch large enough
to take the whole image in one pass; the actuator is decode-bound and wants CPU threads,
because GBNF evaluation runs per sampled token on the CPU. `voltage bench` measures both.

## Testing changes that need real input

`voltage_calibrate` and `dry_run=false` act on a real desktop. Two habits worth keeping:

- Have `voltage stop` ready in another terminal. It writes a file; it does not need the
  session to be healthy.
- Test on a VM or a spare session first if the change touches the executor.

## Style

Match what is there. Comments explain *why*, especially where the reason is a platform
quirk that is invisible from the code — those are the comments that stop the next person
from "simplifying" a workaround back into a bug.
