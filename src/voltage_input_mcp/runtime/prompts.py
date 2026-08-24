"""Prompt construction for the two local models.

Two constraints shape everything here, and they pull in the same direction:

**Prompt-cache locality.** llama.cpp reuses the KV cache for the longest common *prefix*
between consecutive requests. So content is ordered strictly by how often it changes:
fully static rules first, then per-state text (changes on transition), then per-cycle
observation last. Get this backwards -- put the observation near the top -- and every
cycle re-prefills the entire prompt. On a 6 GB card that is the difference between a
250 ms and a 900 ms cycle, from nothing but token ordering.

For the vision model this means the image goes **last**, after the instruction text,
which is the opposite of the usual chat convention. The image is the one thing that
changes every single cycle; anything after it is uncacheable.

**Brevity as latency.** Every token in the prompt is prefill time and every token out is
decode time. The grammar already guarantees the output shape, so the prompt does not need
to describe the format defensively, plead for valid JSON, or give five examples. It needs
to say what to look for and what to do. The syntax block below is the only formatting
instruction, and it exists mainly so the model knows what the grammar is going to let it
say.
"""

from __future__ import annotations

from ..models.observation import Observation
from ..models.playbook import CompiledState

__all__ = [
    "VISION_SYSTEM",
    "ACTUATOR_SYSTEM",
    "vision_prompt",
    "actuator_prompt",
]


VISION_SYSTEM = """\
You read a screenshot and report what is on it. You never guess and never describe \
things that are not visible.

Coordinates are normalised 0-1000 over the whole image: x=0 is the left edge, x=1000 the \
right edge, y=0 the top, y=1000 the bottom. Each box is [x1,y1,x2,y2] and must tightly \
enclose the thing it labels.

Use only labels from the LOOK FOR list. Omit anything you cannot actually see. It is \
correct and useful to return an empty list."""


ACTUATOR_SYSTEM = """\
You convert one instruction plus one screen report into a burst of input events. You do \
not plan, explain, or reason about the wider goal. You emit the next burst.

Burst syntax, chained with ';' :
  g:N        move the pointer to the centre of SEEN item N
  c:l        left click   (c:r right, c:m middle, c:l2 double-click)
  k:ctrl+t   press a key chord
  t:"text"   type text
  m:X,Y      move the pointer to screen pixel X,Y
  s:-3       scroll down 3   (s:+3 up)
  d:shift    hold a key      u:shift  release it
  w:120      wait 120 ms

Reply with exactly three fields separated by '|' :
  <burst>|<next state>|<short note>

Use '.' for the burst when the right move is to wait and look again.
Use '.' for the next state to stay where you are.

Rules:
- Prefer g:N over m:X,Y. Only use m: when SEEN has no item for what you need.
- Put w: between actions that need the screen to catch up, such as after opening a menu.
- One decisive burst beats several timid ones. Chain the whole obvious sequence.
- If SEEN does not contain what TASK needs, emit '.' and wait. Do not hunt around."""


def vision_prompt(state: CompiledState, *, extra_hint: str | None = None) -> str:
    """The per-cycle text for the vision model. Static within a state."""
    watch = state.spec.watch
    lines = ["LOOK FOR: " + (", ".join(watch) if watch else "(anything notable)")]
    if extra_hint:
        lines.append(f"CONTEXT: {extra_hint}")
    return "\n".join(lines)


def actuator_prompt(
    state: CompiledState,
    observation: Observation,
    *,
    variables: dict[str, object],
    targets: list[str],
    screen: tuple[int, int],
    last_burst: str = "",
    last_result: str = "",
    steer_hint: str | None = None,
    cycles_in_state: int = 0,
    probes: dict[str, float] | None = None,
) -> str:
    """The per-cycle text for the actuator.

    Ordered so the semi-static block (task, hint, legal targets, screen) precedes the
    per-cycle block (SEEN, VARS, LAST). Within a state, only the tail changes.
    """
    parts: list[str] = [f"TASK: {state.spec.brief}"]

    if state.spec.hint:
        parts.append(f"HINT: {state.spec.hint}")
    if steer_hint:
        # Mid-run correction from the orchestrator. Placed with the other semi-static
        # text because it persists until changed again.
        parts.append(f"NOTE FROM SUPERVISOR: {steer_hint}")

    parts.append(f"CAN GO TO: {', '.join(targets) if targets else '(nowhere; stay)'}")
    parts.append(f"SCREEN: {screen[0]}x{screen[1]}")

    # --- per-cycle tail ---
    parts.append("")
    parts.append(_render_seen(observation))

    if observation.texts:
        parts.append("TEXT: " + " | ".join(t[:60] for t in observation.texts[:3]))
    if observation.flags:
        parts.append("FLAGS: " + ", ".join(sorted(observation.flags)))

    # Numeric probes, with their rate of change. This is what lets the actuator fly
    # rather than jump and hope: "descending at 170 and 116 m up" is a decision, "there
    # is a screenshot" is not. Rates are shown as +/- per second so the model can read a
    # trend without differencing anything itself.
    if probes:
        readings = []
        for key, value in probes.items():
            if key.startswith("__") or key.endswith("__rate"):
                continue
            rate = probes.get(f"{key}__rate")
            if rate is not None and abs(rate) >= 0.5:
                readings.append(f"{key}={value:g} ({rate:+.0f}/s)")
            else:
                readings.append(f"{key}={value:g}")
        if readings:
            parts.append("READINGS: " + "  ".join(readings[:8]))

    if variables:
        rendered = ", ".join(f"{k}={v}" for k, v in list(variables.items())[:8])
        parts.append(f"VARS: {rendered}")

    if cycles_in_state:
        parts.append(f"TRIES HERE: {cycles_in_state}")

    if last_burst:
        outcome = f" -> {last_result}" if last_result else ""
        parts.append(f"LAST: {last_burst}{outcome}")

    parts.append("")
    parts.append("BURST:")
    return "\n".join(parts)


def _render_seen(observation: Observation) -> str:
    """The indexed element list the `g:` verb refers to.

    Indices here must match the order passed to `parse_burst(elements=...)`, or `g:2`
    resolves to the wrong thing -- silently, and with a click somewhere unintended. The
    session builds both from the same list for exactly this reason.
    """
    if not observation.elements:
        return "SEEN: (nothing from the LOOK FOR list is visible)"
    lines = ["SEEN:"]
    for i, element in enumerate(observation.elements):
        cx, cy = element.center
        lines.append(f"  {i} {element.label} at {cx},{cy} ({element.w}x{element.h})")
    return "\n".join(lines)
