"""The Playbook: the contract between the orchestrator and the two small models.

This is the most important schema in the project, because it is where the division of
labour is actually enforced.

The orchestrating model is smart but slow and remote. The local models are fast but
genuinely limited -- a 1.7B actuator will not reason its way out of an ambiguous
situation, and a 3B vision model will confabulate if asked an open question. So the
orchestrator does not hand them a goal and hope. It hands them a **state machine**:

  * The orchestrator decides *what the states are*, *what to look for in each*, *what is
    permitted in each*, and *exactly when to move on*. That is the thinking.
  * The vision model answers one closed question per cycle: "of these specific things,
    which are on screen and where?"
  * The actuator answers one closed question per cycle: "given that, which inputs?"

Transitions are evaluated by **this process**, not by a model -- they are guard
expressions the orchestrator wrote (see expr.py). The actuator may only *propose* a
transition already listed in the current state. It cannot invent control flow.

Reserved transition targets
---------------------------
    @success   end the run, report success
    @failure   end the run, report failure
    @stop      end the run, no verdict (used for "human should look at this")

Everything else must name a state in `states`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..errors import PlaybookError
from ..expr import Guard
from .burst import Burst, parse_hold
from .template import BurstTemplate

__all__ = [
    "Playbook",
    "State",
    "Transition",
    "ReflexRule",
    "ProbeSpec",
    "Policy",
    "Budget",
    "Perception",
    "Rect",
    "CompiledPlaybook",
    "CompiledState",
    "CompiledReflex",
    "CompiledHold",
    "Tunable",
    "Reward",
    "compile_playbook",
    "TERMINALS",
]

TERMINALS: frozenset[str] = frozenset({"@success", "@failure", "@stop"})

VERBS: tuple[str, ...] = ("k", "d", "u", "t", "g", "m", "r", "c", "p", "e", "s", "h", "w")
VERB_NAMES: dict[str, str] = {
    "k": "key chord",
    "d": "key down",
    "u": "key up",
    "t": "type text",
    "g": "move to seen element",
    "m": "absolute move",
    "r": "relative move",
    "c": "click",
    "p": "button down",
    "e": "button up",
    "s": "scroll",
    "h": "horizontal scroll",
    "w": "wait",
}

Ident = Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]{0,39}$")]


class Rect(BaseModel):
    """An axis-aligned screen region in pixels."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class ProbeSpec(BaseModel):
    """A cheap, model-free screen measurement, evaluated every frame in ~microseconds.

    Probes are what let the loop run faster than the vision model. Anything that can be
    reduced to "is this pixel red" or "has this region changed" should be a probe, not a
    question for the VLM. Reflex rules fire off probes with no model in the path at all.
    """

    model_config = ConfigDict(frozen=True)

    id: Ident
    type: Literal[
        "pixel", "region_mean", "region_diff", "brightness", "template", "number"
    ]
    at: tuple[int, int] | None = Field(default=None, description="pixel probes: (x, y)")
    region: Rect | None = Field(default=None, description="region probes: area to measure")
    expect: str | None = Field(
        default=None, pattern=r"^#?[0-9a-fA-F]{6}$", description="pixel/region_mean: target colour"
    )
    tolerance: int = Field(default=30, ge=0, le=255)
    template: str | None = Field(default=None, description="template: path to a PNG")
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    channel: Literal["rgb", "r", "g", "b", "luma"] = "rgb"

    # -- number probes ---------------------------------------------------------------
    # Reading a HUD number is what turns a guard into a physics condition:
    # `probe('mph') > 120 and probe('meters') < 150` is the difference between flying a
    # character and jumping off and praying. OCR costs ~80-200 ms, far too slow for the
    # reflex path, so it runs on a background worker and every read in between returns
    # the last value at no cost. Reflexes therefore fire on real numbers at full rate.
    ocr_interval_ms: int = Field(
        default=120, ge=20, le=10_000,
        description=(
            "how often to re-read, in milliseconds. Wall-clock rather than a frame count "
            "so the cadence does not change when the loop rate does -- the same probe "
            "belongs to a 2 Hz decision loop and a 20 Hz reflex loop at once."
        ),
    )
    invert: bool = Field(
        default=False, description="set for dark text on a light background"
    )
    glyphs: str | None = Field(
        default=None,
        description=(
            "name of a learned glyph set to read this number with, defaulting to the "
            "probe id. Calibrate one with `voltage learn-digits` and the probe stops "
            "using OCR entirely: reads become ~1 ms and exact instead of 80-200 ms and "
            "approximate, and they run inline rather than on a background worker, so the "
            "value is from this frame rather than one round trip stale. Falls back to OCR "
            "when no set has been learned."
        ),
    )
    scale: float = Field(
        default=1.0, description="multiply the parsed number, e.g. 0.001 for thousands"
    )

    @model_validator(mode="after")
    def _check_shape(self) -> ProbeSpec:
        if self.type == "pixel":
            if self.at is None:
                raise ValueError("pixel probe needs `at`")
            if self.expect is None:
                raise ValueError("pixel probe needs `expect`")
        elif self.type in ("region_mean", "region_diff", "brightness"):
            if self.region is None:
                raise ValueError(f"{self.type} probe needs `region`")
            if self.type == "region_mean" and self.expect is None:
                raise ValueError("region_mean probe needs `expect`")
        elif self.type == "template":
            if self.template is None or self.region is None:
                raise ValueError("template probe needs `template` and a search `region`")
        elif self.type == "number":
            if self.region is None:
                raise ValueError("number probe needs a `region` containing the digits")
        return self


class ReflexRule(BaseModel):
    """A deterministic, model-free rule. Either a one-shot burst or a latched hold.

    Reflexes run on every captured frame, between actuator decisions. This is the second
    half of the latency answer: bursts give you many inputs per decision, reflexes give
    you inputs with no decision at all. A reflex should only depend on probes and cached
    observation state, never on anything that needs fresh vision.

    Two shapes, and the difference matters more than it looks:

    **`do`** is a one-shot. The guard goes true, the burst runs, the cooldown starts. Good
    for discrete reactions -- press the button, dodge, dismiss the dialog.

    **`hold`** is a latch. The guard goes true and the keys go *down*; they stay down,
    across frames and across decisions, until the guard goes false again. This is what
    continuous control needs and what a one-shot cannot express: "hold W while the target
    is ahead" as a one-shot is a stutter of taps at reflex rate, which in a game reads as
    a character twitching in place rather than running.

    A latch is background state, so it never suppresses the actuator regardless of
    `exclusive` -- the point of holding W is that the decision layer keeps thinking while
    the character keeps moving.

    Chatter at the threshold is the failure mode a latch has and a one-shot does not: a
    guard sitting on its boundary flips at reflex rate and machine-guns the key. Both
    remedies are here. `release_when` gives a separate, wider falling-edge condition -- a
    Schmitt trigger, which is the honest fix -- and `min_hold_ms` sets a floor on how long
    a press lasts once made.
    """

    model_config = ConfigDict(frozen=True)

    id: Ident
    when: str = Field(description="guard expression, e.g. probe('health') < 0.25")
    do: str | None = Field(
        default=None,
        description="one-shot burst, e.g. k:q;w:60. May contain {expression} holes.",
    )
    hold: str | None = Field(
        default=None,
        description="keys/buttons to hold while `when` is true, e.g. 'w' or 'shift, btn:r'",
    )
    release_when: str | None = Field(
        default=None,
        description=(
            "hold only: separate guard for letting go. Defaults to `not (when)`. Give it a "
            "wider threshold than `when` to stop the latch chattering on the boundary."
        ),
    )
    min_hold_ms: int = Field(
        default=0, ge=0, le=60_000,
        description="hold only: once pressed, stay down at least this long",
    )
    cooldown_ms: int = Field(default=500, ge=0, le=60_000)
    max_fires: int | None = Field(default=None, ge=1)
    priority: int = Field(default=0, description="higher wins when several fire at once")
    exclusive: bool = Field(
        default=True,
        description=(
            "one-shot only: suppress the next actuator decision when it fires, so a "
            "stale decision cannot override a fresh reaction. Ignored for `hold`."
        ),
    )

    @model_validator(mode="after")
    def _one_shape_only(self) -> ReflexRule:
        if bool(self.do) == bool(self.hold):
            raise ValueError(
                f"reflex {self.id!r} must set exactly one of `do` (a one-shot burst) or "
                f"`hold` (keys held while the guard is true)"
            )
        if not self.hold:
            if self.release_when:
                raise ValueError(
                    f"reflex {self.id!r} sets `release_when`, which only means something "
                    f"for a `hold` -- a one-shot `do` has nothing to let go of"
                )
            if self.min_hold_ms:
                raise ValueError(
                    f"reflex {self.id!r} sets `min_hold_ms`, which only applies to a "
                    f"`hold`; for a one-shot use `cooldown_ms`"
                )
        return self


class Tunable(BaseModel):
    """A guard constant the episodic optimiser is allowed to move.

    Every threshold in a reflex is a number somebody guessed. Declaring it here and
    reading it with `tune('name')` lets the search find a better one against the reward
    the game already prints on screen, at no inference cost -- the output is just a
    better constant, and the guard stays one comparison.
    """

    model_config = ConfigDict(frozen=True)

    default: float = Field(description="starting value; used verbatim when tuning is off")
    min: float = Field(description="lower bound; the search never proposes outside this")
    max: float = Field(description="upper bound")

    @model_validator(mode="after")
    def _sane_bounds(self) -> Tunable:
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be below max ({self.max})")
        if not self.min <= self.default <= self.max:
            raise ValueError(
                f"default {self.default} is outside [{self.min}, {self.max}]"
            )
        return self


class Reward(BaseModel):
    """How to score one episode, read off the screen.

    Without this there is nothing to optimise against and the tuner is inert. `delta` is
    the usual choice: a score, money or progress counter that only goes up, where what
    matters is how much it moved during the episode.
    """

    model_config = ConfigDict(frozen=True)

    probe: str = Field(description="a number probe to score from")
    mode: Literal["delta", "final", "rate"] = Field(
        default="delta",
        description=(
            "delta: value at episode end minus value at start (a score counter). "
            "final: the raw value at the end (a distance, a timer). "
            "rate: delta divided by episode seconds (throughput, which is what you want "
            "when a slower episode earning slightly more is actually worse)."
        ),
    )
    settle_ms: int = Field(
        default=1200, ge=0, le=20_000,
        description=(
            "wait this long after the episode ends before reading the final value. Score "
            "counters animate, and reading one mid-tick scores the animation rather than "
            "the result."
        ),
    )


class Transition(BaseModel):
    """A guarded edge out of a state, evaluated by the runtime, not by a model."""

    model_config = ConfigDict(frozen=True)

    when: str = Field(description="guard expression; the first true transition wins")
    to: str = Field(description="target state name, or @success / @failure / @stop")
    ends_episode: bool = Field(
        default=False,
        description=(
            "taking this edge closes the current episode: the reward is read, the "
            "optimiser is told how the run's constants did, and the next episode starts "
            "with whatever it proposes next."
        ),
    )
    set: dict[str, int | float | str | bool] = Field(
        default_factory=dict, description="variable assignments applied on the way out"
    )
    inc: dict[str, int | float] = Field(
        default_factory=dict, description="variable increments applied on the way out"
    )
    note: str | None = Field(default=None, max_length=200)

    @field_validator("to")
    @classmethod
    def _valid_target(cls, v: str) -> str:
        if v in TERMINALS:
            return v
        if not v.replace("_", "").isalnum() or not v[0].isalpha():
            raise ValueError(f"invalid transition target {v!r}")
        return v


class Perception(BaseModel):
    """How hard to look, and how often.

    `mode` is the main latency lever:

      always     run the vision model every cycle. Highest fidelity, slowest.
      on_change  run it only when a frame-diff probe says the screen materially moved,
                 otherwise reuse the cached observation. Default, and usually 3-6x faster
                 because most cycles in a real task look at a static screen.
      cadence    run it every Nth cycle regardless.
      never      probes and reflexes only. For pure timing/muscle-memory sequences.
    """

    model_config = ConfigDict(frozen=True)

    mode: Literal["always", "on_change", "cadence", "never"] = "on_change"
    cadence: int = Field(default=3, ge=1, le=60)
    change_threshold: float = Field(
        default=0.015, ge=0.0, le=1.0, description="fraction of pixels that must differ"
    )
    max_cache_age_s: float = Field(default=2.5, ge=0.0, le=60.0)
    max_elements: int = Field(
        default=3,
        ge=1,
        le=16,
        description=(
            "how many elements the vision model may report -- THE dominant vision cost. "
            "Decode is the bottleneck (~22 ms/token measured), and each element costs "
            "~21 tokens, so roughly 500 ms each. Measured on Qwen2.5-VL-3B: 2 elements "
            "~1.0 s, 4 elements ~2.2 s. Keep this at the number your guards actually "
            "test for; raising it to 6 costs about 1.5 s per perceived cycle."
        ),
    )
    downscale_to: tuple[int, int] = Field(
        default=(896, 504),
        description=(
            "image size handed to the vision model. Contrary to intuition this is NOT a "
            "significant latency knob: prefill measured ~28 ms and is flat across "
            "448x252 through 896x504, because decode dominates. Shrinking it mostly "
            "hurts -- a blurrier image makes the model less certain and it emits MORE "
            "tokens (measured: 448x252 was 2.5x SLOWER than 896x504). Prefer the largest "
            "size that fits your VRAM and tune max_elements instead. Snapped to "
            "multiples of 28, the model's token block size."
        ),
    )

    @field_validator("downscale_to")
    @classmethod
    def _snap_to_patch_grid(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Round to a multiple of 28, the vision model's effective token block size.

        Qwen2.5-VL uses 14x14 ViT patches with a 2x2 spatial merge, so each output token
        covers 28x28 pixels. Handing it dimensions that are not multiples of 28 makes it
        resize internally: work is wasted on pixels that are then discarded, and the
        rescale shifts grounded boxes slightly relative to what we think we sent. Snapping
        here is free and strictly better.
        """
        w, h = value
        snapped = (max(28, round(w / 28) * 28), max(28, round(h / 28) * 28))
        return snapped
    read_text: bool = Field(default=True, description="ask the VLM to transcribe key text")
    accessibility: bool = Field(
        default=False,
        description=(
            "also read the desktop's accessibility tree, merging its controls into the "
            "observation and exposing them to the `ui(name)` guard. For desktop work this "
            "is strictly better than the vision model: the names come from the "
            "applications themselves, so they cannot be confabulated. Costs ~400 ms for a "
            "full walk, cached for a fraction of a second and run on a worker. Off by "
            "default because it reports nothing for games -- anything drawing its own UI "
            "into a GL surface publishes no tree -- and a sensor that returns nothing is "
            "not worth the walk. Under Wayland only some controls report usable screen "
            "coordinates; ungrounded ones are still exact for `ui()` and are excluded "
            "from the clickable element list."
        ),
    )
    region: Rect | None = Field(
        default=None, description="crop the capture before perception; cheaper and sharper"
    )


class Policy(BaseModel):
    """Hard limits. The governor enforces every field here on every burst.

    Defaults are deliberately restrictive. A playbook opts *into* capability rather than
    out of danger, because the thing generating bursts is a 1.7B model.
    """

    model_config = ConfigDict(frozen=True)

    allow_verbs: list[str] = Field(default_factory=lambda: list(VERBS))
    allow_keys: list[str] | None = Field(
        default=None, description="if set, ONLY these key names may be pressed"
    )
    deny_keys: list[str] = Field(
        default_factory=lambda: [
            "delete", "sysrq", "power", "sleep", "compose", "leftmeta", "rightmeta",
        ]
    )
    deny_chords: list[str] = Field(
        default_factory=lambda: [
            "ctrl+alt+delete", "ctrl+alt+backspace", "alt+f4", "ctrl+q",
            "ctrl+shift+q", "ctrl+alt+f1", "ctrl+alt+f2", "meta+l",
        ]
    )
    deny_text: list[str] = Field(
        default_factory=lambda: [
            r"rm\s+-[rf]", r"\bsudo\b", r"\bmkfs\b", r"\bdd\s+if=", r":\s*\(\)\s*\{.*\|.*&\s*\}",
            r"\bchmod\s+777\b", r"\bcurl\b.*\|\s*(ba)?sh", r"\bgit\s+push\b.*--force",
        ],
        description="regexes; a burst typing anything matching is refused",
    )
    deny_labels: list[str] = Field(
        default_factory=lambda: [
            "delete", "trash", "remove", "confirm", "purchase", "buy", "pay", "send",
            "approve", "allow", "grant", "accept", "uninstall", "format", "erase",
            "sign in", "log in", "password",
        ],
        description="clicking an observed element whose label matches is refused",
    )
    click_allow_regions: list[Rect] = Field(
        default_factory=list, description="if non-empty, clicks must land inside one of these"
    )
    click_deny_regions: list[Rect] = Field(default_factory=list)
    require_target_element: bool = Field(
        default=False,
        description="clicks must land on an element the vision layer actually reported",
    )
    max_actions_per_burst: int = Field(default=32, ge=1, le=128)
    max_burst_ms: int = Field(default=1500, ge=1, le=15_000)
    max_inputs_per_second: float = Field(default=60.0, gt=0, le=500.0)
    max_text_len: int = Field(default=256, ge=0, le=2048)
    allow_held_keys: bool = Field(
        default=True, description="permit d:/p: without a matching release inside the burst"
    )
    max_hold_ms: int = Field(
        default=4000, ge=0, le=30_000,
        description=(
            "auto-release anything held longer. This is the safety net for input nobody "
            "is watching -- a burst that ended with d: and no matching u:. Keys held by "
            "a latched `hold` reflex are exempt, because a guard re-evaluating them "
            "twenty times a second is stricter supervision than a timer."
        ),
    )
    max_latch_ms: int = Field(
        default=12_000, ge=100, le=120_000,
        description=(
            "absolute ceiling on a single `hold`, regardless of what its guard says. "
            "Latched keys are exempt from `max_hold_ms` because a guard re-checked at "
            "reflex rate is stricter supervision than a timer -- but that is only true "
            "while the guard can still see. A number probe whose region gets covered "
            "returns its last value forever rather than failing, so a guard reading it "
            "freezes without noticing and the key never comes up. This is the ceiling on "
            "that. Raise it for a genuinely long hold; do not disable it."
        ),
    )
    pointer_mode: Literal["absolute", "relative"] | None = Field(
        default=None,
        description=(
            "override the machine's pointer mode for this run only. Games with pointer "
            "lock need 'relative' -- absolute positioning fights the game's own camera, "
            "and mouse-look does not work at all. Desktop UI needs 'absolute'. Setting it "
            "here rather than in voltage.toml means a game playbook and a desktop "
            "playbook can run on the same machine without an edit and a restart between "
            "them. Restored when the run ends."
        ),
    )
    dry_run: bool = Field(
        default=True,
        description="parse, validate and journal bursts but never touch the input device",
    )

    @field_validator("allow_verbs")
    @classmethod
    def _known_verbs(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(VERBS))
        if unknown:
            raise ValueError(f"unknown verbs {unknown}; valid: {list(VERBS)}")
        return v


class Budget(BaseModel):
    """Run-level stop conditions. Every one of these is a hard abort."""

    model_config = ConfigDict(frozen=True)

    max_cycles: int = Field(default=240, ge=1, le=100_000)
    max_seconds: float = Field(default=180.0, gt=0, le=86_400)
    max_bursts: int = Field(default=240, ge=1, le=100_000)
    max_rejections: int = Field(
        default=12, ge=0, description="abort after this many governor refusals"
    )
    idle_abort_s: float = Field(
        default=25.0, ge=0, description="abort if the screen never changes for this long"
    )
    deadman_s: float = Field(
        default=6.0, ge=0.5, le=120.0,
        description="abort if a single cycle wedges for this long; releases all held input",
    )


class State(BaseModel):
    """One node of the machine.

    `brief` and `hint` are the entire instruction the actuator receives. Write them as
    imperatives with concrete nouns from `watch`. The actuator has no memory of the goal
    beyond what is in the prompt, and no ability to reason about what the state is *for*.
    """

    model_config = ConfigDict(frozen=True)

    brief: str = Field(max_length=280, description="imperative instruction for the actuator")
    watch: list[str] = Field(
        default_factory=list, max_length=12,
        description="the closed vocabulary of things the vision model may report here",
    )
    hint: str | None = Field(default=None, max_length=280)
    allow_verbs: list[str] | None = Field(
        default=None, description="narrow the policy's verb set for this state only"
    )
    on_enter: str | None = Field(default=None, description="burst run once on entry")
    on_exit: str | None = Field(default=None, description="burst run once on exit")
    reflex: list[ReflexRule] = Field(default_factory=list, max_length=16)
    transitions: list[Transition] = Field(default_factory=list, max_length=16)
    perception: Perception | None = Field(default=None, description="per-state override")
    max_cycles: int | None = Field(default=None, ge=1, le=10_000)
    timeout_s: float | None = Field(default=None, gt=0, le=3600)
    on_timeout: str | None = Field(
        default=None, description="state to jump to when max_cycles/timeout_s trips"
    )
    autonomous: bool = Field(
        default=True,
        description="if false, the actuator is skipped and only reflexes/on_enter run",
    )

    @field_validator("allow_verbs")
    @classmethod
    def _known_verbs(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        unknown = sorted(set(v) - set(VERBS))
        if unknown:
            raise ValueError(f"unknown verbs {unknown}; valid: {list(VERBS)}")
        return v


class Playbook(BaseModel):
    """The complete task specification handed to `voltage.run`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Ident
    goal: str = Field(max_length=600, description="human-readable; not shown to the small models")
    version: int = 1
    initial: str
    states: dict[str, State] = Field(min_length=1, max_length=64)
    vars: dict[str, int | float | str | bool] = Field(default_factory=dict, max_length=32)
    probes: list[ProbeSpec] = Field(default_factory=list, max_length=24)
    tunables: dict[Ident, Tunable] = Field(
        default_factory=dict, max_length=16,
        description=(
            "guard constants the optimiser may move, read with tune('name'). Keep the "
            "count low: the search gets a handful of episodes per minute, and every "
            "extra dimension costs episodes that could have gone into the ones that "
            "matter."
        ),
    )
    reward: Reward | None = Field(
        default=None,
        description="how to score an episode; required for `tunables` to do anything",
    )
    policy: Policy = Field(default_factory=Policy)
    budget: Budget = Field(default_factory=Budget)
    perception: Perception = Field(default_factory=Perception)
    success_when: str | None = Field(
        default=None, description="global guard; checked every cycle before transitions"
    )
    failure_when: str | None = Field(default=None, description="global abort guard")
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("states")
    @classmethod
    def _no_reserved_names(cls, v: dict[str, State]) -> dict[str, State]:
        for name in v:
            if name in TERMINALS or name.startswith("@"):
                raise ValueError(f"state name {name!r} is reserved")
            if not name.replace("_", "").isalnum() or not name[0].isalpha():
                raise ValueError(f"invalid state name {name!r}")
        return v

    @model_validator(mode="after")
    def _initial_exists(self) -> Playbook:
        if self.initial not in self.states:
            raise ValueError(
                f"initial state {self.initial!r} is not defined; have: {sorted(self.states)}"
            )
        return self


# --------------------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledReflex:
    """A one-shot reflex: guard plus the burst template it fires."""

    rule: ReflexRule
    guard: Guard
    template: BurstTemplate


@dataclass(frozen=True, slots=True)
class CompiledHold:
    """A latch: guard, its release condition, and the press/release bursts.

    `release` is stored as its own guard rather than being derived by negating `when` at
    evaluation time, because the useful case is precisely the one where it is *not* the
    negation. `when: probe('meters') < 90` with `release_when: probe('meters') > 110`
    leaves a 20 m dead band in which neither fires and the latch simply keeps its current
    state -- which is what stops it chattering.
    """

    rule: ReflexRule
    guard: Guard
    release: Guard | None
    press: Burst
    release_burst: Burst
    names: tuple[str, ...]
    # Number probes the guards read. When every one of these has stopped producing
    # values the guard is deciding from readings that died, and the latch has to let go
    # on its own -- the guard cannot notice, because a dead number probe returns its last
    # value rather than an error. This is the field that turns a stuck key into a
    # released one.
    depends_on_numbers: frozenset[str] = frozenset()


class CompiledState:
    """A state with its guards and bursts pre-parsed, ready for the hot loop."""

    __slots__ = (
        "name", "spec", "transitions", "reflexes", "holds", "on_enter", "on_exit",
        "allow_verbs",
    )

    def __init__(
        self,
        name: str,
        spec: State,
        transitions: list[tuple[Guard, Transition]],
        reflexes: list[CompiledReflex],
        holds: list[CompiledHold],
        on_enter: BurstTemplate | None,
        on_exit: BurstTemplate | None,
        allow_verbs: frozenset[str],
    ) -> None:
        self.name = name
        self.spec = spec
        self.transitions = transitions
        self.reflexes = reflexes
        self.holds = holds
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.allow_verbs = allow_verbs

    @property
    def vocabulary(self) -> list[str]:
        return list(self.spec.watch)


class CompiledPlaybook:
    """A validated playbook whose guards and bursts are parsed exactly once."""

    __slots__ = ("spec", "states", "success", "failure", "probe_ids", "warnings")

    def __init__(
        self,
        spec: Playbook,
        states: dict[str, CompiledState],
        success: Guard | None,
        failure: Guard | None,
        warnings: list[str],
    ) -> None:
        self.spec = spec
        self.states = states
        self.success = success
        self.failure = failure
        self.probe_ids = frozenset(p.id for p in spec.probes)
        self.warnings = warnings

    def state(self, name: str) -> CompiledState:
        try:
            return self.states[name]
        except KeyError:
            raise PlaybookError(f"no such state {name!r}") from None

    def perception_for(self, name: str) -> Perception:
        override = self.states[name].spec.perception
        return override or self.spec.perception


def _compile_guard(source: str, where: str, errors: list[str]) -> Guard | None:
    try:
        return Guard(source)
    except Exception as exc:  # noqa: BLE001 - collect all errors, do not stop at the first
        errors.append(f"{where}: {exc}")
        return None


def _compile_burst(source: str, where: str, errors: list[str]) -> BurstTemplate | None:
    try:
        return BurstTemplate(source)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{where}: {exc}")
        return None


def _compile_hold(
    rule: ReflexRule, where: str, errors: list[str]
) -> tuple[Burst, Burst, tuple[str, ...]] | None:
    try:
        return parse_hold(rule.hold or "")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{where}.hold: {exc}")
        return None


def _check_probes(guard: Guard, probe_ids: set[str], where: str, errors: list[str]) -> None:
    missing = guard.referenced_probes - probe_ids
    if missing:
        errors.append(f"{where} uses undefined probes {sorted(missing)}")


def _check_tunables(
    guard: Guard, tunable_ids: set[str], where: str, errors: list[str]
) -> None:
    missing = guard.referenced_tunables - tunable_ids
    if missing:
        errors.append(
            f"{where} calls tune({sorted(missing)}) but those are not declared in "
            f"`tunables`. An undeclared name silently falls back to its literal default "
            f"and is never optimised, so the playbook would look like it is learning "
            f"while nothing moves."
        )


def compile_playbook(spec: Playbook) -> CompiledPlaybook:
    """Fully validate a playbook: guards compile, bursts parse, graph is sound.

    Raises `PlaybookError` listing *every* problem found, not just the first -- the
    orchestrator should be able to fix a playbook in one round trip rather than N.
    """
    errors: list[str] = []
    warnings: list[str] = []

    known_states = set(spec.states)
    probe_ids = {p.id for p in spec.probes}
    tunable_ids = set(spec.tunables)
    base_verbs = frozenset(spec.policy.allow_verbs)

    success = _compile_guard(spec.success_when, "success_when", errors) if spec.success_when else None
    failure = _compile_guard(spec.failure_when, "failure_when", errors) if spec.failure_when else None

    compiled: dict[str, CompiledState] = {}
    reachable: set[str] = {spec.initial}

    for name, st in spec.states.items():
        where = f"states.{name}"
        watch = set(st.watch)

        transitions: list[tuple[Guard, Transition]] = []
        for i, tr in enumerate(st.transitions):
            guard = _compile_guard(tr.when, f"{where}.transitions[{i}].when", errors)
            if guard is None:
                continue
            if tr.to not in TERMINALS and tr.to not in known_states:
                errors.append(
                    f"{where}.transitions[{i}].to: unknown state {tr.to!r}; "
                    f"have {sorted(known_states)} plus {sorted(TERMINALS)}"
                )
            else:
                reachable.add(tr.to)
            missing_labels = guard.referenced_labels - watch
            if missing_labels:
                warnings.append(
                    f"{where}.transitions[{i}] tests for {sorted(missing_labels)} but they are "
                    f"not in this state's `watch` list, so the vision model can never report "
                    f"them and this transition can never fire"
                )
            missing_probes = guard.referenced_probes - probe_ids
            if missing_probes:
                errors.append(
                    f"{where}.transitions[{i}] uses undefined probes {sorted(missing_probes)}"
                )
            _check_tunables(guard, tunable_ids, f"{where}.transitions[{i}]", errors)
            transitions.append((guard, tr))

        reflexes: list[CompiledReflex] = []
        holds: list[CompiledHold] = []
        seen_reflex_ids: set[str] = set()
        for i, rx in enumerate(st.reflex):
            at = f"{where}.reflex[{i}]"
            if rx.id in seen_reflex_ids:
                errors.append(f"{at}: duplicate reflex id {rx.id!r}")
            seen_reflex_ids.add(rx.id)

            guard = _compile_guard(rx.when, f"{at}.when", errors)
            if guard is None:
                continue

            if rx.hold:
                parsed = _compile_hold(rx, at, errors)
                release_guard = (
                    _compile_guard(rx.release_when, f"{at}.release_when", errors)
                    if rx.release_when
                    else None
                )
                if parsed is None or (rx.release_when and release_guard is None):
                    continue
                press, release_burst, names = parsed
                _check_probes(guard, probe_ids, at, errors)
                _check_tunables(guard, tunable_ids, at, errors)
                if release_guard is not None:
                    _check_probes(release_guard, probe_ids, f"{at}.release_when", errors)
                    _check_tunables(
                        release_guard, tunable_ids, f"{at}.release_when", errors
                    )
                if rx.max_fires is not None:
                    warnings.append(
                        f"{at} is a hold, so `max_fires` counts how many times it may "
                        f"*engage*, not how many inputs it sends; a latch that has spent "
                        f"its fires stops responding entirely"
                    )
                number_ids = {p.id for p in spec.probes if p.type == "number"}
                reads = set(guard.referenced_numbers)
                if release_guard is not None:
                    reads |= release_guard.referenced_numbers
                holds.append(
                    CompiledHold(
                        rule=rx,
                        guard=guard,
                        release=release_guard,
                        press=press,
                        release_burst=release_burst,
                        names=names,
                        depends_on_numbers=frozenset(reads & number_ids),
                    )
                )
                continue

            template = _compile_burst(rx.do or "", f"{at}.do", errors)
            if template is None:
                continue
            _check_probes(guard, probe_ids, at, errors)
            _check_tunables(guard, tunable_ids, at, errors)
            missing = template.referenced_probes - probe_ids
            if missing:
                errors.append(f"{at}.do interpolates undefined probes {sorted(missing)}")
            reflexes.append(CompiledReflex(rule=rx, guard=guard, template=template))

        reflexes.sort(key=lambda item: -item.rule.priority)
        holds.sort(key=lambda item: -item.rule.priority)

        # Two latches fighting over one key is a stuck-input bug that only shows up when
        # both guards happen to disagree, which is exactly when you are least able to
        # debug it. Catch the overlap statically instead.
        claimed: dict[str, str] = {}
        for hold in holds:
            for target in hold.names:
                owner = claimed.get(target)
                if owner is not None:
                    errors.append(
                        f"{where}: hold reflexes {owner!r} and {hold.rule.id!r} both hold "
                        f"{target!r}; whichever releases first wins and the key state "
                        f"stops tracking either guard"
                    )
                claimed[target] = hold.rule.id

        on_enter = _compile_burst(st.on_enter, f"{where}.on_enter", errors) if st.on_enter else None
        on_exit = _compile_burst(st.on_exit, f"{where}.on_exit", errors) if st.on_exit else None
        for hook, template in (("on_enter", on_enter), ("on_exit", on_exit)):
            if template is None:
                continue
            missing = template.referenced_probes - probe_ids
            if missing:
                errors.append(
                    f"{where}.{hook} interpolates undefined probes {sorted(missing)}"
                )

        verbs = frozenset(st.allow_verbs) & base_verbs if st.allow_verbs else base_verbs
        if st.allow_verbs and not verbs:
            errors.append(
                f"{where}.allow_verbs is disjoint from policy.allow_verbs, so no action is "
                f"possible in this state"
            )

        if st.on_timeout and st.on_timeout not in TERMINALS and st.on_timeout not in known_states:
            errors.append(f"{where}.on_timeout: unknown state {st.on_timeout!r}")
        elif st.on_timeout:
            reachable.add(st.on_timeout)

        if st.autonomous and not st.transitions and not st.max_cycles and not st.timeout_s:
            warnings.append(
                f"{where} has no transitions and no cycle/time limit; it can only be left "
                f"by the run budget expiring"
            )
        if st.autonomous and not st.watch and not probe_ids:
            warnings.append(
                f"{where} is autonomous but has an empty `watch` list and there are no probes, "
                f"so the actuator will act blind"
            )
        if holds and not probe_ids:
            warnings.append(
                f"{where} has hold reflexes but the playbook declares no probes, so every "
                f"latch is driven by cached vision at decision rate. That is the one thing "
                f"a latch cannot do well -- give it a probe to key off"
            )

        compiled[name] = CompiledState(
            name=name,
            spec=st,
            transitions=transitions,
            reflexes=reflexes,
            holds=holds,
            on_enter=on_enter,
            on_exit=on_exit,
            allow_verbs=verbs,
        )

    orphans = known_states - reachable
    if orphans:
        warnings.append(f"unreachable states: {sorted(orphans)}")

    if not spec.success_when and not any(
        tr.to == "@success" for st in spec.states.values() for tr in st.transitions
    ):
        warnings.append(
            "no path to @success: this run can only end via budget, failure, or manual stop"
        )

    if errors:
        raise PlaybookError(
            f"playbook {spec.name!r} has {len(errors)} error(s)",
            errors=errors,
            warnings=warnings,
        )

    return CompiledPlaybook(spec, compiled, success, failure, warnings)


def strip_comments(value: Any) -> Any:
    """Drop underscore-prefixed keys anywhere in the structure.

    JSON has no comment syntax, and a playbook is a document a human and a model both
    have to read. `"_comment_policy": "..."` is the established convention for annotating
    one, so it is honoured here rather than colliding with `extra="forbid"`.
    """
    if isinstance(value, dict):
        return {k: strip_comments(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [strip_comments(v) for v in value]
    return value


def playbook_from_dict(data: dict[str, Any]) -> CompiledPlaybook:
    """Validate + compile in one step, normalising pydantic errors into PlaybookError."""
    try:
        spec = Playbook.model_validate(strip_comments(data))
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError has a huge repr
        raise PlaybookError("playbook failed schema validation", errors=[str(exc)]) from exc
    return compile_playbook(spec)
