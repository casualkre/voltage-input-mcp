"""The governor: the last thing between a 1.7B model and the user's keyboard.

Everything upstream of this is a preference. Grammars make bad output improbable, prompts
make it unlikely, the state machine makes it narrow -- but the governor is the only layer
that is *not* advisory. Every burst passes through `review()` before a single event
reaches `/dev/uinput`, including reflex bursts and bursts the orchestrator wrote by hand.

Design decisions worth stating outright:

**Refusal is whole-burst, never partial.** A burst is one intended act. Executing the
first six actions of a click-then-type sequence and dropping the rest leaves the desktop
in a state nobody planned for, which is worse than doing nothing. If any action is
refused, the whole burst is refused and the cycle is journalled as a rejection.

**Two independent chokepoints on dangerous clicks.** Region fencing catches "do not touch
this part of the screen"; `deny_labels` catches "do not click anything *called* Delete,
wherever it happens to be". The second is what actually protects against the user's stated
concern -- a confirmation dialog appearing somewhere unpredictable and the actuator
cheerfully clicking Confirm.

**The default is restrictive.** `Policy` opts into capability rather than out of danger,
and `dry_run` defaults to True, so a freshly authored playbook journals what it *would*
do and touches nothing until the orchestrator explicitly turns execution on.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..errors import SafetyViolation
from ..inputs import keymap as km
from ..models.burst import (
    Action,
    Burst,
    ButtonDown,
    ButtonUp,
    Click,
    KeyChord,
    KeyDown,
    KeyUp,
    MoveAbs,
    MoveRel,
    Scroll,
    TypeText,
    Wait,
)
from ..models.observation import Element, Observation
from ..models.playbook import Policy

__all__ = ["Governor", "Verdict", "Violation"]


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    detail: str
    action: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"rule": self.rule, "detail": self.detail, "action": self.action}


@dataclass(slots=True)
class Verdict:
    allowed: bool
    violations: list[Violation] = field(default_factory=list)
    burst: Burst | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.allowed:
            return "allowed"
        return "; ".join(f"{v.rule}: {v.detail}" for v in self.violations)

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "violations": [v.as_dict() for v in self.violations],
            "notes": self.notes,
        }

    def raise_if_denied(self) -> None:
        if not self.allowed:
            first = self.violations[0] if self.violations else Violation("unknown", "denied")
            raise SafetyViolation(self.reason, rule=first.rule, action=first.action)


class _RateBucket:
    """Token bucket over input events, so a burst cannot exceed a sustained rate.

    Per-burst caps alone are not enough: twenty bursts of thirty actions each, back to
    back, is six hundred inputs a second regardless of how modest any single burst looks.
    """

    __slots__ = ("_capacity", "_tokens", "_rate", "_last")

    def __init__(self, rate_per_s: float, burst_capacity: float | None = None) -> None:
        self._rate = max(0.1, rate_per_s)
        self._capacity = burst_capacity if burst_capacity is not None else max(rate_per_s, 1.0)
        self._tokens = self._capacity
        self._last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now

    def check(self, count: int) -> tuple[bool, float]:
        """Would `count` inputs fit? Returns (ok, seconds_to_wait)."""
        self._refill()
        if count <= self._tokens:
            return True, 0.0
        deficit = count - self._tokens
        return False, deficit / self._rate

    def consume(self, count: int) -> None:
        self._refill()
        self._tokens = max(0.0, self._tokens - count)

    def reset(self) -> None:
        self._tokens = self._capacity
        self._last = time.monotonic()


class Governor:
    """Stateful policy enforcement for one session."""

    def __init__(
        self,
        policy: Policy,
        *,
        screen: tuple[int, int],
        max_rejections: int = 12,
    ) -> None:
        self.policy = policy
        self.screen = screen
        self.max_rejections = max_rejections

        self._deny_text = _compile_patterns(policy.deny_text, "policy.deny_text")
        self._deny_labels = _compile_label_patterns(policy.deny_labels)
        self._deny_chords = {
            _normalise_chord(c.split("+")) for c in policy.deny_chords
        }
        self._deny_keys = {km.canonical_key(k) for k in policy.deny_keys}
        self._allow_keys = (
            {km.canonical_key(k) for k in policy.allow_keys}
            if policy.allow_keys is not None
            else None
        )
        self._bucket = _RateBucket(policy.max_inputs_per_second)

        self.rejections = 0
        self.approvals = 0
        self._history: list[Violation] = []

    # -- public API ------------------------------------------------------------------

    @property
    def budget_exhausted(self) -> bool:
        return self.rejections >= self.max_rejections

    def recent_violations(self, limit: int = 5) -> list[dict[str, str | None]]:
        return [v.as_dict() for v in self._history[-limit:]]

    def review(
        self,
        burst: Burst,
        *,
        observation: Observation | None = None,
        allow_verbs: Iterable[str] | None = None,
        source: str = "actuator",
    ) -> Verdict:
        """Check a burst against the policy. Does not execute anything."""
        violations: list[Violation] = []
        notes: list[str] = []

        if not burst.actions:
            return Verdict(allowed=True, burst=burst, notes=["empty burst"])

        permitted = (
            set(allow_verbs) if allow_verbs is not None else set(self.policy.allow_verbs)
        )

        if len(burst.actions) > self.policy.max_actions_per_burst:
            violations.append(
                Violation(
                    "max_actions_per_burst",
                    f"{len(burst.actions)} actions exceeds the limit of "
                    f"{self.policy.max_actions_per_burst}",
                )
            )

        if burst.duration_ms > self.policy.max_burst_ms:
            violations.append(
                Violation(
                    "max_burst_ms",
                    f"burst would run for {burst.duration_ms} ms, limit is "
                    f"{self.policy.max_burst_ms} ms",
                )
            )

        ok, wait_s = self._bucket.check(burst.input_count)
        if not ok:
            violations.append(
                Violation(
                    "max_inputs_per_second",
                    f"{burst.input_count} inputs would exceed "
                    f"{self.policy.max_inputs_per_second:.0f}/s; needs {wait_s:.2f}s more headroom",
                )
            )

        cursor = None
        for action in burst.actions:
            rendered = action.render()
            verb = rendered.split(":", 1)[0]
            if verb not in permitted:
                violations.append(
                    Violation(
                        "allow_verbs",
                        f"verb {verb!r} is not permitted in this state "
                        f"(allowed: {' '.join(sorted(permitted))})",
                        rendered,
                    )
                )
                continue
            cursor = self._check_action(action, cursor, observation, violations)

        if not self.policy.allow_held_keys:
            held = burst.held_keys() | {f"btn:{b}" for b in burst.held_buttons()}
            if held:
                violations.append(
                    Violation(
                        "allow_held_keys",
                        f"burst ends with {sorted(held)} still held and this policy "
                        f"forbids cross-burst holds",
                    )
                )
        elif burst.held_keys() or burst.held_buttons():
            notes.append(
                f"leaves held: {sorted(burst.held_keys() | burst.held_buttons())} "
                f"(auto-released after {self.policy.max_hold_ms} ms)"
            )

        allowed = not violations
        if allowed:
            self._bucket.consume(burst.input_count)
            self.approvals += 1
        else:
            self.rejections += 1
            self._history.extend(violations)

        if self.policy.dry_run:
            notes.append("dry_run: parsed and checked, nothing will be injected")

        return Verdict(allowed=allowed, violations=violations, burst=burst, notes=notes)

    def reset_rate(self) -> None:
        self._bucket.reset()

    # -- per-action checks -----------------------------------------------------------

    def _check_action(
        self,
        action: Action,
        cursor: tuple[int, int] | None,
        observation: Observation | None,
        violations: list[Violation],
    ) -> tuple[int, int] | None:
        rendered = action.render()

        if isinstance(action, KeyChord):
            self._check_keys(action.keys, rendered, violations)
            chord = _normalise_chord(action.keys)
            if chord in self._deny_chords:
                violations.append(
                    Violation("deny_chords", f"chord {'+'.join(action.keys)} is denied", rendered)
                )
            return cursor

        if isinstance(action, (KeyDown, KeyUp)):
            self._check_keys((action.key,), rendered, violations)
            return cursor

        if isinstance(action, TypeText):
            self._check_text(action.text, rendered, violations)
            return cursor

        if isinstance(action, MoveAbs):
            self._check_bounds(action.x, action.y, rendered, violations)
            return (action.x, action.y)

        if isinstance(action, MoveRel):
            if cursor is None:
                # Unknown starting point: a relative move cannot be fenced, so a policy
                # that fences clicks must also forbid blind relative motion.
                if self.policy.click_allow_regions:
                    violations.append(
                        Violation(
                            "click_allow_regions",
                            "relative move with unknown cursor position cannot be "
                            "region-checked; emit an absolute m: first",
                            rendered,
                        )
                    )
                return None
            return (
                max(0, min(cursor[0] + action.dx, self.screen[0] - 1)),
                max(0, min(cursor[1] + action.dy, self.screen[1] - 1)),
            )

        if isinstance(action, (Click, ButtonDown)):
            self._check_click(cursor, observation, rendered, violations)
            return cursor

        if isinstance(action, (ButtonUp, Scroll, Wait)):
            return cursor

        return cursor

    def _check_keys(self, keys: Sequence[str], rendered: str, violations: list[Violation]) -> None:
        for name in keys:
            canonical = km.canonical_key(name)
            if km.resolve_key(name) is None:
                violations.append(
                    Violation("unknown_key", f"key {name!r} is not a known key", rendered)
                )
                continue
            if canonical in self._deny_keys:
                violations.append(
                    Violation("deny_keys", f"key {name!r} is denied by policy", rendered)
                )
            if self._allow_keys is not None and canonical not in self._allow_keys:
                violations.append(
                    Violation(
                        "allow_keys",
                        f"key {name!r} is not in this policy's allowlist",
                        rendered,
                    )
                )

    def _check_text(self, text: str, rendered: str, violations: list[Violation]) -> None:
        if len(text) > self.policy.max_text_len:
            violations.append(
                Violation(
                    "max_text_len",
                    f"text is {len(text)} chars, limit is {self.policy.max_text_len}",
                    rendered,
                )
            )
        for pattern in self._deny_text:
            if pattern.search(text):
                violations.append(
                    Violation(
                        "deny_text",
                        f"text matches the denied pattern /{pattern.pattern}/",
                        # Do not echo the full text into the journal; it may be a secret
                        # the model picked up off screen.
                        rendered[:24] + "...",
                    )
                )

    def _check_bounds(self, x: int, y: int, rendered: str, violations: list[Violation]) -> None:
        w, h = self.screen
        if not (0 <= x < w and 0 <= y < h):
            violations.append(
                Violation(
                    "screen_bounds",
                    f"({x},{y}) is outside the {w}x{h} desktop",
                    rendered,
                )
            )

    def _check_click(
        self,
        cursor: tuple[int, int] | None,
        observation: Observation | None,
        rendered: str,
        violations: list[Violation],
    ) -> None:
        if cursor is None:
            if self.policy.click_allow_regions or self.policy.require_target_element:
                violations.append(
                    Violation(
                        "click_position_unknown",
                        "click with no preceding absolute move cannot be checked against "
                        "the click policy",
                        rendered,
                    )
                )
            return

        x, y = cursor

        for rect in self.policy.click_deny_regions:
            if rect.contains(x, y):
                violations.append(
                    Violation(
                        "click_deny_regions",
                        f"({x},{y}) falls inside a denied region {rect.as_tuple()}",
                        rendered,
                    )
                )

        if self.policy.click_allow_regions and not any(
            r.contains(x, y) for r in self.policy.click_allow_regions
        ):
            violations.append(
                Violation(
                    "click_allow_regions",
                    f"({x},{y}) is outside every permitted click region",
                    rendered,
                )
            )

        target: Element | None = observation.at(x, y) if observation else None

        if self.policy.require_target_element and target is None:
            violations.append(
                Violation(
                    "require_target_element",
                    f"({x},{y}) does not land on any element the vision layer reported",
                    rendered,
                )
            )

        if target is not None:
            for pattern in self._deny_labels:
                if pattern.search(target.label):
                    violations.append(
                        Violation(
                            "deny_labels",
                            f"({x},{y}) targets {target.label!r}, which matches the denied "
                            f"label pattern /{pattern.pattern}/",
                            rendered,
                        )
                    )
                    break


# -- helpers -------------------------------------------------------------------------------


def _compile_patterns(patterns: Sequence[str], where: str) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw, re.IGNORECASE))
        except re.error as exc:
            raise SafetyViolation(
                f"{where} contains an invalid regex /{raw}/: {exc}", rule=where
            ) from exc
    return compiled


def _compile_label_patterns(labels: Sequence[str]) -> list[re.Pattern[str]]:
    """Deny-labels are matched as whole words, not substrings.

    Substring matching would refuse to click "Send to Trash" *and* "Resend", but also
    "Undelete" and, more annoyingly, any element whose label happens to contain "buy"
    like "Buyer name". Word boundaries keep the rule tight enough to stay on.
    """
    return [re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE) for label in labels]


def _normalise_chord(keys: Sequence[str]) -> tuple[str, ...]:
    """Canonicalise and sort a chord so ctrl+alt+del == alt+ctrl+delete."""
    return tuple(sorted(km.canonical_key(k) for k in keys))
