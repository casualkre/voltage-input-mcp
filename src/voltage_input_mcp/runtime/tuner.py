"""Episodic optimisation of guard constants against an on-screen reward.

Why this exists
---------------
Every threshold in a reflex is a number somebody guessed. "Brake below 90 metres",
"boost when falling faster than 18 a second", "tuck under 32" -- those came out of my
head, not out of the game, and there is no reason to believe any of them is right. A
playbook full of guessed constants is the reason a fast layer looks like "hold W and
pray": the structure is sound and the numbers are arbitrary.

The fix is not a bigger model. It is to notice that the game already tells you the score.
Broken Bones prints money on the HUD; a file manager either opened the folder or did not;
a crafting loop either produced the item or did not. If a run can be cut into episodes and
each episode scored, the constants can be optimised directly, and the optimiser costs
nothing at inference because its output *is just better constants*. The fast layer stays a
handful of comparisons running at 50 Hz. It simply stops being wrong.

Choice of optimiser
-------------------
A (1+1) evolution strategy with success-based step adaptation. The reasons are all about
the shape of this problem rather than a preference for ES in general:

  * **Very few evaluations.** An episode is a real fall in a real game -- five to fifteen
    seconds. A hundred episodes is twenty minutes. Anything needing thousands of samples
    (policy gradients, most RL) is disqualified before it starts.
  * **Very noisy rewards.** The same constants can earn $18k or $40k depending on where
    the character happened to land. So a proposal is only accepted on a *smoothed*
    comparison, and the step size shrinks when proposals keep failing -- that is what
    stops the search chasing one lucky episode.
  * **Few parameters.** Five to ten. CMA-ES would be the textbook answer above roughly
    twenty; below that its covariance estimate is mostly noise and it needs more samples
    to pay for itself than we will ever have.
  * **Bounded and interpretable.** Every parameter has a declared range and stays a plain
    number the orchestrator can read, reason about, and write back into a playbook. A
    learned weight matrix would be faster to fit and impossible to explain.

`sigma` is a *fraction of each parameter's declared range*, so one step size works across
parameters measured in metres, milliseconds and metres-per-second without any scaling by
hand.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Tunable", "Episode", "Tuner", "tuner_store_path"]

# Step size as a fraction of each parameter's range. 0.25 explores briskly at the start;
# the adaptation below pulls it down as the search converges.
_SIGMA0 = 0.25
_SIGMA_MIN = 0.02
_SIGMA_MAX = 0.6
# Classic 1/5th success rule: grow the step while more than a fifth of proposals win,
# shrink it otherwise. The constants are the standard ones and are not sensitive.
_GROW = 1.28
_SHRINK = 0.87
# How many episodes to average when comparing a proposal against the incumbent. Rewards
# here are noisy enough that a single episode decides nothing.
_REPEATS = 2
# How far to lower a score carried over from a previous session before treating it as the
# bar to beat. Applied as a fraction of magnitude so it lowers the bar for negative
# rewards too.
_STALE_DISCOUNT = 0.25


@dataclass(frozen=True, slots=True)
class Tunable:
    """One optimisable constant, with the bounds that make the search safe."""

    name: str
    default: float
    low: float
    high: float

    def clamp(self, value: float) -> float:
        return min(self.high, max(self.low, value))

    @property
    def span(self) -> float:
        return max(1e-9, self.high - self.low)


@dataclass(slots=True)
class Episode:
    """One scored attempt."""

    index: int
    params: dict[str, float]
    reward: float
    seconds: float
    note: str = ""
    t: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tuner_store_path(playbook: str) -> Path:
    from ..config import state_dir

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in playbook)[:64]
    return state_dir() / "tuned" / f"{safe}.json"


class Tuner:
    """Proposes parameter sets and keeps the best one found.

    The contract with the session is deliberately small: `current()` before an episode,
    `record(reward)` after it. Everything about the search stays in here, so replacing the
    algorithm later touches one file.
    """

    def __init__(
        self,
        tunables: list[Tunable],
        *,
        seed: int | None = None,
        explore: bool = True,
    ) -> None:
        self.tunables = {t.name: t for t in tunables}
        self.explore = explore and bool(tunables)
        self._rng = random.Random(seed)

        self.best = {t.name: float(t.default) for t in tunables}
        self.best_score: float | None = None
        self._candidate: dict[str, float] | None = None
        self._pending: list[float] = []
        self._sigma = _SIGMA0
        self._proposals = 0
        self._accepted = 0
        self.episodes: list[Episode] = []

    # -- the loop --------------------------------------------------------------------

    def current(self) -> dict[str, float]:
        """Parameters for the episode about to start."""
        if not self.explore:
            return dict(self.best)
        if self._candidate is None:
            self._candidate = self._propose()
        return dict(self._candidate)

    def record(self, reward: float, *, seconds: float = 0.0, note: str = "") -> None:
        """Score the episode that just finished and decide what to try next."""
        params = self.current()
        self.episodes.append(
            Episode(index=len(self.episodes) + 1, params=dict(params),
                    reward=float(reward), seconds=round(seconds, 2), note=note)
        )
        if not self.explore:
            return

        # The very first episode establishes the incumbent's score rather than being
        # compared against nothing.
        if self.best_score is None:
            self.best_score = float(reward)
            self._candidate = None
            return

        self._pending.append(float(reward))
        if len(self._pending) < _REPEATS:
            return  # keep the same candidate and sample it again

        score = sum(self._pending) / len(self._pending)
        self._pending.clear()
        self._proposals += 1

        if score > self.best_score:
            self.best = dict(params)
            # Track the incumbent toward the new score rather than jumping to it: the
            # winning episodes were also noisy, and jumping makes the next comparison
            # unfairly hard to beat.
            self.best_score = 0.4 * self.best_score + 0.6 * score
            self._accepted += 1
        self._adapt()
        self._candidate = None

    # -- search ----------------------------------------------------------------------

    def _propose(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, spec in self.tunables.items():
            step = self._rng.gauss(0.0, self._sigma * spec.span)
            out[name] = spec.clamp(self.best[name] + step)
        return out

    def _adapt(self) -> None:
        """The 1/5th success rule: grow the step while winning, shrink while losing."""
        if self._proposals < 4:
            return
        rate = self._accepted / self._proposals
        self._sigma *= _GROW if rate > 0.2 else _SHRINK
        self._sigma = min(_SIGMA_MAX, max(_SIGMA_MIN, self._sigma))

    # -- reporting and persistence ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        rewards = [e.reward for e in self.episodes]
        first = rewards[: max(1, len(rewards) // 3)]
        last = rewards[-max(1, len(rewards) // 3):]
        return {
            "episodes": len(self.episodes),
            "exploring": self.explore,
            "sigma": round(self._sigma, 4),
            "proposals": self._proposals,
            "accepted": self._accepted,
            "best_score": None if self.best_score is None else round(self.best_score, 2),
            "best": {k: round(v, 3) for k, v in self.best.items()},
            "defaults": {k: t.default for k, t in self.tunables.items()},
            "mean_reward": round(sum(rewards) / len(rewards), 2) if rewards else None,
            # First third versus last third is the only honest read on whether the search
            # actually improved anything, given how noisy a single episode is.
            "early_mean": round(sum(first) / len(first), 2) if first else None,
            "late_mean": round(sum(last) / len(last), 2) if last else None,
        }

    def improvement(self) -> float | None:
        """Late-episode mean minus early-episode mean, or None with too little data."""
        s = self.summary()
        if len(self.episodes) < 6 or s["early_mean"] is None:
            return None
        return round(s["late_mean"] - s["early_mean"], 2)

    def save(self, playbook: str) -> Path:
        """Persist the best-known parameters so the next run starts where this ended.

        This is the part that makes it learn across sessions rather than rediscovering
        the same constants every time. Only the parameters and a little provenance are
        stored -- the episode log stays in the run journal, which is where a post-mortem
        would look for it.
        """
        path = tuner_store_path(playbook)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "playbook": playbook,
            "updated": time.time(),
            "best": self.best,
            "best_score": self.best_score,
            "episodes": len(self.episodes),
            "bounds": {k: [t.low, t.high] for k, t in self.tunables.items()},
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load(self, playbook: str) -> bool:
        """Adopt previously learned parameters. Returns whether anything was loaded.

        Stored values are re-clamped to the bounds the *current* playbook declares, and
        names that no longer exist are dropped, so editing a playbook cannot resurrect a
        stale parameter or one that is now out of range.
        """
        path = tuner_store_path(playbook)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        stored = data.get("best")
        if not isinstance(stored, dict):
            return False

        adopted = False
        for name, value in stored.items():
            spec = self.tunables.get(name)
            if spec is None or not isinstance(value, (int, float)):
                continue
            self.best[name] = spec.clamp(float(value))
            adopted = True
        if adopted and isinstance(data.get("best_score"), (int, float)):
            # Deliberately discounted. The stored score came from a previous session --
            # possibly a different map, a different character, a different day -- and
            # adopting it verbatim would make every proposal this session fail to beat a
            # number that no longer describes anything.
            #
            # Subtracting a fraction of the magnitude, rather than scaling: a reward may
            # be negative (a cost, or a `rate` that went backwards), and multiplying a
            # negative score by 0.75 raises it, making the bar *harder* -- the exact
            # opposite of discounting. This lowers the bar whatever the sign.
            stored = float(data["best_score"])
            self.best_score = stored - abs(stored) * _STALE_DISCOUNT
        return adopted


def parse_tunables(spec: dict[str, Any]) -> list[Tunable]:
    """Build Tunables from a playbook's `tunables` block, validating bounds."""
    out: list[Tunable] = []
    for name, body in (spec or {}).items():
        if not isinstance(body, dict):
            raise ValueError(f"tunable {name!r} must be an object with default/min/max")
        try:
            default = float(body["default"])
            low = float(body["min"])
            high = float(body["max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"tunable {name!r} needs numeric `default`, `min` and `max`"
            ) from exc
        if low >= high:
            raise ValueError(f"tunable {name!r}: min ({low}) must be below max ({high})")
        if not low <= default <= high:
            raise ValueError(
                f"tunable {name!r}: default {default} is outside [{low}, {high}]"
            )
        if not all(math.isfinite(v) for v in (default, low, high)):
            raise ValueError(f"tunable {name!r}: bounds must be finite")
        out.append(Tunable(name=name, default=default, low=low, high=high))
    return out
