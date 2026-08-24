"""Distil the actuator into a lookup table, online, while it runs.

The idea
--------
The actuator's decision is a function of a small, mostly-repeating state: which playbook
state we are in, which labels the vision layer reported, and the current probe readings.
In a game loop that state recurs constantly -- every launch looks like the last launch,
every descent passes through the same band of altitudes -- and we pay a full model round
trip to re-derive an answer we have already derived, often within the same minute.

So remember it. On a hit, the burst comes back in microseconds instead of ~300 ms, and it
is not a *worse* answer than the model's: it is literally the model's own previous answer
to a situation this close. The cache is a distillation of the actuator into a table, built
for free out of decisions it was going to make anyway.

This is the honest version of "smarter without reducing speed". Nothing is approximated
and no second model is trained. The first encounter with a situation costs what it always
cost; every later one is free, and the fraction of free cycles climbs as the run goes on.

Why nearest-neighbour rather than an exact key
----------------------------------------------
The readings are continuous. `meters=261.0` will never recur exactly, so an exact key
would never hit. But the *decision* is piecewise-constant over those readings -- what you
do at 261 m is what you do at 258 m -- so a neighbour within a small normalised radius is
the same situation for decision purposes.

The discrete part is matched exactly and never approximated. A cached burst from another
state, or from a moment when different things were on screen, is not a near-miss; it is a
different question. Mixing those is how a cache starts confidently doing the wrong thing.

Normalisation is per-probe, against the range actually observed during the run, because
the readings are wildly heterogeneous: metres run 0-500, a brightness probe runs 0-1, and
an unnormalised Euclidean distance would be entirely decided by whichever probe happened
to have the largest units.

Guard rails
-----------
A cache that learns a mistake and then repeats it forever is worse than no cache, so:

  * **Only successful decisions are stored.** The burst had to parse, pass the governor,
    and execute without error. A refused or unparseable burst teaches nothing worth
    keeping.
  * **A fraction of cycles bypass the cache on purpose.** Without that the table freezes
    at whatever it learned in the first minute and can never notice it was wrong. The
    bypass rate is what buys the ability to change its mind.
  * **Entries can be marked bad from outside.** The episode reward knows things a single
    cycle does not; `penalise()` lets a bad episode retire the entries that fed it.
  * **Bounded.** Least-recently-used eviction, so a long run cannot grow it without limit.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PolicyCache", "CacheEntry", "situation_key"]

# Two situations count as the same when their normalised probe vectors are within this
# distance. Deliberately small: 0.06 over a range-normalised vector is a few percent of
# the span of each reading. Larger and the cache starts answering questions it was not
# asked -- the failure mode that matters here, since a wrong burst is an input to a game.
_DEFAULT_RADIUS = 0.06
# Fraction of cycles that ask the model even when the cache could answer. This is not
# waste; it is the only mechanism by which a stale table gets corrected.
_DEFAULT_BYPASS = 0.12
_MAX_ENTRIES = 512
# An entry must survive this many penalties before it is dropped, so one unlucky episode
# does not retire a decision that is usually right.
_PENALTY_LIMIT = 2
# Floor on the per-probe scale, as a fraction of the readings' own magnitude. Keeps the
# cache usable before it has observed a probe's full range -- see `_scale`.
_MIN_RELATIVE_SPAN = 0.5
_EPS = 1e-9


def situation_key(
    state: str, labels: list[str], targets: list[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """The part of a situation that must match exactly.

    Sorted, because "the vision layer reported a button and a dialog" is the same
    situation as "a dialog and a button" -- the order it happened to emit them in is not
    part of the question.
    """
    return (state, tuple(sorted(set(labels))), tuple(targets))


@dataclass(slots=True)
class CacheEntry:
    vector: dict[str, float]
    burst: str
    proposed_state: str | None
    note: str
    hits: int = 0
    penalties: int = 0
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)


class PolicyCache:
    """Nearest-neighbour recall of the actuator's own past decisions."""

    def __init__(
        self,
        *,
        radius: float = _DEFAULT_RADIUS,
        bypass: float = _DEFAULT_BYPASS,
        max_entries: int = _MAX_ENTRIES,
        seed: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.radius = radius
        self.bypass = bypass
        self.max_entries = max_entries
        self.enabled = enabled
        self._rng = random.Random(seed)
        self._buckets: dict[tuple, list[CacheEntry]] = {}
        # Per-probe observed range, for normalisation. Seeded from the first value seen
        # and widened as the run goes on.
        self._lo: dict[str, float] = {}
        self._hi: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.stored = 0
        self.evicted = 0
        self.retired = 0

    # -- normalisation ---------------------------------------------------------------

    def observe(self, vector: dict[str, float]) -> None:
        """Widen the known range of every reading. Cheap; called every lookup."""
        for key, value in vector.items():
            if not math.isfinite(value):
                continue
            lo = self._lo.get(key)
            if lo is None or value < lo:
                self._lo[key] = value
            hi = self._hi.get(key)
            if hi is None or value > hi:
                self._hi[key] = value

    def _scale(self, key: str, a: float, b: float) -> float:
        """The distance at which two readings of `key` count as different situations.

        Range normalisation alone has a cold-start problem that makes the cache useless
        exactly when it would help most. Two observations of `meters` at 200 and 201 give
        an observed span of 1, so those two readings come out maximally far apart and the
        cache cannot hit until it has already seen the full range -- by which point the
        run is over.

        So the scale is the *larger* of the observed range and a fraction of the readings'
        own magnitude. Early on that means "within a few percent of each other", which is
        the right notion of same-situation when nothing else is known; later the observed
        range takes over and is sharper.
        """
        observed = self._hi.get(key, 0.0) - self._lo.get(key, 0.0)
        magnitude = (abs(a) + abs(b)) / 2.0
        return max(observed, magnitude * _MIN_RELATIVE_SPAN, _EPS)

    def _distance(self, a: dict[str, float], b: dict[str, float]) -> float:
        """Normalised Euclidean distance over the readings both share.

        Normalising per probe is what makes the radius mean the same thing for a reading
        in metres and one in the unit interval. Without it the distance is decided
        entirely by whichever probe has the largest units, and the radius stops being a
        meaningful threshold for anything else.
        """
        keys = a.keys() & b.keys()
        if not keys:
            return math.inf
        total = 0.0
        for key in keys:
            delta = a[key] - b[key]
            if delta:
                total += (delta / self._scale(key, a[key], b[key])) ** 2
        return math.sqrt(total / len(keys))

    # -- the loop --------------------------------------------------------------------

    def lookup(self, key: tuple, vector: dict[str, float]) -> CacheEntry | None:
        """The nearest stored decision for this situation, or None to ask the model."""
        if not self.enabled:
            return None
        self.observe(vector)

        # Ask the model anyway, sometimes. A table that can only ever be added to is a
        # table that cannot notice it was wrong.
        if self._rng.random() < self.bypass:
            self.bypasses += 1
            return None

        bucket = self._buckets.get(key)
        if not bucket:
            self.misses += 1
            return None

        best, best_distance = None, math.inf
        for entry in bucket:
            distance = self._distance(entry.vector, vector)
            if distance < best_distance:
                best, best_distance = entry, distance

        if best is None or best_distance > self.radius:
            self.misses += 1
            return None

        best.hits += 1
        best.last_used = time.monotonic()
        self.hits += 1
        return best

    def store(
        self,
        key: tuple,
        vector: dict[str, float],
        *,
        burst: str,
        proposed_state: str | None = None,
        note: str = "",
    ) -> None:
        """Remember a decision that actually worked.

        Callers are expected to have checked that: the burst parsed, the governor allowed
        it, and it executed without error. Storing anything else teaches the table to
        repeat a failure.
        """
        if not self.enabled or not burst or burst == ".":
            # An empty burst is the actuator declining to act. Caching "do nothing" is
            # how a loop gets stuck doing nothing quickly instead of slowly.
            return
        self.observe(vector)
        bucket = self._buckets.setdefault(key, [])

        # Replace a near-duplicate rather than accumulating a cluster of them.
        for entry in bucket:
            if self._distance(entry.vector, vector) <= self.radius * 0.5:
                entry.vector = dict(vector)
                entry.burst = burst
                entry.proposed_state = proposed_state
                entry.note = note
                entry.penalties = 0
                entry.last_used = time.monotonic()
                return

        bucket.append(
            CacheEntry(
                vector=dict(vector), burst=burst,
                proposed_state=proposed_state, note=note,
            )
        )
        self.stored += 1
        self._evict()

    def penalise(self, key: tuple, vector: dict[str, float]) -> None:
        """Mark the decision nearest this situation as having led somewhere bad.

        Used by the episode layer, which knows what a single cycle cannot: that the run
        this decision belonged to scored badly. Repeated penalties retire the entry.
        """
        bucket = self._buckets.get(key)
        if not bucket:
            return
        best, best_distance = None, math.inf
        for entry in bucket:
            distance = self._distance(entry.vector, vector)
            if distance < best_distance:
                best, best_distance = entry, distance
        if best is None or best_distance > self.radius:
            return
        best.penalties += 1
        if best.penalties >= _PENALTY_LIMIT:
            bucket.remove(best)
            self.retired += 1

    def _evict(self) -> None:
        total = sum(len(b) for b in self._buckets.values())
        if total <= self.max_entries:
            return
        oldest_key, oldest_entry = None, None
        for key, bucket in self._buckets.items():
            for entry in bucket:
                if oldest_entry is None or entry.last_used < oldest_entry.last_used:
                    oldest_key, oldest_entry = key, entry
        if oldest_key is not None and oldest_entry is not None:
            self._buckets[oldest_key].remove(oldest_entry)
            if not self._buckets[oldest_key]:
                del self._buckets[oldest_key]
            self.evicted += 1

    # -- reporting -------------------------------------------------------------------

    @property
    def size(self) -> int:
        return sum(len(b) for b in self._buckets.values())

    def stats(self) -> dict[str, Any]:
        asked = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "entries": self.size,
            "situations": len(self._buckets),
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "retired": self.retired,
            "evicted": self.evicted,
            # The number that says whether this is doing anything: the share of decisions
            # answered without a model round trip.
            "hit_rate": round(self.hits / asked, 3) if asked else None,
            "radius": self.radius,
            "bypass": self.bypass,
        }
