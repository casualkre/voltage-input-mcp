"""Recall: does distilling the actuator into a table actually save calls, safely?

Two things have to be true at once, and they pull against each other. It has to hit often
enough to be worth having, and it must never answer a question it was not asked -- a wrong
burst is an input to a live game, not a wrong search result.
"""

from __future__ import annotations

from voltage_input_mcp.runtime.recall import PolicyCache, situation_key

KEY = situation_key("fall", ["ground"], ["landed"])
OTHER = situation_key("launch", ["ground"], ["fall"])


def cache(**kw) -> PolicyCache:
    kw.setdefault("bypass", 0.0)  # deterministic unless a test is about the bypass
    kw.setdefault("seed", 1)
    return PolicyCache(**kw)


def test_a_repeated_situation_is_answered_without_the_model():
    c = cache()
    c.store(KEY, {"meters": 200.0}, burst="k:shift")
    hit = c.lookup(KEY, {"meters": 201.0})
    assert hit is not None and hit.burst == "k:shift"


def test_a_different_state_is_a_different_question():
    """The discrete part is matched exactly. Approximating it is how a cache starts
    confidently doing the wrong thing in the wrong place."""
    c = cache()
    c.store(KEY, {"meters": 200.0}, burst="k:shift")
    assert c.lookup(OTHER, {"meters": 200.0}) is None


def test_different_visible_elements_are_a_different_question():
    c = cache()
    c.store(situation_key("s", ["button"], ["t"]), {"x": 1.0}, burst="c:l")
    assert c.lookup(situation_key("s", ["dialog"], ["t"]), {"x": 1.0}) is None


def test_a_far_away_reading_is_a_miss():
    c = cache()
    # Establish the range so normalisation is meaningful.
    c.observe({"meters": 0.0})
    c.observe({"meters": 400.0})
    c.store(KEY, {"meters": 380.0}, burst="k:shift")
    assert c.lookup(KEY, {"meters": 20.0}) is None


def test_distance_is_normalised_per_probe():
    """Readings are heterogeneous -- metres run to 400, brightness to 1.

    Without per-probe normalisation the distance is decided entirely by whichever probe
    has the largest units, and the radius stops meaning anything for the others.
    """
    c = cache()
    c.observe({"meters": 0.0, "lit": 0.0})
    c.observe({"meters": 400.0, "lit": 1.0})
    c.store(KEY, {"meters": 200.0, "lit": 0.5}, burst="k:a")
    # Same fractional distance on each axis; both should behave the same way.
    assert c.lookup(KEY, {"meters": 208.0, "lit": 0.52}) is not None
    assert c.lookup(KEY, {"meters": 360.0, "lit": 0.9}) is None


def test_it_never_caches_doing_nothing():
    """Caching '.' is how a loop gets stuck doing nothing quickly instead of slowly."""
    c = cache()
    c.store(KEY, {"meters": 1.0}, burst=".")
    c.store(KEY, {"meters": 1.0}, burst="")
    assert c.size == 0


def test_a_near_duplicate_replaces_rather_than_accumulating():
    c = cache()
    c.observe({"meters": 0.0})
    c.observe({"meters": 400.0})
    for i in range(20):
        c.store(KEY, {"meters": 200.0 + i * 0.1}, burst=f"k:{i}")
    assert c.size == 1, f"{c.size} near-identical entries piled up"


def test_penalties_retire_an_entry_but_one_bad_run_does_not():
    """A single unlucky episode must not retire a decision that is usually right."""
    c = cache()
    c.store(KEY, {"meters": 100.0}, burst="k:shift")
    c.penalise(KEY, {"meters": 100.0})
    assert c.lookup(KEY, {"meters": 100.0}) is not None, "retired after one penalty"
    c.penalise(KEY, {"meters": 100.0})
    assert c.lookup(KEY, {"meters": 100.0}) is None
    assert c.retired == 1


def test_some_cycles_bypass_the_cache_on_purpose():
    """Without this the table freezes at whatever it learned first and cannot be wrong."""
    c = PolicyCache(bypass=0.5, seed=4)
    c.store(KEY, {"meters": 100.0}, burst="k:shift")
    results = [c.lookup(KEY, {"meters": 100.0}) for _ in range(200)]
    misses = sum(1 for r in results if r is None)
    assert 60 < misses < 140, f"{misses}/200 bypassed; not close to the 50% asked for"
    assert c.bypasses > 0


def test_it_is_bounded_and_evicts_the_least_recently_used():
    c = cache(max_entries=10)
    for i in range(60):
        c.store(situation_key(f"s{i}", [], []), {"x": float(i)}, burst=f"k:{i}")
    assert c.size <= 10
    assert c.evicted >= 40


def test_disabled_means_disabled():
    c = PolicyCache(enabled=False)
    c.store(KEY, {"meters": 1.0}, burst="k:a")
    assert c.lookup(KEY, {"meters": 1.0}) is None
    assert c.size == 0


def test_hit_rate_reports_share_of_decisions_that_skipped_the_model():
    c = cache()
    c.store(KEY, {"meters": 100.0}, burst="k:shift")
    for _ in range(9):
        c.lookup(KEY, {"meters": 100.0})
    c.lookup(OTHER, {"meters": 100.0})
    stats = c.stats()
    assert stats["hits"] == 9
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.9


def test_a_probe_that_never_varies_carries_no_information():
    """A constant reading must not dominate the distance, or drown out one that moves."""
    c = cache()
    c.observe({"const": 5.0, "moving": 0.0})
    c.observe({"const": 5.0, "moving": 100.0})
    c.store(KEY, {"const": 5.0, "moving": 10.0}, burst="k:a")
    assert c.lookup(KEY, {"const": 5.0, "moving": 11.0}) is not None
    assert c.lookup(KEY, {"const": 5.0, "moving": 90.0}) is None


def test_it_pays_off_over_a_realistic_run():
    """The number that decides whether this is worth having at all.

    Simulates a descent repeated many times: the same handful of situations recur with
    small variation, which is what a game loop looks like.
    """
    import random

    rng = random.Random(5)
    c = cache()
    calls = 0
    for _ in range(40):                      # 40 falls
        for meters in range(400, 0, -25):    # each passing the same altitude bands
            reading = {"meters": meters + rng.uniform(-4, 4), "rate": -18.0}
            if c.lookup(KEY, reading) is None:
                calls += 1                   # a model round trip we had to pay for
                c.store(KEY, reading, burst="k:shift")
    total = 40 * 16
    assert calls < total * 0.25, (
        f"{calls}/{total} decisions still needed the model; the cache is not paying off"
    )


# -- in a live session -------------------------------------------------------------------


async def test_a_session_stops_calling_the_actuator_for_situations_it_has_seen():
    """The claim, end to end: repeated situations skip the model and still act."""
    import numpy as np

    from voltage_input_mcp.capture.base import CaptureBackend, Frame
    from voltage_input_mcp.inputs import DeviceSet, Executor
    from voltage_input_mcp.llm.base import Backend, GenerationResult
    from voltage_input_mcp.models.playbook import playbook_from_dict
    from voltage_input_mcp.runtime import Session, SessionDeps, SessionOptions

    class Still(CaptureBackend):
        """An unchanging screen, so every cycle is the same situation."""

        name = "stub"

        def __init__(self):
            self.grabs = 0

        def grab(self, region=None):
            self.grabs += 1
            return Frame(pixels=np.full((64, 64, 3), 128, dtype=np.uint8),
                         frame_id=self.grabs, backend=self.name)

        def geometry(self):
            return (1920, 1080)

    class Counting(Backend):
        name = "stub"

        def __init__(self, reply):
            self.reply = reply
            self.calls = 0

        @property
        def supports_grammar(self): return True

        @property
        def supports_vision(self): return False

        async def generate(self, prompt, **kw):
            self.calls += 1
            return GenerationResult(text=self.reply, model="s", backend="s",
                                    latency_ms=1.0)

        async def health(self): return {"ok": True}

    class Sink:
        screen = (1920, 1080)
        is_open = True

        def __init__(self): self.keys = []

        def open(self): ...
        def close(self): ...
        def key(self, name, down): self.keys.append((name, down))
        def button(self, name, down): ...
        def move_abs(self, x, y): ...
        def move_rel(self, dx, dy): ...
        def scroll(self, amount, axis="v"): ...

    playbook = playbook_from_dict({
        "name": "recall_check",
        "goal": "repeat one situation many times",
        "initial": "go",
        "probes": [{"id": "lit", "type": "brightness",
                    "region": {"x": 0, "y": 0, "w": 32, "h": 32}}],
        "perception": {"mode": "never"},
        "policy": {"dry_run": False, "allow_verbs": ["k", "w"],
                   "allow_keys": ["space"]},
        "budget": {"max_cycles": 24, "max_seconds": 10, "idle_abort_s": 0},
        "states": {"go": {"brief": "press space", "watch": [], "transitions": []}},
    })
    actuator = Counting("k:space|.|go")
    sink = Sink()
    deps = SessionDeps(
        capture=Still(), vision=Counting("{}"), actuator=actuator,
        devices=DeviceSet(screen=(1920, 1080)),
        executor=Executor(sink, dry_run=False), screen=(1920, 1080),
    )
    session = Session(playbook, deps, SessionOptions(
        settle_ms=0, watch_physical_input=False, dry_run=False,
        target_period_s=0.01, reflex_hz=0.0, reflex_enabled=False,
    ))
    await session.start()

    stats = session.snapshot()["recall"]
    cycles = session._cycle
    assert cycles >= 20
    # The situation never changes, so after the first decision almost everything should
    # be recalled. The bypass rate is what keeps it from being literally one call.
    assert actuator.calls < cycles * 0.5, (
        f"{actuator.calls} model calls in {cycles} cycles; recall is not saving anything"
    )
    assert stats["hits"] > 0
    # And the recalled decisions still reached the device -- a cache that saves calls by
    # doing nothing would pass every assertion above.
    assert ("space", True) in sink.keys
    assert sink.keys.count(("space", True)) > 1


# -- corroboration: the readings are not enough on their own -------------------------------


def test_matching_readings_on_a_different_screen_are_refused():
    """The failure this exists to prevent.

    200 m during a fall and 200 m on a results panel are the same reading and not the
    same situation. A single-signal cache proceeds on that coincidence and fires a burst
    into the wrong context -- a real input at a real moment in a live game.
    """
    c = cache()
    falling = [0.1] * 40
    results_panel = [0.9] * 40
    c.store(KEY, {"meters": 200.0}, burst="k:shift", fingerprint=falling)

    assert c.lookup(KEY, {"meters": 200.0}, fingerprint=falling) is not None
    assert c.lookup(KEY, {"meters": 200.0}, fingerprint=results_panel) is None
    assert c.conflicts == 1


def test_a_moving_background_still_counts_as_the_same_screen():
    """Tolerance has to survive an animating HUD, or nothing ever hits."""
    import random

    rng = random.Random(2)
    c = cache()
    base = [rng.random() for _ in range(40)]
    jittered = [min(1.0, max(0.0, v + rng.uniform(-0.05, 0.05))) for v in base]
    c.store(KEY, {"meters": 200.0}, burst="k:shift", fingerprint=base)
    assert c.lookup(KEY, {"meters": 200.0}, fingerprint=jittered) is not None


def test_absent_evidence_is_not_treated_as_disagreement():
    """An entry stored before fingerprinting, or a caller that has none, must still hit."""
    c = cache()
    c.store(KEY, {"meters": 200.0}, burst="k:shift")  # no fingerprint
    assert c.lookup(KEY, {"meters": 200.0}, fingerprint=[0.5] * 40) is not None
    c2 = cache()
    c2.store(KEY, {"meters": 200.0}, burst="k:shift", fingerprint=[0.5] * 40)
    assert c2.lookup(KEY, {"meters": 200.0}) is not None


def test_a_fingerprint_is_coarse_enough_to_generalise():
    """A high-resolution print answers "identical frame", which is never true."""
    import numpy as np

    from voltage_input_mcp.runtime.recall import FINGERPRINT_H, FINGERPRINT_W, fingerprint_of

    frame = np.random.default_rng(3).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print_ = fingerprint_of(frame)
    assert len(print_) == FINGERPRINT_W * FINGERPRINT_H
    assert all(0.0 <= v <= 1.0 for v in print_)
