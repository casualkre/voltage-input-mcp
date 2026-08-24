"""The episodic optimiser: does it actually find a better constant, or just move?

A search that wanders and reports motion as progress is worse than no search, because it
looks like learning. So these test the property that matters -- given a reward with a
known optimum, does the best-known parameter end up near it -- rather than testing that
the arithmetic runs.
"""

from __future__ import annotations

import json

import pytest

from voltage_input_mcp.errors import PlaybookError
from voltage_input_mcp.models.playbook import playbook_from_dict
from voltage_input_mcp.runtime.tuner import Tunable, Tuner, parse_tunables


def optimise(reward_fn, *, tunables, episodes=140, seed=7, noise=0.0):
    """Run the tuner against a synthetic reward and return it."""
    import random

    rng = random.Random(seed + 1)
    tuner = Tuner(tunables, seed=seed)
    for _ in range(episodes):
        params = tuner.current()
        value = reward_fn(params)
        if noise:
            value += rng.gauss(0.0, noise)
        tuner.record(value)
    return tuner


def test_it_finds_a_single_optimum():
    """One parameter, smooth reward peaking at 120 in a range of 40-200."""
    tuner = optimise(
        lambda p: -abs(p["brake_at"] - 120.0),
        tunables=[Tunable("brake_at", default=60.0, low=40.0, high=200.0)],
    )
    assert abs(tuner.best["brake_at"] - 120.0) < 12, (
        f"settled on {tuner.best['brake_at']:.1f}, wanted ~120"
    )


def test_it_still_finds_the_optimum_through_heavy_noise():
    """A real episode's reward is dominated by luck; the search must not chase it.

    The signal here is +-40 across the whole parameter range and the noise is 15 per
    episode, which is roughly the ratio a Broken Bones run actually has.
    """
    tuner = optimise(
        lambda p: -abs(p["brake_at"] - 120.0),
        tunables=[Tunable("brake_at", default=190.0, low=40.0, high=200.0)],
        noise=15.0,
        episodes=220,
    )
    assert abs(tuner.best["brake_at"] - 120.0) < 30, (
        f"noise pulled it to {tuner.best['brake_at']:.1f}"
    )


def test_it_optimises_several_parameters_at_once():
    def reward(p):
        return -(abs(p["a"] - 30.0) + abs(p["b"] + 15.0) + abs(p["c"] - 500.0) / 10.0)

    tuner = optimise(
        reward,
        tunables=[
            Tunable("a", default=5.0, low=0.0, high=60.0),
            Tunable("b", default=0.0, low=-40.0, high=10.0),
            Tunable("c", default=100.0, low=0.0, high=1000.0),
        ],
        episodes=300,
    )
    assert abs(tuner.best["a"] - 30.0) < 8
    assert abs(tuner.best["b"] + 15.0) < 8
    assert abs(tuner.best["c"] - 500.0) < 90


def test_it_never_proposes_outside_the_declared_bounds():
    """Bounds are what make the search safe to run against a live game."""
    tuner = Tuner([Tunable("x", default=50.0, low=10.0, high=90.0)], seed=3)
    for _ in range(400):
        value = tuner.current()["x"]
        assert 10.0 <= value <= 90.0, f"proposed {value}, outside [10, 90]"
        # Reward that pulls hard toward a bound, to make it try to escape.
        tuner.record(value)


def test_a_tuner_with_nothing_to_tune_is_inert_but_still_answers():
    """`tune()` must resolve with the search off, so guards need no rewriting."""
    tuner = Tuner([], explore=False)
    assert tuner.current() == {}
    tuner.record(10.0)
    assert tuner.summary()["exploring"] is False


def test_exploration_off_pins_the_parameters():
    tuner = Tuner(
        [Tunable("x", default=50.0, low=0.0, high=100.0)], seed=1, explore=False
    )
    for _ in range(20):
        assert tuner.current() == {"x": 50.0}
        tuner.record(1.0)


def test_improvement_is_none_until_there_is_enough_data():
    """Reporting an improvement off two episodes would be reporting noise."""
    tuner = Tuner([Tunable("x", default=1.0, low=0.0, high=2.0)], seed=1)
    tuner.record(5.0)
    assert tuner.improvement() is None


def test_learned_parameters_survive_a_restart(tmp_path, monkeypatch):
    """Across sessions is the point -- otherwise it relearns the same constants daily."""
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    first = optimise(
        lambda p: -abs(p["brake_at"] - 120.0),
        tunables=[Tunable("brake_at", default=60.0, low=40.0, high=200.0)],
    )
    first.save("bb4")

    second = Tuner([Tunable("brake_at", default=60.0, low=40.0, high=200.0)], seed=2)
    assert second.load("bb4") is True
    assert abs(second.best["brake_at"] - first.best["brake_at"]) < 1e-6
    # The stored score is discounted rather than adopted verbatim: it came from another
    # session, and treating it as the bar to beat makes every new proposal fail.
    # These rewards are negative, which is exactly the case a naive `* 0.75` gets
    # backwards -- scaling a negative score raises it and makes the bar harder.
    assert second.best_score < first.best_score


def test_the_carried_over_score_is_lowered_for_positive_and_negative_rewards(
    tmp_path, monkeypatch
):
    """Discounting must lower the bar whatever the sign of the reward."""
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    for stored in (1000.0, -1000.0):
        tuner = Tuner([Tunable("x", default=1.0, low=0.0, high=2.0)], seed=1)
        tuner.best_score = stored
        tuner.save("signcheck")
        fresh = Tuner([Tunable("x", default=1.0, low=0.0, high=2.0)], seed=1)
        fresh.load("signcheck")
        assert fresh.best_score < stored, (
            f"stored {stored} came back as {fresh.best_score}, which is a harder bar"
        )


def test_loading_re_clamps_to_the_bounds_the_playbook_now_declares(tmp_path, monkeypatch):
    """Editing a playbook must not resurrect a value that is now out of range."""
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    wide = Tuner([Tunable("x", default=180.0, low=0.0, high=200.0)], seed=1)
    wide.best["x"] = 195.0
    wide.save("p")

    narrow = Tuner([Tunable("x", default=30.0, low=0.0, high=50.0)], seed=1)
    assert narrow.load("p") is True
    assert narrow.best["x"] == 50.0, "a stale out-of-range value was adopted verbatim"


def test_loading_drops_parameters_that_no_longer_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    old = Tuner([Tunable("gone", default=1.0, low=0.0, high=2.0)], seed=1)
    old.save("p")
    new = Tuner([Tunable("kept", default=5.0, low=0.0, high=10.0)], seed=1)
    new.load("p")
    assert "gone" not in new.best
    assert new.best == {"kept": 5.0}


def test_a_corrupt_store_is_ignored_rather_than_crashing_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    from voltage_input_mcp.runtime.tuner import tuner_store_path

    path = tuner_store_path("p")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    tuner = Tuner([Tunable("x", default=1.0, low=0.0, high=2.0)], seed=1)
    assert tuner.load("p") is False
    assert tuner.best == {"x": 1.0}


# -- schema -----------------------------------------------------------------------------


def test_parse_tunables_rejects_impossible_bounds():
    with pytest.raises(ValueError, match="min"):
        parse_tunables({"x": {"default": 5, "min": 10, "max": 1}})
    with pytest.raises(ValueError, match="outside"):
        parse_tunables({"x": {"default": 99, "min": 0, "max": 10}})
    with pytest.raises(ValueError, match="numeric"):
        parse_tunables({"x": {"default": "a", "min": 0, "max": 10}})


def _playbook(**extra):
    return {
        "name": "t", "goal": "g", "initial": "s",
        "probes": [{"id": "m", "type": "number",
                    "region": {"x": 0, "y": 0, "w": 10, "h": 10}}],
        "budget": {"max_cycles": 5, "max_seconds": 5, "idle_abort_s": 0},
        "states": {"s": {"brief": "b", "transitions": [
            {"when": "cycles() > 2", "to": "@success", "ends_episode": True}]}},
        **extra,
    }


def test_an_undeclared_tune_is_a_compile_error():
    """It would silently fall back to a literal and never move -- learning that isn't."""
    spec = _playbook()
    spec["states"]["s"]["reflex"] = [
        {"id": "r", "when": "probe('m') < tune('nope')", "do": "k:space"}
    ]
    with pytest.raises(PlaybookError) as caught:
        playbook_from_dict(spec)
    assert "not declared" in " ".join(caught.value.context["errors"])


def test_a_declared_tunable_compiles():
    spec = _playbook(
        tunables={"brake_at": {"default": 90, "min": 40, "max": 200}},
        reward={"probe": "m", "mode": "delta"},
    )
    spec["states"]["s"]["reflex"] = [
        {"id": "r", "when": "probe('m') < tune('brake_at')", "do": "k:space"}
    ]
    compiled = playbook_from_dict(spec)
    assert compiled.spec.tunables["brake_at"].default == 90
    assert compiled.spec.reward.probe == "m"


def test_tunable_bounds_are_validated_by_the_schema():
    with pytest.raises(PlaybookError):
        playbook_from_dict(_playbook(tunables={"x": {"default": 5, "min": 9, "max": 1}}))


def test_the_store_is_plain_readable_json(tmp_path, monkeypatch):
    """The orchestrator reads these back to write better defaults into a playbook."""
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    tuner = Tuner([Tunable("x", default=1.0, low=0.0, high=2.0)], seed=1)
    tuner.record(3.0)
    data = json.loads(tuner.save("p").read_text())
    assert data["playbook"] == "p"
    assert set(data["best"]) == {"x"}
    assert data["bounds"]["x"] == [0.0, 2.0]


# -- wiring into a live session ---------------------------------------------------------


async def test_a_session_runs_episodes_and_moves_its_constants(tmp_path, monkeypatch):
    """End to end: tune() reaches guards, ends_episode scores, the store gets written.

    The algorithm is covered above against synthetic rewards; what this checks is that
    the session actually feeds it -- that an episode boundary fires, a reward is read off
    a probe, and the result survives the run.
    """
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    import numpy as np

    from voltage_input_mcp.capture.base import CaptureBackend, Frame
    from voltage_input_mcp.inputs import DeviceSet, Executor
    from voltage_input_mcp.llm.base import Backend, GenerationResult
    from voltage_input_mcp.runtime import Session, SessionDeps, SessionOptions

    class Ramp(CaptureBackend):
        """Brightness climbs steadily, so `final` mode has something real to score."""

        name = "stub"

        def __init__(self):
            self.grabs = 0

        def grab(self, region=None):
            self.grabs += 1
            level = min(255, 20 + self.grabs // 3)
            return Frame(pixels=np.full((64, 64, 3), level, dtype=np.uint8),
                         frame_id=self.grabs, backend=self.name)

        def geometry(self):
            return (1920, 1080)

    class Quiet(Backend):
        name = "stub"

        @property
        def supports_grammar(self): return True

        @property
        def supports_vision(self): return False

        async def generate(self, prompt, **kw):
            return GenerationResult(text=".|.|", model="s", backend="s", latency_ms=1.0)

        async def health(self): return {"ok": True}

    class Sink:
        screen = (1920, 1080)
        is_open = True

        def open(self): ...
        def close(self): ...
        def key(self, name, down): ...
        def button(self, name, down): ...
        def move_abs(self, x, y): ...
        def move_rel(self, dx, dy): ...
        def scroll(self, amount, axis="v"): ...

    playbook = playbook_from_dict({
        "name": "wiring_check",
        "goal": "run several scored episodes",
        "initial": "go",
        "probes": [{"id": "lit", "type": "brightness",
                    "region": {"x": 0, "y": 0, "w": 32, "h": 32}}],
        "tunables": {"thresh": {"default": 0.5, "min": 0.05, "max": 0.95}},
        "reward": {"probe": "lit", "mode": "final", "settle_ms": 0},
        "perception": {"mode": "never"},
        "policy": {"dry_run": True, "allow_verbs": ["k", "w"], "allow_keys": ["space"]},
        "budget": {"max_cycles": 400, "max_seconds": 6, "idle_abort_s": 0},
        "states": {
            "go": {
                "brief": "x",
                "autonomous": False,
                # Reads the tunable, so a wiring failure shows up as a compile error.
                "reflex": [{"id": "r", "when": "probe('lit') > tune('thresh')",
                            "do": "k:space", "cooldown_ms": 50}],
                "transitions": [
                    {"when": "elapsed() > 0.35", "to": "go", "ends_episode": True},
                ],
            }
        },
    })
    deps = SessionDeps(
        capture=Ramp(), vision=Quiet(), actuator=Quiet(),
        devices=DeviceSet(screen=(1920, 1080)), executor=Executor(Sink(), dry_run=True),
        screen=(1920, 1080),
    )
    session = Session(playbook, deps, SessionOptions(
        settle_ms=0, watch_physical_input=False, dry_run=True,
        target_period_s=0.05, reflex_hz=40.0,
    ))
    await session.start()

    summary = session.tuner_summary()
    assert summary["episodes"] >= 4, f"only {summary['episodes']} episodes ran"
    assert summary["exploring"] is True
    # The value handed to guards stays inside the declared bounds at all times.
    assert 0.05 <= summary["live"]["thresh"] <= 0.95
    # And what was learned outlived the run.
    from voltage_input_mcp.runtime.tuner import tuner_store_path
    assert tuner_store_path("wiring_check").exists()

    events = session.journal.tail(200, kinds={"episode"})
    assert len(events) >= 4
    assert all("reward" in e and "params" in e for e in events)
