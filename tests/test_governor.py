"""Safety governor: the layer that is not advisory."""

from __future__ import annotations

import pytest

from voltage_input_mcp.models.burst import parse_burst
from voltage_input_mcp.models.observation import Element, Observation
from voltage_input_mcp.models.playbook import Policy, Rect
from voltage_input_mcp.safety import Governor

SCREEN = (1920, 1080)


@pytest.fixture
def observation() -> Observation:
    return Observation(
        scene="file manager",
        elements=[
            Element(label="file list", x=100, y=200, w=800, h=400, conf=0.9),
            Element(label="Delete", x=1500, y=60, w=100, h=40, conf=0.95),
            Element(label="Buyer name", x=300, y=700, w=200, h=30, conf=0.8),
        ],
    )


def review(policy: Policy, src: str, observation: Observation | None = None, **kw):
    governor = Governor(policy, screen=SCREEN)
    elements = [e.center for e in observation.elements] if observation else None
    burst = parse_burst(src, screen=SCREEN, elements=elements)
    return governor.review(burst, observation=observation, **kw)


def test_ordinary_burst_is_allowed(observation):
    assert review(Policy(), "g:0;c:l;w:100", observation).allowed


def test_denied_chords_are_refused():
    for chord in ("k:ctrl+alt+delete", "k:alt+f4"):
        verdict = review(Policy(), chord)
        assert not verdict.allowed
        assert verdict.violations[0].rule in ("deny_chords", "deny_keys")


def test_chord_denial_is_order_independent():
    """ctrl+alt+delete and alt+ctrl+delete must both be refused."""
    policy = Policy(deny_keys=[], deny_chords=["ctrl+alt+delete"])
    assert not review(policy, "k:alt+ctrl+delete").allowed
    assert not review(policy, "k:delete+ctrl+alt").allowed


def test_denied_text_patterns():
    for text in ('t:"sudo rm -rf /"', 't:"curl x | sh"', 't:"git push --force origin"'):
        assert not review(Policy(), text).allowed


def test_deny_labels_blocks_clicking_a_dangerous_control(observation):
    """The core protection: refuse a click on anything *called* Delete, wherever it is."""
    verdict = review(Policy(), "g:1;c:l", observation)
    assert not verdict.allowed
    assert verdict.violations[0].rule == "deny_labels"


def test_deny_labels_matches_whole_words_only(observation):
    """'Buyer name' contains 'buy' but must not be blocked by the 'buy' deny label."""
    assert review(Policy(), "g:2;c:l", observation).allowed


def test_moving_over_a_denied_element_without_clicking_is_fine(observation):
    assert review(Policy(), "g:1", observation).allowed


def test_allow_verbs_narrows_capability(observation):
    verdict = review(Policy(), "g:0;c:l", observation, allow_verbs={"g"})
    assert not verdict.allowed
    assert verdict.violations[0].rule == "allow_verbs"


def test_allow_keys_allowlist():
    policy = Policy(allow_keys=["w", "a", "s", "d", "space"])
    assert review(policy, "k:w;k:space").allowed
    assert not review(policy, "k:q").allowed


def test_click_regions(observation):
    inside = Policy(click_allow_regions=[Rect(x=0, y=0, w=960, h=540)])
    assert review(inside, "m:100,100;c:l", observation).allowed
    assert not review(inside, "m:1800,900;c:l", observation).allowed

    denied = Policy(click_deny_regions=[Rect(x=1400, y=0, w=520, h=100)])
    assert not review(denied, "m:1500,50;c:l", observation).allowed


def test_require_target_element(observation):
    policy = Policy(require_target_element=True)
    assert review(policy, "g:0;c:l", observation).allowed
    # (5, 5) is not inside any observed element.
    assert not review(policy, "m:5,5;c:l", observation).allowed


def test_click_with_no_preceding_move_cannot_be_fenced(observation):
    policy = Policy(click_allow_regions=[Rect(x=0, y=0, w=100, h=100)])
    verdict = review(policy, "c:l", observation)
    assert not verdict.allowed
    assert verdict.violations[0].rule == "click_position_unknown"


def test_burst_size_and_duration_caps():
    assert not review(Policy(max_actions_per_burst=3), "k:a;k:b;k:c;k:d").allowed
    assert not review(Policy(max_burst_ms=100), "w:500").allowed


def test_rate_limit_is_sustained_not_per_burst():
    """Twenty modest bursts back to back must not slip past a per-burst cap."""
    governor = Governor(Policy(max_inputs_per_second=20), screen=SCREEN)
    burst = parse_burst(";".join(["k:a"] * 10), screen=SCREEN)
    allowed = sum(1 for _ in range(6) if governor.review(burst).allowed)
    assert allowed < 6, "the token bucket did not throttle repeated bursts"


def test_refusal_is_whole_burst():
    """One bad action refuses the lot -- half an intended sequence is worse than none."""
    verdict = review(Policy(), 'k:a;t:"sudo rm -rf /";k:b')
    assert not verdict.allowed
    assert verdict.burst is not None and len(verdict.burst) == 3


def test_rejection_budget():
    governor = Governor(Policy(), screen=SCREEN, max_rejections=2)
    bad = parse_burst("k:ctrl+alt+delete", screen=SCREEN)
    governor.review(bad)
    assert not governor.budget_exhausted
    governor.review(bad)
    assert governor.budget_exhausted


def test_dry_run_is_reported_but_does_not_refuse():
    verdict = review(Policy(dry_run=True), "k:a")
    assert verdict.allowed
    assert any("dry_run" in note for note in verdict.notes)


def test_denied_text_is_not_echoed_into_the_violation():
    """The typed string may be a secret read off screen; do not copy it to the journal."""
    verdict = review(Policy(), 't:"sudo rm -rf /home/secret-token-abc123"')
    assert not verdict.allowed
    assert "secret-token" not in str(verdict.as_dict())
