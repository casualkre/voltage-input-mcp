"""Playbook compilation: catch authoring mistakes before a run, not during one."""

from __future__ import annotations

import pytest

from voltage_input_mcp.errors import PlaybookError
from voltage_input_mcp.models.playbook import playbook_from_dict
from voltage_input_mcp.reference import EXAMPLE_PLAYBOOK


def minimal(**overrides):
    base = {
        "name": "t",
        "goal": "test",
        "initial": "a",
        "states": {
            "a": {
                "brief": "do the thing",
                "watch": ["target"],
                "transitions": [{"when": "sees('target')", "to": "@success"}],
            }
        },
    }
    base.update(overrides)
    return base


def test_reference_example_compiles_cleanly():
    compiled = playbook_from_dict(EXAMPLE_PLAYBOOK)
    assert compiled.spec.initial in compiled.states
    assert not [w for w in compiled.warnings if "error" in w.lower()]


def test_minimal_playbook_compiles():
    compiled = playbook_from_dict(minimal())
    assert sorted(compiled.states) == ["a"]


def test_unknown_initial_state_is_rejected():
    with pytest.raises(PlaybookError):
        playbook_from_dict(minimal(initial="nope"))


def test_unknown_transition_target_is_rejected():
    bad = minimal()
    bad["states"]["a"]["transitions"] = [{"when": "True", "to": "ghost"}]
    with pytest.raises(PlaybookError) as exc:
        playbook_from_dict(bad)
    assert "ghost" in str(exc.value.context["errors"])


def test_broken_guard_is_rejected_with_a_useful_message():
    bad = minimal()
    bad["states"]["a"]["transitions"] = [{"when": "sees(", "to": "@success"}]
    with pytest.raises(PlaybookError) as exc:
        playbook_from_dict(bad)
    assert "syntax error" in str(exc.value.context["errors"]).lower()


def test_broken_burst_in_a_reflex_is_rejected():
    bad = minimal()
    bad["states"]["a"]["reflex"] = [
        {"id": "r", "when": "True", "do": "k:notarealkey"}
    ]
    with pytest.raises(PlaybookError) as exc:
        playbook_from_dict(bad)
    assert "unknown key" in str(exc.value.context["errors"])


def test_undefined_probe_reference_is_rejected():
    bad = minimal()
    bad["states"]["a"]["transitions"] = [{"when": "probe('nope') > 0", "to": "@success"}]
    with pytest.raises(PlaybookError) as exc:
        playbook_from_dict(bad)
    assert "undefined probes" in str(exc.value.context["errors"])


def test_all_errors_are_reported_at_once():
    """One round trip to fix a playbook, not N."""
    bad = minimal()
    bad["states"]["a"]["transitions"] = [
        {"when": "sees(", "to": "@success"},
        {"when": "True", "to": "ghost"},
        {"when": "probe('nope') > 0", "to": "@success"},
    ]
    with pytest.raises(PlaybookError) as exc:
        playbook_from_dict(bad)
    assert len(exc.value.context["errors"]) >= 3


def test_guard_referencing_a_label_outside_watch_warns():
    """This transition could never fire: the vision grammar cannot emit that label."""
    pb = minimal()
    pb["states"]["a"]["watch"] = ["something else"]
    compiled = playbook_from_dict(pb)
    assert any("watch" in w for w in compiled.warnings)


def test_unreachable_state_warns():
    pb = minimal()
    pb["states"]["orphan"] = {"brief": "never reached", "watch": ["x"],
                              "transitions": [{"when": "True", "to": "@success"}]}
    compiled = playbook_from_dict(pb)
    assert any("unreachable" in w for w in compiled.warnings)


def test_no_path_to_success_warns():
    pb = minimal()
    pb["states"]["a"]["transitions"] = [{"when": "True", "to": "@failure"}]
    compiled = playbook_from_dict(pb)
    assert any("@success" in w for w in compiled.warnings)


def test_reserved_state_names_are_rejected():
    with pytest.raises(PlaybookError):
        playbook_from_dict(minimal(states={"@success": {"brief": "x"}}, initial="@success"))


def test_unknown_verb_in_policy_is_rejected():
    with pytest.raises(PlaybookError):
        playbook_from_dict(minimal(policy={"allow_verbs": ["k", "zzz"]}))


def test_state_verbs_intersect_policy_verbs():
    pb = minimal(policy={"allow_verbs": ["k", "w"]})
    pb["states"]["a"]["allow_verbs"] = ["k", "c"]
    compiled = playbook_from_dict(pb)
    assert compiled.state("a").allow_verbs == frozenset({"k"})


def test_state_verbs_disjoint_from_policy_is_an_error():
    pb = minimal(policy={"allow_verbs": ["k"]})
    pb["states"]["a"]["allow_verbs"] = ["c"]
    with pytest.raises(PlaybookError) as exc:
        playbook_from_dict(pb)
    assert "disjoint" in str(exc.value.context["errors"])


def test_probe_shape_is_validated():
    with pytest.raises(PlaybookError):
        # pixel probes need `at` and `expect`
        playbook_from_dict(minimal(probes=[{"id": "p", "type": "pixel"}]))


def test_defaults_are_restrictive():
    compiled = playbook_from_dict(minimal())
    policy = compiled.spec.policy
    assert policy.dry_run is True
    assert "delete" in policy.deny_keys
    assert any("delete" in label for label in policy.deny_labels)
    assert policy.max_inputs_per_second <= 60
