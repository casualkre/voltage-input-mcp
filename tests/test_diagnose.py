"""Diagnosis over a journal, and the lessons store.

These also guard against a gap that bit once: `diagnose.py` is imported lazily inside an
MCP tool, so nothing loaded it and a syntax error survived a full green test run. Every
module the tools reach needs at least one test that imports it.
"""

from __future__ import annotations

from voltage_input_mcp.diagnose import Lesson, diagnose, load_lessons, save_lesson


def cycle(**kw):
    base = {
        "kind": "cycle", "cycle": 1, "state": "s", "elements": [], "probes": {},
        "perception": "vlm", "burst": "", "allowed": True, "executed": False,
    }
    base.update(kw)
    return base


PLAYBOOK = {
    "states": {
        "s": {
            "watch": ["save button", "dialog"],
            "transitions": [{"when": "sees('save button')", "to": "@success"}],
        }
    }
}


def codes(report) -> set[str]:
    return {f["code"] for f in report["findings"]}


def test_no_cycles():
    report = diagnose([])
    assert report["cycles"] == 0
    assert "no_cycles" in codes(report)


def test_dry_run_is_reported_as_the_reason_nothing_happened():
    report = diagnose([cycle(burst="k:a")], dry_run=True)
    assert "dry_run" in codes(report)
    assert report["findings"][0]["severity"] == "blocker"


def test_labels_the_vision_model_never_reported():
    journal = [cycle(elements=[{"label": "dialog"}]) for _ in range(5)]
    report = diagnose(journal, PLAYBOOK)
    finding = next(f for f in report["findings"] if f["code"] == "label_never_seen")
    assert "save button" in finding["evidence"]["never_seen"]
    assert "dialog" in finding["evidence"]["actually_seen"]


def test_bursts_that_ran_but_changed_nothing():
    """The key distinction: ran-and-did-nothing is not the same as never-ran."""
    journal = [
        cycle(burst="g:0;c:l", executed=True, probes={"__frame_delta__": 0.0})
        for _ in range(6)
    ]
    report = diagnose(journal, PLAYBOOK)
    assert "input_not_landing" in codes(report)


def test_a_burst_that_changed_the_screen_is_not_flagged():
    journal = [
        cycle(burst="g:0;c:l", executed=True, probes={"__frame_delta__": 0.4},
              elements=[{"label": "save button"}, {"label": "dialog"}])
        for _ in range(6)
    ]
    assert "input_not_landing" not in codes(diagnose(journal, PLAYBOOK))


def test_governor_refusals_name_the_rule():
    journal = [
        cycle(allowed=False, violations=[{"rule": "deny_labels"}]) for _ in range(5)
    ]
    report = diagnose(journal, PLAYBOOK)
    finding = next(f for f in report["findings"] if f["code"] == "governor_refusals")
    assert finding["evidence"]["by_rule"]["deny_labels"] == 5
    assert "deny_labels" in finding["what"]


def test_one_action_bursts_are_flagged():
    journal = [
        cycle(burst="c:l", elements=[{"label": "save button"}]) for _ in range(8)
    ]
    assert "timid_bursts" in codes(diagnose(journal, PLAYBOOK))


def test_chained_bursts_are_not_flagged():
    journal = [
        cycle(burst='g:0;c:l;w:150;t:"x";k:enter', elements=[{"label": "save button"}])
        for _ in range(8)
    ]
    assert "timid_bursts" not in codes(diagnose(journal, PLAYBOOK))


def test_state_that_never_transitioned():
    journal = [
        cycle(state="s", elements=[{"label": "save button"}, {"label": "dialog"}])
        for _ in range(12)
    ]
    report = diagnose(journal, PLAYBOOK)
    finding = next(f for f in report["findings"] if f["code"] == "state_never_left")
    assert finding["evidence"]["state"] == "s"


def test_findings_are_ordered_blockers_first():
    journal = [
        cycle(burst="c:l", executed=True, probes={"__frame_delta__": 0.0}) for _ in range(9)
    ]
    report = diagnose(journal, PLAYBOOK, dry_run=False)
    severities = [f["severity"] for f in report["findings"]]
    assert severities == sorted(severities, key=["blocker", "problem", "hint"].index)


def test_every_finding_carries_an_actionable_fix():
    journal = [
        cycle(burst="c:l", executed=True, probes={"__frame_delta__": 0.0},
              allowed=False, violations=[{"rule": "allow_verbs"}])
        for _ in range(9)
    ]
    report = diagnose(journal, PLAYBOOK, dry_run=True)
    assert report["findings"]
    for finding in report["findings"]:
        assert finding["fix"], f"{finding['code']} has no fix"
        assert finding["why"], f"{finding['code']} has no explanation"


def test_a_healthy_run_produces_no_blockers():
    journal = [
        cycle(burst='g:0;c:l;w:120', executed=True, probes={"__frame_delta__": 0.3},
              elements=[{"label": "save button"}, {"label": "dialog"}],
              perception="cache", transition="@success" if i == 5 else None)
        for i in range(6)
    ]
    report = diagnose(journal, PLAYBOOK, dry_run=False)
    assert not [f for f in report["findings"] if f["severity"] == "blocker"]


# -- lessons ---------------------------------------------------------------------------


def test_lesson_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "lessons.json"
    monkeypatch.setattr("voltage_input_mcp.diagnose.lessons_path", lambda: path)

    save_lesson(Lesson(target="Minecraft", note="hotbar is at y=1040", kind="label"))
    save_lesson(Lesson(target="roblox", note="needs w:120 after a jump", kind="timing"))

    assert len(load_lessons()) == 2
    mc = load_lessons("minecraft")          # target matching is slug-based
    assert len(mc) == 1
    assert mc[0]["kind"] == "label"


def test_identical_lessons_do_not_accumulate(tmp_path, monkeypatch):
    path = tmp_path / "lessons.json"
    monkeypatch.setattr("voltage_input_mcp.diagnose.lessons_path", lambda: path)
    for _ in range(4):
        save_lesson(Lesson(target="mc", note="the same thing"))
    assert len(load_lessons("mc")) == 1


def test_missing_lessons_file_is_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "voltage_input_mcp.diagnose.lessons_path", lambda: tmp_path / "nope.json"
    )
    assert load_lessons() == []
