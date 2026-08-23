"""Operator instructions: stored, capped, and surfaced to the orchestrator."""

from __future__ import annotations

import voltage_input_mcp.briefing as briefing
from voltage_input_mcp.briefing import (
    MAX_INSTRUCTIONS,
    TEMPLATES,
    briefing_text,
    load_instructions,
    save_instructions,
)


def redirect(tmp_path, monkeypatch):
    path = tmp_path / "instructions.md"
    monkeypatch.setattr(briefing, "instructions_path", lambda: path)
    return path


def test_round_trip(tmp_path, monkeypatch):
    redirect(tmp_path, monkeypatch)
    save_instructions("  never touch my email client  ")
    assert load_instructions() == "never touch my email client"


def test_absent_file_is_empty_not_an_error(tmp_path, monkeypatch):
    redirect(tmp_path, monkeypatch)
    assert load_instructions() == ""


def test_capped_because_it_sits_in_context_all_session(tmp_path, monkeypatch):
    redirect(tmp_path, monkeypatch)
    save_instructions("x" * (MAX_INSTRUCTIONS * 2))
    assert len(load_instructions()) == MAX_INSTRUCTIONS


def test_clearing_works(tmp_path, monkeypatch):
    redirect(tmp_path, monkeypatch)
    save_instructions("something")
    save_instructions("")
    assert load_instructions() == ""


def test_templates_are_usable_as_written(tmp_path, monkeypatch):
    redirect(tmp_path, monkeypatch)
    for name, body in TEMPLATES.items():
        save_instructions(body)
        assert load_instructions(), f"template {name} saved empty"


def test_instructions_reach_the_briefing_and_are_attributed(tmp_path, monkeypatch):
    """They must be marked as the operator's, not mistakable for observed data."""
    redirect(tmp_path, monkeypatch)
    save_instructions("do not touch the browser")
    text = briefing_text()
    assert "do not touch the browser" in text
    assert "OPERATOR INSTRUCTIONS" in text
    # And it must be clear they cannot loosen enforcement.
    assert "cannot loosen the safety governor" in text


def test_briefing_omits_the_section_entirely_when_unset(tmp_path, monkeypatch):
    redirect(tmp_path, monkeypatch)
    assert "OPERATOR INSTRUCTIONS" not in briefing_text()
