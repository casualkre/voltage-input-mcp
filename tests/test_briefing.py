"""The active-build briefing: only lines that change how a Playbook is written."""

from __future__ import annotations

from voltage_input_mcp.briefing import _check_mismatch, _guidance, briefing_text


def base(**kw):
    info = {
        "platform": "Linux", "engine": "llamacpp", "profile": "lean",
        "grammar_constrained": True, "dry_run_default": True,
        "pointer_mode": "absolute", "running": {}, "mismatch": "",
    }
    info.update(kw)
    return info


def joined(info) -> str:
    return " ".join(_guidance(info))


def test_llamacpp_says_grammars_are_enforced():
    assert "unrepresentable" in joined(base())


def test_ollama_warns_that_bursts_are_unconstrained():
    text = joined(base(engine="ollama", grammar_constrained=False))
    assert "NOT grammar-constrained" in text


def test_hyper_warns_against_building_on_sees():
    text = joined(base(profile="hyper"))
    assert "sees()" in text and "cannot ground" in text


def test_beefy_warns_about_cycle_cost():
    assert "1-2.5 s" in joined(base(profile="beefy"))


def test_windows_notes_elevation_and_layout_free_typing():
    text = joined(base(platform="Windows"))
    assert "elevated" in text and "layout-independent" in text


def test_dry_run_off_is_called_out_loudly():
    assert "inject input" in joined(base(dry_run_default=False))


# -- the mismatch check ----------------------------------------------------------------


def test_mismatch_detected_when_servers_serve_another_profile():
    """Switching profiles edits a file; it does not restart the servers."""
    info = base(
        profile="hyper",
        running={
            "vision": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            "actuator": "Qwen3-1.7B-Q4_K_M.gguf",
        },
    )
    message = _check_mismatch(info)
    assert "does not match what is loaded" in message
    assert "SmolVLM" in message


def test_no_mismatch_when_they_agree():
    info = base(
        profile="lean",
        running={
            "vision": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            "actuator": "Qwen3-1.7B-Q4_K_M.gguf",
        },
    )
    assert _check_mismatch(info) == ""


def test_no_mismatch_claimed_when_servers_are_down():
    """Absence of an answer is not evidence of disagreement."""
    assert _check_mismatch(base(profile="hyper", running={})) == ""


def test_mismatch_suppresses_profile_keyed_guidance():
    """Guidance derived from a profile name is wrong when that profile is not loaded."""
    info = base(
        profile="hyper",
        running={"vision": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "actuator": None},
    )
    info["mismatch"] = _check_mismatch(info)
    text = joined(info)
    assert "MISMATCH" in text
    assert "cannot ground" not in text          # the hyper warning must not fire
    assert "Judge grounding quality from those" in text


def test_briefing_text_is_non_empty_and_names_the_build():
    text = briefing_text()
    assert "ACTIVE BUILD" in text
