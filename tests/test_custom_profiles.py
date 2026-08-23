"""User-defined model profiles: round-trip, merge, and shadowing."""

from __future__ import annotations

import pytest

from voltage_input_mcp.errors import ConfigError
from voltage_input_mcp.llm.custom import delete_custom, load_custom, save_custom
from voltage_input_mcp.llm.profiles import EXPERIMENTAL, PROFILES, ModelSpec, Profile


def make(name: str = "mine", **kw) -> Profile:
    return Profile(
        name=name,
        description=kw.get("description", "test"),
        vision=ModelSpec(
            role="vision", label="VL", hf_repo="o/r", hf_file="v.gguf",
            mmproj_repo="o/r", mmproj_file="mm.gguf",
            params_b=3.0, weights_mb=1900, n_ctx=2048, port=8080,
        ),
        actuator=ModelSpec(
            role="actuator", label="ACT", hf_repo="o/a", hf_file="a.gguf",
            params_b=1.7, weights_mb=1100, n_ctx=2048, port=8081,
        ),
        min_vram_mb=0,
    )


def test_round_trip(tmp_path):
    path = tmp_path / "profiles.toml"
    save_custom(make(), path)
    back = load_custom(path)["mine"]
    assert back.vision.hf_file == "v.gguf"
    assert back.vision.mmproj_file == "mm.gguf"
    assert back.actuator.port == 8081
    assert back.vram_mb > 0


def test_multiple_profiles_survive_each_other(tmp_path):
    """Saving a second profile must not eat the first."""
    path = tmp_path / "profiles.toml"
    save_custom(make("first"), path)
    save_custom(make("second"), path)
    assert sorted(load_custom(path)) == ["first", "second"]


def test_delete(tmp_path):
    path = tmp_path / "profiles.toml"
    save_custom(make("gone"), path)
    delete_custom("gone", path)
    assert "gone" not in load_custom(path)


def test_missing_file_is_not_an_error(tmp_path):
    assert load_custom(tmp_path / "nope.toml") == {}


def test_malformed_file_is_reported_not_swallowed(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text("this is not = valid toml [[[")
    with pytest.raises(ConfigError):
        load_custom(path)


def test_profile_without_a_model_source_is_rejected(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text('[bad]\n[bad.vision]\nlabel = "x"\n[bad.actuator]\nlabel = "y"\n')
    with pytest.raises(ConfigError, match="needs either"):
        load_custom(path)


def test_hf_repo_without_file_is_rejected(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text('[bad]\n[bad.vision]\nhf_repo = "a/b"\n[bad.actuator]\nollama_tag = "x"\n')
    with pytest.raises(ConfigError, match="without hf_file"):
        load_custom(path)


def test_custom_shadows_builtin(tmp_path, monkeypatch):
    """Naming a custom profile `lean` retunes the built-in without forking."""
    path = tmp_path / "profiles.toml"
    save_custom(make("lean", description="my retune"), path)
    monkeypatch.setattr("voltage_input_mcp.llm.custom.profiles_path", lambda: path)
    from voltage_input_mcp.llm.custom import all_profiles

    merged = all_profiles()
    assert merged["lean"].description == "my retune"
    # Every built-in and experimental profile is still present.
    assert set(PROFILES) <= set(merged)
    assert set(EXPERIMENTAL) <= set(merged)


def test_broken_file_does_not_break_all_profiles(tmp_path, monkeypatch):
    """A bad profiles.toml must not make the tool unusable."""
    path = tmp_path / "profiles.toml"
    path.write_text("[[[ broken")
    monkeypatch.setattr("voltage_input_mcp.llm.custom.profiles_path", lambda: path)
    from voltage_input_mcp.llm.custom import all_profiles

    assert set(all_profiles()) == set(PROFILES) | set(EXPERIMENTAL)


def test_recommend_never_returns_an_experimental_profile():
    """`hyper` fits almost any GPU and would win on size every time.

    Each experimental profile trades something away that the user has to consent to
    knowingly -- `hyper` in particular hands over a vision model that cannot ground.
    """
    from voltage_input_mcp.llm.profiles import recommend

    for vram in (2000, 4000, 6144, 12000, 24000, 80000):
        assert recommend(vram).name not in EXPERIMENTAL


def test_every_experimental_profile_carries_a_warning():
    for name, profile in EXPERIMENTAL.items():
        assert profile.warning, f"{name} is experimental but has no warning"
        assert profile.notes, f"{name} has no notes explaining when to use it"


# -- connection info -------------------------------------------------------------------


def test_stdio_json_is_valid_and_carries_the_environment():
    """A config emitted without the session environment connects but cannot see."""
    import json as _json

    from voltage_input_mcp.connect import ConnectionInfo, stdio_json

    info = ConnectionInfo(binary="/x/voltage-input-mcp", env={"DISPLAY": ":0"})
    data = _json.loads(stdio_json(info))
    entry = data["mcpServers"]["voltage-input"]
    assert entry["command"] == "/x/voltage-input-mcp"
    assert entry["env"]["DISPLAY"] == ":0"


def test_claude_code_command_quotes_paths_with_spaces():
    """The repo path here contains spaces; an unquoted command silently truncates."""
    from voltage_input_mcp.connect import ConnectionInfo, claude_code_command

    info = ConnectionInfo(binary="/home/a b/voltage-input-mcp", env={"DISPLAY": ":0"})
    cmd = claude_code_command(info)
    assert '"/home/a b/voltage-input-mcp"' in cmd
    assert "-e DISPLAY=:0" in cmd


def test_every_client_produces_steps():
    from voltage_input_mcp.connect import CLIENTS, ConnectionInfo, instructions

    info = ConnectionInfo(binary="/x/bin", http_url="http://127.0.0.1:8765/mcp")
    for client in CLIENTS:
        steps = instructions(client.key, info)
        assert steps, f"{client.key} produced no instructions"
        assert all(isinstance(text, str) and text for text, _ in steps)
