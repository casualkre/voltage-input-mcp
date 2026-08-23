"""What the orchestrator needs to know about *this* build to author correctly.

The same Playbook is a good idea on one configuration and a bad one on another, and the
orchestrator cannot see which it is running against. Concretely:

  * On Ollama there are no GBNF grammars, so a malformed burst is possible and costs a
    cycle. On llama.cpp it is unrepresentable. That changes how defensively a Playbook
    should be written.
  * On the `hyper` profile the vision model cannot ground reliably. Building states
    around `sees()` there is a wasted run; probes and click regions are the only sound
    approach.
  * On `beefy` a perceived cycle costs seconds, so `max_elements` and `perception.mode`
    matter far more than they do on `lean`.
  * On Windows typing is layout-independent and elevated windows are unreachable. On
    Linux the reverse is true of typing.

So the briefing is not a status dump. Every line either changes what a Playbook should
look like or is left out. It is computed at server startup and repeated by
`voltage_reference`, because the profile can change mid-session and the startup copy then
goes stale.
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["active_build", "briefing_text"]


def active_build() -> dict[str, Any]:
    """Structured description of the running configuration."""
    from .config import load_config
    from .llm.profiles import EXPERIMENTAL, get_profile

    try:
        config = load_config()
    except Exception:  # noqa: BLE001 - a broken config must not break the briefing
        from .config import Config

        config = Config()

    info: dict[str, Any] = {
        "platform": {"linux": "Linux", "win32": "Windows", "darwin": "macOS"}.get(
            sys.platform, sys.platform
        ),
        "engine": config.engine,
        "profile": config.profile,
        "dry_run_default": config.dry_run,
        "target_period_s": config.target_period_s,
        "pointer_mode": config.pointer_mode,
        "grammar_constrained": config.engine == "llamacpp",
    }

    try:
        profile = get_profile(config.profile)
        info["vision_model"] = profile.vision.label
        info["actuator_model"] = profile.actuator.label
        info["experimental"] = profile.name in EXPERIMENTAL
        info["expected_cycle_ms"] = list(profile.expected_cycle_ms)
        if profile.warning:
            info["profile_warning"] = profile.warning
    except Exception:  # noqa: BLE001
        info["vision_model"] = info["actuator_model"] = "unknown"
        info["experimental"] = False

    info["running"] = _running_models(config)
    info["mismatch"] = _check_mismatch(info)
    info["guidance"] = _guidance(info)
    return info


def _running_models(config) -> dict[str, str | None]:
    """Ask the live servers what they actually loaded.

    The configured profile and the running servers can disagree -- switching profiles
    edits a file, it does not restart anything -- and the disagreement is silent. Left
    unchecked the briefing confidently describes models that are not loaded, which is
    worse than saying nothing: it would tell the orchestrator that grounding is
    unreliable while the good vision model is in fact serving, or the reverse.
    """
    import json
    import urllib.error
    import urllib.request

    out: dict[str, str | None] = {"vision": None, "actuator": None}
    if config.engine != "llamacpp":
        return out

    for role, url in (("vision", config.vision_url), ("actuator", config.actuator_url)):
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/props", timeout=0.6) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            continue
        raw = (
            data.get("model_path")
            or data.get("model")
            or (data.get("default_generation_settings") or {}).get("model")
        )
        if raw:
            out[role] = str(raw).replace("\\", "/").rsplit("/", 1)[-1]
    return out


def _check_mismatch(info: dict[str, Any]) -> str:
    """Report a configured-vs-running disagreement in the terms that matter."""
    from .llm.profiles import get_profile

    running = info.get("running") or {}
    if not any(running.values()):
        return ""
    try:
        profile = get_profile(info["profile"])
    except Exception:  # noqa: BLE001
        return ""

    wrong: list[str] = []
    for role, spec in (("vision", profile.vision), ("actuator", profile.actuator)):
        loaded = running.get(role)
        if loaded and spec.hf_file and loaded != spec.hf_file:
            wrong.append(f"{role}: profile expects {spec.hf_file}, server has {loaded}")
    if not wrong:
        return ""
    return (
        f"Profile {info['profile']!r} does not match what is loaded. "
        + "; ".join(wrong)
        + ". Switching profiles edits a file; it does not restart the servers. "
        "Restart them (scripts/serve.sh) or switch the profile back. Until then, "
        "trust the loaded models below, not the profile name."
    )


def _guidance(info: dict[str, Any]) -> list[str]:
    """Only lines that change what a Playbook should look like."""
    lines: list[str] = []
    profile = info.get("profile", "")

    if info.get("mismatch"):
        # Loud and first: every other line below is derived from the profile name, which
        # this says is not what is actually serving.
        lines.append("MISMATCH -- " + info["mismatch"])
        # Guidance keyed to the profile name is untrustworthy here, so say what is
        # actually loaded and stop guessing from the name.
        running = info.get("running") or {}
        lines.append(
            "Loaded right now: vision "
            f"{running.get('vision') or 'unknown'}, actuator "
            f"{running.get('actuator') or 'unknown'}. Judge grounding quality from those."
        )
        profile = ""

    if not info["grammar_constrained"]:
        lines.append(
            "Ollama backend: bursts are NOT grammar-constrained. The actuator can emit a "
            "malformed burst, which costs a cycle. Keep briefs to one instruction, and "
            "expect occasional dropped cycles that are not your Playbook's fault. "
            "llama.cpp makes this class of failure impossible."
        )
    else:
        lines.append(
            "llama.cpp backend: both models are grammar-constrained. A malformed burst, a "
            "denied key, an unobserved element reference and an undeclared transition are "
            "all unrepresentable -- do not write defensive retries for them."
        )

    if profile == "hyper":
        lines.append(
            "PROFILE `hyper`: the vision model cannot ground reliably. Do NOT build states "
            "around sees() -- boxes will come back and they will often be wrong, which "
            "means clicks in the wrong place. Drive this profile with probes and reflexes, "
            "keep `watch` empty or tiny, and fence every click with click_allow_regions "
            "plus require_target_element."
        )
    elif profile in ("beefy", "beefy_moe"):
        lines.append(
            "PROFILE is large: a perceived cycle costs 1-2.5 s. Set perception.mode to "
            "'on_change' and max_elements to 2-3, and do not attempt real-time games."
        )
    elif profile == "fast":
        lines.append(
            "PROFILE `fast`: grounding is unchanged from `lean`, but the actuator is 0.6B. "
            "It stays legal but is likelier to pick a legal-but-wrong action. Keep each "
            "brief to a single instruction and prefer more states over more complex ones."
        )
    elif profile == "cpu_only":
        lines.append(
            "PROFILE `cpu_only`: several seconds per cycle. Useful for validating Playbook "
            "logic in dry_run, not for driving anything in real time."
        )

    if info["platform"] == "Windows":
        lines.append(
            "Windows: typing is layout-independent (UTF-16 direct), so t:\"...\" is safe "
            "for any text. Input cannot reach windows owned by an elevated process, and "
            "that fails silently -- if a burst appears to do nothing over an admin window, "
            "that is why. Fullscreen exclusive games may capture as black; borderless "
            "windowed works."
        )
    else:
        lines.append(
            "Linux: typing sends scancodes, so punctuation depends on the active keyboard "
            "layout. Long or non-ASCII t:\"...\" is routed through the clipboard "
            "automatically; keep typed text short where you can."
        )

    if info.get("pointer_mode") == "relative":
        lines.append(
            "pointer_mode is 'relative': g:N and m:X,Y are emulated by homing and "
            "stepping. Prefer r: deltas for camera control."
        )

    if info.get("dry_run_default"):
        lines.append(
            "dry_run defaults to true. A run will parse, safety-check and journal every "
            "burst while touching nothing. Validate that way first, then pass "
            "dry_run=false explicitly."
        )
    else:
        lines.append(
            "dry_run defaults to FALSE on this machine: a run will inject input "
            "immediately. Pass dry_run=true explicitly when you only mean to validate."
        )

    return lines


def briefing_text() -> str:
    """Compact form for the MCP server's instructions field."""
    try:
        info = active_build()
    except Exception:  # noqa: BLE001
        return ""

    header = (
        f"ACTIVE BUILD: {info['platform']} · {info['engine']} · profile "
        f"{info['profile']}"
        + (" (EXPERIMENTAL)" if info.get("experimental") else "")
    )
    models = (
        f"  vision {info.get('vision_model')} · actuator {info.get('actuator_model')}"
    )
    lines = [header, models]
    running = info.get("running") or {}
    if any(running.values()):
        lines.append(
            f"  loaded: {running.get('vision') or '?'} / {running.get('actuator') or '?'}"
        )
    if info.get("expected_cycle_ms") and any(info["expected_cycle_ms"]):
        low, high = info["expected_cycle_ms"]
        lines.append(f"  expected cycle {low}-{high} ms")
    lines.append("")
    lines.extend(f"- {line}" for line in info["guidance"])
    lines.append("")
    lines.append(
        "This reflects startup. If the profile changes mid-session, "
        "voltage_reference returns the current one."
    )
    return "\n".join(lines)
