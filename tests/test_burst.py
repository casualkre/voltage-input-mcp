"""Burst DSL: parsing, rendering, and the failure modes that matter."""

from __future__ import annotations

import pytest

from voltage_input_mcp.errors import BurstParseError
from voltage_input_mcp.models.burst import (
    Click,
    KeyChord,
    MoveAbs,
    Scroll,
    TypeText,
    Wait,
    parse_burst,
)

SCREEN = (1920, 1080)


def test_roundtrip_preserves_semantics():
    src = 'm:640,360;c:l;w:150;t:"hello";k:ctrl+s;s:-3'
    burst = parse_burst(src, screen=SCREEN)
    assert burst.render() == src
    assert parse_burst(burst.render(), screen=SCREEN).render() == src


def test_action_types_and_order():
    burst = parse_burst('k:ctrl+t;w:80;t:"x";m:10,20;c:r2;s:+2', screen=SCREEN)
    kinds = [type(a) for a in burst]
    assert kinds == [KeyChord, Wait, TypeText, MoveAbs, Click, Scroll]
    assert burst.actions[4].count == 2
    assert burst.actions[4].button == "r"
    assert burst.actions[5].amount == 2


def test_semicolon_inside_quoted_text_is_not_a_separator():
    """The naive split() breaks here, and models emit punctuation constantly."""
    burst = parse_burst('t:"a;b;c";k:enter', screen=SCREEN)
    assert len(burst) == 2
    assert burst.actions[0].text == "a;b;c"


def test_escapes_in_text():
    burst = parse_burst(r't:"say \"hi\"\nnext"', screen=SCREEN)
    assert burst.actions[0].text == 'say "hi"\nnext'


def test_input_count_excludes_waits():
    burst = parse_burst("k:a;w:100;k:b;w:100;k:c", screen=SCREEN)
    assert len(burst) == 5
    assert burst.input_count == 3


def test_duration_accounts_for_typing_and_clicks():
    assert parse_burst("w:250", screen=SCREEN).duration_ms == 250
    # Five characters at the default 8 ms interval.
    assert parse_burst('t:"abcde"', screen=SCREEN).duration_ms == 40
    # Double click: two presses plus one inter-click gap.
    assert parse_burst("c:l2", screen=SCREEN).duration_ms == 2 * 18 + 40


def test_held_keys_tracked_across_the_burst():
    assert parse_burst("d:shift;k:a;u:shift", screen=SCREEN).held_keys() == set()
    assert parse_burst("d:shift;k:a", screen=SCREEN).held_keys() == {"shift"}
    assert parse_burst("p:l;m:10,10", screen=SCREEN).held_buttons() == {"l"}


def test_key_aliases_resolve():
    for alias in ("ctrl", "control", "lctrl"):
        assert parse_burst(f"k:{alias}+a", screen=SCREEN)


def test_empty_burst_is_valid():
    assert len(parse_burst(".", screen=SCREEN)) == 0
    assert len(parse_burst("", screen=SCREEN)) == 0
    assert not parse_burst(".", screen=SCREEN)


# -- g: element references -------------------------------------------------------------


def test_element_reference_resolves_to_centre():
    burst = parse_burst("g:1;c:l", screen=SCREEN, elements=[(10, 20), (505, 58)])
    assert isinstance(burst.actions[0], MoveAbs)
    assert (burst.actions[0].x, burst.actions[0].y) == (505, 58)


def test_element_reference_out_of_range_is_rejected():
    with pytest.raises(BurstParseError, match="out of range"):
        parse_burst("g:3", screen=SCREEN, elements=[(1, 1)])


def test_element_reference_without_elements_is_rejected():
    """Silently clicking nowhere is worse than losing the cycle."""
    with pytest.raises(BurstParseError, match="no elements"):
        parse_burst("g:0", screen=SCREEN)


# -- rejections ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,match",
    [
        ("m:5000,10", "outside the 1920x1080 screen"),
        ("k:notakey", "unknown key"),
        ("x:1", "unknown verb"),
        ("t:unquoted", "double-quoted"),
        ("c:z", "invalid button"),
        ("nocolon", "missing ':'"),
        ('t:"unterminated', "unterminated"),
        ("w:99999", "out of range"),
        ("c:l9", "out of range"),
        ("k:a+b+c+d+e+f", "too long"),
    ],
)
def test_malformed_bursts_are_rejected(src, match):
    with pytest.raises(BurstParseError, match=match):
        parse_burst(src, screen=SCREEN, elements=[(1, 1)])


def test_normalised_coordinates_are_caught():
    """A model that ignored 'screen pixels' and emitted 0-1000 must not silently click."""
    parse_burst("m:500,500", screen=SCREEN)  # legal as pixels
    with pytest.raises(BurstParseError, match="outside the 640x480 screen"):
        parse_burst("m:900,900", screen=(640, 480))


def test_zero_waits_and_scrolls_are_dropped_not_rejected():
    assert len(parse_burst("w:0;s:+0;k:a", screen=SCREEN)) == 1
