"""Grammar generation and the uinput wire encoding.

These are the two places where a silent mistake is most expensive: a grammar that permits
something the policy forbids, and a struct layout that is off by a field.
"""

from __future__ import annotations

import struct

import pytest

from voltage_input_mcp.inputs import keymap as km
from voltage_input_mcp.inputs import uinput
from voltage_input_mcp.llm.grammar import actuator_grammar, observation_grammar

# -- grammar ---------------------------------------------------------------------------


def test_element_index_range_matches_observed_count():
    """An out-of-range g: reference must be unrepresentable, not merely rejected."""
    g = actuator_grammar(allow_verbs=["g", "c"], targets=[], n_elements=3)
    assert 'elidx ::= "0" | "1" | "2"' in g
    assert '"3"' not in g.split("elidx ::=")[1].split("\n")[0]


def test_g_verb_omitted_entirely_when_nothing_was_seen():
    g = actuator_grammar(allow_verbs=["g", "c"], targets=[], n_elements=0)
    assert "gotoel" not in g


def test_denied_keys_are_absent_from_the_grammar():
    """Defence before the governor: the model cannot emit the token at all."""
    g = actuator_grammar(
        allow_verbs=["k"], targets=[], deny_keys=["delete", "leftmeta"], n_elements=0
    )
    keyname = [ln for ln in g.splitlines() if ln.startswith("keyname ::=")][0]
    assert '"delete"' not in keyname
    assert '"meta"' not in keyname
    assert '"enter"' in keyname


def test_allow_keys_restricts_the_grammar():
    g = actuator_grammar(
        allow_verbs=["k"], targets=[], allow_keys=["w", "a", "s", "d"], n_elements=0
    )
    keyname = [ln for ln in g.splitlines() if ln.startswith("keyname ::=")][0]
    assert {keyname.count(f'"{k}"') for k in "wasd"} == {1}
    assert '"q"' not in keyname


def test_only_declared_transitions_are_reachable():
    g = actuator_grammar(allow_verbs=["c"], targets=["alpha", "beta"], n_elements=1)
    target = [ln for ln in g.splitlines() if ln.startswith("target ::=")][0]
    assert target == 'target ::= "." | "alpha" | "beta"'


def test_verbs_outside_the_allowlist_have_no_rule():
    g = actuator_grammar(allow_verbs=["k", "w"], targets=[], n_elements=2)
    for absent in ("click", "typetext", "moveabs", "vscroll", "gotoel"):
        assert absent not in g
    assert "kchord" in g and "wait" in g


def test_no_dangling_rule_references():
    """Every referenced rule must be defined -- llama.cpp errors on an undefined rule."""
    for verbs in (["k"], ["c", "g"], ["m", "r", "s"], ["t", "w"],
                  ["k", "d", "u", "t", "g", "m", "r", "c", "p", "e", "s", "h", "w"]):
        g = actuator_grammar(allow_verbs=verbs, targets=["x"], n_elements=2)
        defined = {ln.split("::=")[0].strip() for ln in g.splitlines() if "::=" in ln}
        for line in g.splitlines():
            if "::=" not in line:
                continue
            body = line.split("::=", 1)[1]
            # Strip quoted literals and character classes before looking for identifiers.
            cleaned = []
            in_str = False
            in_class = False
            escaped = False
            for ch in body:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"' and not in_class:
                    in_str = not in_str
                    continue
                if not in_str and ch == "[":
                    in_class = True
                    continue
                if in_class and ch == "]":
                    in_class = False
                    continue
                if not in_str and not in_class:
                    cleaned.append(ch)
            for token in "".join(cleaned).replace("(", " ").replace(")", " ").split():
                token = token.strip("|?*+")
                if token and token[0].isalpha():
                    assert token in defined, f"rule {token!r} used but never defined ({verbs})"


def test_observation_labels_are_a_closed_vocabulary():
    """Labels are reported as an index, so the vocabulary is closed by construction.

    The model picks a position in the list; it cannot spell a label at all, let alone
    spell one that is not there.
    """
    from voltage_input_mcp.llm.grammar import vision_vocabulary

    watch = ["address bar", "file list"]
    vocabulary = vision_vocabulary(watch)
    assert vocabulary[:2] == watch
    # Generic fallbacks follow, so the model can report a dialog it was not told to
    # look for, but nothing outside the union is representable.
    assert "dialog" in vocabulary

    g = observation_grammar(watch, max_elements=3)
    line = [ln for ln in g.splitlines() if ln.startswith("labelidx ::=")][0]
    assert line.startswith('labelidx ::= "0" | "1"')


def test_observation_coordinates_are_locked_to_0_1000():
    """This is what eliminates the resized-image-pixel class of grounding bug."""
    g = observation_grammar(["x"])
    assert 'coord ::= "0" | [1-9] [0-9]{0,2} | "1000"' in g


def test_grammar_has_no_unsupported_escapes():
    """GBNF does not recognise \\- ; a stray one makes llama.cpp reject the grammar."""
    for g in (
        actuator_grammar(allow_verbs=["k", "t", "m", "c"], targets=["a"], n_elements=1),
        observation_grammar(["a"], read_text=True),
    ):
        assert "\\-" not in g
        assert "\\x" not in g


# -- uinput encoding -------------------------------------------------------------------


def test_struct_layouts_match_the_kernel():
    """These sizes are baked into the ioctl numbers; a mismatch fails silently at runtime."""
    assert struct.calcsize(uinput._EVENT_FMT) == 24        # struct input_event, 64-bit
    assert struct.calcsize(uinput._SETUP_FMT) == 92        # struct uinput_setup
    assert struct.calcsize(uinput._ABS_SETUP_FMT) == 28    # struct uinput_abs_setup


def test_ioctl_numbers_match_known_kernel_values():
    assert uinput.UI_DEV_CREATE == 0x5501
    assert uinput.UI_DEV_DESTROY == 0x5502
    assert uinput.UI_SET_EVBIT == 0x40045564
    assert uinput.UI_SET_KEYBIT == 0x40045565
    assert uinput.UI_DEV_SETUP == 0x405C5503
    assert uinput.UI_ABS_SETUP == 0x401C5504


def test_absolute_axis_scaling_hits_the_endpoints():
    devices = uinput.DeviceSet(screen=(1920, 1080))
    assert devices.to_abs(0, 0) == (0, 0)
    assert devices.to_abs(1919, 1079) == (uinput.ABS_MAX, uinput.ABS_MAX)
    mid = devices.to_abs(960, 540)
    assert abs(mid[0] - uinput.ABS_MAX // 2) < 40
    # Out-of-range input is clamped, never wrapped.
    assert devices.to_abs(99999, -5) == (uinput.ABS_MAX, 0)


def test_pointer_device_is_classified_as_a_mouse_not_a_touchscreen():
    """udev tags ABS_X/ABS_Y + BTN_TOUCH as a touchscreen; we must not advertise BTN_TOUCH."""
    caps = uinput.pointer_abs_caps()
    assert km.ABS_X in caps.abs_axes and km.ABS_Y in caps.abs_axes
    assert km.BTN_LEFT in caps.keys
    assert km.INPUT_PROP_POINTER in caps.props
    assert km.INPUT_PROP_DIRECT not in caps.props
    assert 0x14A not in caps.keys  # BTN_TOUCH


def test_wheel_declares_both_resolutions():
    """Emitting HI_RES without declaring it makes scrolling work in some apps only."""
    caps = uinput.pointer_abs_caps()
    for axis in (km.REL_WHEEL, km.REL_HWHEEL, km.REL_WHEEL_HI_RES, km.REL_HWHEEL_HI_RES):
        assert axis in caps.rel_axes


# -- keymap ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,canonical",
    [("ctrl", "leftctrl"), ("control", "leftctrl"), ("super", "leftmeta"),
     ("win", "leftmeta"), ("esc", "esc"), ("escape", "esc"), ("return", "enter"),
     ("pgup", "pageup"), ("del", "delete"), ("printscreen", "sysrq")],
)
def test_key_aliases(alias, canonical):
    assert km.canonical_key(alias) == canonical
    assert km.resolve_key(alias) == km.KEYS[canonical]


def test_ascii_typing_table():
    assert km.ascii_to_keys("a") == (km.KEYS["a"], False)
    assert km.ascii_to_keys("A") == (km.KEYS["a"], True)
    assert km.ascii_to_keys("!") == (km.KEYS["1"], True)
    assert km.ascii_to_keys(" ") == (km.KEYS["space"], False)
    assert km.ascii_to_keys("\n") == (km.KEYS["enter"], False)
    # Not typeable on a US layout -- must route through the clipboard instead.
    assert km.ascii_to_keys("ğ") is None
    assert km.ascii_to_keys("😀") is None


def test_is_typeable_gates_the_clipboard_fallback():
    assert km.is_typeable("Hello, world! (test) #1")
    assert not km.is_typeable("Ağustos")
    assert not km.is_typeable("naïve")


# -- compact element encoding ----------------------------------------------------------


def test_elements_use_the_compact_array_form():
    """Decode is the vision bottleneck, so elements are [idx, x1, y1, x2, y2]."""
    g = observation_grammar(["address bar"], max_elements=3)
    assert 'element ::= "[" labelidx "," coord "," coord "," coord "," coord "]"' in g
    # The verbose object form must be gone -- it costs ~9 more tokens per element.
    assert '\\"l\\":' not in g


def test_label_index_range_covers_the_whole_vocabulary():
    from voltage_input_mcp.llm.grammar import vision_vocabulary

    watch = ["address bar", "file list"]
    vocabulary = vision_vocabulary(watch)
    g = observation_grammar(watch)
    line = [ln for ln in g.splitlines() if ln.startswith("labelidx ::=")][0]
    assert line.count('"') == 2 * len(vocabulary)
    assert f'"{len(vocabulary) - 1}"' in line
    assert f'"{len(vocabulary)}"' not in line


def test_parser_and_grammar_agree_on_label_ordering():
    """The one way this encoding can fail silently: index drift between the two sides.

    If the grammar and the parser build the vocabulary differently, every element comes
    back with the wrong label and the run misbehaves in a way that looks like the vision
    model hallucinating.
    """
    from voltage_input_mcp.llm.grammar import vision_vocabulary
    from voltage_input_mcp.models.observation import CoordinateMapper, parse_vision_output

    watch = ["address bar", "file list", "save button"]
    vocabulary = vision_vocabulary(watch)
    mapper = CoordinateMapper(capture_size=(1000, 1000), screen_size=(1000, 1000))

    for index, expected in enumerate(vocabulary[:4]):
        raw = f'{{"s":"x","e":[[{index},100,100,200,200]]}}'
        obs = parse_vision_output(raw, mapper, vocabulary=vocabulary)
        assert obs.elements[0].label == expected


def test_parser_still_accepts_the_verbose_form_for_ollama():
    from voltage_input_mcp.models.observation import CoordinateMapper, parse_vision_output

    mapper = CoordinateMapper(capture_size=(1000, 1000), screen_size=(1000, 1000))
    raw = '{"s":"x","e":[{"l":"address bar","b":[100,100,200,200],"c":0.8}]}'
    obs = parse_vision_output(raw, mapper)
    assert obs.elements[0].label == "address bar"
    assert obs.elements[0].conf == 0.8


def test_out_of_range_label_index_is_dropped_not_guessed():
    from voltage_input_mcp.models.observation import CoordinateMapper, parse_vision_output

    mapper = CoordinateMapper(capture_size=(1000, 1000), screen_size=(1000, 1000))
    obs = parse_vision_output(
        '{"s":"x","e":[[99,100,100,200,200]]}', mapper, vocabulary=["only one"]
    )
    assert obs.elements == []


def test_actuator_note_is_short_by_default():
    """The note is diagnostic only and cost 55% of actuator latency at 48 chars."""
    g = actuator_grammar(allow_verbs=["c"], targets=[], n_elements=1)
    assert "{0,12}" in [ln for ln in g.splitlines() if ln.startswith("note ::=")][0]
