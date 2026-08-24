"""GBNF grammar generation -- the mechanism that makes small models reliable.

This is the core answer to "these models are stupid, they need clear instructions".
Instructions are advisory; a grammar is not. llama.cpp masks the logits at every step so
that only tokens which can continue a valid parse have non-zero probability. A 1.7B model
under a grammar cannot emit a malformed burst, cannot invent a key name, cannot name a
state that does not exist, and cannot hallucinate a UI element outside the vocabulary the
orchestrator supplied. Not "usually doesn't" -- *cannot*.

That changes the engineering shape of the whole system. There is no retry loop, no
"please respond in valid JSON", no defensive parsing of half-formed output, and no
temperature-0 superstition. It also makes the output shorter, because the model does not
need to spend tokens on structure the grammar already guarantees.

Grammars are generated per state rather than fixed, because the useful constraints are
state-specific: the vocabulary of visible elements, the set of legal next states, and the
verbs this state is allowed to use. Generation is cheap and the results are cached by the
session.

Three constraints below are worth calling out as deliberate:

  * `coord` admits only 0-1000, which forces normalised coordinates and eliminates the
    entire class of "the model returned resized-image pixels" bugs at the source.
  * `keyname` enumerates the allowed keys, intersected with the policy allowlist. A
    denied key is unreachable rather than merely refused later by the governor.
  * `target` enumerates exactly the transitions this state declares, so the actuator can
    only propose control flow the orchestrator already authorised.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..inputs import keymap as km
from ..models.observation import FLAGS

__all__ = [
    "burst_grammar", "observation_grammar", "actuator_grammar", "escape_literal",
    "vision_vocabulary",
]


def vision_vocabulary(
    labels: Sequence[str], *, include_generic: bool = True
) -> list[str]:
    """The ordered label vocabulary the vision grammar indexes into.

    The parser must build this list identically, because the model reports a label by its
    position here. Both sides call this function so the ordering cannot drift.
    """
    from ..models.observation import GENERIC_LABELS as _GENERIC

    vocabulary = list(dict.fromkeys([*labels, *(_GENERIC if include_generic else ())]))
    return vocabulary or list(_GENERIC)

# Keys the actuator may name. A subset of the full keymap: enough for real UI and game
# control, small enough to keep the grammar compact.
_COMMON_KEYS: tuple[str, ...] = (
    *"abcdefghijklmnopqrstuvwxyz",
    *"0123456789",
    "enter", "esc", "tab", "space", "backspace", "delete", "insert",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "ctrl", "shift", "alt", "meta",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "minus", "equal", "comma", "dot", "slash", "semicolon", "apostrophe",
    "leftbrace", "rightbrace", "backslash", "grave",
)


def escape_literal(text: str) -> str:
    """Escape a Python string for use as a GBNF double-quoted literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _alternation(values: Iterable[str], *, quote: bool = True) -> str:
    items = [f'"{escape_literal(v)}"' if quote else v for v in values]
    return " | ".join(items) if items else '"\\u0000"'  # unmatchable if empty


def _resolve_keys(allow_keys: Sequence[str] | None, deny_keys: Sequence[str]) -> list[str]:
    """Intersect the common key set with the policy, comparing canonical names."""
    denied = {km.canonical_key(k) for k in deny_keys}
    if allow_keys is not None:
        allowed = {km.canonical_key(k) for k in allow_keys}
        keys = [k for k in _COMMON_KEYS if km.canonical_key(k) in allowed]
    else:
        keys = list(_COMMON_KEYS)
    return [k for k in keys if km.canonical_key(k) not in denied]


def burst_grammar(
    *,
    allow_verbs: Sequence[str],
    allow_keys: Sequence[str] | None = None,
    deny_keys: Sequence[str] = (),
    max_actions: int = 16,
    max_text_len: int = 64,
    allow_buttons: Sequence[str] = ("l", "r", "m"),
    n_elements: int = 0,
) -> str:
    """GBNF for a bare burst string. Used for reflex authoring and `execute_burst`."""
    return _burst_rules(
        allow_verbs=allow_verbs,
        allow_keys=allow_keys,
        deny_keys=deny_keys,
        max_actions=max_actions,
        max_text_len=max_text_len,
        allow_buttons=allow_buttons,
        n_elements=n_elements,
        root="root ::= burst",
    )


def actuator_grammar(
    *,
    allow_verbs: Sequence[str],
    targets: Sequence[str],
    allow_keys: Sequence[str] | None = None,
    deny_keys: Sequence[str] = (),
    max_actions: int = 16,
    max_text_len: int = 64,
    allow_buttons: Sequence[str] = ("l", "r", "m"),
    # The note is purely diagnostic -- it goes in the journal and nothing reads it at
    # runtime -- but it is generated token by token on the critical path. Measured on a
    # 1.7B actuator: note_len 48 costs 412 ms per cycle, 12 costs 184 ms, 0 costs 140 ms.
    # 12 characters is enough to say "save button" and cheap enough to keep on.
    note_len: int = 12,
    n_elements: int = 0,
) -> str:
    """GBNF for the actuator's full reply: ``<burst>|<target>|<note>``.

    `targets` are the state names this state may transition to. "." means stay put and is
    always available. Anything not listed is unreachable, which is the point.

    `n_elements` is how many elements the vision layer reported this cycle, which bounds
    the `g:` index. It changes per cycle, so this grammar is regenerated per cycle --
    string building, a few microseconds, and it buys structural impossibility of an
    out-of-range element reference.
    """
    target_options = ['"."', *(f'"{escape_literal(t)}"' for t in targets)]
    extra = [
        f"target ::= {' | '.join(target_options)}",
        # '-' is last in the class so it is a literal, not a range separator, and needs
        # no escape -- GBNF does not recognise \\- as an escape sequence.
        f"note ::= [a-zA-Z0-9 ,.'-]{{0,{note_len}}}",
    ]
    return _burst_rules(
        allow_verbs=allow_verbs,
        allow_keys=allow_keys,
        deny_keys=deny_keys,
        max_actions=max_actions,
        max_text_len=max_text_len,
        allow_buttons=allow_buttons,
        n_elements=n_elements,
        root='root ::= burst "|" target "|" note',
        extra_rules=extra,
    )


def actuator_token_budget(
    *,
    max_actions: int,
    max_text_len: int,
    targets: Sequence[str] = (),
    note_len: int = 12,
) -> int:
    """How many tokens the actuator must be allowed, given what the grammar permits.

    These two numbers have to be derived from one source or they drift apart, and when
    they do the failure is silent and total: the grammar happily generates a 20-action
    burst, the sampler stops at the token limit part-way through action 17, and the reply
    arrives with a half-written action on the end. That does not parse, so the whole cycle
    is wasted -- and it only happens on the cycles where the model had the most to say.

    Capping tokens is also not a way to make the actuator faster. Under a grammar, output
    length is set by the grammar; a lower ceiling does not produce a shorter burst, it
    produces a broken one. To get shorter output, lower `max_actions_per_burst` (which
    regenerates the grammar) or `note_len`.

    Deliberately generous. `max_tokens` is only a ceiling -- a model that stops earlier
    costs nothing -- so the only real error is being too small.
    """
    longest_target = max((len(t) for t in targets), default=1)
    return min(
        1024,
        max(
            64,
            # ~7 tokens for a worst-case action like `k:ctrl+shift+t` plus its separator.
            max_actions * 7
            # Typed text is the one unbounded payload; roughly 3 characters per token.
            + max_text_len // 2
            + longest_target // 2
            + note_len // 2
            # Separators, the reply's two '|' fields, and slack for a tokeniser that
            # splits punctuation more finely than assumed.
            + 24,
        ),
    )


def _burst_rules(
    *,
    allow_verbs: Sequence[str],
    allow_keys: Sequence[str] | None,
    deny_keys: Sequence[str],
    max_actions: int,
    max_text_len: int,
    allow_buttons: Sequence[str],
    root: str,
    n_elements: int = 0,
    extra_rules: Sequence[str] = (),
) -> str:
    verbs = set(allow_verbs)
    keys = _resolve_keys(allow_keys, deny_keys)
    buttons = [b for b in allow_buttons if b in ("l", "r", "m", "4", "5")] or ["l"]

    action_alts: list[str] = []
    rules: list[str] = []

    if "k" in verbs and keys:
        action_alts.append("kchord")
        rules.append('kchord ::= "k:" chord')
        rules.append('chord ::= keyname ("+" keyname){0,3}')
    if "d" in verbs and keys:
        action_alts.append("kdown")
        rules.append('kdown ::= "d:" keyname')
    if "u" in verbs and keys:
        action_alts.append("kup")
        rules.append('kup ::= "u:" keyname')
    if keys:
        rules.append(f"keyname ::= {_alternation(keys)}")

    if "t" in verbs:
        action_alts.append("typetext")
        rules.append('typetext ::= "t:\\"" textchars "\\""')
        # Exclude only the characters that would break the parse: the closing quote, the
        # escape character, and newlines. Kept to escapes llama.cpp's GBNF reader
        # definitely supports rather than a \x00-\x1F range.
        rules.append(f'textchars ::= [^"\\\\\\n\\r]{{0,{max_text_len}}}')

    # `g:` is only offered when there is something to point at. With zero elements the
    # rule is omitted entirely, so the model cannot reference a phantom target.
    if "g" in verbs and n_elements > 0:
        action_alts.append("gotoel")
        rules.append("gotoel ::= \"g:\" elidx")
        rules.append(f"elidx ::= {_alternation(str(i) for i in range(min(n_elements, 10)))}")
    if "m" in verbs:
        action_alts.append("moveabs")
        rules.append('moveabs ::= "m:" px "," px')
    if "r" in verbs:
        action_alts.append("moverel")
        rules.append('moverel ::= "r:" signed "," signed')

    if "c" in verbs:
        action_alts.append("click")
        rules.append('click ::= "c:" button [23]?')
    if "p" in verbs:
        action_alts.append("btndown")
        rules.append('btndown ::= "p:" button')
    if "e" in verbs:
        action_alts.append("btnup")
        rules.append('btnup ::= "e:" button')
    if {"c", "p", "e"} & verbs:
        rules.append(f"button ::= {_alternation(buttons)}")

    if "s" in verbs:
        action_alts.append("vscroll")
        rules.append('vscroll ::= "s:" signed')
    if "h" in verbs:
        action_alts.append("hscroll")
        rules.append('hscroll ::= "h:" signed')

    if "w" in verbs:
        action_alts.append("wait")
        rules.append('wait ::= "w:" millis')
        rules.append('millis ::= [1-9] [0-9]{0,3}')

    if not action_alts:
        raise ValueError("no verbs are permitted, so no grammar can be generated")

    # Only emit the numeric rules that are actually referenced. An unreferenced rule is
    # harmless but makes the grammar harder to read when debugging a refusal.
    if "m" in verbs:
        # Burst coordinates are screen pixels (0-9999), unlike the vision grammar's
        # normalised 0-1000 space. The parser range-checks against the real desktop
        # size; the grammar only bounds the digit count.
        rules.append('px ::= "0" | [1-9] [0-9]{0,3}')
    if {"r", "s", "h"} & verbs:
        rules.append('signed ::= ("+" | "-") [1-9] [0-9]{0,3}')

    body = [
        root,
        f'burst ::= "." | action (";" action){{0,{max(0, max_actions - 1)}}}',
        f"action ::= {' | '.join(action_alts)}",
        *rules,
        *extra_rules,
    ]
    return "\n".join(body) + "\n"


def observation_grammar(
    labels: Sequence[str],
    *,
    max_elements: int = 6,
    read_text: bool = True,
    scene_len: int = 60,
    text_len: int = 48,
    include_generic: bool = True,
) -> str:
    """GBNF for the vision model's reply.

    The label alternation is the important part: the vision model is given a closed
    vocabulary drawn from the current state's `watch` list, so `sees("address bar")`
    downstream is comparing against terms the orchestrator chose rather than whatever
    noun the model happened to pick this cycle.
    """
    vocabulary = vision_vocabulary(labels, include_generic=include_generic)

    root = (
        'root ::= "{\\"s\\":" scene ",\\"e\\":[" elements "]"'
        + (" textpart" if read_text else "")
        + ' flagpart "}"'
    )

    # Elements are emitted as a bare array, [label_index, x1, y1, x2, y2], rather than an
    # object with named keys and a spelled-out label.
    #
    # This is a latency decision, not a style one. Decode is the vision model's
    # bottleneck at roughly 22 ms per output token (measured, not assumed -- prefill is
    # ~28 ms and constant regardless of image size). The object form
    #     {"l":"address bar","b":[120,44,890,72],"c":0.9}
    # costs about 20 tokens; the array form
    #     [0,120,44,890,72]
    # costs about 11. At 22 ms/token that is ~200 ms saved per element, every cycle.
    #
    # Referring to the label by index into the closed `watch` vocabulary is also strictly
    # safer than spelling it out: the model cannot emit a label that is not in the list,
    # and it cannot misspell one.
    rules: list[str] = [
        root,
        f'scene ::= "\\"" [^"\\\\\\n\\r]{{0,{scene_len}}} "\\""',
        f'elements ::= (element ("," element){{0,{max(0, max_elements - 1)}}})?',
        'element ::= "[" labelidx "," coord "," coord "," coord "," coord "]"',
        f"labelidx ::= {_alternation(str(i) for i in range(len(vocabulary)))}",
        # Boxes are [x1, y1, x2, y2] normalised to 0-1000.
        'coord ::= "0" | [1-9] [0-9]{0,2} | "1000"',
    ]

    if read_text:
        rules += [
            'textpart ::= (",\\"t\\":[" texts "]")?',
            'texts ::= (textitem ("," textitem){0,2})?',
            f'textitem ::= "\\"" [^"\\\\\\n\\r]{{0,{text_len}}} "\\""',
        ]

    rules += [
        'flagpart ::= (",\\"f\\":[" flags "]")?',
        'flags ::= (flag ("," flag){0,3})?',
        f"flag ::= {_alternation(FLAGS)}",
    ]
    return "\n".join(rules) + "\n"
