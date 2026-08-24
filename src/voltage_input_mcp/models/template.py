"""Bursts with holes in them: `r:{(probe('tx') - 960) * 0.4},0`.

Why this exists
---------------
A reflex used to be a condition attached to a fixed string. That is enough to *react* --
press space when low -- but not enough to *control*. Control means the size of the action
depends on the size of the error: steer harder when further off course, brake in
proportion to how fast you are falling, scroll by the distance still to travel. With a
fixed string the only way to express that is a ladder of a dozen threshold rules, each
firing a slightly different constant, and the ladder is both unreadable and wrong at the
boundaries.

So a playbook-authored burst may contain `{expression}` holes. Each hole is a guard
expression from expr.py -- the same sandbox, the same function table, the same
compile-time validation -- evaluated against the live context at the moment the burst
fires. `r:{clamp((probe('tx') - 960) * 0.4, -300, 300)},0` is a proportional controller
in one line, and it runs at reflex rate with no model involved.

Who may use it
--------------
Only the orchestrator, in playbook text: reflex `do`, `on_enter`, `on_exit`. The actuator
model still emits plain, literal bursts under a GBNF grammar. That division is deliberate
and load-bearing: the grammar is what makes a malformed or unsafe burst unrepresentable
rather than merely unlikely, and a model that could emit expressions would be a model that
could emit expressions the grammar cannot bound. The orchestrator is trusted and its
playbook is validated up front; the 1.7B is neither.

Types
-----
Every interpolatable position in the burst DSL is an integer -- pixels, milliseconds,
scroll detents, repeat counts -- so a hole must evaluate to a number, and a float is
rounded on the way in. A hole yielding a string is an error rather than a splice: holes
are not scanned inside `t:"..."`, so a string-valued hole is always a mistake, and
allowing one would turn "type what the playbook says" into "type whatever an expression
computed".
"""

from __future__ import annotations

from ..errors import BurstParseError, ExpressionError
from ..expr import Guard, GuardContext
from .burst import Burst, parse_burst

__all__ = ["BurstTemplate"]

# Stand-in value used to type-check the skeleton at compile time. 1 is the only choice
# that is in range everywhere a hole can appear: 0 is an invalid click repeat, and a
# large value is out of range for a scroll.
_PROBE_VALUE = "1"


class BurstTemplate:
    """A burst source with `{expr}` holes, compiled once and rendered per fire.

    A template with no holes is indistinguishable from a plain burst and costs nothing
    extra: `static` is set, the parsed `Burst` is kept, and `burst()` returns it without
    touching the expression machinery.
    """

    __slots__ = ("source", "_literals", "_holes", "_static")

    def __init__(self, source: str) -> None:
        self.source = source
        self._literals, self._holes = _split_holes(source)

        if not self._holes:
            self._static: Burst | None = parse_burst(source)
            return

        # Validate the skeleton now rather than at cycle 40 of a live run. Substituting a
        # representative number proves the surrounding syntax is a real burst; the holes
        # themselves are validated by having compiled as guards at all.
        skeleton = _PROBE_VALUE.join(self._literals)
        try:
            parse_burst(skeleton)
        except BurstParseError as exc:
            raise BurstParseError(
                f"{exc.detail} (checked with every {{...}} standing in as "
                f"{_PROBE_VALUE}; the expressions themselves are fine, the burst around "
                f"them is not)",
                source=source,
            ) from exc
        self._static = None

    # -- introspection ---------------------------------------------------------------

    @property
    def is_dynamic(self) -> bool:
        return self._static is None

    @property
    def referenced_probes(self) -> set[str]:
        probes: set[str] = set()
        for hole in self._holes:
            probes |= hole.referenced_probes
        return probes

    @property
    def referenced_labels(self) -> set[str]:
        labels: set[str] = set()
        for hole in self._holes:
            labels |= hole.referenced_labels
        return labels

    # -- rendering -------------------------------------------------------------------

    def render(self, ctx: GuardContext) -> str:
        if self._static is not None:
            return self._static.source
        out = [self._literals[0]]
        for hole, literal in zip(self._holes, self._literals[1:], strict=True):
            out.append(_as_int_text(hole, ctx))
            out.append(literal)
        return "".join(out)

    def burst(
        self,
        ctx: GuardContext,
        *,
        screen: tuple[int, int] | None = None,
    ) -> Burst:
        """Render and parse. Raises `BurstParseError` if the result is not a valid burst.

        A dynamic burst can go out of range at runtime in a way the skeleton check cannot
        catch -- a steering term that overshoots produces `m:4820,300` on a 1920-wide
        screen. That surfaces here as a parse error carrying the *rendered* text, which is
        the thing the author needs to see; the caller journals it and skips the fire
        rather than moving the pointer somewhere arbitrary.
        """
        if self._static is not None:
            return self._static
        return parse_burst(self.render(ctx), screen=screen)


def _as_int_text(hole: Guard, ctx: GuardContext) -> str:
    try:
        value = hole.evaluate(ctx)
    except ExpressionError:
        raise
    except Exception as exc:  # noqa: BLE001 - a hole must fail as an expression error
        raise ExpressionError(f"{hole.source!r} failed at runtime: {exc}") from exc

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
            raise ExpressionError(
                f"{hole.source!r} produced {value}, which is not a usable coordinate; "
                f"guard the division or wrap it in clamp()"
            )
        return str(int(round(value)))
    raise ExpressionError(
        f"{hole.source!r} produced {type(value).__name__}, but a burst hole must be a "
        f"number -- every interpolatable field is an integer"
    )


def _split_holes(source: str) -> tuple[list[str], list[Guard]]:
    """Split `a{x}b{y}c` into (['a', 'b', 'c'], [Guard(x), Guard(y)]).

    Braces inside a quoted text payload are literal. `t:"{}"` types two braces; it does
    not interpolate. Scanning has to track quotes for the same reason `_split_actions`
    does -- the moment a task involves punctuation, a model emits some.
    """
    literals: list[str] = []
    holes: list[Guard] = []
    buf: list[str] = []
    in_quotes = False
    escaped = False
    i = 0
    while i < len(source):
        ch = source[i]
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == "\\" and in_quotes:
            buf.append(ch)
            escaped = True
        elif ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
        elif ch == "{" and not in_quotes:
            end = source.find("}", i + 1)
            if end == -1:
                raise BurstParseError(
                    "unterminated '{' in burst; an expression hole needs a closing '}'",
                    source=source,
                )
            expression = source[i + 1 : end].strip()
            if not expression:
                raise BurstParseError("empty '{}' in burst", source=source)
            try:
                holes.append(Guard(expression))
            except ExpressionError as exc:
                raise BurstParseError(
                    f"expression {{{expression}}} is not valid: {exc}", source=source
                ) from exc
            literals.append("".join(buf))
            buf = []
            i = end + 1
            continue
        elif ch == "}" and not in_quotes:
            raise BurstParseError(
                "stray '}' in burst; expression holes look like {probe('x') * 2}",
                source=source,
            )
        else:
            buf.append(ch)
        i += 1

    if in_quotes:
        raise BurstParseError("unterminated quoted text payload", source=source)
    literals.append("".join(buf))
    return literals, holes
