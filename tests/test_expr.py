"""Guard expressions: evaluation semantics and the sandbox boundary."""

from __future__ import annotations

import pytest

from voltage_input_mcp.errors import ExpressionError
from voltage_input_mcp.expr import Guard, GuardContext


@pytest.fixture
def ctx() -> GuardContext:
    return GuardContext(
        elements=[
            {"label": "address bar", "conf": 0.9},
            {"label": "file list", "conf": 0.45},
        ],
        texts=["Downloads - Dolphin", "3 items"],
        flags={"loading"},
        scene="file manager, list view",
        probes={"health": 0.18, "__frame_delta__": 0.05, "__static_for__": 4.2},
        vars={"attempts": 2, "path": "/home"},
        state={"name": "nav", "cycles": 7, "elapsed": 3.4},
        run={"cycles": 40, "elapsed": 22.0, "bursts": 12, "rejections": 1,
             "last_burst_ok": True},
    )


@pytest.mark.parametrize(
    "src,expected",
    [
        ("sees('address bar')", True),
        ("sees('nonexistent')", False),
        ("sees('file list', 0.9)", False),          # below min_conf
        ("sees('file list', 0.4)", True),
        ("count('address bar') == 1", True),
        ("conf('address bar') > 0.8", True),
        ("conf('nothing') == 0.0", True),
        ("text('downloads')", True),                 # case-insensitive
        ("text('^3 items$')", True),                 # regex
        ("text('missing')", False),
        ("flag('loading')", True),
        ("flag('error')", False),
        ("probe('health') < 0.25", True),
        ("probe('absent') == 0.0", True),
        ("elapsed() > 3", True),
        ("cycles() == 7", True),
        ("run_cycles() == 40", True),
        ("stalled(4.0)", True),
        ("stalled(9.0)", False),
        ("changed(0.01)", True),
        ("last_burst_ok()", True),
        ("rejections() == 1", True),
        ("vars.attempts < 3", True),
        ("vars.path == '/home'", True),
        ("'file' in obs.scene", True),
        ("obs.n == 2", True),
        ("'address bar' in obs.labels", True),
        ("not sees('x') and cycles() > 5", True),
        ("min(cycles(), 3) == 3", True),
        ("(1 + 2) * 3 == 9", True),
        ("3 if sees('address bar') else 4", 3),
    ],
)
def test_evaluation(ctx, src, expected):
    assert Guard(src).evaluate(ctx) == expected


def test_missing_variable_compares_false_rather_than_raising():
    """`vars.attempts < 3` must be safe before `attempts` is first assigned."""
    empty = GuardContext()
    assert Guard("vars.never_set < 3").test(empty) is False
    assert Guard("vars.never_set > 3").test(empty) is False
    assert Guard("vars.never_set == None").test(empty) is True


def test_chained_comparison(ctx):
    assert Guard("0 < probe('health') < 1").evaluate(ctx) is True


def test_division_by_zero_yields_zero_not_an_exception(ctx):
    assert Guard("1 / var('nope', 0) == 0").evaluate(ctx) is True


def test_referenced_labels_are_extractable():
    """compile_playbook uses this to warn about guards that can never fire."""
    guard = Guard("sees('a') or count('b') > 0 or conf('c') > 0.5")
    assert guard.referenced_labels == {"a", "b", "c"}
    assert Guard("probe('hp') < 1").referenced_probes == {"hp"}


# -- sandbox ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "__import__('os').system('id')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "exec('x=1')",
        "eval('1')",
        "[x for x in range(10)]",
        "{k: 1 for k in 'ab'}",
        "lambda: 1",
        "vars.__class__",
        "obs._private",
        "'a' * 999999999",
        "globals()",
        "sees",                      # bare function reference
        "unknown_function()",
        "nosuchnamespace.field",
        "x := 1",
        "vars['attempts']",          # subscript is not permitted
    ],
)
def test_sandbox_rejects_at_compile_time(src):
    with pytest.raises(ExpressionError):
        Guard(src)


def test_complexity_and_length_limits():
    with pytest.raises(ExpressionError, match="limit is"):
        Guard("1 + " * 200 + "1")
    with pytest.raises(ExpressionError, match="empty"):
        Guard("   ")


def test_bad_regex_is_reported(ctx):
    with pytest.raises(ExpressionError, match="bad regex"):
        Guard("text('[unclosed')").test(ctx)
