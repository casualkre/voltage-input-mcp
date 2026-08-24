"""Sandboxed guard expressions.

Playbook transitions and reflex rules need conditions like::

    obs.sees("address bar") and vars.attempts < 3
    probe("health") < 0.25
    elapsed() > 4.0 or flag("dialog")

These are authored by the orchestrating model and evaluated many times per second, so
they must be (a) safe -- never `eval()` -- and (b) statically checkable, so a malformed
guard is caught by `voltage.validate_playbook` rather than at cycle 40 of a live run.

Implementation is an allowlisted walk over the `ast` module's parse tree. Anything not
explicitly permitted raises `ExpressionError` at compile time. There are no loops,
comprehensions, lambdas, imports, attribute writes, subscripts, or arbitrary calls --
only the function table below, plain arithmetic, comparisons, and boolean logic.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ExpressionError

__all__ = ["Guard", "GuardContext", "GUARD_FUNCTIONS", "GUARD_NAMESPACES"]

MAX_SOURCE_LEN = 600
MAX_NODES = 220

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.IfExp,
    ast.List,
    ast.Tuple,
    ast.Load,
)

# Namespaces reachable via dotted access. Values come from GuardContext at eval time.
GUARD_NAMESPACES: frozenset[str] = frozenset({"obs", "vars", "probes", "state", "run"})


# --------------------------------------------------------------------------------------
# Function table
# --------------------------------------------------------------------------------------


def _fn_sees(ctx: GuardContext, label: str, min_conf: float = 0.4) -> bool:
    """True if the vision layer reported an element with this label above `min_conf`."""
    return any(
        e.get("label") == label and float(e.get("conf", 1.0)) >= min_conf
        for e in ctx.elements
    )


def _fn_count(ctx: GuardContext, label: str, min_conf: float = 0.4) -> int:
    return sum(
        1 for e in ctx.elements
        if e.get("label") == label and float(e.get("conf", 1.0)) >= min_conf
    )


def _fn_conf(ctx: GuardContext, label: str) -> float:
    """Highest confidence for `label`, or 0.0 if absent."""
    vals = [float(e.get("conf", 1.0)) for e in ctx.elements if e.get("label") == label]
    return max(vals) if vals else 0.0


def _fn_text(ctx: GuardContext, pattern: str) -> bool:
    """Case-insensitive regex search over all text the vision layer read this cycle."""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ExpressionError(f"bad regex {pattern!r}: {exc}") from exc
    return any(rx.search(t) for t in ctx.texts)


def _fn_flag(ctx: GuardContext, name: str) -> bool:
    return name in ctx.flags


def _fn_probe(ctx: GuardContext, name: str, default: float = 0.0) -> float:
    value = ctx.probes.get(name, default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fn_var(ctx: GuardContext, name: str, default: Any = None) -> Any:
    return ctx.vars.get(name, default)


def _fn_elapsed(ctx: GuardContext) -> float:
    """Seconds since entering the current state."""
    return float(ctx.state.get("elapsed", 0.0))


def _fn_cycles(ctx: GuardContext) -> int:
    """Loop cycles spent in the current state."""
    return int(ctx.state.get("cycles", 0))


def _fn_run_elapsed(ctx: GuardContext) -> float:
    return float(ctx.run.get("elapsed", 0.0))


def _fn_run_cycles(ctx: GuardContext) -> int:
    return int(ctx.run.get("cycles", 0))


def _fn_changed(ctx: GuardContext, threshold: float = 0.02) -> bool:
    """True if the frame differs from the previous one by more than `threshold` (0-1)."""
    return float(ctx.probes.get("__frame_delta__", 0.0)) > threshold


def _fn_stalled(ctx: GuardContext, seconds: float = 3.0) -> bool:
    """True if the screen has not materially changed for `seconds`."""
    return float(ctx.probes.get("__static_for__", 0.0)) >= seconds


def _fn_last_burst_ok(ctx: GuardContext) -> bool:
    return bool(ctx.run.get("last_burst_ok", True))


def _fn_rejections(ctx: GuardContext) -> int:
    """How many bursts the safety governor has refused so far this run."""
    return int(ctx.run.get("rejections", 0))


def _fn_rate(ctx: GuardContext, name: str, default: float = 0.0) -> float:
    """Rate of change of a number probe, per second.

    Sugar for `probe('<name>__rate')`, which the probe engine publishes alongside every
    number probe. Worth having as its own function because rate is what most continuous
    control actually keys off: `rate('meters') < -60` is "falling fast", and expressing
    that as a string suffix invites a typo that silently reads 0.0 forever.
    """
    return _fn_probe(ctx, f"{name}__rate", default)


def _fn_stale_reading(ctx: GuardContext, name: str, seconds: float = 1.0) -> bool:
    """True if this number probe has not produced a fresh value for `seconds`.

    A number probe returns its last reading forever once the thing it was reading is
    covered up -- by a modal, a scene change, a menu. The value alone cannot distinguish
    "genuinely steady" from "stopped arriving", and a guard that cannot tell the
    difference will happily keep a key held against a measurement that died a minute ago.
    """
    return float(ctx.probes.get(f"{name}__age", 0.0)) >= seconds


def _fn_held(ctx: GuardContext, name: str) -> bool:
    """True if the executor is currently holding this key or button (`btn:l` for mouse).

    Lets a guard avoid fighting itself: `not held('w')` stops a rule re-pressing what a
    latch already has down.
    """
    return name in ctx.held


def _fn_latched(ctx: GuardContext, rule_id: str) -> bool:
    """True if the named hold reflex is currently engaged."""
    return rule_id in ctx.latched


def _fn_clamp(value: Any, low: Any, high: Any) -> Any:
    """Constrain a value to [low, high].

    Present because the alternative in a steering expression is `min(max(v, lo), hi)`,
    which the small models and humans both get inside out, and because an unclamped
    servo term is the standard way an interpolated burst produces an off-screen move.
    """
    if low > high:
        low, high = high, low
    return low if value < low else (high if value > high else value)


def _fn_sign(value: Any) -> int:
    return (value > 0) - (value < 0)


# Signature note: context-taking functions are bound at eval time; pure ones are not.
_CTX_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sees": _fn_sees,
    "count": _fn_count,
    "conf": _fn_conf,
    "text": _fn_text,
    "flag": _fn_flag,
    "probe": _fn_probe,
    "rate": _fn_rate,
    "stale_reading": _fn_stale_reading,
    "var": _fn_var,
    "held": _fn_held,
    "latched": _fn_latched,
    "elapsed": _fn_elapsed,
    "cycles": _fn_cycles,
    "run_elapsed": _fn_run_elapsed,
    "run_cycles": _fn_run_cycles,
    "changed": _fn_changed,
    "stalled": _fn_stalled,
    "last_burst_ok": _fn_last_burst_ok,
    "rejections": _fn_rejections,
}

_PURE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "int": int,
    "float": float,
    "round": round,
    "bool": bool,
    "any": any,
    "all": all,
    "clamp": _fn_clamp,
    "sign": _fn_sign,
}

GUARD_FUNCTIONS: frozenset[str] = frozenset(_CTX_FUNCTIONS) | frozenset(_PURE_FUNCTIONS)

_CONSTANTS: dict[str, Any] = {"True": True, "False": False, "None": None}


# --------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class GuardContext:
    """Everything a guard may read. Constructed fresh each cycle by the session loop."""

    elements: list[dict[str, Any]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    scene: str = ""
    probes: dict[str, Any] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    # Keys and buttons the executor has down right now (`btn:l` for mouse buttons), and
    # the ids of hold reflexes currently engaged. Both are needed so a guard can see the
    # effect of the previous tick -- without them a latch cannot express hysteresis and a
    # rule cannot tell whether it already did the thing it is about to do again.
    held: set[str] = field(default_factory=set)
    latched: set[str] = field(default_factory=set)

    def namespace(self, name: str) -> Mapping[str, Any]:
        if name == "obs":
            return {
                "scene": self.scene,
                "n": len(self.elements),
                "labels": [e.get("label") for e in self.elements],
                "text": " | ".join(self.texts),
                "flags": sorted(self.flags),
            }
        if name == "vars":
            return self.vars
        if name == "probes":
            return self.probes
        if name == "state":
            return self.state
        if name == "run":
            return self.run
        raise ExpressionError(f"unknown namespace {name!r}")


# --------------------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------------------


class Guard:
    """A compiled, validated guard expression.

    Compilation is strict (raises on anything disallowed). Evaluation is forgiving:
    unknown attributes resolve to None rather than raising, because an observation
    legitimately may not contain every field on every cycle. Static name errors are
    caught at compile time instead.
    """

    __slots__ = ("source", "_tree", "_names")

    def __init__(self, source: str) -> None:
        source = (source or "").strip()
        if not source:
            raise ExpressionError("guard expression is empty")
        if len(source) > MAX_SOURCE_LEN:
            raise ExpressionError(
                f"guard expression is {len(source)} chars, limit is {MAX_SOURCE_LEN}"
            )
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"syntax error in guard {source!r}: {exc.msg}") from exc

        names: set[str] = set()
        # Name nodes appearing as the callee of a Call, tracked by identity so a bare
        # `sees` (an authoring slip) is rejected during validation rather than blowing
        # up mid-run.
        callee_ids: set[int] = {
            id(n.func) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        node_count = 0
        for node in ast.walk(tree):
            node_count += 1
            if node_count > MAX_NODES:
                raise ExpressionError(f"guard expression too complex (>{MAX_NODES} nodes)")
            if not isinstance(node, _ALLOWED_NODES):
                raise ExpressionError(
                    f"{type(node).__name__} is not allowed in a guard expression"
                )
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ExpressionError("only plain function calls are allowed in guards")
                if node.func.id not in GUARD_FUNCTIONS:
                    raise ExpressionError(
                        f"unknown guard function {node.func.id!r}; "
                        f"available: {', '.join(sorted(GUARD_FUNCTIONS))}"
                    )
                if any(k.arg is None for k in node.keywords):
                    raise ExpressionError("**kwargs is not allowed in a guard call")
                if any(isinstance(a, ast.Starred) for a in node.args):
                    raise ExpressionError("*args is not allowed in a guard call")
            if isinstance(node, ast.Attribute):
                # Attribute access resolves through Mapping.get, never getattr, so this
                # is defence in depth rather than the only barrier -- but rejecting
                # dunders at compile time turns a silent None into a clear error.
                if node.attr.startswith("_"):
                    raise ExpressionError(
                        f"attribute {node.attr!r} is not allowed; guard namespaces expose "
                        f"plain data keys only"
                    )
                root = node
                while isinstance(root, ast.Attribute):
                    root = root.value  # type: ignore[assignment]
                if not isinstance(root, ast.Name):
                    raise ExpressionError("attribute access must start from a namespace name")
                if root.id not in GUARD_NAMESPACES:
                    raise ExpressionError(
                        f"unknown namespace {root.id!r}; "
                        f"available: {', '.join(sorted(GUARD_NAMESPACES))}"
                    )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                # Sequence repetition is the one arithmetic operation that can allocate
                # unboundedly. The runtime blocks it too; this catches the literal form
                # during validation.
                for operand in (node.left, node.right):
                    if isinstance(operand, ast.Constant) and isinstance(
                        operand.value, (str, bytes)
                    ):
                        raise ExpressionError(
                            "sequence repetition is not allowed in a guard expression"
                        )
            if isinstance(node, ast.Name):
                if node.id in GUARD_FUNCTIONS and id(node) not in callee_ids:
                    raise ExpressionError(
                        f"{node.id!r} is a function and must be called: "
                        f"write {node.id}(...) rather than a bare {node.id}"
                    )
                names.add(node.id)

        for name in names:
            if name in GUARD_NAMESPACES or name in GUARD_FUNCTIONS or name in _CONSTANTS:
                continue
            raise ExpressionError(
                f"unknown name {name!r} in guard; use a namespace "
                f"({', '.join(sorted(GUARD_NAMESPACES))}), a function, or a literal"
            )

        self.source = source
        self._tree = tree.body
        self._names = frozenset(names)

    def __repr__(self) -> str:
        return f"Guard({self.source!r})"

    @property
    def referenced_labels(self) -> set[str]:
        """String literals passed to sees()/count()/conf(), for cross-checking `watch`."""
        labels: set[str] = set()
        for node in ast.walk(ast.Expression(body=self._tree)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("sees", "count", "conf")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                labels.add(node.args[0].value)
        return labels

    @property
    def referenced_probes(self) -> set[str]:
        """Probe ids this guard reads, for cross-checking against `playbook.probes`.

        `rate('x')` counts as a reference to `x`: the derivative is published by the
        probe engine, so what has to exist in the playbook is the base probe. The
        `__rate` suffix is stripped for the same reason -- someone who writes
        `probe('meters__rate')` by hand still needs `meters` declared.
        """
        probes: set[str] = set()
        for node in ast.walk(ast.Expression(body=self._tree)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("probe", "rate", "stale_reading")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                name = node.args[0].value
                probes.add(name.removesuffix("__rate").removesuffix("__age"))
        return probes

    @property
    def referenced_numbers(self) -> set[str]:
        """Probe ids read through `probe`, `rate` or `stale_reading`.

        The runtime uses this to work out which measurements a guard actually depends on,
        so it can tell when every one of them has stopped arriving and the guard is
        deciding from stale numbers without any way to notice.
        """
        return self.referenced_probes

    def evaluate(self, ctx: GuardContext) -> Any:
        return self._eval(self._tree, ctx)

    def test(self, ctx: GuardContext) -> bool:
        """Evaluate to a bool. A guard that raises is False, not an abort.

        A single bad guard should not kill a run mid-flight; the session records the
        failure in the journal so it surfaces in `voltage.status`.
        """
        try:
            return bool(self.evaluate(ctx))
        except ExpressionError:
            raise
        except Exception as exc:  # noqa: BLE001 - guards must never crash the loop
            raise ExpressionError(f"guard {self.source!r} failed at runtime: {exc}") from exc

    # -- interpreter ---------------------------------------------------------------

    def _eval(self, node: ast.AST, ctx: GuardContext) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            if node.id in GUARD_NAMESPACES:
                return ctx.namespace(node.id)
            raise ExpressionError(f"name {node.id!r} is not a value")

        if isinstance(node, ast.Attribute):
            base = self._eval(node.value, ctx)
            if isinstance(base, Mapping):
                return base.get(node.attr)
            return None

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result: Any = True
                for value in node.values:
                    result = self._eval(value, ctx)
                    if not result:
                        return result
                return result
            result = False
            for value in node.values:
                result = self._eval(value, ctx)
                if result:
                    return result
            return result

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, ctx)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            return +operand

        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left, ctx), self._eval(node.right, ctx)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                # Block string/list repetition -- it is the only memory-blowup vector left.
                if isinstance(left, (str, list, tuple)) or isinstance(right, (str, list, tuple)):
                    raise ExpressionError("sequence repetition is not allowed in a guard")
                return left * right
            if isinstance(op, ast.Div):
                return left / right if right else 0.0
            if isinstance(op, ast.FloorDiv):
                return left // right if right else 0
            if isinstance(op, ast.Mod):
                return left % right if right else 0
            raise ExpressionError(f"operator {type(op).__name__} is not allowed")

        if isinstance(node, ast.Compare):
            left = self._eval(node.left, ctx)
            # ast guarantees these are parallel, but strict= makes a malformed tree fail
            # loudly instead of silently dropping a comparison from a guard.
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._eval(comparator, ctx)
                if not self._compare(op, left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.IfExp):
            return (
                self._eval(node.body, ctx)
                if self._eval(node.test, ctx)
                else self._eval(node.orelse, ctx)
            )

        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(e, ctx) for e in node.elts]

        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)  # enforced at compile time
            name = node.func.id
            args = [self._eval(a, ctx) for a in node.args]
            kwargs = {k.arg: self._eval(k.value, ctx) for k in node.keywords if k.arg}
            if name in _CTX_FUNCTIONS:
                return _CTX_FUNCTIONS[name](ctx, *args, **kwargs)
            return _PURE_FUNCTIONS[name](*args, **kwargs)

        raise ExpressionError(f"cannot evaluate {type(node).__name__}")

    @staticmethod
    def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
        try:
            if isinstance(op, ast.Eq):
                return bool(left == right)
            if isinstance(op, ast.NotEq):
                return bool(left != right)
            if isinstance(op, ast.In):
                return left in right if right is not None else False
            if isinstance(op, ast.NotIn):
                return left not in right if right is not None else True
            # Ordered comparisons against a missing value are False rather than TypeError:
            # `vars.attempts < 3` should not explode before `attempts` is first set.
            if left is None or right is None:
                return False
            if isinstance(op, ast.Lt):
                return bool(left < right)
            if isinstance(op, ast.LtE):
                return bool(left <= right)
            if isinstance(op, ast.Gt):
                return bool(left > right)
            if isinstance(op, ast.GtE):
                return bool(left >= right)
        except TypeError:
            return False
        raise ExpressionError(f"comparison {type(op).__name__} is not allowed")
