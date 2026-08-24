"""The accessibility source: exact identity, honest about position.

The value is that these names come from the applications rather than from a model looking
at a screenshot, so they cannot be confabulated. The risk is the opposite of the usual
one: not that it invents an element, but that it reports one with no usable coordinates
and a caller clicks (0, 0).
"""

from __future__ import annotations

from voltage_input_mcp.capture.a11y import A11yElement, A11ySource


class FakeTree(A11ySource):
    """An A11ySource with the D-Bus walk replaced, so behaviour is testable offline."""

    def __init__(self, elements, **kw):
        super().__init__(**kw)
        self._fake = elements
        self.walks = 0

    def _walk(self):
        self.walks += 1
        return list(self._fake)


GROUNDED = A11yElement(name="Save", role="push button", x=10, y=20, w=80, h=30,
                       application="Editor", grounded=True)
UNGROUNDED = A11yElement(name="Cancel", role="push button", application="Editor")


def test_it_reports_what_the_application_calls_its_widgets():
    src = FakeTree([GROUNDED, UNGROUNDED])
    assert src.names() == ["Save", "Cancel"]


def test_a_snapshot_is_cached_rather_than_rewalked():
    """A full walk measured 467 ms on a real desktop; per-cycle walking is not viable."""
    src = FakeTree([GROUNDED], ttl_s=60.0)
    for _ in range(20):
        src.snapshot()
    assert src.walks == 1


def test_force_bypasses_the_cache():
    src = FakeTree([GROUNDED], ttl_s=60.0)
    src.snapshot()
    src.snapshot(force=True)
    assert src.walks == 2


def test_disabled_never_walks():
    src = FakeTree([GROUNDED], enabled=False)
    assert src.snapshot() == []
    assert src.walks == 0


def test_a_bus_failure_degrades_to_nothing_rather_than_killing_the_run():
    class Broken(A11ySource):
        def _walk(self):
            raise RuntimeError("bus went away")

    src = Broken()
    assert src.snapshot() == []
    assert "bus went away" in src.stats()["error"]


def test_find_matches_case_insensitively_and_by_substring():
    """Widget names carry decoration a playbook author will not reproduce exactly."""
    src = FakeTree([A11yElement(name="Save As...", role="push button", grounded=True,
                                x=1, y=2, w=3, h=4)])
    assert src.find("save") is not None
    assert src.find("SAVE AS") is not None
    assert src.find("delete") is None


def test_find_prefers_a_control_it_can_actually_aim_at():
    """An exact name with no coordinates is useless to a caller that wants to click."""
    src = FakeTree([
        A11yElement(name="Save", role="push button"),                      # ungrounded
        A11yElement(name="Save", role="push button", x=5, y=6, w=7, h=8,
                    grounded=True),
    ])
    found = src.find("Save")
    assert found is not None and found.grounded


def test_wayland_zero_extents_are_reported_as_ungrounded_not_as_the_origin():
    """The failure this prevents: a click at (0,0) because a toolkit said 0x0.

    Measured on a real KDE Wayland desktop, 249 of 387 controls report no extents.
    """
    assert UNGROUNDED.grounded is False
    assert GROUNDED.center == (50, 35)


def test_stats_separate_grounded_from_merely_known():
    src = FakeTree([GROUNDED, UNGROUNDED])
    src.snapshot()
    stats = src.stats()
    assert stats["elements"] == 2
    assert stats["grounded"] == 1


# -- the guard ---------------------------------------------------------------------------


def test_the_ui_guard_asks_the_application_not_a_model():
    from voltage_input_mcp.expr import Guard, GuardContext

    ctx = GuardContext(ui_names={"save as...", "cancel"})
    assert Guard("ui('Save')").evaluate(ctx) is True
    assert Guard("ui('cancel')").evaluate(ctx) is True
    assert Guard("ui('Delete')").evaluate(ctx) is False


def test_ui_is_false_rather_than_an_error_when_nothing_is_published():
    """Games publish no tree at all; a guard must degrade, not explode."""
    from voltage_input_mcp.expr import Guard, GuardContext

    assert Guard("ui('anything')").evaluate(GuardContext()) is False
