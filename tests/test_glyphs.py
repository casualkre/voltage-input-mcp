"""Glyph-matched HUD digits: is it exact, and does it learn its own templates?

The bar is not "better than tesseract on average". A guard reading 4114 instead of 414
fires the wrong reflex, so the requirement is exactness on the fixed-font case, and a
refusal rather than a guess everywhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from voltage_input_mcp.capture.glyphs import (
    GlyphLearner,
    GlyphSet,
    binarise,
    segment_glyphs,
)

pytest.importorskip("PIL")


def render(text: str, *, fg=(60, 230, 120), bg=(225, 232, 238), size=30, pad=6):
    """A HUD-ish number: coloured text on a light, slightly noisy panel."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default(size=size)
    probe = Image.new("RGB", (10, 10))
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (box[2] - box[0] + pad * 2, box[3] - box[1] + pad * 2), bg)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, font=font, fill=fg)
    arr = np.asarray(img, dtype=np.int16)
    rng = np.random.default_rng(len(text))
    arr = np.clip(arr + rng.integers(-4, 5, arr.shape), 0, 255)
    return arr.astype(np.uint8)


def learn(samples: list[str], **kw) -> GlyphSet:
    """Teach a glyph set from rendered samples, with a perfect teacher."""
    learner = GlyphLearner(**kw)
    for text in samples:
        learner.observe(render(text), teacher_text=text)
    result = learner.result()
    assert result is not None, f"learning failed: {learner.report()}"
    return result


DIGITS = ["0123456789", "9876543210", "4150", "72", "308", "641", "999", "100"]


# -- segmentation ------------------------------------------------------------------------


def test_it_finds_one_component_per_digit():
    glyphs = segment_glyphs(binarise(render("41508")))
    assert len(glyphs) == 5, f"segmented into {len(glyphs)}"


def test_it_separates_ink_from_a_light_background():
    mask = binarise(render("123"))
    # Text is a minority of a HUD patch; a mask claiming most of it has found background.
    assert 0.02 < mask.mean() < 0.55


def test_it_handles_light_text_on_a_dark_panel_too():
    """Both polarities without the caller having to declare which one this HUD is."""
    glyphs = segment_glyphs(binarise(render("507", fg=(255, 255, 255), bg=(18, 20, 26))))
    assert len(glyphs) == 3


def test_an_empty_region_segments_to_nothing_rather_than_noise():
    blank = np.full((40, 120, 3), 210, dtype=np.uint8)
    assert segment_glyphs(binarise(blank)) == []


# -- reading -----------------------------------------------------------------------------


def test_it_reads_the_numbers_tesseract_got_wrong():
    """The two real misreads from the live HUD: 414 -> 4114 and 0 -> 636."""
    glyphs = learn(DIGITS)
    for text, expected in (("414", 414.0), ("0", 0.0)):
        value, confidence = glyphs.read(render(text))
        assert value == expected, f"read {text!r} as {value} (conf {confidence:.2f})"


@pytest.mark.parametrize(
    "text", ["0", "7", "42", "414", "1031132", "999999", "10", "205", "3000"]
)
def test_it_reads_a_range_of_values_exactly(text):
    value, confidence = learn(DIGITS).read(render(text))
    assert value == float(text), f"read {text!r} as {value}"
    assert confidence >= 0.72


def test_confidence_is_the_worst_glyph_not_the_average():
    """One bad digit ruins the number -- 414 read as 814 is wrong, not 2/3 right.

    Checked against the per-glyph scores from the *same* read. Comparing against separate
    renders would be comparing two different images and measuring rendering noise.
    """
    glyphs = learn(DIGITS)
    patch = render("1031132")
    scored = glyphs.read_glyphs(patch)
    _, confidence = glyphs.read(patch)
    assert len(scored) == 7
    assert confidence == pytest.approx(min(s for _, s in scored))
    mean = sum(s for _, s in scored) / len(scored)
    assert confidence <= mean


def test_an_unknown_glyph_is_refused_rather_than_guessed():
    """A sensor that invents a number is worse than one that admits it cannot see."""
    partial = learn(["01", "10", "0", "1"])
    value, _ = partial.read(render("8"))
    assert value is None


def test_a_blank_region_reads_as_nothing():
    glyphs = learn(DIGITS)
    blank = np.full((40, 120, 3), 210, dtype=np.uint8)
    assert glyphs.read(blank)[0] is None


# -- learning ----------------------------------------------------------------------------


def test_it_learns_every_digit_from_a_counter_passing_through():
    glyphs = learn(DIGITS)
    assert set(glyphs.templates) >= set("0123456789")


def test_a_misreading_teacher_cannot_decide_a_label_on_its_own():
    """The teacher is unreliable by assumption -- that is why it is being replaced.

    Here it is wrong a third of the time, in the way tesseract actually is: extra digits.
    Positional votes across frames have to outvote the noise.
    """
    learner = GlyphLearner()
    for i, text in enumerate(DIGITS * 4):
        lie = text + "1" if i % 3 == 0 else text
        learner.observe(render(text), teacher_text=lie)
    glyphs = learner.result()
    assert glyphs is not None
    value, _ = glyphs.read(render("414"))
    assert value == 414.0


def test_frames_where_the_teacher_disagrees_on_length_are_skipped():
    """Positional alignment is meaningless when the counts differ, so do not vote."""
    learner = GlyphLearner()
    for text in DIGITS:
        learner.observe(render(text), teacher_text="12345678901234")
    assert learner.report()["frames_with_teacher"] == 0
    assert learner.result() is None


def test_learning_needs_more_than_one_confirmation_per_glyph():
    learner = GlyphLearner()
    learner.observe(render("07"), teacher_text="07")
    assert learner.result() is None, "labelled a glyph off a single frame"


def test_no_teacher_at_all_yields_nothing_rather_than_wrong_labels():
    learner = GlyphLearner()
    for text in DIGITS:
        learner.observe(render(text), teacher_text=None)
    assert learner.result() is None
    assert learner.report()["clusters"] > 0  # shapes were still collected


# -- persistence ---------------------------------------------------------------------------


def test_a_learned_set_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    learn(DIGITS).save("meters")
    reloaded = GlyphSet.load("meters")
    assert reloaded is not None
    assert reloaded.read(render("414"))[0] == 414.0


def test_a_missing_or_corrupt_set_loads_as_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "voltage_input_mcp.config.state_dir", lambda: tmp_path, raising=True
    )
    assert GlyphSet.load("nope") is None
    from voltage_input_mcp.capture.glyphs import glyphs_path

    path = glyphs_path("bad")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert GlyphSet.load("bad") is None


# -- the point of the exercise -------------------------------------------------------------


def test_reading_is_orders_of_magnitude_cheaper_than_ocr():
    """OCR costs 80-200 ms and forced a background worker and a staleness window.

    If this is not dramatically cheaper there is no reason to prefer it, so the threshold
    is deliberately strict rather than a token assertion.
    """
    import time

    glyphs = learn(DIGITS)
    patch = render("1031132")
    glyphs.read(patch)  # warm

    t0 = time.perf_counter()
    for _ in range(100):
        glyphs.read(patch)
    per_read_ms = (time.perf_counter() - t0) / 100 * 1000
    assert per_read_ms < 5.0, f"{per_read_ms:.2f} ms per read is not fast enough to matter"
