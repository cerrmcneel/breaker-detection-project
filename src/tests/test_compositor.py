import pytest

# ============================================================
#  TDD — test_compositor.py
#
#  The Compositor has two responsibilities:
#    1. calculate_layout(panel) — PURE function: converts a Panel
#       object into a flat list of pixel-coordinate annotations.
#       No images, no cv2. Fully testable.
#    2. compose(panel) — pastes seed images onto a canvas.
#       Requires cv2/numpy; tested separately when cv2 is available.
#
#  These tests cover calculate_layout only.
# ============================================================

from src.data_gen.grammar import Panel, Rail, Breaker
from src.data_gen.compositor import Compositor


# Helper — build a Compositor with predictable dimensions
# module_width_px=40  → 1 MCB  = 40px wide
# rail_height=200      → 1 rail = 200px tall
def make_compositor():
    return Compositor(
        seed_library=None,  # not needed for calculate_layout
        img_width=640,
        img_height=640,
        module_width_px=40,
    )


# -----------------------------------------------------------------
# Test 1 — Single MCB on a single rail.
#   MCB width=1 → pixel width = 1 * 40 = 40
#   Rail 0 → y = 0, h = rail_height = 200
#   No previous breakers → x = 0
# -----------------------------------------------------------------
def test_single_mcb_single_rail():
    """
    One MCB on Rail 0 should produce exactly one annotation
    at x=0, y=0, w=40, h=200.
    """
    panel = Panel(rails_count=1, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert len(layout) == 1
    ann = layout[0]
    assert ann["class_id"] == Compositor.CLASS_MAP["MCB"]
    assert ann["x"] == 0
    assert ann["y"] == 0
    assert ann["w"] == 40
    assert ann["h"] == 200


# -----------------------------------------------------------------
# Test 2 — Two MCBs on the same rail: verify x-cursor advances.
#   MCB 1: x=0,  w=40
#   MCB 2: x=40, w=40  (cursor moved right by 40px)
# -----------------------------------------------------------------
def test_two_mcbs_x_cursor_advances():
    """
    A second MCB must start where the first one ended.
    """
    panel = Panel(rails_count=1, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert len(layout) == 2
    assert layout[0]["x"] == 0
    assert layout[1]["x"] == 40  # cursor advanced by 1 * module_width_px


# -----------------------------------------------------------------
# Test 3 — MainBreaker (width=4) followed by an MCB.
#   MainBreaker: x=0,   w=4*40=160
#   MCB:         x=160, w=40
# -----------------------------------------------------------------
def test_mainbreaker_then_mcb_x_positions():
    """
    A MainBreaker occupies 4 module slots; the next component
    must start at x=160 (4 * 40).
    """
    panel = Panel(rails_count=1, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MAINBREAKER', width=4))
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert len(layout) == 2
    assert layout[0]["x"] == 0
    assert layout[0]["w"] == 160  # 4 * 40
    assert layout[1]["x"] == 160
    assert layout[1]["w"] == 40


# -----------------------------------------------------------------
# Test 4 — Two rails: verify y-offset resets per rail.
#   Rail 0 → y = 0
#   Rail 1 → y = rail_height = 200
#   x-cursor must reset to 0 for each new rail.
# -----------------------------------------------------------------
def test_two_rails_y_offsets():
    """
    Each rail starts at a different y. The x cursor must reset
    to 0 at the start of each rail.
    """
    panel = Panel(rails_count=2, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))
    panel.rails[1].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert len(layout) == 2
    assert layout[0]["y"] == 0    # Rail 0
    assert layout[1]["y"] == 200  # Rail 1
    assert layout[0]["x"] == 0   # x resets for each rail
    assert layout[1]["x"] == 0


# -----------------------------------------------------------------
# Test 5 — class_id mapping is correct for both classes.
# -----------------------------------------------------------------
def test_class_id_mapping():
    """
    MAINBREAKER and MCB must map to different integer class_ids,
    and those IDs must match Compositor.CLASS_MAP.
    """
    panel = Panel(rails_count=1, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MAINBREAKER', width=4))
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert layout[0]["class_id"] == Compositor.CLASS_MAP["MAINBREAKER"]
    assert layout[1]["class_id"] == Compositor.CLASS_MAP["MCB"]
    assert layout[0]["class_id"] != layout[1]["class_id"]
