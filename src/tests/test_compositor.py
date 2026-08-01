import pytest

from src.data_gen.compositor import Compositor

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
#
#  NOTE: calculate_layout centres the panel on the canvas.
#    Horizontal: cursor_x = (img_width - total_rail_width) // 2
#    Vertical:   start_y  = (img_height - total_height) // 2
#  All expected coordinates below account for this centering.
# ============================================================
from src.data_gen.grammar import Breaker, Panel, Rail


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
#   1 rail × 200px = 200px total height → start_y = (640-200)//2 = 220
#   1 MCB × 40px  = 40px rail width     → cursor_x = (640-40)//2 = 300
# -----------------------------------------------------------------
def test_single_mcb_single_rail():
    """
    One MCB on Rail 0 should produce exactly one annotation,
    centred on the 640×640 canvas.
    """
    panel = Panel(rails_count=1, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert len(layout) == 1
    ann = layout[0]
    assert ann["class_id"] == Compositor.CLASS_MAP["MCB"]
    assert ann["x"] == 300   # (640 - 40) // 2
    assert ann["y"] == 220   # (640 - 200) // 2
    assert ann["w"] == 40
    assert ann["h"] == 200


# -----------------------------------------------------------------
# Test 2 — Two MCBs on the same rail: verify x-cursor advances.
#   Total rail width = 2*40 = 80  → cursor_x starts at (640-80)//2 = 280
#   MCB 1: x=280,  w=40
#   MCB 2: x=320,  w=40  (cursor moved right by 40px)
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
    assert layout[0]["x"] == 280  # (640 - 80) // 2
    assert layout[1]["x"] == 320  # 280 + 40


# -----------------------------------------------------------------
# Test 3 — MainBreaker (width=4) followed by an MCB.
#   Total rail width = 4*40 + 1*40 = 200  → cursor_x starts at (640-200)//2 = 220
#   MainBreaker: x=220,  w=160
#   MCB:         x=380,  w=40
# -----------------------------------------------------------------
def test_mainbreaker_then_mcb_x_positions():
    """
    A MainBreaker occupies 4 module slots; the next component
    must start at x = start + 160 (4 * 40).
    """
    panel = Panel(rails_count=1, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MAINBREAKER', width=4))
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert len(layout) == 2
    assert layout[0]["x"] == 220   # (640 - 200) // 2
    assert layout[0]["w"] == 160   # 4 * 40
    assert layout[1]["x"] == 380   # 220 + 160
    assert layout[1]["w"] == 40


# -----------------------------------------------------------------
# Test 4 — Two rails: verify y-offset and x-cursor reset.
#   2 rails × 200px = 400px total height → start_y = (640-400)//2 = 120
#   Rail 0 → y = 120
#   Rail 1 → y = 120 + 200 = 320
#   Each rail has 1 MCB (40px) → cursor_x = (640-40)//2 = 300
# -----------------------------------------------------------------
def test_two_rails_y_offsets():
    """
    Each rail starts at a different y. The x cursor must reset
    for each new rail.
    """
    panel = Panel(rails_count=2, rail_height=200)
    panel.rails[0].add_component(Breaker(cls='MCB', width=1))
    panel.rails[1].add_component(Breaker(cls='MCB', width=1))

    compositor = make_compositor()
    layout = compositor.calculate_layout(panel)

    assert len(layout) == 2
    assert layout[0]["y"] == 120    # (640 - 400) // 2
    assert layout[1]["y"] == 320    # 120 + 200
    assert layout[0]["x"] == 300    # each rail centred independently
    assert layout[1]["x"] == 300


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

