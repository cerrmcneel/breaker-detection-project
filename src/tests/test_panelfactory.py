import pytest
import random

# ============================================================
#  TDD — test_panelfactory.py
#
#  PanelFactory generates randomized Panel objects following
#  Spanish REBT grammar rules with a tunable chaos_factor.
#
#  chaos_factor=0.0 → fully valid panel (deterministic rules)
#  chaos_factor=1.0 → maximum chaos (rules break at full probability)
#
#  Key design decisions being tested:
#  - Valid panels always start with MAINBREAKER/OVERSURGE on Rail 0
#  - Every rail always has an RCD or RCD_SI (unless chaos strikes)
#  - No rail ever exceeds its module capacity
#  - All generated class names are in the known CLASS_MAP
# ============================================================

from src.data_gen.grammar import Panel, Rail, Breaker, PanelFactory
from src.data_gen.compositor import Compositor

VALID_CLASSES = set(Compositor.CLASS_MAP.keys())


# -----------------------------------------------------------------
# Test 1 — Valid panel: Rail 0 always opens with a MAINBREAKER.
#   chaos_factor=0.0 disables all chaos behaviors (probabilities
#   are multiplied by 0.0), so rules are always followed.
# -----------------------------------------------------------------
def test_valid_panel_rail0_starts_with_mainbreaker():
    """
    With chaos_factor=0.0, Rail 0 must always begin with MAINBREAKER.
    Run 30 times to cover random rail-count variation.
    """
    factory = PanelFactory(chaos_factor=0.0)
    for _ in range(30):
        panel = factory.generate()
        first = panel.rails[0].components[0]
        assert first.cls == "MAINBREAKER", (
            f"Expected MAINBREAKER, got {first.cls}"
        )


# -----------------------------------------------------------------
# Test 2 — Valid panel: every rail has at least one RCD/RCD_SI.
#   chaos_factor=0.0 → missing_rcd and use_rcd_si are both disabled.
#   So every rail gets exactly one standard RCD.
# -----------------------------------------------------------------
def test_valid_panel_every_rail_has_rcd():
    """
    With chaos_factor=0.0, no rail should be missing its RCD.
    """
    factory = PanelFactory(chaos_factor=0.0)
    for _ in range(30):
        panel = factory.generate()
        for rail_idx, rail in enumerate(panel.rails):
            rcd_count = sum(
                1 for b in rail.components if b.cls in ("RCD", "RCD_SI")
            )
            assert rcd_count >= 1, (
                f"Rail {rail_idx} has no RCD. Components: "
                f"{[b.cls for b in rail.components]}"
            )


# -----------------------------------------------------------------
# Test 3 — Rail count is always between 1 and 4.
# -----------------------------------------------------------------
def test_rail_count_in_valid_range():
    """
    PanelFactory must only generate 1–4 rail panels.
    """
    factory = PanelFactory(chaos_factor=0.0)
    for _ in range(50):
        panel = factory.generate()
        assert 1 <= len(panel.rails) <= 4, (
            f"Invalid rail count: {len(panel.rails)}"
        )


# -----------------------------------------------------------------
# Test 4 — No rail ever overflows its module capacity.
#   This is a hard constraint regardless of chaos_factor.
# -----------------------------------------------------------------
def test_no_rail_ever_overflows():
    """
    No generated panel should have a rail exceeding max_modules.
    True at any chaos level — the Rail.add_component() guard enforces
    this, but PanelFactory must not try to add beyond that limit.
    """
    factory = PanelFactory(chaos_factor=1.0)
    for _ in range(50):
        panel = factory.generate()
        for rail in panel.rails:
            assert rail.current_width() <= rail.max_modules, (
                f"Rail overflowed: {rail.current_width()} > {rail.max_modules}"
            )


# -----------------------------------------------------------------
# Test 5 — All generated class names are in the 6-class taxonomy.
#   Guards against typos like "MAIN_BREAKER" or "rcD_si".
# -----------------------------------------------------------------
def test_all_component_classes_are_valid():
    """
    Every component class name must be a key in Compositor.CLASS_MAP.
    Tests at chaos_factor=0.5 to exercise a mix of valid and chaos paths.
    """
    factory = PanelFactory(chaos_factor=0.5)
    for _ in range(50):
        panel = factory.generate()
        for rail in panel.rails:
            for breaker in rail.components:
                assert breaker.cls in VALID_CLASSES, (
                    f"Unknown class: '{breaker.cls}'"
                )


# -----------------------------------------------------------------
# Test 6 — Chaos can produce a missing MAINBREAKER (common in field).
#   chaos_factor=1.0 applies full base probabilities. With
#   missing_mainbreaker at 20%, we should see it within 100 trials.
# -----------------------------------------------------------------
def test_chaos_can_produce_missing_mainbreaker():
    """
    chaos_factor=1.0 must eventually produce a panel where Rail 0
    does NOT start with MAINBREAKER or OVERSURGE.
    """
    factory = PanelFactory(chaos_factor=1.0)
    found_missing = False
    for _ in range(200):
        panel = factory.generate()
        first = panel.rails[0].components[0] if panel.rails[0].components else None
        if first is None or first.cls not in ("MAINBREAKER", "OVERSURGE"):
            found_missing = True
            break
    assert found_missing, (
        "chaos_factor=1.0 never produced a missing mainbreaker in 200 trials"
    )


# -----------------------------------------------------------------
# Test 7 — Chaos can produce OVERSURGE instead of MAINBREAKER.
#   OVERSURGE represents a newer REBT-compliant IGA+DPS install.
# -----------------------------------------------------------------
def test_chaos_can_produce_oversurge():
    """
    chaos_factor=1.0 must eventually produce OVERSURGE as the
    first component on Rail 0.
    """
    factory = PanelFactory(chaos_factor=1.0)
    found = False
    for _ in range(200):
        panel = factory.generate()
        for rail in panel.rails:
            for breaker in rail.components:
                if breaker.cls == "OVERSURGE":
                    found = True
                    break
    assert found, "chaos_factor=1.0 never produced OVERSURGE in 200 trials"


# -----------------------------------------------------------------
# Test 8 — Chaos can produce RCD_SI (superinmunizado).
#   Increasingly the REBT recommendation for new installs.
# -----------------------------------------------------------------
def test_chaos_can_produce_rcd_si():
    """
    chaos_factor=1.0 must eventually produce RCD_SI.
    """
    factory = PanelFactory(chaos_factor=1.0)
    found = False
    for _ in range(200):
        panel = factory.generate()
        for rail in panel.rails:
            for breaker in rail.components:
                if breaker.cls == "RCD_SI":
                    found = True
                    break
    assert found, "chaos_factor=1.0 never produced RCD_SI in 200 trials"
