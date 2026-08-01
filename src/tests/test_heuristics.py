"""
Tests for SpatialHeuristicEngine.apply_logic's spatial reasoning: fragmented-box
merging (step 2) and grid extrapolation (step 3).

All tests pass use_hmm=False deliberately. The HMM/Viterbi stage runs AFTER these
steps and can rewrite pred["class"], which would mask what these tests assert.
The HMM itself is covered separately in test_hmm.py.
"""
import pytest

from src.model.heuristics import SpatialHeuristicEngine, compute_iou


@pytest.fixture
def engine():
    return SpatialHeuristicEngine()


def mcb(x1, x2, y1=50, y2=200, conf=0.90, cls="MCB"):
    """Build a prediction dict. Default y-range keeps everything on one rail."""
    return {"box": [x1, y1, x2, y2], "class": cls, "conf": conf}


def synthetic(preds):
    return [p for p in preds if p.get("heuristic_correction") == "GRID_EXTRAPOLATION"]


# --- compute_iou (the merge predicate) ---

def test_iou_identical_boxes_is_one():
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert compute_iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0


def test_iou_zero_area_boxes_does_not_divide_by_zero():
    # Degenerate boxes are reachable from a collapsed YOLO detection; the guard in
    # compute_iou must return 0 rather than raising ZeroDivisionError.
    assert compute_iou([5, 5, 5, 5], [5, 5, 5, 5]) == 0


# --- Step 2: fragmented box merging ---

def test_merge_combines_overlapping_boxes_of_same_class(engine):
    # IoU here is ~0.78, comfortably over the 0.4 merge threshold.
    result = engine.apply_logic(
        [mcb(100, 180, conf=0.80), mcb(110, 190, conf=0.91)],
        use_hmm=False,
    )

    assert len(result) == 1
    merged = result[0]
    # Merged box is the union of both inputs.
    assert merged["box"] == [100, 50, 190, 200]
    # Confidence takes the max of the fragments, not the first or the last.
    assert merged["conf"] == 0.91
    assert merged["heuristic_applied"] is True
    assert merged["heuristic_correction"] == "MERGE_FRAGMENTED"


def test_merge_does_not_combine_different_classes(engine):
    # Same geometry as the merge test (high IoU), but the class differs, so the
    # merge must not fire -- an RCD overlapping an MCB is a real misdetection to
    # surface, not two fragments of one device.
    result = engine.apply_logic(
        [mcb(100, 180), mcb(110, 190, cls="RCD")],
        use_hmm=False,
    )

    assert len(result) == 2
    assert {p["class"] for p in result} == {"MCB", "RCD"}
    assert all(p.get("heuristic_correction") != "MERGE_FRAGMENTED" for p in result)


def test_merge_does_not_combine_low_overlap_boxes(engine):
    # Adjacent breakers on a rail touch slightly but are distinct devices;
    # IoU here is ~0.03, well under the threshold.
    result = engine.apply_logic(
        [mcb(100, 180), mcb(175, 255)],
        use_hmm=False,
    )

    assert len(result) == 2
    assert all(p.get("heuristic_correction") != "MERGE_FRAGMENTED" for p in result)


# --- Step 3: grid extrapolation ---

def test_extrapolation_fills_a_single_module_gap(engine):
    # Two 40px MCBs -> median_width 40, so the gap threshold is 44px.
    # The 60px gap between them fits exactly one synthetic module.
    result = engine.apply_logic([mcb(100, 140), mcb(200, 240)], use_hmm=False)

    assert len(result) == 3
    filled = synthetic(result)
    assert len(filled) == 1

    ghost = filled[0]
    assert ghost["class"] == "MCB"
    # Low confidence is the signal that this box was inferred, not detected.
    assert ghost["conf"] == 0.50
    assert ghost["heuristic_applied"] is True
    # Sits in the gap, inheriting the rail's vertical extent.
    assert ghost["box"] == [142, 50, 182, 200]

    # Real detections must not be marked as heuristic output.
    real = [p for p in result if p.get("heuristic_correction") is None]
    assert len(real) == 2


def test_extrapolation_does_not_fill_a_gap_below_threshold(engine):
    # 40px modules -> 44px threshold; a 20px gap is normal rail spacing.
    result = engine.apply_logic([mcb(100, 140), mcb(160, 200)], use_hmm=False)

    assert len(result) == 2
    assert synthetic(result) == []


def test_extrapolation_fills_a_wide_gap_repeatedly(engine):
    # A 160px gap between 40px modules fits multiple missing breakers. The loop
    # re-examines each inserted box against the next real one, so it keeps
    # filling left-to-right until the remaining gap drops under threshold.
    result = engine.apply_logic([mcb(100, 140), mcb(300, 340)], use_hmm=False)

    filled = synthetic(result)
    assert len(filled) == 3
    assert len(result) == 5

    # Boxes come back in left-to-right order with no overlaps.
    xs = [p["box"][0] for p in result]
    assert xs == sorted(xs)
    for left, right in zip(result, result[1:]):
        assert left["box"][2] <= right["box"][0]


def test_extrapolation_clamps_synthetic_box_to_avoid_overlap(engine):
    # Tuned so the gap clears the threshold (17.9 > 17.6) but is narrower than
    # the synthetic module would be (2 + 16 = 18), forcing the clamp branch.
    result = engine.apply_logic(
        [mcb(100, 116), mcb(133.9, 150.0)],
        use_hmm=False,
    )

    assert len(result) == 3
    ghost = synthetic(result)[0]

    # Unclamped the right edge would have landed at 134, past the next device.
    assert ghost["box"][2] == pytest.approx(131.9)
    assert ghost["box"][2] <= result[2]["box"][0]


def test_extrapolation_is_scoped_per_rail(engine):
    # Two rails, each with its own gap. Rows are processed independently, so each
    # gets exactly one synthetic box carrying its own rail's y-coordinates.
    result = engine.apply_logic(
        [
            mcb(100, 140, y1=50, y2=200),
            mcb(200, 240, y1=50, y2=200),
            mcb(100, 140, y1=500, y2=650),
            mcb(200, 240, y1=500, y2=650),
        ],
        use_hmm=False,
    )

    filled = synthetic(result)
    assert len(filled) == 2

    # A synthetic box must never inherit the wrong rail's vertical extent.
    tops = sorted(g["box"][1] for g in filled)
    assert tops == [50, 500]
    assert sorted(g["box"][3] for g in filled) == [200, 650]


# --- edge cases ---

def test_apply_logic_returns_empty_for_no_predictions(engine):
    assert engine.apply_logic([], use_hmm=False) == []


def test_apply_logic_handles_single_prediction(engine):
    result = engine.apply_logic([mcb(100, 140)], use_hmm=False)

    assert len(result) == 1
    assert result[0].get("heuristic_correction") is None
