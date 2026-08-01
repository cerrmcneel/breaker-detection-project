# src/tests/test_hmm.py
import pytest

from src.model.hmm_corrector import HMMCorrector


def test_hmm_corrector_initialization():
    corrector = HMMCorrector()
    assert len(corrector.classes) == 6
    assert "MCB" in corrector.transition_matrix
    assert "main" in corrector.initial_probs

def test_viterbi_corrects_typical_yolo_errors():
    corrector = HMMCorrector()
    
    # Mock sequence: MCB -> RCD -> RCD (conf 0.7, width 1, OCR C16) -> MCB -> MCB
    # The 3rd component is a visual error; physically it is an MCB.
    sequence = [
        {"class": "MAINBREAKER", "conf": 0.90, "width": 2, "ocr_text": ""},
        {"class": "RCD", "conf": 0.85, "width": 2, "ocr_text": ""},
        {"class": "RCD", "conf": 0.70, "width": 1, "ocr_text": "C16"},
        {"class": "MCB", "conf": 0.95, "width": 1, "ocr_text": ""},
        {"class": "MCB", "conf": 0.95, "width": 1, "ocr_text": ""}
    ]
    
    corrected = corrector.decode_rail(sequence, rail_type="main")
    
    # Assert third item was corrected to MCB
    assert corrected[0] == "MAINBREAKER"
    assert corrected[1] == "RCD"
    assert corrected[2] == "MCB"
    assert corrected[3] == "MCB"
    assert corrected[4] == "MCB"

def test_viterbi_corrects_leftmost_mainbreaker():
    corrector = HMMCorrector()
    
    # Scenario: YOLO detected a MAINBREAKER at low confidence (0.55) on the
    # leftmost position of the main rail. Even at low conf, position prior +
    # width (2 modules) should confirm MAINBREAKER over MCB.
    sequence = [
        {"class": "MAINBREAKER", "conf": 0.55, "width": 2, "ocr_text": ""},
        {"class": "RCD",         "conf": 0.90, "width": 2, "ocr_text": ""},
        {"class": "MCB",         "conf": 0.95, "width": 2, "ocr_text": "C16"}
    ]

    corrected = corrector.decode_rail(sequence, rail_type="main")
    assert corrected[0] == "MAINBREAKER", (
        "Low-conf MAINBREAKER at rail-head position should be confirmed, not overridden"
    )
    assert corrected[1] == "RCD"
    assert corrected[2] == "MCB"

def test_hmm_bidirectional_reversibility():
    corrector = HMMCorrector()
    
    # Left-to-right mock sequence:
    seq_l2r = [
        {"class": "MAINBREAKER", "conf": 0.90, "width": 2, "ocr_text": ""},
        {"class": "RCD", "conf": 0.85, "width": 2, "ocr_text": ""},
        {"class": "MCB", "conf": 0.95, "width": 1, "ocr_text": ""}
    ]
    
    # Right-to-left (reversed wiring) mock sequence:
    seq_r2l = list(reversed(seq_l2r))
    
    decoded_l2r = corrector.decode_rail(seq_l2r, rail_type="main")
    decoded_r2l = corrector.decode_rail(seq_r2l, rail_type="main")
    
    # The right-to-left decoding should be the exact reverse of the left-to-right decoding
    assert decoded_l2r == ["MAINBREAKER", "RCD", "MCB"]
    assert decoded_r2l == ["MCB", "RCD", "MAINBREAKER"]
