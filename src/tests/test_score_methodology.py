"""
Tests for score methodology public documentation synchronization.

Ensures that app/frontend/metodologia/index.html is always kept in sync with
the actual scoring algorithm implemented in app.main.grade_panel_layout.
When the scoring logic changes, the SHA-256 hash of grade_panel_layout changes,
causing this test to fail until the methodology page and its version stamp
are updated.
"""

import hashlib
import inspect
import re
from pathlib import Path

from app.main import grade_panel_layout


def test_score_methodology_hash_matches_code():
    # 1. Compute expected hash prefix from grade_panel_layout source
    source = inspect.getsource(grade_panel_layout)
    full_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    expected_stamp = full_sha[:12]

    # 2. Locate and load app/frontend/metodologia/index.html
    repo_root = Path(__file__).resolve().parent.parent.parent
    html_path = repo_root / "app" / "frontend" / "metodologia" / "index.html"
    assert html_path.exists(), f"Methodology page not found at {html_path}"

    html_content = html_path.read_text(encoding="utf-8")

    # 3. Locate <code id="score-version">...</code> element
    tag_match = re.search(
        r'<code[^>]*id=["\']score-version["\'][^>]*>(.*?)</code>',
        html_content,
        re.DOTALL | re.IGNORECASE,
    )
    assert tag_match is not None, (
        "Element <code id='score-version'> was not found in app/frontend/metodologia/index.html"
    )

    badge_text = tag_match.group(1).strip()

    # 4. Extract 12-char hex version string
    hash_match = re.search(r"\b([0-9a-f]{12})\b", badge_text)
    assert hash_match is not None, (
        f"Could not extract a 12-character hex hash from <code id='score-version'>: {badge_text!r}"
    )

    extracted_stamp = hash_match.group(1)

    # 5. Assert hash equality with clear diagnostic message
    assert extracted_stamp == expected_stamp, (
        f"Score methodology version mismatch!\n"
        f"The source code of app.main.grade_panel_layout has SHA-256 prefix '{expected_stamp}',\n"
        f"but app/frontend/metodologia/index.html is stamped with '{extracted_stamp}'.\n"
        f"If the scoring rules or logic were updated, you must update the documentation in\n"
        f"app/frontend/metodologia/index.html and update the stamp in <code id=\"score-version\">."
    )


def test_score_methodology_independence_and_disclaimer():
    repo_root = Path(__file__).resolve().parent.parent.parent
    html_path = repo_root / "app" / "frontend" / "metodologia" / "index.html"
    assert html_path.exists(), f"Methodology page not found at {html_path}"

    html_content = html_path.read_text(encoding="utf-8")

    # Verify conflict-of-interest statement
    assert "panelsafe.cv" in html_content
    assert "electricista" in html_content.lower()

    # Verify non-boletín / non-inspection disclaimer
    assert "boletín" in html_content.lower() or "boletin" in html_content.lower()
    assert "instalador autorizado" in html_content.lower() or "instalador electricista" in html_content.lower()
