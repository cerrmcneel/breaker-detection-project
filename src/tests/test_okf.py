import os
import pathlib
import tempfile

import pytest
import yaml

from src.tools.okf_updater import update_frontmatter
from src.tools.validate_okf import validate_bundle, validate_concept


def test_okf_updater_preserves_body_and_updates_frontmatter():
    # Create a temporary file representing a mock OKF concept
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
        file_path = f.name
        f.write("---\ntype: Test Model\ntitle: Old Title\nmetrics:\n  acc: 0.85\n---\n\n# Header\n\nSome body text here.\n- List item\n")

    try:
        # Perform update
        updates = {
            "title": "New Title",
            "metrics": {"acc": 0.95, "loss": 0.05},
            "timestamp": "2026-06-22T16:50:00Z"
        }
        update_frontmatter(file_path, updates)

        # Read back content
        content = pathlib.Path(file_path).read_text(encoding="utf-8")
        
        # Parse YAML and verify
        parts = content.split("---", 2)
        meta = yaml.safe_load(parts[1])
        
        assert meta["type"] == "Test Model"
        assert meta["title"] == "New Title"
        assert meta["metrics"]["acc"] == 0.95
        assert meta["metrics"]["loss"] == 0.05
        assert meta["timestamp"] == "2026-06-22T16:50:00Z"
        
        # Verify body section remains untouched
        assert "# Header" in content
        assert "Some body text here." in content
        assert "- List item" in content
    finally:
        os.remove(file_path)

def test_validate_concept_detects_missing_type_and_broken_links():
    # Test valid concept
    valid_content = "---\ntype: Standard\ntitle: Safe\n---\n[index](/index.md)\n"
    # Root dir of docs mock
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        # Create target link target
        (temp_path / "index.md").write_text("# Index", encoding="utf-8")
        
        # Validate
        errors = validate_concept(valid_content, "/mock/path.md", temp_path)
        assert len(errors) == 0

        # Test missing type frontmatter
        missing_type_content = "---\ntitle: Safe\n---\n[index](/index.md)\n"
        errors = validate_concept(missing_type_content, "/mock/path.md", temp_path)
        assert any("missing required field 'type'" in e for e in errors)

        # Test broken link
        broken_link_content = "---\ntype: Standard\n---\n[missing](/does_not_exist.md)\n"
        errors = validate_concept(broken_link_content, "/mock/path.md", temp_path)
        assert any("Broken link to '/does_not_exist.md'" in e for e in errors)
