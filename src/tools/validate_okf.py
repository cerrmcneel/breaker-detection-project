import pathlib
import re
import sys

import yaml


def validate_concept(content, file_path, root_dir):
    """
    Validates a single OKF concept string content.
    Returns a list of error/warning strings.
    """
    errors = []
    
    # 1. Check YAML Frontmatter block
    if not content.startswith("---"):
        errors.append(f"[{file_path}] Concept does not start with a YAML frontmatter block.")
        return errors

    parts = content.split("---", 2)
    if len(parts) < 3:
        errors.append(f"[{file_path}] Concept has incomplete YAML frontmatter block.")
        return errors

    fm_text = parts[1]
    body_text = parts[2]

    try:
        meta = yaml.safe_load(fm_text) or {}
    except Exception as e:
        errors.append(f"[{file_path}] Failed to parse YAML frontmatter: {e}")
        return errors

    # 2. Check required 'type' field
    if "type" not in meta:
        errors.append(f"[{file_path}] missing required field 'type' in frontmatter.")

    # 3. Find and check cross-links
    # OKF links are usually absolute-style links starting with '/' like [orders](/tables/orders.md)
    # We match any link inside parenthesis that ends with .md, optionally including anchor links
    links = re.findall(r"\]\((/[^)]+\.md)(?:#[^)]*)?\)", body_text)
    
    for link in links:
        # Resolve target path relative to root_dir
        # Strip leading slash to make path relative
        rel_link_path = link.lstrip("/")
        target_file = pathlib.Path(root_dir) / rel_link_path
        
        if not target_file.exists():
            errors.append(f"[{file_path}] Broken link to '{link}'. Target file '{target_file.name}' not found.")
            
    return errors

def validate_bundle(root_dir):
    """
    Scans and validates all OKF markdown files inside root_dir.
    Prints a report and returns True if bundle is valid, False if warnings are found.
    """
    root_path = pathlib.Path(root_dir)
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: directory '{root_dir}' does not exist.")
        return False

    all_errors = []
    file_count = 0

    for path in root_path.rglob("*.md"):
        file_count += 1
        try:
            content = path.read_text(encoding="utf-8")
            # Generate a relative path string for display
            rel_path = str(path.relative_to(root_path))
            errors = validate_concept(content, rel_path, root_path)
            all_errors.extend(errors)
        except Exception as e:
            all_errors.append(f"[{path}] Error reading file: {e}")

    print(f"\n--- OKF Validation Report ({file_count} files scanned) ---")
    if all_errors:
        print(f"Warnings found ({len(all_errors)} issues):")
        for err in all_errors:
            print(f"  [WARN] {err}")
        print("----------------------------------------------------\n")
        return False
    else:
        print("  [SUCCESS] All files are OKF v0.1 compliant! Link graph is healthy.")
        print("----------------------------------------------------\n")
        return True

if __name__ == "__main__":
    # If run as CLI, default to standard docs directory in the repository
    default_root = pathlib.Path(__file__).parent.parent.parent / "docs" / "okf"
    target_dir = sys.argv[1] if len(sys.argv) > 1 else str(default_root)
    validate_bundle(target_dir)
