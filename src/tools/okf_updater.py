import pathlib

import yaml


def update_frontmatter(file_path, updates):
    """
    Reads an OKF markdown file, parses and updates its YAML frontmatter,
    and writes it back preserving the body content exactly.
    """
    path = pathlib.Path(file_path)
    content = path.read_text(encoding="utf-8")
    
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body_text = parts[2]
            meta = yaml.safe_load(fm_text) or {}
        else:
            meta = {}
            body_text = content
    else:
        meta = {}
        body_text = content

    # Apply updates recursively or simply override top-level keys
    for k, v in updates.items():
        if isinstance(v, dict) and k in meta and isinstance(meta[k], dict):
            meta[k].update(v)
        else:
            meta[k] = v

    # Dump YAML back (removing extra newlines from safe_dump)
    new_fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=False)
    
    # Reassemble file contents
    new_content = f"---\n{new_fm}---\n{body_text}"
    path.write_text(new_content, encoding="utf-8")
