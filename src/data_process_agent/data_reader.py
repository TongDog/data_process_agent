from __future__ import annotations

import json
import os


def get_directory_structure_json(target_dir: str, max_files: int = 20) -> str:
    """Return a compact directory snapshot for the architect agent."""
    structure_summary: list[dict[str, object]] = []
    for root, dirs, files in os.walk(target_dir):
        files = [f for f in files if not f.startswith(".")]
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        displayed_files = files[:max_files]
        remainder = len(files) - len(displayed_files)
        structure_summary.append(
            {
                "path": root,
                "name": os.path.basename(root),
                "files_sample": displayed_files,
                "file_count": len(files),
                "note": f"{remainder} more files omitted" if remainder > 0 else "all files listed",
            }
        )
    return json.dumps(structure_summary, ensure_ascii=False, indent=2)
