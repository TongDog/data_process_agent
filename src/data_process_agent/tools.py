from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import tool


def get_structure_summary(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: get_structure_summary(value) for key, value in data.items()}
    if isinstance(data, list):
        return [get_structure_summary(data[0])] if data else []
    return type(data).__name__


@tool
def read_json_structure(file_path: str) -> Any:
    """Read one JSON file and return only its structural summary."""
    if not os.path.exists(file_path):
        return f"Error: file not found: {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive branch
        return f"Error: {exc}"
    return get_structure_summary(data)
