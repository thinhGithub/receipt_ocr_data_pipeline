"""Structured and intermediate data storage components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(record: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
