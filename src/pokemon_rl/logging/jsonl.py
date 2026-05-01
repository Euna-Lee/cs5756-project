from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class JsonlLogger:
    """Append one JSON object per line (JSONL). Safe for concurrent appends only if using separate files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, default=_json_default, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
