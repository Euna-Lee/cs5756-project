from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def get_git_commit_hash(repo_root: str | Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def save_run_manifest(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, default=str)


def argv_list() -> list[str]:
    return list(sys.argv)


def common_run_metadata(
    script: str,
    repo_root: str | Path,
    *,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "script": script,
        "git_commit": get_git_commit_hash(repo_root),
        "argv": argv_list(),
    }
    if seed is not None:
        d["seed"] = seed
    if extra:
        d.update(dict(extra))
    return d
