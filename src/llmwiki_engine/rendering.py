from __future__ import annotations

from pathlib import Path

from .profiles import safe_filename


def source_title_for_raw(raw_path: str) -> str:
    path = Path(raw_path)
    if path.parts and path.parts[0] == "raw":
        path = Path(*path.parts[1:])
    stem = path.with_suffix("").as_posix().replace("/", "_")
    return f"Source_{safe_filename(stem)}"
