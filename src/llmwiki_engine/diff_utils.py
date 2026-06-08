from __future__ import annotations

from difflib import unified_diff


def render_update_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    return "".join(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=old_name,
            tofile=new_name,
        )
    )
