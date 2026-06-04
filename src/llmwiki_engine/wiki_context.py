from __future__ import annotations

from pathlib import Path

from .hash_utils import sha256_file
from .models import WikiContextSnapshot


def wiki_context_drift_messages(vault: Path, snapshot: WikiContextSnapshot) -> list[str]:
    messages: list[str] = []
    for entry in snapshot.entries:
        path = vault / entry.path
        if entry.expected_state == "missing":
            if path.exists():
                messages.append(f"wiki context appeared after planning: {entry.path}")
            continue
        if not path.exists():
            messages.append(f"wiki context disappeared after planning: {entry.path}")
            continue
        if entry.preimage_sha256 is None:
            messages.append(f"wiki context snapshot is missing sha256: {entry.path}")
            continue
        if sha256_file(path) != entry.preimage_sha256:
            messages.append(f"wiki context changed after planning: {entry.path}")
    return messages
