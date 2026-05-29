from __future__ import annotations

import hashlib
from pathlib import Path

from .models import ArtifactRef, ArtifactVisibility


MISSING_SHA = "__MISSING__"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_ref(
    *,
    base: Path,
    path: Path,
    kind: str,
    producer_step: str,
    schema_version: str | None = None,
    required_for_resume: bool = True,
    visibility: ArtifactVisibility = ArtifactVisibility.run_cache,
) -> ArtifactRef:
    return ArtifactRef(
        relative_path=path.relative_to(base).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        kind=kind,
        schema_version=schema_version,
        producer_step=producer_step,
        required_for_resume=required_for_resume,
        visibility=visibility,
    )

