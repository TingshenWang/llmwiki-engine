from __future__ import annotations

from pathlib import Path

from .hash_utils import sha256_file
from .models import OperationManifest, VerificationStatus, VerifyIssue, VerifyResult
from .workspace import RunStore


class VerifyError(RuntimeError):
    def __init__(self, result: VerifyResult):
        self.result = result
        super().__init__("; ".join(issue.message for issue in result.issues) or "verification failed")


def verify_run(vault: Path, manifest: OperationManifest) -> VerifyResult:
    issues: list[VerifyIssue] = []
    store = RunStore(vault)
    run_dir = store.run_dir(manifest.operation_id)
    for raw in manifest.raw_bindings:
        raw_path = vault / raw.relative_path
        if not raw_path.exists():
            issues.append(VerifyIssue(code=VerificationStatus.missing, path=raw.relative_path, message="raw file is missing"))
        elif sha256_file(raw_path) != raw.sha256:
            issues.append(VerifyIssue(code=VerificationStatus.raw_changed, path=raw.relative_path, message="raw file hash changed"))
    for step in manifest.steps:
        for ref in step.outputs:
            if not ref.required_for_resume:
                continue
            path = run_dir / ref.relative_path
            if not path.exists():
                issues.append(VerifyIssue(code=VerificationStatus.missing, path=ref.relative_path, message="required artifact is missing"))
                continue
            if not path.is_file():
                issues.append(VerifyIssue(code=VerificationStatus.drift, path=ref.relative_path, message="required artifact is not a file"))
            elif sha256_file(path) != ref.sha256:
                issues.append(VerifyIssue(code=VerificationStatus.drift, path=ref.relative_path, message="artifact hash changed"))
    return VerifyResult(ok=not issues, issues=issues)


def require_verified(vault: Path, manifest: OperationManifest) -> None:
    result = verify_run(vault, manifest)
    if not result.ok:
        raise VerifyError(result)
