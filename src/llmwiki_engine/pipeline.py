from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from rich.console import Console

from .events import EventLogger
from .io import read_model, write_json, write_yaml
from .models import (
    ClaimsArtifact,
    OperationManifest,
    PagePlanArtifact,
    RawIndexArtifact,
    RawSpan,
    SemanticAggregationArtifact,
    StepRecord,
    utc_now,
)
from .profiles import load_profile
from .providers import ProviderRegistry
from .rendering import render_drafts
from .structured import StructuredModelCall
from .validators import validate_aggregation, validate_claims, validate_page_plan, validate_raw_index


PIPELINE_STEPS = [
    "create_operation",
    "raw_index",
    "semantic_aggregation",
    "claim_extraction",
    "page_planning",
    "draft_rendering",
    "validation",
]


def init_vault(vault: Path, *, profile_name: str = "project_basic") -> None:
    profile = load_profile(profile_name)
    (vault / "raw").mkdir(parents=True, exist_ok=True)
    (vault / "stage" / "ingest").mkdir(parents=True, exist_ok=True)
    (vault / "logs").mkdir(parents=True, exist_ok=True)
    for spec in profile.page_types.values():
        (vault / "wiki" / spec.directory).mkdir(parents=True, exist_ok=True)
    write_yaml(vault / "profiles" / "profile.yaml", profile.model_dump(mode="json"))
    write_yaml(
        vault / "llmwiki.yaml",
        {
            "profile": profile.name,
            "providers": {
                "semantic_aggregation": "mock:fixture",
                "claim_extraction": "mock:fixture",
                "page_planning": "mock:fixture",
                "critic": "mock:fixture",
            },
        },
    )


def run_simplified_ingest(
    *,
    vault: Path,
    raw_file: Path,
    fixture_dir: Path,
    profile_name: str = "project_basic",
    slug: str | None = None,
    console: Console | None = None,
) -> OperationManifest:
    raw_path = raw_file if raw_file.is_absolute() else vault / raw_file
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    operation_id = f"ING-{utc_now().replace(':', '').replace('+0000', 'Z')}-{slug or raw_path.stem}"
    stage_dir = vault / "stage" / "ingest" / operation_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(operation_id, stage_dir / "events.jsonl", console=console)
    profile = load_profile(profile_name)
    manifest = OperationManifest(
        operation_id=operation_id,
        operation_type="ingest",
        profile=profile.name,
        steps=[StepRecord(name=name) for name in PIPELINE_STEPS],
    )

    with logger.step("create_operation"):
        write_json(stage_dir / "manifest.json", manifest)

    with logger.step("raw_index"):
        raw_index = build_raw_index(vault, raw_path)
        write_json(stage_dir / "raw_index.json", raw_index)

    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)
    caller = StructuredModelCall(provider, output_dir=stage_dir / "model_calls")

    with logger.step("semantic_aggregation"):
        aggregation, _ = caller.run("semantic_aggregation", raw_index.model_dump(mode="json"), SemanticAggregationArtifact)
        write_json(stage_dir / "semantic_aggregation.json", aggregation)

    with logger.step("claim_extraction"):
        claims_payload = {"raw_index": raw_index.model_dump(mode="json"), "semantic_aggregation": aggregation.model_dump(mode="json")}
        claims, _ = caller.run("claim_extraction", claims_payload, ClaimsArtifact)
        write_json(stage_dir / "claims.json", claims)

    with logger.step("page_planning"):
        plan_payload = {"profile": profile.model_dump(mode="json"), "claims": claims.model_dump(mode="json")}
        plan, _ = caller.run("page_planning", plan_payload, PagePlanArtifact)
        write_json(stage_dir / "page_plan.json", plan)

    with logger.step("draft_rendering"):
        outputs = render_drafts(
            draft_root=stage_dir / "draft_pages",
            profile=profile,
            raw_index=raw_index,
            claims=claims,
            plan=plan,
        )
        logger.emit("draft_rendering", "outputs", data={"count": len(outputs)})

    with logger.step("validation"):
        validate_raw_index(raw_index)
        validate_aggregation(raw_index, aggregation)
        validate_claims(raw_index, aggregation, claims)
        validate_page_plan(profile, claims, plan)

    manifest.status = "drafted"
    manifest.updated_at = utc_now()
    manifest.steps = [StepRecord(name=name, status="completed") for name in PIPELINE_STEPS]
    write_json(stage_dir / "manifest.json", manifest)
    return manifest


def build_raw_index(vault: Path, raw_path: Path) -> RawIndexArtifact:
    text = raw_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    spans: list[RawSpan] = []
    cursor = 0
    for part in [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]:
        start = text.find(part, cursor)
        end = start + len(part)
        cursor = end
        spans.append(
            RawSpan(
                span_id=f"S{len(spans) + 1:03d}",
                raw_path=str(raw_path.relative_to(vault)) if raw_path.is_relative_to(vault) else str(raw_path),
                raw_sha256=digest,
                start_char=start,
                end_char=end,
                text_sha256=hashlib.sha256(part.encode("utf-8")).hexdigest(),
                text=part,
            )
        )
    return RawIndexArtifact(
        raw_path=str(raw_path.relative_to(vault)) if raw_path.is_relative_to(vault) else str(raw_path),
        raw_sha256=digest,
        spans=spans,
    )


def status(vault: Path, operation_id: str) -> OperationManifest:
    return read_model(vault / "stage" / "ingest" / operation_id / "manifest.json", OperationManifest)


def latest_operation(vault: Path) -> str | None:
    root = vault / "stage" / "ingest"
    if not root.exists():
        return None
    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    return candidates[-1].name if candidates else None


def copy_fixture_raw(vault: Path, fixture_raw: Path) -> Path:
    target = vault / "raw" / fixture_raw.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture_raw, target)
    return target

