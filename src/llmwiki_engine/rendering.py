from __future__ import annotations

from pathlib import Path

from .models import ClaimsArtifact, PagePlanArtifact, PagePlanItem, ProfileSpec, RawIndexArtifact
from .profiles import page_output_path, profile_template_text, safe_filename


def render_drafts(
    *,
    draft_root: Path,
    profile: ProfileSpec,
    raw_index: RawIndexArtifact,
    claims: ClaimsArtifact,
    plan: PagePlanArtifact,
) -> list[Path]:
    claim_by_id = {claim.claim_id: claim for claim in claims.claims}
    outputs: list[Path] = []
    for page in normalized_page_plan(raw_index, claims, plan).pages:
        out = page_output_path(draft_root, profile, page.page_type, page.title)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_page(profile, page, raw_index, claim_by_id), encoding="utf-8")
        outputs.append(out)
    return outputs


def normalized_page_plan(raw_index: RawIndexArtifact, claims: ClaimsArtifact, plan: PagePlanArtifact) -> PagePlanArtifact:
    source_identity = raw_index.original_raw_path or raw_index.raw_path
    source_title = source_title_for_raw(source_identity)
    source_claim_ids = [claim.claim_id for claim in claims.claims]
    pages = [
        page
        for page in plan.pages
        if not (page.page_type == "source" and page.source_raw_path in {None, raw_index.raw_path, source_identity})
    ]
    pages.insert(
        0,
        PagePlanItem(
            page_type="source",
            title=source_title,
            claim_ids=source_claim_ids,
            source_raw_path=raw_index.raw_path,
            summary=next((page.summary for page in plan.pages if page.page_type == "source"), f"Source for {raw_index.raw_path}"),
        ),
    )
    return PagePlanArtifact(pages=pages)


def source_title_for_raw(raw_path: str) -> str:
    path = Path(raw_path)
    if path.parts and path.parts[0] == "raw":
        path = Path(*path.parts[1:])
    stem = path.with_suffix("").as_posix().replace("/", "_")
    return f"Source_{safe_filename(stem)}"


def _render_page(profile: ProfileSpec, page: PagePlanItem, raw_index: RawIndexArtifact, claim_by_id: dict) -> str:
    template = profile_template_text(profile, page.page_type)
    claims = [claim_by_id[claim_id] for claim_id in page.claim_ids if claim_id in claim_by_id]
    claim_lines = "\n".join(
        f"- `{claim.claim_id}` {claim.text}  \n  Evidence: `{claim.evidence_quote}`"
        for claim in claims
    ) or "- No claims."
    source_line = "\n".join(
        [
            f"- Prepared Raw: `{raw_index.raw_path}`",
            f"- Prepared Raw SHA256: `{raw_index.raw_sha256}`",
            f"- Original Raw: `{raw_index.original_raw_path}`" if raw_index.original_raw_path else "- Original Raw: same as prepared raw",
        ]
    )
    return (
        template.replace("{{title}}", page.title)
        .replace("{{summary}}", page.summary or "No summary.")
        .replace("{{source_info}}", source_line)
        .replace("{{claims}}", claim_lines)
    )
