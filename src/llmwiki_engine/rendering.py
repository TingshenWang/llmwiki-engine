from __future__ import annotations

from pathlib import Path

from .models import ClaimsArtifact, PagePlanArtifact, PagePlanItem, ProfileSpec, RawIndexArtifact
from .profiles import page_output_path, profile_template_text


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
    for page in plan.pages:
        out = page_output_path(draft_root, profile, page.page_type, page.title)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_page(profile, page, raw_index, claim_by_id), encoding="utf-8")
        outputs.append(out)
    return outputs


def _render_page(profile: ProfileSpec, page: PagePlanItem, raw_index: RawIndexArtifact, claim_by_id: dict) -> str:
    template = profile_template_text(profile, page.page_type)
    claims = [claim_by_id[claim_id] for claim_id in page.claim_ids if claim_id in claim_by_id]
    claim_lines = "\n".join(
        f"- `{claim.claim_id}` {claim.text}  \n  Evidence: `{claim.evidence_quote}`"
        for claim in claims
    ) or "- No claims."
    source_line = f"- Raw: `{raw_index.raw_path}`"
    return (
        template.replace("{{title}}", page.title)
        .replace("{{summary}}", page.summary or "No summary.")
        .replace("{{source_info}}", source_line)
        .replace("{{claims}}", claim_lines)
    )

