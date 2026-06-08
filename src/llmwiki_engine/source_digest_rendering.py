from __future__ import annotations

from .models import SourceDigestArtifact, SourceDigestCandidate, WeakOrNoiseItem
from .system_pages import format_markdown_table


def render_source_digest_markdown(digest: SourceDigestArtifact) -> str:
    sections = [
        "# 来源消化",
        "",
        f"- 原始材料: `{digest.source_raw_path}`",
        f"- 摘要: {digest.summary}",
        "",
        "## 关键收获",
        "",
        "\n".join(f"- {item}" for item in digest.key_takeaways) or "- 暂无关键收获记录。",
    ]
    for title, candidates in [
        ("实体", digest.entities),
        ("概念", digest.concepts),
        ("设计", digest.designs),
        ("对比", digest.comparisons),
        ("未决问题", digest.open_questions),
        ("预算延后候选", digest.budget_deferred_candidates),
        ("弱相关或噪声项", digest.weak_or_noise_items),
    ]:
        sections.extend(["", f"## {title}", "", _render_candidate_table(candidates)])
    return "\n".join(sections).rstrip() + "\n"


def _render_candidate_table(candidates: list[SourceDigestCandidate] | list[WeakOrNoiseItem]) -> str:
    if not candidates:
        return "_暂无。_"
    return format_markdown_table(
        ["ID", "类型", "名称", "摘要", "重复风险"],
        [[f"`{item.candidate_id}`", item.type, item.name, item.one_sentence_summary, item.duplicate_risk] for item in candidates],
    )
