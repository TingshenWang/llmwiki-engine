from __future__ import annotations

from pathlib import Path
from typing import Literal

from . import apply_guards as _apply_guards
from .hash_utils import sha256_file
from .io import read_json, read_model
from .models import DraftApproval, DraftRenderingArtifact, DraftWriteManifest, DraftWriteTarget, UpdateMergeReport
from .system_pages import format_markdown_table


def build_draft_approval(
    run_dir: Path,
    approved_manifest_path: Path,
    *,
    decision: Literal["approved", "pending", "rejected"],
    auto_approved: bool,
    notes: str,
) -> DraftApproval:
    _apply_guards.require_draft_rendering_sidecars(run_dir)
    approved_manifest = read_model(approved_manifest_path, DraftWriteManifest)
    markdown_hashes: dict[str, str] = {}
    for target in approved_manifest.targets:
        draft = run_dir / target.draft_path
        if draft.suffix == ".md" and draft.exists():
            markdown_hashes[target.draft_path] = sha256_file(draft)
    for rel_path in _apply_guards.REQUIRED_DRAFT_RENDERING_SIDECARS:
        markdown_hashes[rel_path] = sha256_file(run_dir / rel_path)
    return DraftApproval(
        decision=decision,
        auto_approved=auto_approved,
        approved_draft_json_sha256=sha256_file(approved_manifest_path),
        approved_markdown_sha256=markdown_hashes,
        notes=notes,
    )


def pending_draft_approval(reason: str) -> DraftApproval:
    return DraftApproval(decision="pending", auto_approved=False, notes=reason)


def render_draft_review_prompt(run_dir: Path, draft_manifest: DraftWriteManifest) -> str:
    manual_resolution_count = _update_manual_resolution_count(run_dir)
    reinforcement_count = _update_reinforcement_count(run_dir)
    manual_resolution_note = (
        f"是（{manual_resolution_count} 段旧页保留观察需人工消化、改写或确认删除）"
        if manual_resolution_count
        else "否"
    )
    reinforcement_note = f"是（{reinforcement_count} 段旧页知识已由系统本地补强并记录）" if reinforcement_count else "否"
    warning = (
        "## 旧页保留观察警示\n\n"
        f"Update 合并报告包含 {manual_resolution_count} 段旧页保留观察。批准前需要人工消化："
        "把仍有价值的旧知识自然改写进新页，或明确确认删除。\n\n"
        if manual_resolution_count
        else ""
    )
    reinforcement_warning = (
        "## 本地旧知识补强提示\n\n"
        f"Draft rendering 本地补强了 {reinforcement_count} 段旧页知识，并记录在 {_update_reinforcement_report_ref(run_dir)}。"
        "如本轮还因其他问题进入人工审核，"
        "可顺手检查这些桥接语是否自然。\n\n"
        if reinforcement_count
        else ""
    )
    rows = [
        [
            target.action,
            f"`{target.target_path}`",
            f"`{target.draft_path}`",
            _draft_diff_ref(run_dir, target),
            _draft_change_summary(run_dir, target),
            target.expected_state,
            target.preimage_sha256 or "",
        ]
        for target in draft_manifest.targets
    ]
    return (
        "# 草稿审核\n\n"
        "审查这一步回答：具体写什么、是否应批准写入。\n\n"
        f"- 需要 Grounding 人工确认：{'是' if draft_manifest.requires_grounding_review else '否'}\n"
        f"- 旧页保留观察需人工消化：{manual_resolution_note}\n"
        f"- 本地旧知识补强已执行：{reinforcement_note}\n\n"
        f"{warning}"
        f"{reinforcement_warning}"
        "## 核心判断\n\n"
        "- create/update 的正文是否忠实于 raw 和已召回旧页？\n"
        "- update diff 是否符合你的理解，没有覆盖掉旧页中仍然重要的内容？\n"
        "- 未被来源支持的细节是否放在“矛盾与未决问题/待补来源”，而不是写成事实？\n"
        "- Related 是否少而准，单页主动连接不超过 3 条？\n\n"
        "## 下一步命令\n\n"
        "- 批准：`uv run llmwiki ingest approve \"$VAULT\" \"$OP\" draft_review`\n"
        "- 重新生成/修订：`uv run llmwiki ingest revise \"$VAULT\" \"$OP\" draft_review`\n"
        "- 批准后继续：`uv run llmwiki ingest resume \"$VAULT\" \"$OP\"`\n"
        "- Apply：`uv run llmwiki ingest apply \"$VAULT\" \"$OP\"`\n\n"
        "## 关键文件\n\n"
        "- Update 合并报告：`draft_rendering/update_merge_report.md`\n"
        "- Grounding 审查：`draft_rendering/draft_grounding_review.md`\n"
        "- Related 合并报告：`draft_rendering/related_merge_report.md`\n"
        "- 草稿目录：`draft_rendering/draft_pages/`\n"
        "- Diff 目录：`draft_rendering/diffs/`\n\n"
        "## 草稿清单\n\n"
        + format_markdown_table(["动作", "目标", "草稿", "Diff", "变更摘要", "预期状态", "Preimage"], rows)
        + "\n"
    )


def draft_review_requires_manual(run_dir: Path, draft_manifest: DraftWriteManifest) -> bool:
    return bool(draft_manifest.requires_grounding_review or _update_manual_resolution_count(run_dir))


def draft_review_reason(run_dir: Path, draft_manifest: DraftWriteManifest) -> str:
    reasons: list[str] = []
    if draft_manifest.requires_grounding_review:
        reasons.append("Grounding review 发现 unsupported new_fact，需要人工确认。")
    manual_resolution_count = _update_manual_resolution_count(run_dir)
    if manual_resolution_count:
        reasons.append(f"Update 合并报告包含 {manual_resolution_count} 段旧页保留观察，需人工消化、改写或确认删除。")
    return " ".join(reasons) or "草稿需要显式人工批准。"


def _update_manual_resolution_count(run_dir: Path) -> int:
    report_path = run_dir / "draft_rendering" / "update_merge_report.json"
    if not report_path.exists():
        return 0
    try:
        report = read_model(report_path, UpdateMergeReport)
    except Exception:
        return 0
    return sum(1 for page in report.pages for section in page.sections if section.needs_manual_resolution)


def _update_reinforcement_count(run_dir: Path) -> int:
    draft_root = run_dir / "draft_rendering"
    report_path = draft_root / "update_preservation_reinforcement_report.json"
    if report_path.exists():
        try:
            return int(read_json(report_path).get("reinforced_section_count", 0))
        except Exception:
            return 0
    batch_report_path = draft_root / "draft_rendering_batch_report.json"
    if not batch_report_path.exists():
        return 0
    try:
        report = read_json(batch_report_path)
        return sum(int(batch.get("reinforced_section_count", 0)) for batch in report.get("batches", []))
    except Exception:
        return 0


def _update_reinforcement_report_ref(run_dir: Path) -> str:
    draft_root = run_dir / "draft_rendering"
    if (draft_root / "update_preservation_reinforcement_report.md").exists():
        return "`draft_rendering/update_preservation_reinforcement_report.md`"
    if (draft_root / "draft_rendering_batch_report.md").exists():
        return "`draft_rendering/draft_rendering_batch_report.md`"
    return "`draft_rendering/`"


def _draft_diff_ref(run_dir: Path, target: DraftWriteTarget) -> str:
    if not target.page_plan_id:
        return ""
    diff = run_dir / "draft_rendering" / "diffs" / f"{target.page_plan_id}.diff"
    return f"`{diff.relative_to(run_dir).as_posix()}`" if diff.exists() else ""


def _draft_change_summary(run_dir: Path, target: DraftWriteTarget) -> str:
    if not target.page_plan_id:
        return ""
    draft_json = run_dir / "draft_rendering" / "draft_rendering.json"
    if not draft_json.exists():
        return ""
    try:
        draft = read_model(draft_json, DraftRenderingArtifact)
    except Exception:
        return ""
    for page in draft.pages:
        if page.page_plan_id == target.page_plan_id:
            return page.change_summary
    return ""
