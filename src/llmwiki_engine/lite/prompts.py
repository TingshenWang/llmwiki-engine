from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .models import (
    CandidateContexts,
    CandidateMergePlan,
    CandidateMergeUnit,
    CandidatePages,
    CandidatePagesWarmup,
    CompositionItem,
    CompositionPlan,
    FinalPages,
    MergePlan,
    SourceDigest,
    WikiSnapshot,
)
from .profile import Profile
from .providers import PromptRequest


SYSTEM_PROMPT = """你是 llmwiki-engine Lite 的严格知识编译组件。
只返回符合 schema 的 JSON，不要用 Markdown 包裹 JSON。
/raw 是证据层：不要编造来源、hash 或路径。
所有给人阅读的文本必须使用中文，包括摘要、理由、正文、标题、问题、warnings 和 Markdown 内容。
结构字段名、路径、hash、id、枚举值必须保持 schema 要求的英文/机器值。
如果不确定，选择最安全的自动动作，并把不确定性写入中文 warnings 或中文未决问题。
"""


CHINESE_OUTPUT_RULES = [
    "所有给人阅读的文本必须使用中文；不要输出英文段落、英文说明或英文模板句。",
    "schema 字段名、id、路径、hash、action 枚举值保持原样；这些机器字段不需要翻译。",
    "标题可以保留必要产品名/专有名词，但解释性文字必须是中文。",
    "Markdown 章节标题必须使用中文，例如 摘要、核心内容、矛盾与未决问题。",
]


CANDIDATE_PAGE_SHARED_INSTRUCTIONS = [
    *CHINESE_OUTPUT_RULES,
    "这是候选知识页生成请求组共享的缓存前缀。",
    "必须以 raw_text 为最高优先级证据写正文，不要只复述 source_digest 摘要。",
    "不要在这里判断 create/update/noop。",
    "不要读取或假设旧 wiki；旧 wiki 只能在后续 embedding 召回之后进入。",
    "不要包含 frontmatter。",
    "不要伪装成写入最终 target path；proposed_path_hint 只是提示。",
    "每个 candidate page 必须保留 source_refs。",
]


def source_digest_prompt(*, raw_path: str, raw_sha256: str, raw_text: str, profile: Profile) -> PromptRequest:
    return _request(
        step="source_digest",
        model=SourceDigest,
        payload={
            "raw_path": raw_path,
            "raw_sha256": raw_sha256,
            "raw_text": raw_text,
            "profile": profile.model_dump(mode="json"),
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "source_raw_path 必须严格等于 raw_path。",
                "raw_sha256 必须严格等于 raw_sha256。",
                "提取值得写入 wiki 知识页的候选项。",
                "每个候选项必须包含至少一个 source_ref，包含 raw_path、raw_sha256 和有用 locator。",
                "如果 suggested_page_title 不确定，使用 name；不要让 name 为空。",
                "related_candidates 只用于本 source 内部 candidate_id 关系，例如上下游、补充、反例、使用场景或方法依赖。",
                "不要在 related_candidates 放 wiki 路径，也不要在本步骤判断 create/update/noop。",
            ],
        },
    )


def candidate_merge_prompt(*, digest: SourceDigest, profile: Profile) -> PromptRequest:
    return _request(
        step="candidate_merge",
        model=CandidateMergePlan,
        payload={
            "source_digest": digest.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "本步骤只做 source_digest 内部候选项的去重、近义合并和页面类型确认。",
                "不要读取或假设旧 wiki；不要判断 create/update/noop。",
                "每个 unit 表示后续要生成的一篇候选页。",
                "source_candidate_ids 必须来自 source_digest 中已有 candidate_id，不能编造。",
                "page_type 必须来自 profile.page_types，且不能是 source。",
                "path_hint 必须位于 page_type 对应目录内，且以 .md 结尾。",
                "must_cover_points 必须列出后续候选页必须覆盖的中文要点。",
                "source_refs 必须合并对应 source candidates 的来源引用。",
            ],
        },
    )


def candidate_page_prompt(*, digest: SourceDigest, candidate_unit: CandidateMergeUnit, raw_path: str, raw_sha256: str, raw_text: str, profile: Profile) -> PromptRequest:
    return _request(
        step="candidate_pages",
        model=CandidatePages,
        cache_prefix_payload=_candidate_page_cache_prefix(digest=digest, raw_path=raw_path, raw_sha256=raw_sha256, raw_text=raw_text, profile=profile),
        payload={
            "candidate_unit": candidate_unit.model_dump(mode="json"),
            "parallel_generation_contract": {
                "mode": "per_candidate_unit",
                "expected_candidate_unit_id": candidate_unit.candidate_unit_id,
                "expected_page_count": 1,
            },
            "instructions": [
                "这是并发 candidate-page 生成请求组中的一个请求。",
                "只为 candidate_unit 生成一个候选页面。",
                "返回的 pages 数组必须只有一个 item。",
                "page.candidate_unit_id 必须等于 expected_candidate_unit_id。",
                "page.source_candidate_ids 必须等于或覆盖 candidate_unit.source_candidate_ids。",
                "必须对照 cache_prefix.raw_text 写正文。",
                "必须使用 cache_prefix.source_digest、cache_prefix.profile 和 cache_prefix.instructions。",
            ],
        },
    )


def candidate_pages_warmup_prompt(*, digest: SourceDigest, raw_path: str, raw_sha256: str, raw_text: str, profile: Profile) -> PromptRequest:
    return _request(
        step="candidate_pages_warmup",
        model=CandidatePagesWarmup,
        cache_prefix_payload=_candidate_page_cache_prefix(digest=digest, raw_path=raw_path, raw_sha256=raw_sha256, raw_text=raw_text, profile=profile),
        payload={
            "warmup": True,
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "这是 candidate_pages 并发生成前的缓存预热请求。",
                "不要生成候选页，不要复述 raw_text，不要解释。",
                "只返回 status=OK。",
            ],
        },
    )


def merge_plan_prompt(*, candidate_pages: CandidatePages, candidate_contexts: CandidateContexts, profile: Profile) -> PromptRequest:
    return _request(
        step="merge_plan",
        model=MergePlan,
        payload={
            "candidate_pages": candidate_pages.model_dump(mode="json"),
            "candidate_contexts": candidate_contexts.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "allowed_actions": ["create", "update", "noop"],
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "只能查看每个 candidate_page 对应的 candidate_contexts top-k 旧 wiki 页面；不要假设还有其他旧页。",
                "每个 candidate_page_id 必须至少产生一个 decision；同一个候选页可以拆成多个 decision。",
                "如果候选页的一部分应更新 top-k 中某篇旧页，另一部分应新建页面，就输出一个 update decision 和一个 create decision。",
                "每个 decision 必须填写全局唯一 decision_id，例如 MD-001。",
                "每个 decision 必须填写 title、page_type、content_scope、candidate_path_index 和 source_refs。",
                "content_scope 用中文说明这个 decision 消费候选页中的哪一部分内容，避免拆分后重复或遗漏。",
                "candidate_path_index 必须列出候选页中被此 decision 消费的章节、要点或证据定位。",
                "新知识使用 create，匹配 top-k 旧页使用 update，已经覆盖才使用 noop。",
                "update 的 target_path 必须来自同一 candidate_page_id 的 candidate_contexts.hits.path。",
                "create 的 target_path 必须留在 profile 路由目录内。",
                "每个候选项选择 related_pages 前必须检查 top-k context hits。",
                "related_pages 在这里决定，不在 final writing 决定。最多建议 2 个 related_pages；引擎可能保留旧链接，总上限 3 个。",
                "相关页可以来自 source 内部 related_candidates、已检查旧 wiki context、或精确 title/alias 命中。",
                "不要链接 source pages、log/index pages、未知路径或页面自身。",
                "没有合适相关页时，设置 related_absence_reason，并保证理由字段使用中文。",
                "重新计算 action_counts，确保它与 decisions 一致。",
            ],
        },
    )


def composition_plan_prompt(
    *,
    merge_plan: MergePlan,
    candidate_pages: CandidatePages,
    profile: Profile,
) -> PromptRequest:
    return _request(
        step="composition_plan",
        model=CompositionPlan,
        payload={
            "merge_plan": merge_plan.model_dump(mode="json"),
            "candidate_pages": candidate_pages.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "每个可写 create/update decision 必须进入一个 composition item；同一 target_path 的多个 decision 可以合并为一个 item。",
                "不要为 noop decisions 创建 composition item。",
                "merge_decision_ids 必须覆盖进入该 item 的 merge decision。",
                "candidate_page_ids 必须来自对应 merge decisions。",
                "update target 必须保留有价值的旧内容。",
                "每个 item 必须包含 source_ref_rules，且规则文本用中文。",
                "把 merge decisions 中的 related_pages、related_absence_reason、related_unresolved 带入匹配的 composition item。",
            ],
        },
    )


def final_page_prompt(
    *,
    composition_item: CompositionItem,
    candidate_pages: CandidatePages,
    snapshot: WikiSnapshot,
    profile: Profile,
) -> PromptRequest:
    candidate_ids = set(composition_item.candidate_page_ids)
    relevant_candidates = CandidatePages(pages=[page for page in candidate_pages.pages if page.candidate_page_id in candidate_ids])
    relevant_paths = {composition_item.target_path, *composition_item.existing_page_refs}
    relevant_entries = [entry for entry in snapshot.entries if entry.path in relevant_paths]
    return _request(
        step="final_pages",
        model=FinalPages,
        payload={
            "composition_item": composition_item.model_dump(mode="json"),
            "candidate_pages": relevant_candidates.model_dump(mode="json"),
            "wiki_snapshot_entries": [entry.model_dump(mode="json") for entry in relevant_entries],
            "profile": profile.model_dump(mode="json"),
            "parallel_generation_contract": {
                "mode": "per_composition_item",
                "expected_final_page_id": composition_item.final_page_id,
                "expected_target_path": composition_item.target_path,
                "expected_page_count": 1,
            },
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "这是并发 final-page 生成请求组中的一个请求。",
                "只为 composition_item 返回一个最终页面。",
                "返回的 pages 数组必须只有一个 item。",
                "page.final_page_id 和 target_path 必须匹配 expected values。",
                "引擎会替换最终 frontmatter；专注于中文正文和稳定的中文标题结构。",
                "每个页面必须保留 source_refs。",
                "update 页面必须保留有价值的旧笔记，并加入有来源支撑的新材料。",
                "不要写 Related 或 相关页面章节；引擎会在 validation 之后渲染官方相关页面章节。",
                "不要创建自链接。",
                "content_sha256 可以为空；引擎会重新计算。",
                "不要包含模型自我说明。",
            ],
        },
    )


def _candidate_page_cache_prefix(*, digest: SourceDigest, raw_path: str, raw_sha256: str, raw_text: str, profile: Profile) -> dict[str, Any]:
    return {
        "task": "candidate_pages",
        "raw_path": raw_path,
        "raw_sha256": raw_sha256,
        "raw_text": raw_text,
        "source_digest": digest.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "instructions": CANDIDATE_PAGE_SHARED_INSTRUCTIONS,
    }


def _request(*, step: str, model: type[BaseModel], payload: dict[str, Any], cache_prefix_payload: dict[str, Any] | None = None) -> PromptRequest:
    schema_name = f"llmwiki_lite_{step}"
    return PromptRequest(
        step=step,
        schema_name=schema_name,
        system_prompt=SYSTEM_PROMPT,
        cache_prefix_payload=cache_prefix_payload,
        user_payload=payload,
        response_schema=model.model_json_schema(),
        json_output_example=_json_example(step),
    )


def _json_example(step: str) -> dict[str, Any]:
    examples: dict[str, dict[str, Any]] = {
        "source_digest": {
            "source_raw_path": "raw/example.md",
            "raw_sha256": "sha256",
            "summary": "这是一段来源材料的中文摘要。",
            "key_takeaways": ["一条有来源支撑的中文收获。"],
            "entities": [],
            "concepts": [
                {
                    "candidate_id": "CAND-001",
                    "kind": "concept",
                    "name": "示例概念",
                    "suggested_page_title": "示例概念",
                    "summary": "说明这个概念为什么值得写入 wiki。",
                    "source_basis": "候选项对应的具体来源依据。",
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                    "related_candidates": [],
                }
            ],
            "designs": [],
            "comparisons": [],
            "open_questions": [],
            "budget_deferred_candidates": [],
            "weak_or_noise_items": [],
        },
        "candidate_pages": {
            "pages": [
                {
                    "candidate_page_id": "CP-001",
                    "candidate_unit_id": "CM-001",
                    "source_candidate_ids": ["CAND-001"],
                    "title": "示例概念",
                    "proposed_page_type": "concept",
                    "proposed_path_hint": "concepts/Concept_Example_Concept.md",
                    "summary": "一段有来源支撑的中文摘要。",
                    "body_markdown": "## 摘要\n\n一段有来源支撑的中文摘要。\n",
                    "open_questions": [],
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                    "evidence_notes": ["whole_file"],
                    "confidence": 0.7,
                }
            ],
            "skipped_candidate_ids": [],
        },
        "candidate_pages_warmup": {"status": "OK"},
        "candidate_merge": {
            "units": [
                {
                    "candidate_unit_id": "CM-001",
                    "source_candidate_ids": ["CAND-001"],
                    "title": "示例概念",
                    "page_type": "concept",
                    "path_hint": "concepts/Concept_Example_Concept.md",
                    "summary": "把同义候选合并成一个待生成页面。",
                    "merge_reason": "只有一个候选项，直接保留为一个页面单元。",
                    "must_cover_points": ["说明示例概念的定义和来源依据。"],
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                }
            ],
            "skipped_candidate_ids": [],
            "warnings": [],
        },
        "merge_plan": {
            "decisions": [
                {
                    "decision_id": "MD-001",
                    "candidate_page_id": "CP-001",
                    "action": "create",
                    "target_path": "concepts/Concept_Example_Concept.md",
                    "title": "示例概念",
                    "page_type": "concept",
                    "content_scope": "写入候选页中关于示例概念定义和来源依据的全部内容。",
                    "candidate_path_index": ["摘要", "核心内容"],
                    "matched_existing_paths": [],
                    "inspected_context_paths": [],
                    "strongest_overlap": 0.0,
                    "reason": "没有已有页面覆盖这个有来源支撑的概念。",
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                    "related_pages": [],
                    "related_absence_reason": "no_related_candidate_after_filter",
                    "related_unresolved": [],
                    "warnings": [],
                }
            ],
            "action_counts": {"create": 1, "update": 0, "noop": 0},
            "warnings": [],
        },
        "composition_plan": {
            "items": [
                {
                    "final_page_id": "FP-001",
                    "target_path": "concepts/Concept_Example_Concept.md",
                    "action": "create",
                    "merge_decision_ids": ["MD-001"],
                    "candidate_page_ids": ["CP-001"],
                    "existing_page_refs": [],
                    "section_order": ["摘要", "核心内容", "相关页面", "矛盾与未决问题"],
                    "preserve_rules": [],
                    "insert_rules": ["写入有来源支撑的候选材料。"],
                    "delete_rules": [],
                    "source_ref_rules": ["保留 raw 来源引用。"],
                    "readability_goal": "生成可读的中文 wiki 笔记。",
                    "related_pages": [],
                    "related_absence_reason": "no_related_candidate_after_filter",
                    "related_unresolved": [],
                    "warnings": [],
                }
            ]
        },
        "final_pages": {
            "pages": [
                {
                    "final_page_id": "FP-001",
                    "target_path": "concepts/Concept_Example_Concept.md",
                    "action": "create",
                    "title": "示例概念",
                    "page_type": "concept",
                    "content_sha256": "",
                    "markdown": "# 示例概念\n\n## 摘要\n\n一段有来源支撑的中文摘要。\n",
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                    "preimage_sha256": None,
                    "warnings": [],
                }
            ],
            "warnings": [],
        },
    }
    return examples.get(step, {})
