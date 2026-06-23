from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .models import (
    CandidateContexts,
    CandidatePages,
    CandidatePagesWarmup,
    CompositionItem,
    CompositionPlan,
    FinalPages,
    MergePlan,
    SourceDigest,
    SourceGranularityStats,
    SourcePageUnit,
    WikiKnowledgeEntry,
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


def source_digest_prompt(
    *,
    raw_path: str,
    raw_sha256: str,
    raw_text: str,
    profile: Profile,
    granularity_stats: SourceGranularityStats | None = None,
) -> PromptRequest:
    return _request(
        step="source_digest",
        model=SourceDigest,
        payload={
            "raw_path": raw_path,
            "raw_sha256": raw_sha256,
            "raw_text": raw_text,
            **({"granularity_stats": granularity_stats.model_dump(mode="json")} if granularity_stats is not None else {}),
            "profile": profile.model_dump(mode="json"),
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "source_raw_path 必须严格等于 raw_path。",
                "raw_sha256 必须严格等于 raw_sha256。",
                "直接从 raw_text 规划本次应该生成哪些 page_units；每个 page_unit 表示后续要生成的一篇可读 wiki 候选页，而不是一个零散知识点。",
                "优先使用粗粒度页面：同一主体对象、同一读者任务、同一页面类型下的 what/why/how/example/version 应合并为同一 page_unit 的不同段落。",
                "只有主体对象不同、读者任务不同、或页面类型明显不同，才拆成多个 page_units。",
                "如果提供了 granularity_stats，page_unit 数量应落在 suggested_min_page_units 到 suggested_max_page_units 的经验区间附近；超出区间必须有强拆分理由。",
                "在本步骤内完成去重、近义合并和页面类型确认；不要额外输出候选项再交给后续合并。",
                "page_unit_id 使用 PU-001、PU-002 这样的稳定格式，不要编造 wiki 旧页路径。",
                "page_type 必须来自 profile.page_types，且不能是 source。",
                "path_hint 必须位于 page_type 对应目录内，且以 .md 结尾。",
                "page_units 必须覆盖 raw_text 中所有值得进入知识库的有效内容；不要把重要信息漏掉。",
                "must_cover_points 必须列出后续候选页必须覆盖的中文要点。",
                "多个 page_units 时，每个 page_unit 的 split_rationale 必须用中文说明为什么它不能并入其他 page_unit；如果说不清，应合并。",
                "每个 page_unit 必须包含至少一个 source_ref，包含 raw_path、raw_sha256 和有用 locator。",
                "不值得入库、噪声、重复或证据不足的内容放入 weak_or_noise_items，并用中文说明原因。",
                "不要在本步骤读取或假设旧 wiki；不要判断 create/update/noop。",
            ],
        },
    )


def source_digest_retry_prompt(
    *,
    raw_path: str,
    raw_sha256: str,
    raw_text: str,
    profile: Profile,
    previous_digest: SourceDigest,
    validation_error: str,
    granularity_stats: SourceGranularityStats | None = None,
) -> PromptRequest:
    request = source_digest_prompt(raw_path=raw_path, raw_sha256=raw_sha256, raw_text=raw_text, profile=profile, granularity_stats=granularity_stats)
    payload = dict(request.user_payload)
    instructions = list(payload.get("instructions") or [])
    payload.update(
        {
            "previous_invalid_output": previous_digest.model_dump(mode="json"),
            "validation_error": validation_error,
            "instructions": [
                *instructions,
                "上一轮 source_digest 输出没有通过系统校验；本轮必须返回修正后的完整 JSON，不要解释。",
                "所有 summary、key_takeaways、title、content_scope、must_cover_points、weak_or_noise reason 等用户可读字段必须使用中文。",
                "必要产品名、框架名、API 名可以保留英文专有名词，但解释性句子必须是中文。",
                "source_raw_path 和 raw_sha256 必须保持与输入完全一致。",
                "不要为了通过中文校验而删除有价值 page_unit；应把英文说明改写成中文说明。",
                "如果 validation_error 指出 page_unit 过多，请优先合并同主体、同读者任务、同页面类型的页面单元，而不是删掉有效内容。",
                "如果 validation_error 指出 0 个 page_unit 但原文不是噪声，请生成至少一个粗粒度 page_unit。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


def candidate_page_prompt(*, digest: SourceDigest, page_unit: SourcePageUnit, raw_path: str, raw_sha256: str, raw_text: str, profile: Profile) -> PromptRequest:
    return _request(
        step="candidate_pages",
        model=CandidatePages,
        cache_prefix_payload=_candidate_page_cache_prefix(digest=digest, raw_path=raw_path, raw_sha256=raw_sha256, raw_text=raw_text, profile=profile),
        payload={
            "page_unit": page_unit.model_dump(mode="json"),
            "parallel_generation_contract": {
                "mode": "per_page_unit",
                "expected_page_unit_id": page_unit.page_unit_id,
                "expected_page_count": 1,
            },
            "instructions": [
                "这是并发 candidate-page 生成请求组中的一个请求。",
                "只为 page_unit 生成一个候选页面。",
                "返回的 pages 数组必须只有一个 item。",
                "page.page_unit_id 必须等于 expected_page_unit_id。",
                "必须对照 cache_prefix.raw_text 写正文。",
                "必须使用 cache_prefix.source_digest、cache_prefix.profile 和 cache_prefix.instructions。",
            ],
        },
    )


def candidate_page_retry_prompt(
    *,
    digest: SourceDigest,
    page_unit: SourcePageUnit,
    raw_path: str,
    raw_sha256: str,
    raw_text: str,
    profile: Profile,
    previous_pages: CandidatePages | None,
    validation_error: str,
) -> PromptRequest:
    request = candidate_page_prompt(
        digest=digest,
        page_unit=page_unit,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        raw_text=raw_text,
        profile=profile,
    )
    payload = dict(request.user_payload)
    instructions = list(payload.get("instructions") or [])
    payload.update(
        {
            "validation_error": validation_error,
            "previous_invalid_output": previous_pages.model_dump(mode="json") if previous_pages is not None else None,
            "instructions": [
                *instructions,
                "上一轮 candidate_pages 输出没有通过系统校验；本轮必须返回修正后的完整 JSON，不要解释。",
                "顶层必须是对象，必须包含 pages 数组和 skipped_page_unit_ids 数组。",
                "pages 数组必须且只能包含 1 个候选页。",
                "该候选页必须对应 expected_page_unit_id，且必须包含中文 title、summary、body_markdown、evidence_notes 和 source_refs。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


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
                "每个 decision 必须填写 title、page_type、content_scope、candidate_content_locators 和 source_refs。",
                "content_scope 用中文说明这个 decision 消费候选页中的哪一部分内容，避免拆分后重复或遗漏。",
                "candidate_content_locators 必须列出候选页中被此 decision 消费的章节、要点或证据定位。",
                "新知识使用 create，匹配 top-k 旧页使用 update，已经覆盖才使用 noop。",
                "update 的 target_path 必须来自同一 candidate_page_id 的 candidate_contexts.hits.path。",
                "create 的 target_path 必须留在 profile 路由目录内。",
                "不要在 merge_plan 中决定 Related；相关页面由后续计算步骤按 embedding 相似度生成。",
                "重新计算 action_counts，确保它与 decisions 一致。",
            ],
        },
    )


def merge_plan_retry_prompt(
    *,
    candidate_pages: CandidatePages,
    candidate_contexts: CandidateContexts,
    profile: Profile,
    previous_plan: MergePlan,
    validation_error: str,
) -> PromptRequest:
    request = merge_plan_prompt(candidate_pages=candidate_pages, candidate_contexts=candidate_contexts, profile=profile)
    payload = dict(request.user_payload)
    instructions = list(payload.get("instructions") or [])
    allowed_update_targets = {
        item.candidate_page_id: [hit.path for hit in item.hits]
        for item in candidate_contexts.items
    }
    payload.update(
        {
            "allowed_update_targets_by_candidate_page": allowed_update_targets,
            "previous_invalid_output": previous_plan.model_dump(mode="json"),
            "validation_error": validation_error,
            "instructions": [
                *instructions,
                "上一轮 merge_plan 输出没有通过系统校验；本轮必须返回修正后的完整 JSON，不要解释。",
                "update decision 的 target_path 必须来自同一个 candidate_page_id 的 allowed_update_targets_by_candidate_page。",
                "update decision 必须填写 matched_existing_paths，并且至少包含本次 update 的 target_path。",
                "如果候选页只有部分内容能更新旧页，其余新内容必须拆成 create decision，并用 candidate_content_locators 标明各自消费的候选内容。",
                "不要为了通过校验而遗漏 candidate_page；每个 candidate_page_id 仍必须至少有一个 decision。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


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
                "不要在 composition_plan 中决定 Related；相关页面由后续计算步骤按 embedding 相似度生成。",
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
    target_entry = next((entry for entry in snapshot.entries if entry.path == composition_item.target_path), None)
    preimage_requirements = preimage_coverage_requirements(target_entry) if composition_item.action == "update" else []
    allowed_body_link_targets = [
        {
            "path": entry.path,
            "title": entry.title,
            "page_type": entry.page_type,
            "summary": entry.summary,
        }
        for entry in snapshot.entries
        if entry.path != composition_item.target_path
    ]
    return _request(
        step="final_pages",
        model=FinalPages,
        payload={
            "composition_item": composition_item.model_dump(mode="json"),
            "candidate_pages": relevant_candidates.model_dump(mode="json"),
            "wiki_snapshot_entries": [entry.model_dump(mode="json") for entry in relevant_entries],
            "preimage_coverage_requirements": preimage_requirements,
            "allowed_body_link_targets": allowed_body_link_targets,
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
                "update 页面禁止只围绕本次 candidate 重写旧页；必须把 preimage_coverage_requirements 中每条旧页覆盖要求保留或合并进最终正文。",
                "update 页面必须填写 preimage_coverage_report；每个 requirement_id 必须逐条出现一次，status 只能是 preserved 或 merged。",
                "preimage_coverage_report.final_anchor 必须是最终正文中真实存在的具体中文小节或锚点，不能只写“摘要”“核心内容”“矛盾与未决问题”等泛化位置。",
                "preimage_coverage_report.evidence 必须用中文说明旧页内容被保留或合并到了哪里。",
                "正文可以写 0-2 条 Obsidian wikilink，但只在阅读语境确实需要跳转理解时使用，不要为了凑数而链接。",
                "正文 wikilink 必须从 allowed_body_link_targets.path 中选择，不能链接未知页面或本页面。",
                "不要写 Related 或 相关页面章节；引擎会在 final_pages 之后按 embedding 相似度计算唯一 Related。",
                "不要创建自链接。",
                "content_sha256 可以为空；引擎会重新计算。",
                "不要包含模型自我说明。",
            ],
        },
    )


def final_page_retry_prompt(
    *,
    composition_item: CompositionItem,
    candidate_pages: CandidatePages,
    snapshot: WikiSnapshot,
    profile: Profile,
    previous_pages: FinalPages,
    validation_error: str,
) -> PromptRequest:
    request = final_page_prompt(composition_item=composition_item, candidate_pages=candidate_pages, snapshot=snapshot, profile=profile)
    payload = dict(request.user_payload)
    instructions = list(payload.get("instructions") or [])
    payload.update(
        {
            "previous_invalid_output": previous_pages.model_dump(mode="json"),
            "validation_error": validation_error,
            "instructions": [
                *instructions,
                "上一轮 final_pages 输出没有通过系统校验；本轮必须只修正当前 composition_item，并返回完整 JSON。",
                "仍然只能返回 1 页，final_page_id 和 target_path 必须匹配 expected values。",
                "正文 Obsidian wikilink 最多 2 条，且必须来自 allowed_body_link_targets.path；也可以不写正文链接。",
                "正文中不得包含 raw、sources、logs、index、Source_* 或 source page 的 wikilink、Markdown link、HTML href。",
                "不要写 Related 或 相关页面章节；引擎会统一生成。",
                "保留有来源支撑的中文正文和 source_refs，不要为了修复链接而删除核心信息。",
                "如果 validation_error 指出旧页覆盖丢失，必须把对应旧页小节或主题补回最终正文，并更新 preimage_coverage_report。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


def preimage_coverage_requirements(entry: WikiKnowledgeEntry | None) -> list[dict[str, object]]:
    if entry is None:
        return []
    requirements: list[dict[str, object]] = []
    if entry.summary.strip():
        requirements.append(
            {
                "requirement_id": "OLD-SUMMARY",
                "kind": "summary",
                "description": f"旧页摘要覆盖必须保留或合并：{entry.summary.strip()}",
                "source_raw_paths": entry.source_raw_paths,
            }
        )
    for index, heading in enumerate(_coverage_headings(entry.text_excerpt, entry.title), start=1):
        requirements.append(
            {
                "requirement_id": f"OLD-SECTION-{index:03d}",
                "kind": "section",
                "description": f"旧页小节或主题必须保留或合并：{heading}",
                "anchor": heading,
                "source_raw_paths": entry.source_raw_paths,
            }
        )
    if not requirements:
        requirements.append(
            {
                "requirement_id": "OLD-BODY",
                "kind": "body",
                "description": f"旧页 `{entry.path}` 的正文已有知识必须保留或合并，不得被本次 update 清空。",
                "source_raw_paths": entry.source_raw_paths,
            }
        )
    return requirements


def _coverage_headings(markdown: str, title: str) -> list[str]:
    generic = {"摘要", "核心内容", "矛盾与未决问题", "未决问题", "相关页面", "related"}
    headings: list[str] = []
    title_key = _coverage_key(title)
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        level, _, raw_heading = stripped.partition(" ")
        if not raw_heading or len(level) > 3:
            continue
        heading = raw_heading.strip()
        key = _coverage_key(heading)
        if not key or key == title_key or heading.lower() in generic:
            continue
        if heading not in headings:
            headings.append(heading)
        if len(headings) >= 8:
            break
    return headings


def _coverage_key(text: str) -> str:
    import re

    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower()))


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
            "page_units": [
                {
                    "page_unit_id": "PU-001",
                    "title": "示例概念",
                    "page_type": "concept",
                    "path_hint": "concepts/Concept_Example_Concept.md",
                    "summary": "说明这个概念为什么值得写入 wiki。",
                    "content_scope": "覆盖 raw 中关于示例概念的定义、用途和来源依据。",
                    "must_cover_points": ["说明示例概念的定义和来源依据。"],
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                }
            ],
            "weak_or_noise_items": [],
        },
        "candidate_pages": {
            "pages": [
                {
                    "candidate_page_id": "CP-001",
                    "page_unit_id": "PU-001",
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
            "skipped_page_unit_ids": [],
        },
        "candidate_pages_warmup": {"status": "OK"},
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
                    "candidate_content_locators": ["摘要", "核心内容"],
                    "matched_existing_paths": [],
                    "inspected_context_paths": [],
                    "strongest_overlap": 0.0,
                    "reason": "没有已有页面覆盖这个有来源支撑的概念。",
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
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
