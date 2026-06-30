from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .models import (
    CandidateContexts,
    CandidatePages,
    CandidatePagesWarmup,
    ClaimRepairResult,
    CompositionItem,
    CompositionPlan,
    CoverageJudge,
    FinalPages,
    MergePlan,
    PageUpdatePlanItem,
    SourceDigest,
    SourceGranularityStats,
    SourceContentUnit,
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

GENERIC_PREIMAGE_ANCHORS = ["摘要", "核心内容", "矛盾与未决问题", "未决问题", "相关页面", "Related"]


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
                "先从 raw_text 抽取 claims；claim 是有效知识原子，不是页面。",
                "每个有效 claim 必须包含 claim_id、中文 text、kind、importance、concept_terms、raw_locator 和 source_refs。",
                "claim_id 使用 C-001、C-002 这样的稳定格式；importance 取 1-5，5 表示核心知识，1 表示边缘但仍值得保留的信息。",
                "再把 claims 聚合成 content_units；content_unit 是有效内容单元，不一定独立成页。",
                "每个 content_unit 必须包含 content_unit_id、title、content_role、absorption_decision、anchor_unit_id、section_hint、page_type、path_hint、summary、absorption_reason、content_scope、claim_ids、source_refs。",
                "content_unit_id 使用 CU-001、CU-002 这样的稳定格式，不要编造 wiki 旧页路径。",
                "content_role 只能是：主干、附属、工具性。",
                "absorption_decision 只能是：独立成页、并入主干、降级为段落。",
                "所有有效 content_unit 必须有 anchor_unit_id；主干的 anchor_unit_id 必须指向自己，附属/工具性内容的 anchor_unit_id 必须指向一个真实存在的主干 content_unit。",
                "只有 content_role=主干 且 absorption_decision=独立成页 的 content_unit 会生成候选页；附属和工具性内容会随 anchor 主干一起吸收。",
                "如果 raw 中只有安装、价格、API 生命周期、experimental 单点、单 claim 比较、论文 benchmark 数字等附属/工具性内容，也必须先规划一个可承载它的主干对象，再把这些内容挂到该 anchor 下。",
                "优先使用稳定主干对象：概念、框架、协议、架构、机制、方法论、长期可维护能力。",
                "默认不要让单工具、安装指南、价格、API 生命周期、experimental 特性、单一比较、论文实验数字独立成页；它们通常应并入主干或降级为段落。",
                "如果提供了 granularity_stats，主干候选页数量应落在 suggested_min_candidate_pages 到 suggested_max_candidate_pages 的经验区间附近；超出区间必须减少主干 anchor 数量。",
                "在本步骤内完成去重、近义合并、主干/附属/工具性判断和归属确认；不要额外输出候选项再交给后续合并。",
                "page_type 必须来自 profile.page_types，且不能是 source；附属/工具性 content_unit 的 page_type/path_hint 应与其 anchor 主干保持一致。",
                "path_hint 必须位于 page_type 对应目录内，且以 .md 结尾。",
                "content_units 必须通过 claim_ids 覆盖 raw_text 中所有值得进入知识库的有效 claims；不要把重要信息漏掉。",
                "每个 claim_id 必须且只能归入一个 content_unit；如果一个 claim 完全无法归属，不要静默忽略，应调整 content_units。",
                "每个 content_unit 必须包含至少一个 source_ref，包含 raw_path、raw_sha256 和有用 locator。",
                "如果 raw_text 明显是 404、Page not found、不可访问、空页面、导航页或只有网站菜单，不要把“页面失效”写成 event/知识页；claims 和 content_units 必须为空，只在 weak_or_noise_items 说明原因。",
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
                "所有 summary、key_takeaways、claim text、title、content_scope、weak_or_noise reason 等用户可读字段必须使用中文。",
                "必要产品名、框架名、API 名可以保留英文专有名词，但解释性句子必须是中文。",
                "source_raw_path 和 raw_sha256 必须保持与输入完全一致。",
                "不要为了通过中文校验而删除有价值 content_unit；应把英文说明改写成中文说明。",
                "如果 validation_error 指出主干候选页过多，请减少 content_role=主干 的数量，把附属或工具性内容挂到已有 anchor。",
                "如果 validation_error 指出 0 个候选页但原文不是噪声，请生成至少一个主干 content_unit，并让 anchor_unit_id 指向自己。",
                "如果 validation_error 指出 raw 只能记录来源，请清空 claims 和 content_units，只保留 weak_or_noise_items 说明 404、不可访问、空页面或导航噪声原因。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


def candidate_page_prompt(
    *,
    digest: SourceDigest,
    anchor_unit: SourceContentUnit,
    attached_units: list[SourceContentUnit],
    raw_path: str,
    raw_sha256: str,
    raw_text: str,
    profile: Profile,
) -> PromptRequest:
    return _request(
        step="candidate_pages",
        model=CandidatePages,
        cache_prefix_payload=_candidate_page_cache_prefix(digest=digest, raw_path=raw_path, raw_sha256=raw_sha256, raw_text=raw_text, profile=profile),
        payload={
            "anchor_content_unit": anchor_unit.model_dump(mode="json"),
            "attached_content_units": [unit.model_dump(mode="json") for unit in attached_units],
            "parallel_generation_contract": {
                "mode": "per_anchor_content_unit",
                "expected_content_unit_id": anchor_unit.content_unit_id,
                "expected_page_count": 1,
            },
            "instructions": [
                "这是并发 candidate-page 生成请求组中的一个请求。",
                "只为 anchor_content_unit 生成一个候选页面。",
                "返回的 pages 数组必须只有一个 item。",
                "page.content_unit_id 必须等于 expected_content_unit_id。",
                "必须对照 cache_prefix.raw_text 写正文。",
                "必须覆盖 anchor_content_unit 和 attached_content_units 中所有 claim_ids 对应的 source_digest.claims，不要只复述 summary。",
                "attached_content_units 中 absorption_decision=并入主干 的内容应写入主干页相应小节；absorption_decision=降级为段落 的内容应按 section_hint 压缩为段落或备注。",
                "不得把 attached_content_units 写成单独候选页。",
                "必须使用 cache_prefix.source_digest、cache_prefix.profile 和 cache_prefix.instructions。",
            ],
        },
    )


def candidate_page_retry_prompt(
    *,
    digest: SourceDigest,
    anchor_unit: SourceContentUnit,
    attached_units: list[SourceContentUnit],
    raw_path: str,
    raw_sha256: str,
    raw_text: str,
    profile: Profile,
    previous_pages: CandidatePages | None,
    validation_error: str,
) -> PromptRequest:
    request = candidate_page_prompt(
        digest=digest,
        anchor_unit=anchor_unit,
        attached_units=attached_units,
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
                "顶层必须是对象，必须包含 pages 数组和 skipped_content_unit_ids 数组。",
                "pages 数组必须且只能包含 1 个候选页。",
                "该候选页必须对应 expected_content_unit_id，且必须包含中文 title、summary、body_markdown、evidence_notes 和 source_refs。",
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
                "当某个 candidate_page 的 top1 旧页 score >= 0.80 时，默认必须至少输出一个 update decision 指向该 top1 旧页；如果只有部分内容重叠，可以同时输出 update 和 create。",
                "当 top1 score >= 0.80 且你认为旧页已完整覆盖候选内容时，输出 noop decision 并用 target_path 或 matched_existing_paths 指向该 top1 旧页；禁止只输出 create。",
                "create decision 仍必须填写 inspected_context_paths 和 strongest_overlap，并在 reason 中说明为什么不能更新已检查旧页。",
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
    writable_targets = sorted({decision.target_path for decision in merge_plan.decisions if decision.action != "noop" and decision.target_path})
    return _request(
        step="composition_plan",
        model=CompositionPlan,
        payload={
            "merge_plan": merge_plan.model_dump(mode="json"),
            "candidate_pages": candidate_pages.model_dump(mode="json"),
            "expected_writable_targets": writable_targets,
            "profile": profile.model_dump(mode="json"),
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "每个可写 create/update decision 必须进入一个 composition item；同一 target_path 的多个 decision 可以合并为一个 item。",
                "composition item 的 target_path 必须逐字使用 expected_writable_targets 中的路径，禁止翻译、改写、重命名或重新生成路径。",
                "不要为 noop decisions 创建 composition item。",
                "merge_decision_ids 必须覆盖进入该 item 的 merge decision。",
                "candidate_page_ids 必须来自对应 merge decisions。",
                "update target 必须保留有价值的旧内容。",
                "每个 item 必须包含 source_ref_rules，且规则文本用中文。",
                "不要在 composition_plan 中决定 Related；相关页面由后续计算步骤按 embedding 相似度生成。",
            ],
        },
    )


def composition_plan_retry_prompt(
    *,
    merge_plan: MergePlan,
    candidate_pages: CandidatePages,
    profile: Profile,
    previous_plan: CompositionPlan,
    validation_error: str,
) -> PromptRequest:
    request = composition_plan_prompt(merge_plan=merge_plan, candidate_pages=candidate_pages, profile=profile)
    payload = dict(request.user_payload)
    instructions = list(payload.get("instructions") or [])
    payload.update(
        {
            "previous_invalid_output": previous_plan.model_dump(mode="json"),
            "validation_error": validation_error,
            "instructions": [
                *instructions,
                "上一轮 composition_plan 输出没有通过系统校验；本轮必须返回修正后的完整 JSON，不要解释。",
                "不要改变 merge_plan 已决定的 target_path；只能从 expected_writable_targets 逐字复制。",
                "如果 validation_error 中出现 expected/actual，以 expected 为准重建 composition items。",
                "仍然必须覆盖所有可写 merge_decision_ids，不得遗漏任何 create/update decision。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


def final_page_prompt(
    *,
    composition_item: CompositionItem,
    candidate_pages: CandidatePages,
    snapshot: WikiSnapshot,
    profile: Profile,
    page_update_plan_item: PageUpdatePlanItem | None = None,
) -> PromptRequest:
    candidate_ids = set(composition_item.candidate_page_ids)
    relevant_candidates = CandidatePages(pages=[page for page in candidate_pages.pages if page.candidate_page_id in candidate_ids])
    relevant_paths = {composition_item.target_path, *composition_item.existing_page_refs}
    relevant_entries = [entry for entry in snapshot.entries if entry.path in relevant_paths]
    target_entry = next((entry for entry in snapshot.entries if entry.path == composition_item.target_path), None)
    has_page_state_claims = bool(page_update_plan_item and page_update_plan_item.previous_active_claims)
    preimage_requirements = [] if has_page_state_claims else preimage_coverage_requirements(target_entry) if composition_item.action == "update" else []
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
            "page_update_plan": page_update_plan_item.model_dump(mode="json") if page_update_plan_item is not None else None,
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
                "page_update_plan 是页面维护状态计划；旧页面 Markdown 才是旧内容语境，page_update_plan 只用于说明结构责任、旧活跃知识点和新增内容应该如何补入。",
                "如果 page_update_plan.previous_active_claims 非空，它们取代 preimage_coverage_requirements；不要从旧知识点重新生成旧内容，而要以旧页面正文为语境，确保这些处于“活跃”状态的旧知识没有无声消失。",
                "如果 page_update_plan.incoming_content_units 非空，应按其 content_role、absorption_decision、anchor_unit_id 和 section_hint 把新 candidate 内容补入对应主干或分支，不要直接覆盖旧页。",
                "如果 preimage_coverage_requirements 非空，说明本页还没有 page_state，必须把其中每条旧页覆盖要求保留或合并进最终正文。",
                "只有 preimage_coverage_requirements 非空时才需要填写 preimage_coverage_report；每个 requirement_id 必须逐条出现一次，status 只能是 preserved 或 merged。若 preimage_coverage_requirements 为空，preimage_coverage_report 可以为空数组。",
                "preimage_coverage_report.final_anchor 必须是最终正文中真实存在的具体中文小节或锚点，不能只写“摘要”“核心内容”“矛盾与未决问题”等泛化位置。",
                "preimage_coverage_report.final_anchor 必须是读者可见的中文标题文本；不要填写 HTML anchor id、英文 slug、#锚点 或隐藏标记。",
                "OLD-SUMMARY 也不能把 final_anchor 写成“摘要”；如果旧页摘要被合入摘要，必须在正文中保留或新建一个更具体的中文小节承接该旧知识，并把 final_anchor 指向这个具体小节。",
                "preimage_coverage_requirements 中若有 anchor_candidates，优先从其中选择已保留的具体小节；如果小节被合并或改名，final_anchor 必须写最终正文里承接该旧知识的具体中文小节。",
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
    page_update_plan_item: PageUpdatePlanItem | None = None,
    coverage_repair_claims: list[dict[str, object]] | None = None,
) -> PromptRequest:
    request = final_page_prompt(
        composition_item=composition_item,
        candidate_pages=candidate_pages,
        snapshot=snapshot,
        profile=profile,
        page_update_plan_item=page_update_plan_item,
    )
    payload = dict(request.user_payload)
    instructions = list(payload.get("instructions") or [])
    if coverage_repair_claims:
        payload["coverage_repair_claims"] = coverage_repair_claims
        instructions.extend(
            [
                "这是 coverage_judge 后的覆盖修复请求。",
                "必须优先补齐 coverage_repair_claims 中 status/judge_status 为 partial、missing 或 contradicted 的知识点。",
                "每条 coverage_repair_claims.claim.text 都必须在最终正文中有明确中文落点；不要只写泛化总结。",
                "如果 claim 包含数字、限制条件、例子、机制或对象，修复后必须保留这些关键信息。",
                "不要删除已经正确覆盖的内容；只做必要补充和局部改写。",
            ]
        )
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
                "如果 validation_error 指出 final_anchor 有问题，不要改成“摘要”“核心内容”等泛化位置；必须选择或新增一个读者可见的具体中文小节标题。",
                "final_anchor 必须精确等于最终 Markdown 中出现的中文标题文本，不要写 HTML anchor id、英文 slug、路径或 #锚点。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


def coverage_judge_prompt(*, digest: SourceDigest, final_pages: FinalPages) -> PromptRequest:
    return _request(
        step="coverage_judge",
        model=CoverageJudge,
        payload={
            "source_raw_path": digest.source_raw_path,
            "raw_sha256": digest.raw_sha256,
            "claims": [claim.model_dump(mode="json") for claim in digest.claims],
            "content_units": [unit.model_dump(mode="json") for unit in digest.content_units],
            "final_pages": [
                {
                    "final_page_id": page.final_page_id,
                    "target_path": page.target_path,
                    "title": page.title,
                    "action": page.action,
                    "markdown": page.markdown,
                    "source_refs": [ref.model_dump(mode="json") for ref in page.source_refs],
                }
                for page in final_pages.pages
            ],
            "allowed_status": ["covered", "partial", "missing", "contradicted"],
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "逐条判断 claims 是否被 final_pages 明确覆盖。",
                "claim_results 必须且只能包含每个 claim_id 一条结果，不能遗漏、不能新增。",
                "covered：最终页面明确写入了 claim 的事实含义，允许中文改写。",
                "partial：最终页面写到了主题但缺少 claim 的关键限定、数字、例子、机制或对象。",
                "missing：最终页面没有覆盖该 claim。",
                "contradicted：最终页面与 claim 含义冲突。",
                "covered_by 写 target_path#中文小节；如果缺失或冲突可为空数组。",
                "evidence 必须引用最终页面中的中文表达；reason 必须中文说明判断理由。",
                "evidence 和 reason 都不能为空；即使 status 是 missing、partial 或 contradicted，也必须写明最终页面没有覆盖、部分覆盖或冲突的具体依据。",
                "不要因为 source_refs 或 frontmatter 包含 raw 路径就判 covered；必须看正文内容。",
            ],
        },
    )


def coverage_judge_retry_prompt(
    *,
    digest: SourceDigest,
    final_pages: FinalPages,
    previous_judge: CoverageJudge,
    validation_error: str,
) -> PromptRequest:
    request = coverage_judge_prompt(digest=digest, final_pages=final_pages)
    payload = dict(request.user_payload)
    instructions = list(payload.get("instructions") or [])
    payload.update(
        {
            "previous_invalid_output": previous_judge.model_dump(mode="json"),
            "validation_error": validation_error,
            "instructions": [
                *instructions,
                "上一轮 coverage_judge 输出没有通过系统校验；本轮必须返回修正后的完整 JSON，不要解释。",
                "claim_results 的 claim_id 集合必须与输入 claims 完全一致。",
                "每条 claim_results 的 evidence 和 reason 都不能为空。",
                "所有 evidence、reason、warnings 等用户可读字段必须使用中文。",
            ],
        }
    )
    return request.model_copy(update={"user_payload": payload})


def claim_repair_prompt(
    *,
    raw_path: str,
    raw_sha256: str,
    raw_text: str,
    digest: SourceDigest,
    coverage_report: dict[str, object],
    repair_claim_ids: list[str],
) -> PromptRequest:
    return _request(
        step="claim_repair",
        model=ClaimRepairResult,
        payload={
            "raw_path": raw_path,
            "raw_sha256": raw_sha256,
            "raw_text": raw_text,
            "source_digest": digest.model_dump(mode="json"),
            "coverage_report": coverage_report,
            "repair_claim_ids": repair_claim_ids,
            "instructions": [
                *CHINESE_OUTPUT_RULES,
                "这是 coverage_judge 发现 source_digest claim 可能有事实错误后的局部修复请求。",
                "只允许为 repair_claim_ids 中的 claim 返回 patches；不要返回未点名 claim 的 patch。",
                "必须以 raw_text 为最高优先级证据判断原 claim 是否真的错误，不要盲目信任 final_pages 或 coverage_judge。",
                "如果 raw_text 支持原 claim，只返回空 patches，并在 warnings 用中文说明应该修最终页而不是修 claim。",
                "如果原 claim 把范围、数量、对象、机制、限制条件写错或写窄，则返回 replacement_claim。",
                "replacement_claim.claim_id 必须与 patch.claim_id 完全一致；不要改 claim_id，不要拆分 claim，不要合并 claim。",
                "replacement_claim.source_refs 只能引用当前 raw_path/raw_sha256，raw_locator 必须指向 raw_text 中支撑修正的段落、小节或片段。",
                "replacement_claim.text、concept_terms、reason、warnings 等用户可读字段必须使用中文。",
                "未出现在 repair_claim_ids 中的 claims 必须由引擎原样保留；你不要尝试重写它们。",
            ],
        },
    )


def preimage_coverage_requirements(entry: WikiKnowledgeEntry | None) -> list[dict[str, object]]:
    if entry is None:
        return []
    requirements: list[dict[str, object]] = []
    anchor_candidates = _coverage_headings(entry.text_excerpt, entry.title)
    if entry.summary.strip():
        requirements.append(
            {
                "requirement_id": "OLD-SUMMARY",
                "kind": "summary",
                "description": f"旧页摘要覆盖必须保留或合并：{entry.summary.strip()}",
                "anchor_candidates": anchor_candidates,
                "forbidden_final_anchors": GENERIC_PREIMAGE_ANCHORS,
                "final_anchor_rule": "OLD-SUMMARY 的 final_anchor 不能写“摘要”或隐藏锚点；必须指向最终正文中承接旧页摘要知识的具体中文小节。",
                "source_raw_paths": entry.source_raw_paths,
            }
        )
    for index, heading in enumerate(anchor_candidates, start=1):
        requirements.append(
            {
                "requirement_id": f"OLD-SECTION-{index:03d}",
                "kind": "section",
                "description": f"旧页小节或主题必须保留或合并：{heading}",
                "anchor": heading,
                "forbidden_final_anchors": GENERIC_PREIMAGE_ANCHORS,
                "final_anchor_rule": "如果旧小节被保留，final_anchor 优先写该中文小节名；如果被合并或改名，必须写最终正文里承接该旧知识的具体中文小节名。",
                "source_raw_paths": entry.source_raw_paths,
            }
        )
    if not requirements:
        requirements.append(
            {
                "requirement_id": "OLD-BODY",
                "kind": "body",
                "description": f"旧页 `{entry.path}` 的正文已有知识必须保留或合并，不得被本次 update 清空。",
                "anchor_candidates": anchor_candidates,
                "forbidden_final_anchors": GENERIC_PREIMAGE_ANCHORS,
                "final_anchor_rule": "final_anchor 必须指向最终正文中承接旧页正文知识的具体中文小节；不能写“摘要”或隐藏锚点。",
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
            "claims": [
                {
                    "claim_id": "C-001",
                    "text": "示例概念说明了一个有来源支撑的定义和用途。",
                    "kind": "concept",
                    "importance": 4,
                    "concept_terms": ["示例概念"],
                    "raw_locator": "whole_file",
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                }
            ],
            "content_units": [
                {
                    "content_unit_id": "CU-001",
                    "title": "示例概念",
                    "content_role": "主干",
                    "absorption_decision": "独立成页",
                    "anchor_unit_id": "CU-001",
                    "section_hint": "核心概念",
                    "page_type": "concept",
                    "path_hint": "concepts/Concept_Example_Concept.md",
                    "summary": "说明这个概念为什么值得写入 wiki。",
                    "absorption_reason": "该内容是 raw 的主要知识对象，适合作为长期维护的知识页。",
                    "content_scope": "覆盖 raw 中关于示例概念的定义、用途和来源依据。",
                    "claim_ids": ["C-001"],
                    "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                }
            ],
            "weak_or_noise_items": [],
        },
        "candidate_pages": {
            "pages": [
                {
                    "candidate_page_id": "CP-001",
                    "content_unit_id": "CU-001",
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
            "skipped_content_unit_ids": [],
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
        "coverage_judge": {
            "claim_results": [
                {
                    "claim_id": "C-001",
                    "status": "covered",
                    "covered_by": ["concepts/Concept_Example_Concept.md#摘要"],
                    "evidence": "最终页面说明了示例概念的定义和用途。",
                    "reason": "正文明确覆盖了 claim 的事实含义。",
                }
            ],
            "warnings": [],
        },
        "claim_repair": {
            "patches": [
                {
                    "claim_id": "C-001",
                    "replacement_claim": {
                        "claim_id": "C-001",
                        "text": "示例概念包含定义、用途和限制条件，需要按 raw 证据完整表述。",
                        "kind": "concept",
                        "importance": 4,
                        "concept_terms": ["示例概念"],
                        "raw_locator": "whole_file",
                        "source_refs": [{"raw_path": "raw/example.md", "raw_sha256": "sha256", "locator": "whole_file"}],
                    },
                    "reason": "原 claim 遗漏了 raw 中明确出现的限制条件。",
                }
            ],
            "warnings": [],
        },
    }
    return examples.get(step, {})
