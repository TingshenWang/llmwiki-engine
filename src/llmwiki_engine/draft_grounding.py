from __future__ import annotations

import re
import unicodedata
from typing import Any

from . import draft_validation as _draft_validation
from . import markdown_utils as _markdown_utils
from . import page_sections as _page_sections
from . import source_excerpt as _source_excerpt
from . import update_preservation as _update_preservation
from .models import (
    DraftGroundingReview,
    DraftPageItem,
    DraftRenderingArtifact,
    GroundingClaim,
    WikiContextEntry,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
)
from .system_pages import format_markdown_table
from .validators import looks_like_untranslated_english
from .wiki_context import snapshot_entry

DRAFT_RENDERING_GROUNDING_RISK_RULES = (
    "Do not wrap paraphrases, inferred concept labels, or rewritten source ideas in Chinese/English quotation marks; "
    "use quotes only for text that exact-matches source_excerpt_pack, approved_prepared_markdown, or inspected wiki context.",
    "For conversational source text, treat speaker-like wording as paraphrase unless the exact span is present; "
    "prefer indirect attribution such as 访谈中提到、她描述、团队讨论.",
    "The user has already approved this source for ingest. Do not reject or avoid a domain just because it is medical, legal, financial, security, account, password, payment, or privacy related.",
    "The only hard grounding boundary is contradiction with the approved source: absence of support is a warning, not a blocker.",
    "When turning source-local capabilities or examples into popularity/adoption/authority claims, prefer source-local "
    "wording such as 本材料提到、访谈中讨论、团队成员提到、本材料将该说法用于解释 unless the source/wiki context clearly supports broader phrasing.",
    "In open_questions, unsupported adoption/authority premises such as 公认、最佳实践、行业最佳、广泛采用、业界普遍 are allowed as hypotheses, "
    "but phrase them with uncertainty or 待补来源 when the source does not establish them.",
    "For causal/scope terms such as 导致、造成、证明、表明、必然、长期来看、用户会、影响到, keep the wording proportional to the source. "
    "If the source only gives a tradeoff or concern, write 可能伴随、需要权衡、访谈中提到, or move the claim to open_questions.",
)
UNSUPPORTED_BACKING_MARKERS = (
    "被广泛应用",
    "被广泛使用",
    "被广泛采用",
    "广泛应用",
    "广泛使用",
    "广泛采用",
    "被多个",
    "被广泛",
    "公认",
    "最佳实践",
    "行业最佳",
    "业界普遍",
    "多个社区",
)
EXTERNAL_BACKING_EQUIVALENTS = (
    "widely used",
    "widely adopted",
    "widely adopt",
    "widely applied",
    "widely across tasks",
    "use widely",
    "used widely",
    "commonly used",
    "extensively used",
    "broadly used",
    "de-facto standard",
    "de facto standard",
)
EXTERNAL_BACKING_ZH_EN_ANCHORS = (
    ("评估", "evaluat"),
    ("基准", "benchmark"),
    ("智能体", "agent"),
    ("模型", "model"),
    ("网络安全", "cybersecurity"),
    ("安全", "security"),
    ("作弊", "cheat"),
    ("污染", "contamination"),
    ("产品", "product"),
    ("开发", "develop"),
)
EXTERNAL_BACKING_GENERIC_ANCHORS = {
    "agent",
    "agents",
    "benchmark",
    "benchmarks",
    "bench",
    "evaluat",
    "evaluation",
    "evaluating",
    "framework",
    "frameworks",
    "llm",
    "llms",
    "model",
    "models",
    "product",
    "products",
    "system",
    "systems",
    "develop",
}


def render_grounding_paraphrase_rewrite_report(report: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for page in report.get("pages", []):
        if not isinstance(page, dict):
            continue
        for field in page.get("fields", []):
            if not isinstance(field, dict):
                continue
            for rewrite in field.get("rewrites", []):
                if not isinstance(rewrite, dict):
                    continue
                rows.append(
                    [
                        page.get("page_plan_id", ""),
                        page.get("target_path", ""),
                        field.get("field", ""),
                        rewrite.get("original_quote", ""),
                        rewrite.get("replacement", ""),
                        rewrite.get("source_sentence", ""),
                    ]
                )
    return (
        "# Grounding Paraphrase Rewrite Report\n\n"
        f"- Changed: `{str(bool(report.get('changed'))).lower()}`\n"
        f"- Rewrite count: `{report.get('rewrite_count', 0)}`\n\n"
        + (
            format_markdown_table(["页面计划", "目标", "段落", "原引号短语", "替换文本", "来源句"], rows)
            if rows
            else "_无需本地改写。_"
        )
        + "\n"
    )


def draft_grounding_sections(page: DraftPageItem) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = [("summary", _update_preservation.draft_page_summary(page))]
    sections.extend(body_markdown_grounding_sections(_update_preservation.draft_page_core_markdown(page)))
    sections.append(("open_questions", _update_preservation.draft_page_open_questions(page)))
    return [(section_key, body) for section_key, body in sections if body.strip()]


def body_markdown_grounding_sections(body: str) -> list[tuple[str, str]]:
    body = body.strip()
    if not body:
        return []
    chunks: list[tuple[str, list[str]]] = [("detail", [])]
    current_lines = chunks[0][1]
    fence_char = ""
    fence_length = 0
    for line in body.splitlines():
        if fence_char:
            current_lines.append(line)
            if _draft_validation.closing_fence_line(line, fence_char, fence_length):
                fence_char = ""
                fence_length = 0
            continue
        if match := _draft_validation.opening_fence_line(line):
            marker = match.group("marker")
            fence_char = marker[0]
            fence_length = len(marker)
            current_lines.append(line)
            continue
        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading_match:
            title = re.sub(r"\s+", " ", heading_match.group(1).strip()).strip("#:： ")
            section_key = draft_body_heading_section_key(title)
            current_lines = [title] if draft_body_heading_title_should_scan(title, section_key) else []
            chunks.append((section_key, current_lines))
            continue
        current_lines.append(line)
    merged: dict[str, str] = {}
    for section_key, lines in chunks:
        text = "\n".join(lines).strip()
        if text:
            merged[section_key] = _markdown_utils.merge_markdown_blocks(merged.get(section_key, ""), text)
    return list(merged.items()) or [("detail", body)]


def draft_body_heading_section_key(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title.strip()).strip("#:： ")
    mapped = _page_sections.SECTION_TITLE_TO_KEY.get(normalized)
    if mapped is None and normalized.isascii():
        mapped = _page_sections.ENGLISH_SECTION_TITLE_TO_KEY.get(normalized.casefold())
    if mapped in {"examples", "value_points", "additional_notes"}:
        return mapped
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", normalized)).casefold()
    lowered = unicodedata.normalize("NFKC", normalized).casefold()
    if any(marker in compact for marker in ["例子", "示例", "案例", "使用场景"]):
        return "examples"
    if re.search(r"\b(?:examples?|use cases?|case studies|scenarios?)\b", lowered):
        return "examples"
    if any(marker in compact for marker in ["价值点", "价值", "意义", "为什么重要"]):
        return "value_points"
    if re.search(r"\b(?:value points?|why it matters|importance)\b", lowered):
        return "value_points"
    if any(marker in compact for marker in ["补充观察", "补充", "观察", "备注"]):
        return "additional_notes"
    if re.search(r"\b(?:additional notes?|notes?|observations?)\b", lowered):
        return "additional_notes"
    return "detail"


def draft_body_heading_title_should_scan(title: str, section_key: str) -> bool:
    normalized = re.sub(r"\s+", " ", title.strip()).strip("#:： ")
    if not normalized:
        return False
    if section_key == "detail":
        return True
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", normalized)).casefold()
    structural_titles = {
        "例子",
        "示例",
        "案例",
        "使用场景",
        "examples",
        "example",
        "usecases",
        "usecase",
        "scenarios",
        "scenario",
        "价值点",
        "价值",
        "为什么重要",
        "valuepoints",
        "valuepoint",
        "whyitmatters",
        "importance",
        "补充观察",
        "补充",
        "观察",
        "备注",
        "additionalnotes",
        "additionalnote",
        "notes",
        "note",
        "observations",
        "observation",
    }
    if compact in structural_titles:
        return False
    return bool(
        severe_factual_claim_marker(normalized)
        or unsupported_backing_marker(normalized)
        or contains_hard_fact_marker(compact)
        or contains_short_fact_marker(compact)
        or re.search(r"\d", normalized)
    )


def grounding_issue_message(claim: GroundingClaim) -> str:
    reason = claim.reason or "unsupported new_fact"
    text = re.sub(r"\s+", " ", claim.text).strip()
    if claim.section_key == "examples":
        reason = (
            f"{reason} 例子区不应换一个具体用户事实继续尝试；"
            "请改成抽象占位符（如 `某个用户`、`用户偏好 X`、`user_id`、`memory`）或删除该例子。"
            "如果是 CLI/API/code 示例，命令参数要么照抄来源 literal，要么改成 `<memory_text>`、`<user_id>`、`<memory_query>` 这类占位符；"
            "不要把被拒绝的具体偏好、用户 ID、查询或命令参数换成另一个具体值。"
        )
    if grounding_external_backing_issue(claim):
        reason = (
            f"{reason} 这是非阻塞提醒："
            "采用度、流行度、行业共识或最佳实践这类 adoption/authority 表达最好有来源意识；"
            "如果想更严谨，可以改成 source-local 表达（如 本材料提到、访谈中讨论、材料将其作为例子）。"
        )
    if not text:
        return reason
    return f"{reason} 触发文本：{text[:240]}"


def grounding_external_backing_issue(claim: GroundingClaim) -> bool:
    return (
        claim.support == "unsupported"
        and claim.action in {"needs_review", "warn"}
        and "新增外部背书/强事实标记" in claim.reason
    )


def unsupported_backing_marker(text: str) -> str | None:
    markers = unsupported_backing_markers(text)
    return markers[0] if markers else None


def unsupported_backing_markers(text: str) -> list[str]:
    markers: list[str] = []
    for marker in UNSUPPORTED_BACKING_MARKERS:
        if marker == "被多个" and not re.search(
            r"被多个(?:社区|团队|公司|机构|组织|项目|产品|用户|客户|开发者|研究|论文|媒体|开源项目).{0,12}(?:引用|采用|使用|验证|复现|报道|认可|采纳)",
            text,
        ):
            continue
        if marker in text:
            markers.append(marker)
    return markers


def external_backing_supported_by_context(
    text: str,
    marker: str,
    approved_raw_text: str,
    existing_wiki_text: str,
) -> tuple[bool, str | None]:
    if grounding_text_supported_by_context(text, approved_raw_text, existing_wiki_text):
        return True, "raw" if quote_supported_by_text(text, approved_raw_text) else "existing_wiki"
    if external_backing_supported_by_text(text, marker, approved_raw_text):
        return True, "raw"
    if external_backing_supported_by_text(text, marker, existing_wiki_text):
        return True, "existing_wiki"
    if external_backing_supported_by_retained_existing_fact(text, marker, existing_wiki_text):
        return True, "existing_wiki"
    return False, None


def external_backing_supported_by_retained_existing_fact(text: str, marker: str, existing_wiki_text: str) -> bool:
    if not text or not marker or not existing_wiki_text:
        return False
    marker_equivalents = _markdown_utils.dedupe_strings(
        [_source_excerpt.normalized_source_match_text(marker), *[_source_excerpt.normalized_source_match_text(phrase) for phrase in EXTERNAL_BACKING_EQUIVALENTS]]
    )
    specific_anchors, generic_anchors = external_backing_topic_anchors(text)
    normalized_text = _source_excerpt.normalized_source_match_text(text)
    bridge_anchors = [
        _source_excerpt.normalized_source_match_text(anchor)
        for anchor in [
            "旧页",
            "旧页视角",
            "existing wiki",
            "Managed Agents / 托管智能体",
            "Managed Agents",
            "托管智能体",
        ]
    ]
    has_bridge_context = any(anchor and anchor in normalized_text for anchor in bridge_anchors)
    if not has_bridge_context:
        return False
    useful_specific = [anchor for anchor in specific_anchors if anchor not in {"managed", "agents"}]
    all_anchors = [*useful_specific, *generic_anchors]
    if len(all_anchors) < 2:
        return False
    for sentence in external_backing_source_sentences(existing_wiki_text):
        normalized_sentence = _source_excerpt.normalized_source_match_text(sentence)
        if not any(equivalent and equivalent in normalized_sentence for equivalent in marker_equivalents):
            continue
        specific_hits = external_backing_anchor_hit_count(useful_specific, normalized_sentence)
        all_hits = external_backing_anchor_hit_count(all_anchors, normalized_sentence)
        if specific_hits >= 1 and all_hits >= 2:
            return True
    return False


def external_backing_supported_by_text(text: str, marker: str, source_text: str) -> bool:
    if not text or not marker or not source_text:
        return False
    marker_equivalents = _markdown_utils.dedupe_strings(
        [_source_excerpt.normalized_source_match_text(marker), *[_source_excerpt.normalized_source_match_text(phrase) for phrase in EXTERNAL_BACKING_EQUIVALENTS]]
    )
    specific_anchors, generic_anchors = external_backing_topic_anchors(text)
    if not specific_anchors:
        return False
    all_anchors = [*specific_anchors, *generic_anchors]
    if len(all_anchors) < 2:
        return False
    sentences = external_backing_source_sentences(source_text)
    for index, sentence in enumerate(sentences):
        normalized_sentence = _source_excerpt.normalized_source_match_text(sentence)
        if not any(equivalent and equivalent in normalized_sentence for equivalent in marker_equivalents):
            continue
        context = " ".join(sentences[max(0, index - 1) : index + 2])
        normalized_context = _source_excerpt.normalized_source_match_text(context)
        if external_backing_anchor_hit_count(specific_anchors, normalized_context) < 1:
            continue
        if external_backing_anchor_hit_count(all_anchors, normalized_context) >= 2:
            return True
    return False


def external_backing_topic_anchors(text: str) -> tuple[list[str], list[str]]:
    specific_anchors: list[str] = []
    generic_anchors: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.+_-]{1,}", unicodedata.normalize("NFKC", text)):
        normalized = token.lower().strip("._-+")
        if len(normalized) < 2 or normalized in {"the", "and", "for", "with", "from", "into", "this", "that"}:
            continue
        target = generic_anchors if normalized in EXTERNAL_BACKING_GENERIC_ANCHORS else specific_anchors
        if normalized not in target:
            target.append(normalized)
    for zh_marker, english_anchor in EXTERNAL_BACKING_ZH_EN_ANCHORS:
        if zh_marker in text and english_anchor not in generic_anchors:
            generic_anchors.append(english_anchor)
    return specific_anchors[:6], generic_anchors[:8]


def external_backing_anchor_hit_count(anchors: list[str], normalized_sentence: str) -> int:
    hits = 0
    for anchor in anchors:
        if anchor and anchor in normalized_sentence:
            hits += 1
    return hits


def external_backing_source_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in re.split(r"(?<=[。！？!?\.])\s+|\n+", normalized) if sentence.strip()]


def sentence_with_marker(text: str, marker: str) -> str:
    marker_index = text.find(marker)
    stripped = text.strip(" -*\t")
    if marker_index < 0:
        return stripped
    boundary_chars = "。！？!?；;\n"
    start = 0
    for index in range(marker_index - 1, -1, -1):
        if text[index] in boundary_chars:
            start = index + 1
            break
    end = len(text)
    for index in range(marker_index + len(marker), len(text)):
        if text[index] in boundary_chars:
            end = index + 1
            break
    return text[start:end].strip(" -*\t") or stripped


def unsupported_scope_speculation_marker(text: str) -> str | None:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    speculative = ("可能", "也许", "或许", "推测", "疑似")
    impact = ("受到影响", "受影响", "波及", "涉及", "牵涉", "导致", "造成", "影响到", "关联")
    if not any(marker in compact for marker in speculative):
        return None
    for marker in impact:
        if marker in compact:
            return marker
    return None


def scope_speculation_supported_by_context(text: str, approved_raw_text: str, existing_wiki_text: str) -> tuple[bool, str | None]:
    if grounding_text_supported_by_context(text, approved_raw_text, existing_wiki_text):
        return True, "raw" if quote_supported_by_text(text, approved_raw_text) else "existing_wiki"
    if scope_speculation_supported_by_text(text, approved_raw_text):
        return True, "raw"
    if scope_speculation_supported_by_text(text, existing_wiki_text):
        return True, "existing_wiki"
    return False, None


def scope_speculation_supported_by_text(text: str, source_text: str) -> bool:
    if not text or not source_text:
        return False
    anchors = scope_speculation_anchors(text)
    if len(anchors) < 2:
        return False
    sentences = external_backing_source_sentences(source_text)
    for index, sentence in enumerate(sentences):
        context = " ".join(sentences[max(0, index - 1) : index + 2])
        normalized_context = _source_excerpt.normalized_source_match_text(context)
        if sum(1 for anchor in anchors if anchor in normalized_context) >= min(3, len(anchors)):
            return True
    return False


def scope_speculation_anchors(text: str) -> list[str]:
    normalized_text = _source_excerpt.normalized_source_match_text(text)
    anchors: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.+_-]{1,}", unicodedata.normalize("NFKC", text)):
        normalized = token.lower().strip("._-+")
        if len(normalized) >= 3 and normalized not in {"the", "and", "for", "with", "from", "into", "this", "that"}:
            anchors.append(normalized)
    for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9（）()·]{2,}", text):
        normalized = _source_excerpt.normalized_source_match_text(phrase)
        if len(normalized) >= 2 and normalized not in {"可能", "也许", "或许", "推测", "疑似", "受到影响", "受影响", "波及", "涉及", "牵涉", "导致", "造成", "影响到", "关联"}:
            anchors.append(normalized)
    deduped: list[str] = []
    for anchor in anchors:
        if anchor and anchor in normalized_text and anchor not in deduped:
            deduped.append(anchor)
    return deduped[:8]


def grounding_text_supported_by_context(text: str, approved_raw_text: str, existing_wiki_text: str) -> bool:
    return quote_supported_by_text(text, approved_raw_text) or quote_supported_by_text(text, existing_wiki_text)


def unsupported_quote_grounding_reason(body: str, quote: str, *, quote_start: int, section_key: str) -> str:
    risk_marker = unsupported_quote_risk_marker(body, quote, quote_start=quote_start, section_key=section_key)
    if risk_marker:
        return f"直接引用未在 raw 或已有 wiki 中 exact match；{risk_marker} 作为非阻塞提醒保留，必要时可人工回看来源。"
    if explicit_direct_quote_context(body, quote, quote_start=quote_start) or attributed_quote_context(body, quote_start=quote_start):
        return "写成直接引用/作者归因的引号内容未在 raw 或已有 wiki 中 exact match；作为非阻塞提醒保留，必要时可改成转述或人工回看来源。"
    normalized_quote = re.sub(r"\s+", "", unicodedata.normalize("NFKC", quote.strip()))
    normalized_sentence = re.sub(r"\s+", "", unicodedata.normalize("NFKC", sentence_around_index(body, quote_start).strip()))
    if contains_short_fact_marker(normalized_quote) or contains_hard_fact_marker(normalized_quote):
        return "引号内数字、指标、规模、日期或其他硬事实未 exact match；作为非阻塞提醒保留，不阻塞用户已选择材料的 ingest。"
    if contains_short_fact_marker(normalized_sentence) or contains_hard_fact_marker(normalized_sentence):
        return "引号所在句包含数字、指标、规模、日期或其他硬事实且未 exact match；作为非阻塞提醒保留。"
    return "低风险未支撑引号内容仅记录为 warning，不阻塞自动 ingest；如需严谨可人工回看来源。"


def unsupported_quote_risk_marker(body: str, quote: str, *, quote_start: int, section_key: str) -> str:
    sentence = sentence_around_index(body, quote_start)
    normalized_quote = re.sub(r"\s+", "", unicodedata.normalize("NFKC", quote.strip()))
    if severe_factual_claim_marker(quote) or severe_factual_claim_marker(sentence):
        return "该表述包含专名关系、发布、收购、隶属、身份或因果等严重事实关系。"
    if section_key == "examples" and (unsupported_backing_marker(quote) or unsupported_backing_marker(sentence)):
        return ""
    if section_key == "examples" and examples_quote_has_unsafe_marker_for_bypass(normalized_quote, quote):
        return ""
    return ""


def sentence_around_index(text: str, index: int) -> str:
    if index < 0:
        return ""
    start = 0
    end = len(text)
    for pos in range(index - 1, -1, -1):
        if text[pos] in "\n。！？!?；;":
            start = pos + 1
            break
    for pos in range(index, len(text)):
        if text[pos] in "\n。！？!?；;":
            end = pos + 1
            break
    return text[start:end].strip(" -*\t")


def severe_factual_claim_marker(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", text)
    compact = re.sub(r"\s+", "", normalized)
    lowered = normalized.lower()
    strong_relation_markers = [
        "收购",
        "发布",
        "推出",
        "创立",
        "创建",
        "隶属",
        "属于",
        "担任",
        "任职",
        "宣布",
        "提出",
    ]
    strong_marker = next(
        (marker for marker in strong_relation_markers if marker in compact and not severe_relation_marker_meta_usage(compact, marker)),
        "",
    )
    if strong_marker and severe_factual_claim_has_named_entity(normalized):
        return strong_marker
    english_strong_pattern = (
        r"\b(?:acquired|acquires|acquire|released|releases|launched|launches|founded|created|"
        r"announced|owned\s+by|developed\s+by|built\s+by|proposed\s+by|ceo|cto|founder)\b"
    )
    match = re.search(english_strong_pattern, lowered, re.IGNORECASE)
    if match and severe_factual_claim_has_named_entity(normalized):
        return match.group(0)
    weak_relation_markers = ["证明", "导致", "造成", "取代", "替代", "支持", "不支持", "由"]
    weak_marker = next((marker for marker in weak_relation_markers if marker in compact), "")
    if weak_marker and severe_weak_factual_relation_requires_review(normalized, compact):
        return weak_marker
    english_weak_pattern = r"\b(?:proves?|causes?|caused|replaces?|replaced|supports?|unsupported|does\s+not\s+support)\b"
    weak_match = re.search(english_weak_pattern, lowered, re.IGNORECASE)
    if weak_match and severe_weak_factual_relation_requires_review(normalized, compact):
        return weak_match.group(0)
    return None


def severe_relation_marker_meta_usage(compact: str, marker: str) -> bool:
    if marker == "发布":
        return bool(
            re.search(r"(?:产品)?发布(?:节奏|流程|计划|策略|周期|管理|评审|窗口|阶段|一致性)", compact)
            or re.search(r"(?:快速|持续|连续)发布", compact)
            or re.search(r"(?:框架持续更新|近期版本|版本包括).{0,40}发布", compact)
            or re.search(r"(?:评测基准|基准|基准测试|开源基准|benchmark|Benchmark).{0,40}发布", compact)
            or re.search(r"(?:评估|测试).{0,16}(?:规划能力|Agent能力|智能体能力).{0,40}发布", compact)
        )
    if marker == "创建":
        return bool(
            re.search(
                r"创建(?:、更新)?(?:、删除)?(?:相关)?(?:wiki|Wiki)?(?:文档|页面|知识页|内容|文件|草稿|记录|摘要页面|摘要页)",
                compact,
            )
            or re.search(r"创建.{0,12}(?:文档|页面|知识页|内容|文件|草稿|记录|摘要页面|摘要页)", compact)
            or re.search(r"创建.{0,4}示例|创建示例", compact)
            or ("Assistant(" in compact and "创建" in compact)
            or re.search(r"创建.{0,20}(?:Docker)?(?:隔离)?容器", compact)
            or (
                any(context in compact for context in ["开发者", "用户", "代码", "示例", "框架", "使用", "通过", "注册", "配置", "实例化"])
                and re.search(r"创建.{0,20}(?:工具|智能体|Agent|Assistant|应用|实例|函数|类|服务器|后端|服务|图像生成工具|容器)", compact)
            )
        )
    if marker == "提出":
        return bool(re.search(r"提出(?:问题|请求|查询|疑问|检索需求|用户问题)", compact))
    if marker in {"推出", "宣布"}:
        return bool(re.search(rf"{marker}(?:计划|策略|流程|节奏|安排)", compact))
    return False


def severe_factual_claim_has_named_entity(text: str) -> bool:
    return bool(severe_factual_named_entities(text))


def grounding_claim_contradicts_approved_raw(claim_text: str, approved_raw_text: str) -> bool:
    if not claim_text.strip() or not approved_raw_text.strip():
        return False
    if quote_supported_by_text(claim_text, approved_raw_text):
        return False
    claim_marker = severe_factual_claim_marker(claim_text)
    if not claim_marker:
        return False
    claim_group = severe_factual_relation_group(claim_marker)
    if not claim_group:
        return False
    claim_entities = severe_factual_entities_in_text_order(claim_text)
    if len(claim_entities) < 2:
        return False
    for sentence in external_backing_source_sentences(approved_raw_text):
        source_marker = severe_factual_claim_marker(sentence)
        if not source_marker or severe_factual_relation_group(source_marker) != claim_group:
            continue
        source_entities = severe_factual_entities_in_text_order(sentence)
        if len(source_entities) < 2:
            continue
        if severe_factual_relations_contradict(
            claim_text=claim_text,
            source_text=sentence,
            claim_entities=claim_entities,
            source_entities=source_entities,
            relation_group=claim_group,
        ):
            return True
    return False


def severe_factual_relation_group(marker: str) -> str:
    normalized = marker.lower().strip()
    groups = {
        "acquire": {"收购", "acquired", "acquires", "acquire"},
        "release": {"发布", "推出", "released", "releases", "launched", "launches"},
        "create": {"创立", "创建", "created", "founded", "built by", "developed by"},
        "ownership": {"隶属", "属于", "由", "owned by"},
        "role": {"担任", "任职", "ceo", "cto", "founder"},
        "announce": {"宣布", "announced"},
        "propose": {"提出", "proposed by"},
    }
    for group, markers in groups.items():
        if normalized in markers:
            return group
    return ""


def severe_factual_entities_in_text_order(text: str) -> list[str]:
    entities = []
    for entity in severe_factual_named_entities(text):
        index = text.find(entity)
        if index >= 0:
            entities.append((index, entity))
    entities.sort(key=lambda item: (item[0], -len(item[1]), item[1]))
    return _markdown_utils.dedupe_strings([entity for _index, entity in entities])


def severe_factual_relations_contradict(
    *,
    claim_text: str,
    source_text: str,
    claim_entities: list[str],
    source_entities: list[str],
    relation_group: str,
) -> bool:
    claim_set = set(claim_entities)
    source_set = set(source_entities)
    common = claim_set & source_set
    if not common:
        return False
    claim_relation = severe_factual_relation_tuple(claim_text, claim_entities, relation_group)
    source_relation = severe_factual_relation_tuple(source_text, source_entities, relation_group)
    if claim_relation and source_relation:
        claim_actor, claim_object = claim_relation
        source_actor, source_object = source_relation
        if severe_factual_relation_polarity(claim_text) != severe_factual_relation_polarity(source_text) and (
            claim_actor == source_actor and claim_object == source_object
        ):
            return True
        if claim_actor == source_object and claim_object == source_actor:
            return True
        if claim_object == source_object and claim_actor != source_actor:
            return True
        if relation_group in {"ownership", "role"} and claim_actor == source_actor and claim_object != source_object:
            return True
        return False
    if severe_factual_relation_polarity(claim_text) != severe_factual_relation_polarity(source_text) and (
        claim_set == source_set or len(common) >= 2
    ):
        return True
    if len(common) >= 2 and [entity for entity in claim_entities if entity in common] != [
        entity for entity in source_entities if entity in common
    ]:
        return True
    if (
        relation_group in {"acquire", "release", "create", "ownership", "role", "announce", "propose"}
        and claim_entities[-1] == source_entities[-1]
        and claim_entities[0] != source_entities[0]
    ):
        return True
    if (
        relation_group in {"acquire", "release", "create", "announce", "propose"}
        and severe_factual_relation_by_actor_form(claim_text, relation_group)
        and severe_factual_relation_by_actor_form(source_text, relation_group)
        and claim_entities[0] == source_entities[0]
        and set(claim_entities[1:]) - source_set
        and set(source_entities[1:]) - claim_set
    ):
        return True
    if relation_group in {"ownership", "role"} and (
        claim_entities[0] == source_entities[0] and set(claim_entities[1:]) - source_set and set(source_entities[1:]) - claim_set
    ):
        return True
    return False


def severe_factual_relation_tuple(text: str, entities: list[str], relation_group: str) -> tuple[str, str] | None:
    if len(entities) < 2:
        return None
    if relation_group in {"acquire", "release", "create", "announce", "propose"}:
        if severe_factual_relation_by_actor_form(text, relation_group) or severe_factual_relation_passive_form(text, relation_group):
            return entities[1], entities[0]
        return entities[0], entities[1]
    if relation_group == "ownership":
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
        lowered = unicodedata.normalize("NFKC", text).lower()
        if "属于" in compact or "隶属" in compact or re.search(r"\bowned\s+by\b", lowered):
            return entities[1], entities[0]
        return entities[0], entities[1]
    if relation_group == "role":
        return entities[0], entities[1]
    return None


def severe_factual_relation_polarity(text: str) -> int:
    normalized = unicodedata.normalize("NFKC", text)
    compact = re.sub(r"\s+", "", normalized)
    lowered = normalized.lower()
    if re.search(r"(?:没有|并未|未曾|从未|不再|不是|并非|未).{0,12}(?:收购|发布|推出|创立|创建|隶属|属于|担任|任职|宣布|提出|支持|证明|导致|造成|取代|替代)", compact):
        return -1
    if re.search(r"(?:收购|发布|推出|创立|创建|隶属|属于|担任|任职|宣布|提出|支持|证明|导致|造成|取代|替代).{0,8}(?:不成立|并不成立|没有发生|未发生)", compact):
        return -1
    if re.search(
        r"\b(?:not|never|no longer|did not|does not|do not|has not|have not|had not|was not|were not|is not|are not|isn't|aren't|wasn't|weren't)\b.{0,80}"
        r"\b(?:acquir|releas|launch|found|creat|own|develop|built|propos|support|cause|replace|ceo|cto|founder)\b",
        lowered,
        re.IGNORECASE,
    ):
        return -1
    return 1


def severe_factual_relation_by_actor_form(text: str, relation_group: str) -> bool:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    lowered = unicodedata.normalize("NFKC", text).lower()
    group_markers = {
        "acquire": ("收购",),
        "release": ("发布", "推出"),
        "create": ("创立", "创建"),
        "announce": ("宣布",),
        "propose": ("提出",),
    }
    markers = group_markers.get(relation_group, ())
    if "由" in compact and any(re.search(rf"由.{{1,40}}{re.escape(marker)}", compact) for marker in markers):
        return True
    english_patterns = {
        "acquire": r"\bacquired\s+by\b",
        "release": r"\b(?:released|launched)\s+by\b",
        "create": r"\b(?:created|founded|built|developed)\s+by\b",
        "announce": r"\bannounced\s+by\b",
        "propose": r"\bproposed\s+by\b",
    }
    pattern = english_patterns.get(relation_group)
    return bool(pattern and re.search(pattern, lowered, re.IGNORECASE))


def severe_factual_relation_passive_form(text: str, relation_group: str) -> bool:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    group_markers = {
        "acquire": ("收购",),
        "release": ("发布", "推出"),
        "create": ("创立", "创建"),
        "announce": ("宣布",),
        "propose": ("提出",),
    }
    markers = group_markers.get(relation_group, ())
    return any(re.search(rf"被.{{1,40}}{re.escape(marker)}", compact) for marker in markers)


def severe_weak_factual_relation_requires_review(text: str, compact: str) -> bool:
    entities = severe_factual_named_entities(text)
    if len(entities) < 2:
        return False
    if weak_relation_technical_capability_usage(text, compact):
        return False
    relation_context_markers = [
        "公司",
        "团队",
        "产品",
        "模型",
        "系统",
        "CEO",
        "CTO",
        "创始人",
        "发布方",
        "开发方",
        "母公司",
        "子公司",
    ]
    if any(marker in compact for marker in relation_context_markers):
        return True
    known_count = sum(1 for entity in entities if entity in SEVERE_FACTUAL_KNOWN_ENTITIES)
    low_risk_technical_markers = ["编程", "异步", "缓存", "语义", "组成", "包括", "包含", "能力", "特性", "工具", "记忆"]
    return known_count >= 2 and not any(marker in compact for marker in low_risk_technical_markers)


def weak_relation_technical_capability_usage(text: str, compact: str) -> bool:
    if not any(marker in compact for marker in ["支持", "不支持", "由"]):
        lowered = text.lower()
        if not re.search(r"\b(?:supports?|unsupported|does\s+not\s+support)\b", lowered):
            return False
    capability_markers = [
        "组件",
        "自定义工具",
        "流式输出",
        "函数调用",
        "并行工具调用",
        "工具调用",
        "工具输出",
        "评估",
        "评测",
        "基准",
        "基准测试",
        "开源基准",
        "评测基准",
        "规划能力",
        "智能体能力",
        "Agent能力",
        "文件读取",
        "模板",
        "多种模板",
        "参数",
        "参数配置",
        "默认",
        "推荐",
        "fncall_prompt_type",
        "API",
        "OpenAI API",
        "接入",
        "模型服务",
        "DashScope",
        "阿里云",
        "开源",
        "后端运行",
        "Qwen Chat",
        "解析",
        "vLLM",
        "接口",
        "方法",
        "功能",
        "能力",
        "后端",
        "服务端解析",
        "原生工具调用",
        "可选依赖",
        "依赖",
        "安装",
        "集成",
        "协议",
        "模型上下文协议",
        "代码解释器",
        "GUI",
        "Gradio",
        "RAG",
        "MCP",
        "Node.js",
        "uv",
        "Git",
        "README",
        "BaseChatModel",
        "chat方法",
        "use_raw_api",
    ]
    if not any(marker in text or marker in compact for marker in capability_markers):
        return False
    support_index = min((index for marker in ["支持", "不支持", "由"] if (index := compact.find(marker)) >= 0), default=-1)
    supported_fragment = compact[support_index:] if support_index >= 0 else compact
    supported_fragment = (
        supported_fragment.replace("OpenAI-compatible", "")
        .replace("OpenAICompatible", "")
        .replace("OpenAI兼容", "")
        .replace("openai-compatible", "")
    )
    known_entities_after_support = [
        entity
        for entity in SEVERE_FACTUAL_KNOWN_ENTITIES
        if entity in supported_fragment and entity.lower() not in {"openai"}
    ]
    technical_context_for_entities = supported_fragment + compact
    if known_entities_after_support and re.search(
        r"(?:模板|参数|配置|默认|推荐|fncall_prompt_type|自定义工具|代码解释器|MCP|组件|后端|模型服务|接入|API|DashScope|阿里云|工具输出|解析|vLLM|评估|评测|基准|规划能力|智能体能力|Agent能力)",
        technical_context_for_entities,
    ):
        known_entities_after_support = [
            entity
            for entity in known_entities_after_support
            if not (re.match(r"Qwen(?:\d|\b)", entity) or entity in {"阿里", "阿里巴巴"})
        ]
    if known_entities_after_support:
        return False
    if "OpenAI" in supported_fragment and not any(marker in supported_fragment for marker in ["API", "接口", "兼容"]):
        return False
    return True


SEVERE_FACTUAL_KNOWN_ENTITIES = {
    "OpenAI",
    "Anthropic",
    "Claude",
    "Google",
    "Microsoft",
    "Meta",
    "Karpathy",
    "Andrej",
    "DeepMind",
    "Redis",
    "Qwen",
    "阿里",
    "阿里巴巴",
    "腾讯",
    "字节",
    "百度",
    "华为",
}


def severe_factual_named_entities(text: str) -> list[str]:
    known_entities = [
        "OpenAI",
        "Anthropic",
        "Claude",
        "Google",
        "Microsoft",
        "Meta",
        "Karpathy",
        "Andrej",
        "DeepMind",
        "Redis",
        "Qwen",
        "阿里",
        "阿里巴巴",
        "腾讯",
        "字节",
        "百度",
        "华为",
        "Claude",
        "OpenAI",
        "Anthropic",
        "Google",
    ]
    entities: list[str] = [entity for entity in known_entities if entity in text]
    entities.extend(re.findall(r"\b[A-Z][A-Za-z0-9.+_-]{2,}(?:\s+[A-Z][A-Za-z0-9.+_-]{2,})?\b", text))
    chinese_known_entities = [entity for entity in SEVERE_FACTUAL_KNOWN_ENTITIES if re.search(r"[\u4e00-\u9fff]", entity)]
    for entity in re.findall(r"[\u4e00-\u9fff]{2,}(?:公司|团队|模型|系统|产品|CEO|CTO|创始人)", text):
        if any(known_entity in entity for known_entity in chinese_known_entities):
            entities.append(entity)
    return _markdown_utils.dedupe_strings(entities)


def rewrite_grounding_sensitive_paraphrases(
    artifact: DraftRenderingArtifact,
    approved_raw_text: str,
) -> tuple[DraftRenderingArtifact, dict[str, Any]]:
    report_pages: list[dict[str, Any]] = []
    rewritten_pages: list[DraftPageItem] = []
    rewrite_count = 0
    source_sentences = grounding_rewrite_source_sentences(approved_raw_text)
    for page in artifact.pages:
        updates: dict[str, str] = {}
        page_sections: list[dict[str, Any]] = []
        for field_name, body in [
            ("summary", page.summary),
            ("body_markdown", page.body_markdown),
            ("open_questions", page.open_questions),
        ]:
            rewritten_body, section_rewrites = rewrite_grounding_sensitive_body(body, source_sentences)
            updates[field_name] = rewritten_body
            if section_rewrites:
                rewrite_count += len(section_rewrites)
                page_sections.append({"field": field_name, "rewrites": section_rewrites})
        if page_sections:
            report_pages.append(
                {
                    "page_plan_id": page.page_plan_id,
                    "target_path": page.canonical_target_path,
                    "fields": page_sections,
                }
            )
            rewritten_pages.append(page.model_copy(update=updates))
        else:
            rewritten_pages.append(page)
    report = {
        "schema_version": "grounding_paraphrase_rewrite_report.v1",
        "changed": rewrite_count > 0,
        "rewrite_count": rewrite_count,
        "pages": report_pages,
    }
    if rewrite_count == 0:
        return artifact, report
    return artifact.model_copy(update={"pages": rewritten_pages}), report


def rewrite_grounding_sensitive_body(body: str, source_sentences: list[str]) -> tuple[str, list[dict[str, str]]]:
    rewritten = body
    rewrites: list[dict[str, str]] = []
    source_text = " ".join(source_sentences)
    rewritten, internal_rewrites = rewrite_internal_artifact_references(rewritten)
    rewrites.extend(internal_rewrites)
    for quote, _quote_start in iter_grounding_quote_spans(body):
        known_translation = grounding_known_english_quote_translation(quote)
        if known_translation:
            for quote_start, quoted_text in grounding_quoted_literals(rewritten, quote):
                replacement = grounding_dequoted_source_replacement(rewritten, quote_start, known_translation)
                rewritten = rewritten[:quote_start] + replacement + rewritten[quote_start + len(quoted_text) :]
                rewrites.append(
                    {
                        "original_quote": quote,
                        "replacement": replacement,
                        "source_sentence": known_translation,
                        "reason": "已知英文来源短语改写为中文意译，避免 zh-CN 页面粘贴英文概括。",
                    }
                )
                break
            continue
        source_sentence = numeric_reliability_source_sentence(quote, source_sentences)
        if source_sentence:
            for quote_start, quoted_text in grounding_quoted_literals(rewritten, quote):
                replacement = grounding_dequoted_source_replacement(rewritten, quote_start, source_sentence)
                rewritten = rewritten[:quote_start] + replacement + rewritten[quote_start + len(quoted_text) :]
                rewrites.append(
                    {
                        "original_quote": quote,
                        "replacement": replacement,
                        "source_sentence": source_sentence,
                        "reason": "百分比可靠性短语改回 raw 中更具体的来源表述，避免把 paraphrase 写成直接引语。",
                    }
                )
                break
            continue
        for quote_start, quoted_text in grounding_quoted_literals(rewritten, quote):
            if not dequotable_grounding_paraphrase(rewritten, quote, quote_start=quote_start, source_text=source_text):
                continue
            replacement = quote.strip()
            rewritten = rewritten[:quote_start] + replacement + rewritten[quote_start + len(quoted_text) :]
            rewrites.append(
                {
                    "original_quote": quote,
                    "replacement": replacement,
                    "source_sentence": "",
                    "reason": "非显式直接引用的长概括去除引号，避免把 paraphrase 当成 raw exact quote。",
                }
            )
            break
    return rewritten, rewrites


def rewrite_internal_artifact_references(body: str) -> tuple[str, list[dict[str, str]]]:
    rewrites: list[dict[str, str]] = []
    rewritten = body
    patterns = [
        (
            re.compile(
                r"对应\s+approved_digest\s+中\s+`?[A-Za-z0-9_-]+`?\s+的\s+`?[A-Za-z0-9_]+`?\s+描述[:：]"
            ),
            "对应的来源要点是：",
        ),
        (
            re.compile(r"approved_digest\s+中\s+`?[A-Za-z0-9_-]+`?\s+的\s+`?[A-Za-z0-9_]+`?"),
            "来源要点",
        ),
    ]
    for pattern, replacement in patterns:
        matches = list(pattern.finditer(rewritten))
        if not matches:
            continue
        rewritten = pattern.sub(replacement, rewritten)
        for match in matches:
            rewrites.append(
                {
                    "original_quote": match.group(0),
                    "replacement": replacement,
                    "source_sentence": "",
                    "reason": "移除面向模型的内部 artifact 名称，改成用户可读的来源要点表达。",
                }
            )
    return rewritten, rewrites


def grounding_known_english_quote_translation(quote: str) -> str:
    normalized = _source_excerpt.normalized_source_match_text(quote)
    translations = {
        "anexcellentharnessthatprovidesafocusedcodingexperience": "一种优秀的 harness，提供聚焦的编码体验",
    }
    return translations.get(normalized, "")


def dequotable_grounding_paraphrase(body: str, quote: str, *, quote_start: int, source_text: str) -> bool:
    normalized = re.sub(r"\s+", "", quote.strip())
    if (
        len(normalized) < 18
        and not dequotable_source_local_concept_paraphrase(normalized)
        and not dequotable_short_slogan_or_label(normalized)
    ):
        return False
    if quote_supported_by_text(quote, source_text):
        return False
    if re.search(r"\d", normalized):
        return False
    if len(normalized) <= 32 and contains_short_fact_marker(normalized) and not dequotable_open_question_quote(normalized):
        return False
    if looks_like_untranslated_english(quote):
        return False
    if attributed_quote_context(body, quote_start=quote_start):
        return False
    if strict_direct_quote_context(body, quote_start=quote_start) and not dequotable_source_local_concept_paraphrase(normalized):
        return False
    return True


def dequotable_short_slogan_or_label(normalized: str) -> bool:
    if len(normalized) > 24:
        return False
    if not any(separator in normalized for separator in ["，", ",", "、", "/"]):
        return False
    if re.search(r"\d|[%％$￥¥]", normalized):
        return False
    if contains_short_fact_marker(normalized) or contains_hard_fact_marker(normalized):
        return False
    sentence_markers = [
        "认为",
        "表示",
        "指出",
        "发现",
        "证明",
        "承诺",
        "宣布",
        "导致",
        "因为",
        "所以",
        "已经",
        "正在",
        "应该",
        "必须",
        "需要",
        "推出",
        "发布",
        "上线",
    ]
    return not any(marker in normalized for marker in sentence_markers)


def dequotable_source_local_concept_paraphrase(normalized: str) -> bool:
    concept_pairs = [
        ("会话", "上下文窗口"),
        ("会话日志", "上下文窗口"),
        ("大脑", "双手"),
        ("harness", "容器"),
    ]
    return any(first in normalized and second in normalized for first, second in concept_pairs)


def dequotable_open_question_quote(normalized: str) -> bool:
    if not normalized.endswith(("?", "？")) and "？" not in normalized:
        return False
    question_markers = ["如何", "是否", "什么", "哪", "为何", "为什么", "能否", "需要"]
    return any(marker in normalized for marker in question_markers)


def strict_direct_quote_context(body: str, *, quote_start: int) -> bool:
    prefix = body[max(0, quote_start - 36) : quote_start]
    strict_markers = [
        "原文",
        "直接引用",
        "引用",
        "作者",
        "论文",
        "研究",
        "他说",
        "她说",
        "对方说",
        "Cat Wu指出",
        "Cat Wu表示",
        "Cat Wu说",
        "Boris指出",
        "Boris表示",
        "Boris说",
    ]
    return any(marker in prefix for marker in strict_markers)


def attributed_quote_context(body: str, *, quote_start: int) -> bool:
    prefix = body[max(0, quote_start - 28) : quote_start]
    explicit_attributors = [
        "文中",
        "原文",
        "作者",
        "论文",
        "研究",
        "访谈",
        "报告",
        "他说",
        "她说",
        "对方说",
        "Cat Wu",
        "Boris",
    ]
    if not any(marker in prefix for marker in explicit_attributors):
        return False
    return bool(re.search(r"(?:称|指出|表示|写道|说)\s*[：:，,]?\s*[“\"]?$", prefix))


def grounding_quoted_literals(body: str, quote: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for quoted in [f"“{quote}”", f'"{quote}"']:
        start = body.find(quoted)
        if start >= 0:
            matches.append((start, quoted))
    return sorted(matches, key=lambda item: item[0])


def grounding_dequoted_source_replacement(body: str, quote_start: int, source_sentence: str) -> str:
    source = source_sentence.strip().rstrip("。！？!?")
    prefix = body[max(0, quote_start - 16) : quote_start]
    if prefix.endswith(("所说", "说", "提到", "指出", "表示")):
        return f"，{source}"
    return source


def grounding_rewrite_source_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [sentence.strip() for sentence in re.split(r"(?<=[。！？!?])", normalized) if sentence.strip()]


def numeric_reliability_source_sentence(quote: str, source_sentences: list[str]) -> str | None:
    percentages = re.findall(r"\d+(?:\.\d+)?\s*%", quote)
    if not percentages:
        return None
    normalized_quote = _source_excerpt.normalized_source_match_text(quote)
    if not any(marker in normalized_quote for marker in ["失败", "没价值", "不够", "不是自动化", "可靠", "有效"]):
        return None
    normalized_percentages = {normalized_percentage_token(percentage) for percentage in percentages}
    best: str | None = None
    for sentence in source_sentences:
        normalized_sentence = _source_excerpt.normalized_source_match_text(sentence)
        sentence_percentages = {
            normalized_percentage_token(percentage)
            for percentage in re.findall(r"\d+(?:\.\d+)?\s*%", unicodedata.normalize("NFKC", sentence))
        }
        if not normalized_percentages or not normalized_percentages <= sentence_percentages:
            continue
        if "自动化" not in normalized_sentence:
            continue
        if not any(marker in normalized_sentence for marker in ["价值", "有效", "准确率", "不够", "放弃", "100"]):
            continue
        if not any(marker in normalized_sentence for marker in ["不", "没", "不是", "不够"]):
            continue
        if best is None or len(sentence) < len(best):
            best = sentence
    return best


def normalized_percentage_token(percentage: str) -> str:
    token = unicodedata.normalize("NFKC", percentage).replace(" ", "")
    match = re.match(r"(\d+(?:\.\d+)?)%", token)
    return f"{match.group(1)}%" if match else token


def compact_paraphrase_supported_by_context(text: str, approved_raw_text: str, existing_wiki_text: str) -> tuple[bool, str | None]:
    if compact_paraphrase_supported_by_text(text, approved_raw_text) or method_goal_paraphrase_supported_by_text(
        text, approved_raw_text
    ):
        return True, "raw"
    if compact_paraphrase_supported_by_text(text, existing_wiki_text) or method_goal_paraphrase_supported_by_text(
        text, existing_wiki_text
    ):
        return True, "existing_wiki"
    return False, None


def iter_grounding_quote_spans(body: str) -> list[tuple[str, int]]:
    spans: list[tuple[str, int]] = []
    for match in re.finditer(r"“([^“”\n]{2,})”", body):
        quote = match.group(1)
        normalized_quote = re.sub(r"\s+", "", quote.strip())
        if len(quote) >= 6 or quote_has_dynamic_sensitive_query_marker(normalized_quote, quote):
            spans.append((quote, match.start()))
    index = 0
    while index < len(body):
        quote_start = body.find('"', index)
        if quote_start < 0:
            break
        if not plausible_ascii_open_quote(body, quote_start):
            quote_end = body.find('"', quote_start + 1)
            if quote_end < 0:
                break
            quote = body[quote_start + 1 : quote_end]
            normalized_quote = re.sub(r"\s+", "", quote.strip())
            if (
                quote_has_dynamic_sensitive_query_marker(normalized_quote, quote)
                and "\n" not in quote
                and not any(char in quote for char in '“”"')
            ):
                spans.append((quote, quote_start))
            index = quote_end + 1
            continue
        quote_end = body.find('"', quote_start + 1)
        if quote_end < 0:
            break
        quote = body[quote_start + 1 : quote_end]
        normalized_quote = re.sub(r"\s+", "", quote.strip())
        if (
            (len(quote) >= 6 or quote_has_dynamic_sensitive_query_marker(normalized_quote, quote))
            and "\n" not in quote
            and not any(char in quote for char in '“”"')
        ):
            spans.append((quote, quote_start))
        index = quote_end + 1
    return sorted(spans, key=lambda span: span[1])


def plausible_ascii_open_quote(body: str, quote_start: int) -> bool:
    prefix = body[max(0, quote_start - 12) : quote_start]
    if any(marker in prefix for marker in ["原文", "直接引用", "引用", "他说", "她说", "对方说", "访谈中说"]):
        return True
    if quote_start == 0:
        return True
    previous = body[quote_start - 1]
    return previous.isspace() or previous in "([{<（【《:：,，;；.。!！?？\n\r\t-—"


def remove_grounding_quote_spans_for_scan(text: str) -> str:
    if not text:
        return text
    chars = list(text)
    for quote, quote_start in iter_grounding_quote_spans(text):
        quote_end = min(len(chars), quote_start + len(quote) + 2)
        for index in range(max(0, quote_start), quote_end):
            chars[index] = " "
    return "".join(chars)


def collect_grounding_claims(
    *,
    item: WikiMergePlanItem,
    page: DraftPageItem,
    existing_entry: WikiContextEntry,
    approved_raw_text: str,
    claims: list[GroundingClaim],
) -> None:
    for section_key, body in draft_grounding_sections(page):
        for quote, quote_start in iter_grounding_quote_spans(body):
            normalized_quote = re.sub(r"\s+", "", quote.strip())
            is_explicit_quote = explicit_direct_quote_context(body, quote, quote_start=quote_start)
            is_illustrative_example = illustrative_example_context(body, quote, quote_start=quote_start)
            is_memory_example = illustrative_memory_example_context(body, quote, quote_start=quote_start)
            raw_supported = quote_supported_by_text(quote, approved_raw_text)
            existing_supported = quote_supported_by_text(quote, existing_entry.content)
            supported = raw_supported or existing_supported
            quote_has_external_backing_marker = unsupported_backing_marker(quote) is not None
            quote_has_dynamic_sensitive_query = quote_has_dynamic_sensitive_query_marker(normalized_quote, quote)
            examples_unsafe_bypass_quote = (
                section_key == "examples"
                and not supported
                and examples_quote_has_unsafe_marker_for_bypass(normalized_quote, quote)
            )
            is_concept_label_quote = (
                looks_like_concept_phrase(quote)
                or looks_like_abstract_trend_label(re.sub(r"\s+", "", quote.strip()))
            ) and (
                not supported
                and not quote_has_external_backing_marker
                and not quote_has_dynamic_sensitive_query
                and not severe_factual_claim_marker(quote)
                and not severe_factual_claim_marker(sentence_around_index(body, quote_start))
                and not strict_direct_quote_context(body, quote_start=quote_start)
                and not attributed_quote_context(body, quote_start=quote_start)
            )
            if section_key == "examples" and is_concept_label_quote and (
                examples_quote_has_concrete_marker(normalized_quote, quote) or examples_unsafe_bypass_quote
            ):
                is_concept_label_quote = False
            if examples_unsafe_bypass_quote:
                is_illustrative_example = False
                is_memory_example = False
            if (
                section_key == "examples"
                and not supported
                and not examples_unsafe_bypass_quote
                and examples_quote_should_warn_not_infer(normalized_quote, quote)
            ):
                is_illustrative_example = False
                is_memory_example = False
            is_abstract_placeholder_example = (
                section_key == "examples"
                and not supported
                and not is_explicit_quote
                and not strict_direct_quote_context(body, quote_start=quote_start)
                and not attributed_quote_context(body, quote_start=quote_start)
                and not examples_unsafe_bypass_quote
                and examples_abstract_placeholder_quote(normalized_quote, quote)
            )
            is_generic_prompt_example = (
                section_key == "examples"
                and not supported
                and not is_explicit_quote
                and not strict_direct_quote_context(body, quote_start=quote_start)
                and not attributed_quote_context(body, quote_start=quote_start)
                and not examples_unsafe_bypass_quote
                and examples_generic_prompt_quote(normalized_quote, quote)
            )
            is_memory_query_example = (
                section_key == "examples"
                and not supported
                and not is_explicit_quote
                and not strict_direct_quote_context(body, quote_start=quote_start)
                and not attributed_quote_context(body, quote_start=quote_start)
                and not examples_unsafe_bypass_quote
                and examples_memory_query_quote(body, normalized_quote, quote, quote_start=quote_start)
            )
            is_query_template_example = (
                section_key == "examples"
                and not supported
                and not is_explicit_quote
                and not strict_direct_quote_context(body, quote_start=quote_start)
                and not attributed_quote_context(body, quote_start=quote_start)
                and not examples_unsafe_bypass_quote
                and examples_query_template_quote(body, normalized_quote, quote, quote_start=quote_start)
            )
            if (
                section_key == "examples"
                and not supported
                and not examples_unsafe_bypass_quote
                and examples_quote_should_warn_not_infer(normalized_quote, quote)
            ):
                is_abstract_placeholder_example = False
                is_generic_prompt_example = False
                is_memory_query_example = False
                is_query_template_example = False
            if (
                section_key != "examples"
                and not supported
                and is_illustrative_example
                and (
                    len(normalized_quote) > 12
                    or contains_short_fact_marker(normalized_quote)
                    or contains_hard_fact_marker(normalized_quote)
                    or examples_quote_should_warn_not_infer(normalized_quote, quote)
                )
            ):
                is_illustrative_example = False
            if (
                not is_explicit_quote
                and (
                    is_illustrative_example
                    or is_memory_example
                    or is_abstract_placeholder_example
                    or is_generic_prompt_example
                    or is_memory_query_example
                    or is_query_template_example
                )
            ) or is_concept_label_quote:
                reason = "短标题/概念短语按概念标签处理，不要求 raw exact match。"
                if is_abstract_placeholder_example:
                    reason = "例子区的抽象占位符示例按 illustrative example 处理，不要求 raw exact match。"
                elif is_memory_query_example:
                    reason = "例子区的抽象记忆查询样例按 illustrative example 处理，不要求 raw exact match。"
                elif is_memory_example:
                    reason = "记忆评估中的短问句/用户偏好/对话样例按 illustrative example 处理，不要求 raw exact match。"
                elif is_query_template_example:
                    reason = "例子区的短查询/请求模板按 illustrative example 处理，不要求 raw exact match。"
                elif is_generic_prompt_example:
                    reason = "例子区的通用问题/指令示例按 illustrative example 处理，不要求 raw exact match。"
                elif section_key == "examples":
                    reason = "例子区的通用示例句按 illustrative example 处理，不要求 raw exact match。"
                elif is_illustrative_example:
                    reason = "由如/例如/比如引出的通用示例句按 illustrative example 处理，不要求 raw exact match。"
                claims.append(
                    GroundingClaim(
                        page_plan_id=page.page_plan_id,
                        target_path=item.canonical_target_path,
                        section_key=section_key,
                        claim_type="inference",
                        text=quote,
                        support="inference",
                        action="kept",
                        reason=reason,
                    )
                )
                continue
            compact_supported, compact_support = (
                (False, None)
                if is_explicit_quote or supported
                else compact_paraphrase_supported_by_context(quote, approved_raw_text, existing_entry.content)
            )
            if compact_supported:
                claims.append(
                    GroundingClaim(
                        page_plan_id=page.page_plan_id,
                        target_path=item.canonical_target_path,
                        section_key=section_key,
                        claim_type="inference",
                        text=quote,
                        support=compact_support or "raw",
                        action="kept",
                        reason="引号内压缩概括已被 raw 或已有 wiki 的邻近片段支撑，不按直接引用 exact match 拦截。",
                    )
                )
                continue
            contradicts_raw = grounding_claim_contradicts_approved_raw(quote, approved_raw_text) if not supported else False
            claims.append(
                GroundingClaim(
                    page_plan_id=page.page_plan_id,
                    target_path=item.canonical_target_path,
                    section_key=section_key,
                    claim_type="new_fact",
                    text=quote,
                    support="raw" if raw_supported else ("existing_wiki" if existing_supported else "unsupported"),
                    action=(
                        "kept"
                        if supported
                        else (
                            "needs_review"
                            if contradicts_raw
                            else "warn"
                        )
                    ),
                    reason=(
                        "直接引用已在 raw 或已有 wiki 中规范化 exact match。"
                        if supported
                        else (
                            "该表述与 Approved Raw 中的同类事实关系明显不符；需要人工处理。"
                            if contradicts_raw
                            else unsupported_quote_grounding_reason(body, quote, quote_start=quote_start, section_key=section_key)
                        )
                    ),
                )
            )
        for line in body.splitlines():
            if line.lstrip().startswith("#"):
                continue
            text = line.strip(" -*")
            if not text or len(text) < 8:
                continue
            scope_marker = None if section_key == "open_questions" else unsupported_scope_speculation_marker(text)
            unquoted_scan_text = remove_grounding_quote_spans_for_scan(text)
            severe_marker = None if section_key == "open_questions" else severe_factual_claim_marker(unquoted_scan_text)
            if severe_marker:
                unsupported_text = sentence_with_marker(text, severe_marker)
                raw_supported = quote_supported_by_text(unsupported_text, approved_raw_text)
                existing_supported = quote_supported_by_text(unsupported_text, existing_entry.content)
                supported = raw_supported or existing_supported
                contradicts_raw = (not supported) and grounding_claim_contradicts_approved_raw(
                    unsupported_text,
                    approved_raw_text,
                )
                claims.append(
                    GroundingClaim(
                        page_plan_id=page.page_plan_id,
                        target_path=item.canonical_target_path,
                        section_key=section_key,
                        claim_type="new_fact",
                        text=unsupported_text,
                        support="raw" if raw_supported else ("existing_wiki" if existing_supported else "unsupported"),
                        action="kept" if supported else ("needs_review" if contradicts_raw else "warn"),
                        reason=(
                            "严重事实关系已在 raw 或已有 wiki 中规范化 exact match。"
                            if supported
                            else "该表述与 Approved Raw 中的同类事实关系明显不符；需要人工处理。"
                            if contradicts_raw
                            else (
                                f"新增严重事实关系 `{severe_marker}` 缺少 raw 或 inspected wiki 同句级支撑；"
                                "作为非阻塞提醒保留，必要时可人工回看来源。"
                            )
                        ),
                    )
                )
            if scope_marker:
                unsupported_text = sentence_with_marker(text, scope_marker)
                supported, _support_source = scope_speculation_supported_by_context(
                    unsupported_text,
                    approved_raw_text,
                    existing_entry.content,
                )
                if not supported:
                    claims.append(
                        GroundingClaim(
                            page_plan_id=page.page_plan_id,
                            target_path=item.canonical_target_path,
                            section_key=section_key,
                            claim_type="new_fact",
                            text=unsupported_text,
                            support="unsupported",
                            action="warn",
                            reason=(
                                f"新增影响范围/受影响对象推测 `{scope_marker}` 未被 raw 或 inspected wiki 同句级支撑；"
                                "作为非阻塞提醒保留，必要时可人工回看来源。"
                            ),
                        )
                    )
            for marker in unsupported_backing_markers(text):
                if unsupported_backing_marker_only_inside_quote(text, marker):
                    continue
                if unsupported_backing_marker_only_inside_supported_quote(
                    text,
                    marker,
                    approved_raw_text,
                    existing_entry.content,
                ):
                    continue
                unsupported_text = sentence_with_marker(text, marker)
                backing_context_text = text if len(text) <= 600 else unsupported_text
                supported, _support_source = external_backing_supported_by_context(
                    backing_context_text,
                    marker,
                    approved_raw_text,
                    existing_entry.content,
                )
                if not supported:
                    claims.append(
                        GroundingClaim(
                            page_plan_id=page.page_plan_id,
                            target_path=item.canonical_target_path,
                            section_key=section_key,
                            claim_type="new_fact",
                            text=unsupported_text,
                            support="unsupported",
                            action="warn",
                            reason=(
                                f"新增外部背书/强事实标记 `{marker}` 未在 raw 或 inspected wiki 中出现；"
                                "作为非阻塞提醒保留，必要时可改写为 source-local 表达。"
                            ),
                        )
                    )
                    break
    if item.action == "update" and existing_entry.content:
        claims.append(
            GroundingClaim(
                page_plan_id=page.page_plan_id,
                target_path=item.canonical_target_path,
                claim_type="retained_fact",
                text="旧页事实作为 existing wiki 背景参与更新。",
                support="existing_wiki",
                action="kept",
            )
                )


def unsupported_backing_marker_only_inside_quote(text: str, marker: str) -> bool:
    marker_count = text.count(marker)
    if marker_count <= 0:
        return False
    quoted_count = sum(quote.count(marker) for quote, _quote_start in iter_grounding_quote_spans(text))
    return quoted_count == marker_count


def unsupported_backing_marker_only_inside_supported_quote(
    text: str,
    marker: str,
    approved_raw_text: str,
    existing_wiki_text: str,
) -> bool:
    marker_count = text.count(marker)
    if marker_count <= 0:
        return False
    supported_count = 0
    for quote, _quote_start in iter_grounding_quote_spans(text):
        if marker not in quote:
            continue
        if quote_supported_by_text(quote, approved_raw_text) or quote_supported_by_text(quote, existing_wiki_text):
            supported_count += quote.count(marker)
    return supported_count == marker_count


def quote_supported_by_text(quote: str, text: str) -> bool:
    if not quote or not text:
        return False
    if quote in text:
        return True
    normalized_quote_variants = normalized_quote_support_variants(quote)
    normalized_text_variants = normalized_quote_support_variants(text)
    for normalized_quote in normalized_quote_variants:
        for normalized_text in normalized_text_variants:
            if len(normalized_quote) >= 16 and normalized_quote in normalized_text:
                if re.search(r"\d", normalized_quote) and not quote_numeric_tokens_are_exactly_present(quote, text):
                    continue
                return True
            if short_quote_supported_by_normalized_text(quote, normalized_quote, text, normalized_text):
                return True
    return False


def normalized_quote_support_variants(text: str) -> list[str]:
    variants = [_source_excerpt.normalized_source_match_text(text)]
    range_normalized = normalize_numeric_range_connectors(text)
    if range_normalized != text:
        variants.append(_source_excerpt.normalized_source_match_text(range_normalized))
    enumerated_range_normalized = normalize_paired_temporal_enumerated_ranges(text)
    if enumerated_range_normalized != text:
        variants.append(_source_excerpt.normalized_source_match_text(enumerated_range_normalized))
    stripped = strip_inline_term_translation_parentheticals(text)
    if stripped != text:
        variants.append(_source_excerpt.normalized_source_match_text(stripped))
    if re.search(r"\d", unicodedata.normalize("NFKC", text)):
        variants.extend(normalized_direct_quote_elision_variant(variant) for variant in list(variants))
    return _markdown_utils.dedupe_strings([variant for variant in variants if variant])


def normalize_numeric_range_connectors(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:[-~至到])\s*(\d+(?:\.\d+)?)(?=\s*(?:个?月|年|天|周|小时|分钟|秒|%|％|倍|人|个|项|种|类|步))",
        r"\1到\2",
        normalized,
    )


def normalize_paired_temporal_enumerated_ranges(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)

    def replacement(match: re.Match[str]) -> str:
        tail = normalized[match.end() : match.end() + 24]
        if re.match(r"\s*[、,，]\s*\d", tail):
            return match.group(0)
        return f"{match.group(1)}到{match.group(3)}{match.group('unit')}{match.group('suffix') or ''}"

    return re.sub(
        r"(\d+(?:\.\d+)?)\s*(?P<unit>个月|年|天|周|小时|分钟|秒)\s*[、,，]\s*"
        r"(\d+(?:\.\d+)?)\s*(?P=unit)(?P<suffix>后|前|内|间|之间|左右|以后|之内)?",
        replacement,
        normalized,
    )


def normalized_direct_quote_elision_variant(normalized_text: str) -> str:
    if len(normalized_text) < 16:
        return normalized_text
    return re.sub(r"那个|这个|这些|那些|该|其|的", "", normalized_text)


def strip_inline_term_translation_parentheticals(text: str) -> str:
    return re.sub(
        r"(?P<term>[A-Za-z][A-Za-z0-9.+#/-]*)\s*[（(][\u4e00-\u9fffA-Za-z0-9\s/+.-]{1,32}[）)]",
        r"\g<term>",
        unicodedata.normalize("NFKC", text),
    )


def short_quote_supported_by_normalized_text(
    quote: str,
    normalized_quote: str,
    text: str,
    normalized_text: str,
) -> bool:
    if len(normalized_quote) < 6 or normalized_quote not in normalized_text:
        return False
    if not quote_numeric_tokens_are_exactly_present(quote, text):
        return False
    if re.search(r"\d", normalized_quote):
        return len(re.sub(r"\d+", "", normalized_quote)) >= 3
    if looks_like_named_concept_label(normalized_quote):
        return True
    domain_anchors = [
        "agent",
        "claude",
        "claudecode",
        "cowork",
        "eval",
        "harness",
        "langflow",
        "managedagents",
        "mcp",
        "n8n",
        "rag",
        "sandbox",
        "session",
        "workflow",
    ]
    return bool(re.search(r"[a-z]", normalized_quote)) and any(anchor in normalized_quote for anchor in domain_anchors)


def quote_numeric_tokens_are_exactly_present(quote: str, text: str) -> bool:
    tokens = re.findall(r"\d+(?:\.\d+)?", unicodedata.normalize("NFKC", quote))
    if not tokens:
        return True
    normalized_text = unicodedata.normalize("NFKC", text)
    source_tokens = set(re.findall(r"\d+(?:\.\d+)?", normalized_text))
    return all(token in source_tokens for token in tokens)


def compact_paraphrase_supported_by_text(quote: str, text: str) -> bool:
    if not quote or not text:
        return False
    if not any(separator in quote for separator in ["，", ",", "；", ";", "、"]):
        return False
    normalized_quote = _source_excerpt.normalized_source_match_text(quote)
    if len(normalized_quote) < 16 or len(normalized_quote) > 96:
        return False
    if re.search(r"\d", normalized_quote):
        return False
    segments = [
        segment
        for segment in (_source_excerpt.normalized_source_match_text(part) for part in re.split(r"[，,；;、]", quote))
        if len(segment) >= 4
    ]
    if len(segments) < 2:
        return False
    normalized_text = _source_excerpt.normalized_source_match_text(text)
    segment_hits: list[tuple[list[int], int, bool]] = []
    for segment in segments:
        exact_position = normalized_text.find(segment)
        if exact_position >= 0:
            segment_hits.append(([exact_position], 1, True))
            continue
        anchors = compact_paraphrase_anchor_matches(segment, normalized_text)
        if len(anchors) < 2:
            return False
        segment_hits.append(([position for _anchor, position in anchors], 2, False))
    all_positions = sorted({position for positions, _required, _exact in segment_hits for position in positions})
    if not all_positions:
        return False
    for center in all_positions:
        window_start = center - 350
        window_end = center + 350
        total_match_units = 0
        window_ok = True
        for positions, required, exact in segment_hits:
            hit_count = sum(1 for position in positions if window_start <= position <= window_end)
            if hit_count < required:
                window_ok = False
                break
            total_match_units += 2 if exact else hit_count
        if window_ok and total_match_units >= 4:
            return True
    return False


def method_goal_paraphrase_supported_by_text(quote: str, text: str) -> bool:
    if not quote or not text:
        return False
    normalized_quote = _source_excerpt.normalized_source_match_text(quote)
    if len(normalized_quote) < 12 or len(normalized_quote) > 64:
        return False
    if re.search(r"\d", normalized_quote) or contains_hard_fact_marker(normalized_quote):
        return False
    if not any(marker in normalized_quote for marker in ["方法", "路径", "方式", "流程", "目标", "原则", "模式", "用例"]):
        return False
    if re.search(r"发布(?:了|过)|推出(?:了|过)|上线(?:了|过)", normalized_quote):
        return False
    normalized_text = _source_excerpt.normalized_source_match_text(text)
    anchors = compact_paraphrase_anchor_matches(normalized_quote, normalized_text)
    anchor_occurrences = [
        (anchor, compact_paraphrase_anchor_occurrences(anchor, normalized_text))
        for anchor, _position in anchors
    ]
    anchor_occurrences = [(anchor, positions) for anchor, positions in anchor_occurrences if positions]
    strong_anchor_count = sum(1 for anchor, _positions in anchor_occurrences if len(anchor) >= 4)
    if len(anchor_occurrences) < 3 or strong_anchor_count < 2:
        return False
    all_positions = sorted({position for _anchor, positions in anchor_occurrences for position in positions})
    for center in all_positions:
        window_start = center - 350
        window_end = center + 350
        hit_count = 0
        strong_hit_count = 0
        for anchor, positions in anchor_occurrences:
            if any(window_start <= position <= window_end for position in positions):
                hit_count += 1
                if len(anchor) >= 4:
                    strong_hit_count += 1
        if hit_count >= 3 and strong_hit_count >= 2:
            return True
    return False


def compact_paraphrase_anchor_occurrences(anchor: str, normalized_text: str, *, limit: int = 30) -> list[int]:
    positions: list[int] = []
    start = 0
    while len(positions) < limit:
        position = normalized_text.find(anchor, start)
        if position < 0:
            break
        positions.append(position)
        start = position + max(1, len(anchor))
    return positions


def compact_paraphrase_anchor_matches(segment: str, normalized_text: str) -> list[tuple[str, int]]:
    matches: list[tuple[str, int]] = []
    max_len = min(8, len(segment))
    for length in range(max_len, 1, -1):
        for start in range(0, len(segment) - length + 1):
            anchor = segment[start : start + length]
            if compact_paraphrase_anchor_is_noise(anchor):
                continue
            if any(anchor in existing or existing in anchor for existing, _position in matches):
                continue
            position = normalized_text.find(anchor)
            if position >= 0:
                matches.append((anchor, position))
        if len(matches) >= 3:
            break
    return matches


def compact_paraphrase_anchor_is_noise(anchor: str) -> bool:
    if len(anchor) < 2:
        return True
    if all(char in "的是了和与及或并把被在为对从到中上下一种一个这个那个其" for char in anchor):
        return True
    return anchor in {
        "主要",
        "问题",
        "核心",
        "用户",
        "团队",
        "目标",
        "产品",
        "功能",
        "这个",
        "那个",
        "一种",
        "一个",
    }


def explicit_direct_quote_context(body: str, quote: str, *, quote_start: int | None = None) -> bool:
    index = quote_start if quote_start is not None else body.find(quote)
    if index < 0:
        return False
    if quoted_label_context(body, quote, quote_start=index):
        return False
    prefix = body[max(0, index - 24) : index]
    return any(
        marker in prefix
        for marker in [
            "原文",
            "直接引用",
            "引用",
            "他说",
            "她说",
            "对方说",
            "访谈中说",
            "所说",
            "指出",
            "表示",
            "提到",
            "写道",
            "称",
        ]
    )


def quoted_label_context(body: str, quote: str, *, quote_start: int | None = None) -> bool:
    index = quote_start if quote_start is not None else body.find(quote)
    if index < 0:
        return False
    normalized_quote = re.sub(r"\s+", "", quote.strip())
    if not looks_like_concept_phrase(normalized_quote) and not looks_like_abstract_trend_label(normalized_quote):
        return False
    prefix = body[max(0, index - 32) : index]
    quote_end = index + len(quote) + (2 if body[index : index + 1] in {'"', "“"} else 0)
    suffix = body[quote_end : quote_end + 16]
    if re.search(r"(?:关于|围绕|主题为|标题为|名为|所谓|称为|叫做)[“\"]?$", prefix):
        return True
    if re.search(r"(?:提到|讨论|涉及|聚焦|描述|概括)的[“\"]?$", prefix) and re.match(
        r"(?:的)?(?:讨论|趋势|概念|问题|主题|选择|定位|框架|方法|模式|说法|标题|标签)",
        suffix,
    ):
        return True
    if re.match(r"(?:的)?(?:讨论|趋势|概念|问题|主题|选择|定位|框架|方法|模式|说法|标题|标签)", suffix) and any(
        marker in prefix for marker in ["关于", "围绕", "源自", "来自", "作为", "可称为"]
    ):
        return True
    return False


def illustrative_example_context(body: str, quote: str, *, quote_start: int | None = None) -> bool:
    normalized = re.sub(r"\s+", "", quote.strip())
    index = quote_start if quote_start is not None else body.find(quote)
    if index < 0:
        return False
    prefix = body[max(0, index - 18) : index]
    direct_markers = ["原文", "直接引用", "引用", "指出", "表示", "论文", "研究", "作者"]
    if any(marker in prefix for marker in direct_markers):
        return False
    if not any(marker in prefix for marker in ["如", "例如", "比如", "示例", "例子", "e.g.", "for example"]):
        return False
    if quote_has_dynamic_sensitive_query_marker(normalized, quote):
        return False
    return True


def illustrative_memory_example_context(body: str, quote: str, *, quote_start: int | None = None) -> bool:
    normalized = re.sub(r"\s+", "", quote.strip())
    if len(normalized) > 48:
        return False
    index = quote_start if quote_start is not None else body.find(quote)
    if index < 0:
        return False
    prefix = body[max(0, index - 32) : index]
    if any(marker in prefix for marker in ["原文", "直接引用", "引用", "指出", "表示", "论文", "研究", "作者"]):
        return False
    if quote_has_dynamic_sensitive_query_marker(normalized, quote):
        return False
    if not any(marker in prefix for marker in ["如", "例如", "比如", "示例", "例子", "问题", "问句", "评估"]):
        return False
    window = body[max(0, index - 72) : index + len(quote) + 24]
    if not memory_example_context_marker(window):
        return False
    if memory_example_question(normalized):
        return True
    if memory_example_user_preference(normalized) and not contains_hard_fact_marker(normalized):
        return True
    if memory_example_utterance(normalized):
        return True
    return False


def memory_example_context_marker(text: str) -> bool:
    markers = [
        "记忆",
        "智能体",
        "MemBench",
        "评估",
        "事实",
        "反思",
        "参与场景",
        "观察场景",
        "单跳",
        "多跳",
        "知识更新",
        "情感",
        "偏好",
        "任务类型",
    ]
    return any(marker in text for marker in markers)


def memory_example_question(normalized: str) -> bool:
    if not normalized.endswith(("?", "？")):
        return False
    return any(marker in normalized for marker in ["什么", "谁", "哪", "多少", "是否", "吗", "何时", "几", "名字", "多大", "年龄", "态度"])


def memory_example_user_preference(normalized: str) -> bool:
    return any(marker in normalized for marker in ["用户喜欢", "用户偏好", "用户讨厌", "我喜欢", "我讨厌", "偏好"])


def memory_example_utterance(normalized: str) -> bool:
    return any(marker in normalized.lower() for marker in ["用户", "我", "我的", "表哥", "表弟", "cousin", "ethan", "assistant", "智能体"])


def examples_abstract_placeholder_quote(normalized: str, original: str = "") -> bool:
    if not normalized:
        return False
    placeholder_markers = [
        "某家店",
        "某家门店",
        "某家餐厅",
        "某个地点",
        "某类产品",
        "某种产品",
        "某种饮品",
        "某类内容",
        "某项任务",
        "某次交互",
        "某段记忆",
        "某条记忆",
        "某种偏好",
        "某些偏好",
        "用户偏好X",
        "user_id",
        "example_id",
        "time_period",
        "memory",
    ]
    lowered_original = original.lower()
    has_placeholder = any(marker in normalized for marker in placeholder_markers) or any(
        marker in lowered_original for marker in ["user_id", "example_id", "time_period", "memory"]
    )
    if not has_placeholder:
        return False
    if examples_quote_has_concrete_marker(normalized, original):
        return False
    return True


def examples_generic_prompt_quote(normalized: str, original: str = "") -> bool:
    if not normalized:
        return False
    if examples_quote_has_concrete_marker(normalized, original):
        return False
    if memory_example_question(normalized):
        return True
    if looks_like_instructional_example(normalized, "示例"):
        return True
    generic_question_markers = ["什么", "如何", "是否", "哪", "何时", "为什么", "吗"]
    if normalized.endswith(("?", "？")) and any(marker in normalized for marker in generic_question_markers):
        return True
    generic_instruction_markers = ["你是一位", "请", "回答", "说明", "解释", "写一段", "生成"]
    return any(marker in normalized for marker in generic_instruction_markers)


def examples_memory_query_quote(body: str, normalized: str, original: str = "", *, quote_start: int | None = None) -> bool:
    if not normalized or len(normalized) > 48:
        return False
    if examples_quote_has_concrete_marker(normalized, original):
        return False
    if not memory_query_call_argument_context(body, quote_start):
        return False
    query_markers = [
        "用户",
        "记忆",
        "偏好",
        "上下文",
        "问题",
        "工单",
        "任务",
        "项目",
        "截止日期",
        "历史",
        "状态",
        "信息",
    ]
    return any(marker in normalized for marker in query_markers)


def examples_query_template_quote(body: str, normalized: str, original: str = "", *, quote_start: int | None = None) -> bool:
    if not normalized or len(normalized) > 40:
        return False
    if quote_start is None or quote_start < 0:
        return False
    if examples_query_template_has_unsafe_marker(normalized, original):
        return False
    if not examples_query_template_local_context(body, quote_start, original):
        return False
    query_template_markers = [
        "方法",
        "步骤",
        "怎么",
        "如何",
        "查询",
        "搜索",
        "请求",
        "问题",
        "问句",
        "片段",
        "信息",
        "记忆",
        "安装",
        "设置",
        "配置",
        "调用",
        "接入",
        "教程",
        "指南",
    ]
    lowered_original = original.lower()
    return any(marker in normalized for marker in query_template_markers) or any(
        marker in lowered_original
        for marker in ["how to", "install", "setup", "configure", "config", "configuration", "query", "search"]
    )


def examples_query_template_local_context(body: str, quote_start: int, original: str) -> bool:
    prefix = re.sub(r"\s+", "", body[max(0, quote_start - 28) : quote_start])
    quote_end = examples_quote_end_index(body, quote_start, original)
    suffix = re.sub(r"\s+", "", body[quote_end : quote_end + 24])
    if re.search(r"(?:问及|提问|询问|查询|请求|搜索|类似|例如|比如|示例|例子|如果用|可以用|输入)$", prefix):
        return True
    return bool(re.match(r"(?:的)?(?:请求|问题|问句|查询|搜索|询问|提问|输入|query|prompt|request)", suffix, re.IGNORECASE))


def examples_quote_end_index(body: str, quote_start: int, original: str) -> int:
    if quote_start < 0 or quote_start >= len(body):
        return max(0, quote_start) + len(original)
    opening = body[quote_start]
    closing = "”" if opening == "“" else '"'
    quote_end = body.find(closing, quote_start + 1)
    if quote_end >= 0:
        return quote_end + 1
    return quote_start + len(original) + 2


def examples_query_template_has_unsafe_marker(normalized: str, original: str = "") -> bool:
    if examples_quote_has_unsafe_marker_for_bypass(normalized, original):
        return True
    if looks_like_user_id_literal(normalized):
        return True
    if re.search(r"https?://|www\.|@|[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{8,}", original):
        return True
    if re.search(r"`[^`]*(?:--|=|/|\\|\d)[^`]*`", original):
        return True
    lowered_original = original.lower()
    unsafe_word_pattern = (
        r"\b(?:token|api[_-]?key|password|passwd|secret|credential|account|permission|payment|refund)\b"
    )
    if re.search(unsafe_word_pattern, lowered_original):
        return True
    if examples_quote_has_factual_eval_marker(original):
        return True
    if examples_quote_has_chinese_factual_eval_marker(normalized):
        return True
    unsafe_markers = ["订单", "交易", "付款", "支付", "退款", "删除", "凭证", "密码", "密钥", "账户", "账号", "权限", "收入", "状态", "张三", "李四", "王五"]
    if any(marker in normalized for marker in unsafe_markers):
        return True
    return bool(re.search(r"\b(?:Alice|Bob|Ethan|Zhang|Li|Wang)\b", original))


def quote_has_dynamic_sensitive_query_marker(normalized: str, original: str = "") -> bool:
    return dynamic_sensitive_query_marker_text(normalized, original) is not None


def dynamic_sensitive_query_marker_text(normalized: str, original: str = "") -> str | None:
    compact = normalized or re.sub(r"\s+", "", original)
    lowered = unicodedata.normalize("NFKC", original).lower()
    if not compact and not lowered:
        return None
    chinese_patterns = [
        r"(?:账户|账号|银行卡|信用卡)?余额",
        r"(?:重置|找回|忘记|忘了|忘掉|修改|更改).{0,6}密码",
        r"密码.{0,8}(?:重置|找回|找不回|忘记|忘了|忘掉|修改|更改)",
        r"(?:无法|不能).{0,6}(?:登录|登陆|登入|账号|账户)",
        r"(?:登录|登陆|登入|账号|账户).{0,6}(?:无法|不能|失败|异常|出错|报错|打不开|登不上)",
        r"(?:登录|登陆|登入)(?:问题|故障|异常)",
        r"(?:用户|客户).{0,2}(?:登录|登陆|登入)(?:问题|故障|异常)",
        r"(?:账号|账户)(?:有)?(?:问题|故障|异常)",
        r"(?:账号|账户)(?:登录|登陆|登入)(?:问题|故障|异常)",
        r"(?:登不上|登录不了|登陆不了|登入不了)",
        r"(?:支付|付款|退款|扣款|转账|充值|提现)",
        r"(?:订单|工单|票据).{0,6}(?:状态|进度|查询|查看|取消|退款)",
        r"(?:登录|登陆).{0,6}(?:记录|日志|状态|会话|session)",
        r"(?:凭证|密钥|令牌|token|cookie|api.?key)",
        r"(?:查询|查看|获取|读取|搜索).{0,6}(?:手机号|手机号码|电话号码|邮箱|邮件地址|住址|个人资料|用户资料|客户资料)",
        r"(?:用户|客户|个人|某个用户|某个客户).{0,8}(?:手机号|手机号码|电话号码|邮箱|邮件地址|住址|个人资料|用户资料|客户资料)",
    ]
    for pattern in chinese_patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            return match.group(0)
    login_verb_pattern = r"(?:login|log\s+in|log-in|signin|sign\s+in|sign-in)"
    login_outcome_pattern = r"(?:failed|failures?|errors?|issues?|problems?)"
    english_patterns = [
        r"\baccount\s+balance\b",
        r"\b(?:reset|forgot|change|update|recover)\s+(?:my|a|the|user|customer|account)?\s*password\b|\bpassword\s+(?:reset|recovery|change|update)\b",
        rf"\b(?:cannot|can't|can not|unable to|failed to)\s+{login_verb_pattern}\b",
        rf"\b(?:(?:user|customer|account)\s+)?{login_verb_pattern}\s+{login_outcome_pattern}\b|\b{login_outcome_pattern}\s+{login_verb_pattern}\b",
        r"(?<!service\s)\b(?:account|user\s+account|customer\s+account)\s+(?:problems?|issues?|errors?)\b",
        r"\b(?:payment|refund|charge|transfer|deposit|withdrawal)\b",
        r"\b(?:order|ticket|issue)\s+(?:status|progress|lookup|query|refund|cancel)\b",
        r"\b(?:login|signin|sign-in|session)\s+(?:record|log|status|history|cookie)\b",
        r"\b(?:credential|credentials|api[_ -]?key|secret|token|cookie|cookies)\b",
        r"\b(?:query|lookup|find|get|search)\s+(?:a\s+)?(?:user|users|user's|users'|customer|customers|customer's|customers'|person|persons|person's|persons'|people|people's).{0,24}\b(?:phones?|emails?|addresses?|profiles?)\b",
        r"\b(?:personal\s+data|user\s+data|customer\s+data)\b",
    ]
    for pattern in english_patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def examples_quote_has_unsafe_marker_for_bypass(normalized: str, original: str = "") -> bool:
    if quote_has_dynamic_sensitive_query_marker(normalized, original):
        return True
    if looks_like_user_id_literal(normalized):
        return True
    if examples_quote_has_personal_name_reference(normalized, original):
        return True
    if examples_quote_has_sensitive_user_data_marker(normalized, original):
        return True
    if severe_factual_claim_marker(original) or severe_factual_claim_marker(normalized):
        return True
    if re.search(r"https?://|www\.|@|[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{8,}", original):
        return True
    return False


def examples_quote_should_warn_not_infer(normalized: str, original: str = "") -> bool:
    return bool(
        unsupported_backing_marker(original)
        or unsupported_backing_marker(normalized)
        or examples_quote_has_factual_eval_marker(original)
        or examples_quote_has_chinese_factual_eval_marker(normalized)
        or contains_short_fact_marker(normalized)
        or contains_hard_fact_marker(normalized)
        or looks_like_mixed_unsupported_example_fact(normalized)
    )


def examples_quote_has_factual_eval_marker(original: str = "") -> bool:
    lowered_original = original.lower()
    return bool(
        re.search(
            r"\b(?:support|supports|supported|improve|improves|improved|best|better|recommend|recommended|"
            r"recommends|prove|proves|proved|cause|causes|caused|release|released|releases|launch|"
            r"launched|launches)\b",
            lowered_original,
        )
    )


def examples_quote_has_chinese_factual_eval_marker(normalized: str) -> bool:
    factual_eval_markers = [
        "最佳",
        "推荐",
        "证明",
        "导致",
        "造成",
        "提升",
        "适合",
        "优于",
        "已经",
        "发布",
        "推出",
        "上线",
        "支持",
        "发现",
        "认为",
        "应该",
        "必须",
    ]
    return any(marker in normalized for marker in factual_eval_markers)


def examples_quote_has_personal_name_reference(normalized: str, original: str = "") -> bool:
    abstracted = normalized
    for marker in ["某个用户", "某位用户", "该用户", "某个客户", "某位客户", "该客户"]:
        abstracted = abstracted.replace(marker, "<abstract_user>")
    surnames = (
        "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏"
        "水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪"
        "汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛"
        "汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强"
        "贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经"
        "房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆荣翁荀羊惠甄"
        "曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯蓬全班秋仲伊"
        "宫宁仇栾甘厉戎祖武符刘詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索"
        "咸籍赖卓蔺屠蒙池乔胥苍双闻党翟谭贡劳姬申冉雍桑桂牛寿边扈燕冀浦"
        "尚农温别庄晏柴瞿阎充慕连茹习艾鱼容向古易戈廖终居衡步都耿满弘匡"
        "国文寇广禄东欧沃利蔚越师巩聂晁辛阚那简饶曾沙养鞠丰关相查荆游竺权益桓公"
    )
    person_objects = (
        "记忆|片段|对话|摘要|工单|订单|偏好|账户|账号|手机号|手机号码|电话|邮箱|"
        "邮件|登录|记录|地址|住址|凭证|密码|密钥|权限|身份证|证件|银行卡|信用卡"
    )
    direct_person_objects = (
        "记忆|片段|对话|摘要|工单|订单|偏好|账户|账号|手机号|手机号码|电话|邮箱|"
        "邮件|登录|记录|凭证|密码|密钥|权限|身份证|证件|银行卡|信用卡"
    )
    person_query_verbs = "查询|搜索|查看|读取|获取|查找|检索"
    chinese_person_ref = bool(
        re.search(rf"[{surnames}][\u4e00-\u9fff]{{1,2}}的(?:{person_objects})", abstracted)
        or re.search(
            rf"(?:{person_query_verbs})[{surnames}][\u4e00-\u9fff]{{1,2}}(?:{direct_person_objects})",
            abstracted,
        )
    )
    if chinese_person_ref:
        return True
    latin_person = r"[A-Za-z][A-Za-z]{1,31}"
    english_person_objects = r"memory|memories|ticket|order|account|phone|email|login|session|cookie|credential|credentials"
    latin_patterns = [
        rf"({latin_person})的({person_objects})",
        rf"(?:{person_query_verbs})({latin_person})(?:的)?({direct_person_objects})",
        rf"\b(?:query|search|lookup|find|get)\s+({latin_person})\s+({english_person_objects})\b",
        rf"\b({latin_person})(?:'s|’s)\s+({english_person_objects})\b",
    ]
    for pattern in latin_patterns:
        for match in re.finditer(pattern, original, re.IGNORECASE):
            if examples_latin_token_is_technical_memory_topic(match.group(1), match.group(2)):
                continue
            return True
    return False


def examples_latin_token_is_technical_memory_topic(token: str, object_name: str) -> bool:
    technical_tokens = {"redis", "mem0", "qwen", "openai", "anthropic", "claude", "mcp", "oauth", "api", "sdk"}
    memory_objects = {"memory", "memories", "记忆"}
    return token.lower() in technical_tokens and object_name.lower() in memory_objects


def examples_quote_has_sensitive_user_data_marker(normalized: str, original: str = "") -> bool:
    lowered_original = original.lower()
    if re.search(
        r"\b(?:payment|refund|credential|credentials|account|permission|password|passwd|secret|"
        r"api[ _-]?keys?|token|tokens|session|cookie|cookies|email|emails|phone|login|passwords|"
        r"secrets|passwds)\b",
        lowered_original,
    ):
        return True
    if examples_english_sensitive_user_data_query(lowered_original):
        return True
    sensitive_markers = [
        "手机号",
        "手机号码",
        "电话号码",
        "电话",
        "邮箱",
        "邮件地址",
        "电子邮件",
        "登录记录",
        "登录日志",
        "登录",
        "身份证",
        "证件",
        "银行卡",
        "信用卡",
        "凭证",
        "密码",
        "密钥",
        "令牌",
        "账户",
        "账号",
        "权限",
        "订单",
        "交易",
        "付款",
        "支付",
        "退款",
    ]
    if any(marker in normalized for marker in sensitive_markers):
        return True
    personal_address_patterns = [
        r"(?:家庭地址|收货地址|通信地址|联系地址|住址)",
        r"(?:用户|客户|个人|某个用户|某个客户|该用户|该客户).{0,6}地址",
        r"地址.{0,6}(?:用户|客户|个人)",
    ]
    return any(re.search(pattern, normalized) for pattern in personal_address_patterns)


def examples_english_sensitive_user_data_query(lowered_original: str) -> bool:
    user_markers = (
        r"users'|user's|users|user|customers'|customer's|customers|customer|people's|people|"
        r"persons'|person's|persons|person|personal"
    )
    sensitive_objects = (
        r"ip\s+address|ips|ip|email\s+address|email|emails|addresses|address|names|name|birthdays|birthday|"
        r"birth\s*date|birthdate|profiles|profile|locations|location|ids|id|credit\s+cards|credit\s+card|"
        r"cards|card|tokens|token|sessions|session|passwords|password|credentials|credential|"
        r"accounts|account|permissions|permission|cookies|cookie|ssns|ssn|social\s+security\s+numbers|"
        r"social\s+security\s+number|passport\s+numbers|passport\s+number|license\s+numbers|license\s+number|"
        r"api\s+keys|api\s+key|secrets|secret|passwds|passwd"
    )
    return bool(
        re.search(rf"\b(?:{user_markers})\b.{{0,32}}\b(?:{sensitive_objects})\b", lowered_original)
        or re.search(rf"\b(?:{sensitive_objects})\b.{{0,32}}\b(?:{user_markers})\b", lowered_original)
    )


def memory_query_call_argument_context(body: str, quote_start: int | None) -> bool:
    if quote_start is None or quote_start < 0:
        return False
    prefix = body[max(0, quote_start - 48) : quote_start]
    return bool(re.search(r"(?:^|[^\w])(?:[\w.]+\.)?(?:recall|search_context)\s*\(\s*$", prefix))


def examples_quote_has_concrete_marker(normalized: str, original: str = "") -> bool:
    if examples_quote_has_sensitive_user_data_marker(normalized, original):
        return True
    if re.search(r"\d|[%％$￥¥]|https?://|www\.|@|[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{8,}", original):
        return True
    lowered_original = original.lower()
    if re.search(
        r"\b(?:build|order|ticket|issue|status|success|failed|error|token|api[_-]?key|password|"
        r"payment|refund|credential|credentials|account|permission|session|cookie|email|phone|login)\b",
        lowered_original,
    ):
        return True
    if re.search(r"`[^`]*(?:--|=|/|\\|\d)[^`]*`", original):
        return True
    placeholder_safe_original = re.sub(r"用户偏好\s*X", "用户偏好", original, flags=re.IGNORECASE)
    if re.search(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", placeholder_safe_original):
        return True
    concrete_markers = [
        "蓝色",
        "红色",
        "绿色",
        "黄色",
        "科幻电影",
        "笔记本电脑",
        "型号",
        "星巴克",
        "华为",
        "小米",
        "苹果手机",
        "北京",
        "南京",
        "上海",
        "深圳",
        "广州",
        "MacBook",
        "XPS",
        "iPhone",
        "OpenAI",
        "Anthropic",
        "Claude",
        "Alice",
        "Bob",
        "张三",
        "李四",
        "王五",
        "订单",
        "交易",
        "付款",
        "支付",
        "退款",
        "删除",
        "凭证",
        "密码",
        "密钥",
        "token",
    ]
    return any(marker.lower() in lowered_original for marker in concrete_markers) or any(
        marker in normalized for marker in ["购买了华为", "北京门店", "星巴克"]
    )

def contains_hard_fact_marker(normalized: str) -> bool:
    if re.search(r"\d|[0-9]+(?:%|％)?", normalized):
        return True
    hard_markers = [
        "增长",
        "下降",
        "增加",
        "减少",
        "降低",
        "提升",
        "裁撤",
        "裁员",
        "超过",
        "超出",
        "少于",
        "高于",
        "低于",
        "达到",
        "收入",
        "销量",
        "预算",
        "成本",
        "团队",
        "市场份额",
        "融资",
        "估值",
        "一半",
        "三倍",
        "两倍",
        "数倍",
        "百万",
        "千万",
        "上亿",
        "亿元",
        "万美元",
        "人民币",
    ]
    return any(marker in normalized for marker in hard_markers)


def looks_like_instructional_example(normalized: str, prefix: str) -> bool:
    context_markers = ["风格", "示例", "例子", "指令", "要求", "条件性", "规定性", "禁止性", "解释性", "描述性"]
    if not any(marker in prefix for marker in context_markers):
        return False
    instruction_markers = [
        "如果",
        "若",
        "请",
        "必须",
        "不要",
        "禁止",
        "运行",
        "使用",
        "避免",
        "优先",
        "更新",
        "拆分",
        "should",
        "must",
        "donot",
        "don't",
    ]
    return any(marker in normalized.lower() for marker in instruction_markers)


def looks_like_concept_phrase(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text.strip())
    if not normalized:
        return False
    if re.search(r"[。！？!?；;，,：:]", normalized):
        return False
    if looks_like_evaluation_question_template(normalized):
        return True
    if looks_like_named_concept_label(normalized):
        return True
    if len(normalized) > 18:
        return False
    if contains_short_fact_marker(normalized):
        return False
    if re.search(r"发布(?:了|过|出|到|为|成)|推出(?:了|过)|上线(?:了|过)", normalized):
        return False
    sentence_markers = [
        "认为",
        "表示",
        "指出",
        "发现",
        "证明",
        "承诺",
        "宣布",
        "导致",
        "因为",
        "所以",
        "已经",
        "正在",
        "应该",
        "必须",
    ]
    return not any(marker in normalized for marker in sentence_markers)


def looks_like_abstract_trend_label(normalized: str) -> bool:
    if len(normalized) > 24:
        return False
    if re.search(r"\d|[%％$￥¥]|[。！？!?；;，,：:]", normalized):
        return False
    if not any(marker in normalized for marker in ["降低", "提升", "变化", "融合", "模糊", "吞噬", "收敛"]):
        return False
    abstract_subjects = [
        "技术壁垒",
        "进入门槛",
        "代码成本",
        "模型能力",
        "角色边界",
        "产品边界",
        "产品一致性",
        "产品功能",
        "工程成本",
        "能力边界",
    ]
    return any(subject in normalized for subject in abstract_subjects)


def looks_like_named_concept_label(normalized: str) -> bool:
    if len(normalized) > 32:
        return False
    if not re.search(r"[a-zA-Z]", normalized):
        return False
    if re.search(r"[%％$￥¥]|\d+(?:\.\d+)?(?:倍|万|亿|元|美元|%|％)", normalized):
        return False
    if re.search(r"(?:有|含|包含|包括|分为|需要)\d+|\d+(?:个|类|种|步|步骤|层|点|项|条|大|次|年|月|日)", normalized):
        return False
    if any(marker in normalized for marker in ["增长", "下降", "增加", "减少", "超过", "达到", "裁撤", "裁员", "预算", "收入", "销量"]):
        return False
    label_markers = [
        "claude",
        "cowork",
        "工作流",
        "agent",
        "rag",
        "系统",
        "架构",
        "对比",
        "工程",
        "构建",
        "技术栈",
        "框架",
        "模型",
        "定位",
        "选择",
        "n8n",
        "langflow",
    ]
    return any(marker in normalized.lower() for marker in label_markers)


def looks_like_evaluation_question_template(normalized: str) -> bool:
    if not any(marker in normalized for marker in ["是否", "多少", "几", "如何", "什么", "哪"]):
        return False
    evaluation_markers = ["满意", "信任", "接受", "成功", "正确", "失败", "质量", "评估", "比例"]
    return any(marker in normalized for marker in evaluation_markers)


def contains_short_fact_marker(normalized: str) -> bool:
    if re.search(r"\d|[0-9]+(?:%|％)?", normalized):
        return True
    metric_markers = [
        "增长",
        "下降",
        "增加",
        "减少",
        "降低",
        "提升",
        "裁撤",
        "裁员",
        "超过",
        "超出",
        "少于",
        "高于",
        "低于",
        "达到",
        "收入",
        "销量",
        "预算",
        "成本",
        "用户",
        "团队",
        "市场份额",
        "融资",
        "估值",
    ]
    quantity_markers = ["一半", "三倍", "两倍", "数倍", "百万", "千万", "上亿", "亿元", "万美元", "人民币"]
    return any(marker in normalized for marker in [*metric_markers, *quantity_markers])


def build_draft_grounding_review(
    artifact: DraftRenderingArtifact,
    plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    approved_raw_text: str,
) -> DraftGroundingReview:
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    claims: list[GroundingClaim] = []
    for page in artifact.pages:
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            continue
        grounding_page = draft_page_for_grounding(page)
        entry = snapshot_entry(snapshot, f"wiki/{page.canonical_target_path}")
        collect_grounding_claims(
            item=item,
            page=grounding_page,
            existing_entry=entry,
            approved_raw_text=approved_raw_text,
            claims=claims,
        )
    return draft_grounding_review_from_claims(claims)


def draft_page_for_grounding(page: DraftPageItem) -> DraftPageItem:
    return page.model_copy(
        update={
            "summary": page.summary.strip(),
            "body_markdown": _draft_validation.normalize_core_body_markdown(page.body_markdown),
            "open_questions": page.open_questions.strip(),
        }
    )


def looks_like_mixed_unsupported_example_fact(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    compact = re.sub(r"\s+", "", normalized)
    lowered = normalized.lower()
    if looks_like_metric_or_outcome_literal(compact):
        return True
    markers = [
        "导致",
        "造成",
        "引发",
        "影响",
        "失败",
        "成功",
        "完成",
        "错误",
        "异常",
        "购买",
        "买了",
        "使用",
        "访问",
        "消费",
        "下单",
        "删除",
        "退款",
        "付款",
        "支付",
        "交易",
        "订单",
        "状态",
        "收入",
        "成本",
        "凭证",
        "密码",
        "账户",
        "账号",
        "权限",
        "授权",
        "敏感",
        "purchased",
        "bought",
        "uses",
        "used",
        "visited",
        "visit",
        "deleted",
        "delete",
        "completed",
        "status",
        "success",
        "failed",
        "failure",
        "error",
        "refund",
        "payment",
        "order",
        "revenue",
        "credential",
        "credentials",
        "password",
        "account",
        "permission",
        "sensitive",
    ]
    return any(marker in lowered for marker in markers)


def looks_like_user_id_literal(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return bool(
        re.fullmatch(r"(?:user|uid|customer|account)[-_ ]?\d+", normalized, re.IGNORECASE)
        or re.fullmatch(r"(?:用户|客户|账户|账号)\s*\d+", normalized)
        or re.fullmatch(r"(?:客户|用户)\s+(?:user|uid)[-_ ]?\d+", normalized, re.IGNORECASE)
    )


def looks_like_metric_or_outcome_literal(compact: str) -> bool:
    if not compact:
        return False
    metric_markers = [
        "增长",
        "下降",
        "增加",
        "减少",
        "降低",
        "提升",
        "裁撤",
        "裁员",
        "超过",
        "达到",
        "销量",
        "收入",
        "预算",
        "市场份额",
        "三倍",
        "两倍",
        "一半",
        "百万",
        "千万",
        "上亿",
    ]
    return any(marker in compact for marker in metric_markers)


def draft_grounding_review_from_claims(claims: list[GroundingClaim]) -> DraftGroundingReview:
    unsupported_new_facts = [
        claim
        for claim in claims
        if claim.claim_type == "new_fact" and claim.support == "unsupported" and claim.action == "needs_review"
    ]
    warnings = [
        claim
        for claim in claims
        if claim.claim_type == "new_fact" and claim.support == "unsupported" and claim.action == "warn"
    ]
    return DraftGroundingReview(
        unsupported_new_facts=unsupported_new_facts,
        warnings=warnings,
        claims=claims,
        requires_review=bool(unsupported_new_facts),
    )

def render_draft_grounding_review(review: DraftGroundingReview) -> str:
    if review.requires_review:
        summary = "需要人工确认"
    elif review.warnings:
        summary = "通过，有非阻塞提醒"
    else:
        summary = "通过"
    rows = [
        [
            claim.page_plan_id,
            f"`{claim.target_path}`",
            claim.section_key,
            claim.claim_type,
            claim.support,
            claim.action,
            claim.reason,
            claim.text[:240],
        ]
        for claim in review.claims
    ]
    unsupported_rows = [
        [
            claim.page_plan_id,
            f"`{claim.target_path}`",
            claim.section_key,
            claim.reason,
            claim.text[:240],
        ]
        for claim in review.unsupported_new_facts
    ]
    warning_rows = [
        [
            claim.page_plan_id,
            f"`{claim.target_path}`",
            claim.section_key,
            claim.reason,
            claim.text[:240],
        ]
        for claim in review.warnings
    ]
    sections = [
        "# 草稿来源支撑审查",
        "",
        f"- 结果：{summary}",
        f"- 未支撑新增事实数量：{len(review.unsupported_new_facts)}",
        f"- 非阻塞提醒数量：{len(review.warnings)}",
        "",
        "## 需要确认的新事实",
        "",
        format_markdown_table(["页面计划", "目标", "段落", "原因", "文本"], unsupported_rows) if unsupported_rows else "暂无。",
        "",
        "## 非阻塞提醒",
        "",
        format_markdown_table(["页面计划", "目标", "段落", "原因", "文本"], warning_rows) if warning_rows else "暂无。",
        "",
        "## 全部分类",
        "",
        format_markdown_table(["页面计划", "目标", "段落", "类型", "支持", "处理", "原因", "文本"], rows) if rows else "暂无分类记录。",
    ]
    return "\n".join(sections).rstrip() + "\n"
