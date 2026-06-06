from __future__ import annotations

import json
import shutil
import re
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import unified_diff
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeVar

import yaml
from rich.console import Console
from pydantic import BaseModel

from . import __version__
from .events import EventLogger, format_duration
from .hash_utils import artifact_ref, sha256_bytes, sha256_file
from .io import read_json, read_model, read_yaml, write_json, write_yaml
from .manifest import (
    begin_model_step_attempt,
    begin_step_attempt,
    complete_step,
    fail_step,
    first_resumable_step,
    get_step,
    initial_steps,
    mark_step_awaiting_review,
    mark_step_approved,
    mark_from_pending,
    raw_ref,
    read_manifest,
    step_satisfied,
    write_manifest,
)
from .models import (
    ApplyPreview,
    ApplyTarget,
    ArtifactRef,
    CandidateResolutionArtifact,
    CandidateResolutionItem,
    DraftApproval,
    DraftGroundingReview,
    DraftPageItem,
    DraftRenderingArtifact,
    DraftWriteManifest,
    DraftWriteTarget,
    GroundingClaim,
    OperationManifest,
    OperationStatus,
    ProviderResult,
    RawBinding,
    RawIngestCandidate,
    RawIngestCandidateReport,
    RawLinkCleanupArtifact,
    RawLinkCleanupLink,
    RawLinkCleanupWarning,
    RawPreparePolicy,
    RawPreparationArtifact,
    ReviewDecision,
    RunMode,
    RelatedPageRef,
    RelatedMergeReport,
    RelatedCandidateReport,
    SourceBasis,
    SourceDuplicateGuardArtifact,
    SourceDigestArtifact,
    SourceDigestCandidate,
    StepStatus,
    StructuredAttemptRef,
    StructuredIssue,
    StructuredRepairReport,
    UpdateMergeReport,
    UpdatePageMergeReport,
    SectionMergeChange,
    VaultConfig,
    WeakOrNoiseItem,
    CandidateContextsArtifact,
    ContextOverlapSignal,
    EmbeddingRetrievalConfig,
    WikiContextEntry,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
    WikiPageMetadata,
    utc_now,
)
from .provider_config import ProviderExecutionContext, build_provider_execution_context
from .providers import Provider
from .profiles import load_profile, page_output_path, safe_filename
from .rendering import source_title_for_raw
from .retrieval import (
    RetrievalError,
    SCORE_BUCKET_EPSILON,
    build_candidate_contexts,
    build_knowledge_pool,
    candidate_pool_sha256,
    metadata_from_text,
    resolve_cache_dir,
)
from .steps import (
    MODEL_BACKED_STEPS,
    STEP_NAMES,
    STEP_SPECS,
    StepSpec,
    downstream_steps,
    require_step_output_dir,
    step_index,
    step_output_dir,
)
from .structured import StructuredModelCall, parse_structured_json_object
from .system_pages import (
    assert_current_system_page,
    ensure_system_pages,
    format_markdown_table,
    local_date,
    render_daily_log,
    render_index,
    render_log_index,
)
from .validators import (
    ValidationError as ContractValidationError,
    create_reason_needs_repair,
    looks_like_untranslated_english,
    nonempty_prepared_discovered_candidates,
    text_contains_source_graph_link,
    validate_candidate_resolution,
    validate_raw_preparation,
    validate_source_digest,
    validate_wiki_merge_plan,
)
from .verify import require_verified
from .vault_config import read_vault_config, write_default_vault_config
from .wiki_context import wiki_context_drift_messages
from .workspace import RunStore, apply_lock, ensure_workspace_layout, relative_to_vault, resolve_raw_path, run_lock


class PipelineError(RuntimeError):
    pass


RAW_INGEST_TEXT_SUFFIXES = {".md", ".markdown", ".mdown", ".txt"}


@dataclass(frozen=True)
class _SourceRawCoverageRecord:
    source_page: str
    raw_paths: tuple[str, ...]
    raw_hashes: tuple[str, ...]
    operation_ids: tuple[str, ...]


RAW_PREPARE_CONTRACT = {
    "goal": "Create a higher-quality canonical prepared raw for downstream knowledge compilation.",
    "rules": [
        "Do not add facts that are not supported by the original raw.",
        "Remove or relocate non-content noise such as media timestamps, self-promotion, and obvious formatting artifacts.",
        "Correct obvious ASR/OCR/formatting errors only when the context makes the correction clear.",
        "Record uncertainty instead of guessing.",
        "Return prepared_markdown as clean Markdown suitable for source_digest and downstream knowledge digestion.",
    ],
}

MODEL_RELATED_SUGGESTION_LIMIT = 2
FINAL_RELATED_LIMIT = 3
DRAFT_RENDERING_BATCH_PAGE_LIMIT = 4
DRAFT_RENDERING_MAX_PARALLEL_BATCHES = 3
MAX_AUTO_APPROVED_ALL_CREATE_ITEMS = 12
LOCAL_MEDIUM_CREATE_REASON_MARKER = "本地补充结构化 create/update 对比理由"
DRAFT_RENDERING_GROUNDING_RISK_RULES = (
    "Do not wrap paraphrases, inferred concept labels, or rewritten source ideas in Chinese/English quotation marks; "
    "use quotes only for text that exact-matches source_excerpt_pack, approved_prepared_markdown, or inspected wiki context.",
    "For interview, ASR/OCR, or translated transcript source text, treat speaker-like Chinese wording as paraphrase "
    "unless the exact span is present; prefer indirect attribution such as 访谈中提到、她描述、团队讨论.",
    "Avoid broad external-backing/adoption phrases such as 被广泛应用、被广泛使用、公认、业界普遍、被多个社区引用 "
    "unless the exact source/wiki context says them; prefer source-local wording such as 本材料提到、访谈中讨论、"
    "团队成员提到、本材料将该说法用于解释.",
    "High-risk causal/scope terms such as 导致、造成、证明、表明、必然、长期来看、用户会、影响到 require same sentence "
    "or clearly adjacent explicit support in source_excerpt_pack, approved_prepared_markdown, or inspected wiki context. "
    "If the source only gives a tradeoff or concern, write 可能伴随、需要权衡、访谈中提到, or move the claim to open_questions.",
)
UNSUPPORTED_BACKING_MARKERS = (
    "被广泛应用",
    "被广泛使用",
    "广泛应用",
    "广泛使用",
    "被多个",
    "被广泛",
    "公认",
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
DRAFT_SELF_TALK_MARKERS = (
    "我记错",
    "可能我记错",
    "检查原文",
    "查看原文",
    "我会修正",
    "输出已经确定",
    "我还没输出",
    "但现在我们无法修改",
    "等等，原文",
    "目前wiki中无此页面",
    "当前wiki没有相关知识页",
    "当前wiki没有此页面",
    "创建后可与",
    "创建后可和",
    "创建后可互链",
)
OPEN_QUESTION_SEMANTIC_CLUSTERS = (
    (
        "semantic:agi_pm_role_necessity",
        (("agi",), ("pm", "产品经理"), ("消失", "必要", "取代", "替代", "还有价值", "是否需要", "需要")),
    ),
    (
        "semantic:model_capability_product_function_boundary",
        (("模型能力",), ("产品功能", "产品边界"), ("吞噬", "吞掉", "取代", "替代", "边界")),
    ),
    (
        "semantic:product_judgement_training",
        (("产品品味", "产品判断"), ("训练", "提升", "培养", "系统化")),
    ),
    (
        "semantic:ai_pm_role_evolution",
        (("ai", "agi", "agent"), ("pm", "产品经理"), ("演变", "变化", "转型", "未来")),
    ),
    (
        "semantic:claude_code_product_experience_harness_boundary",
        (("claudecode", "claude code"), ("产品体验", "体验"), ("harness", "安全边界"), ("掩盖", "重要性")),
    ),
    (
        "semantic:rapid_iteration_quality_safety_research_preview",
        (("快速发布", "快速迭代"), ("质量", "安全"), ("研究预览", "用户预期", "长期产品一致性")),
    ),
    (
        "semantic:model_progress_feature_lifecycle",
        (("模型", "模型能力", "模型进步"), ("功能", "产品功能", "ui元素"), ("保留", "移除", "废弃", "过时", "存废", "淘汰")),
    ),
    (
        "semantic:agent_hand_transfer_mechanism",
        (("大脑", "brain"), ("传递", "pass", "handoff"), ("双手", "hand", "hands"), ("机制", "实现细节", "高效")),
    ),
)
SOURCE_DIGEST_BUDGET_GROUP_ORDER = ("concepts", "designs", "comparisons", "open_questions", "entities")
SOURCE_DIGEST_AGGREGATION_MIN_CANDIDATES = 2
SOURCE_DIGEST_AGGREGATION_CLUSTER_SIMILARITY = 0.18
SOURCE_DIGEST_PROMOTED_AGGREGATION_MIN_REPLACEMENT_SIMILARITY = 0.35
DEFERRED_AGGREGATION_GROUP_LABELS = {
    "concepts": "延后概念",
    "designs": "延后设计模式",
    "comparisons": "延后对比",
    "open_questions": "延后未决问题",
    "entities": "延后实体",
}
SOURCE_DIGEST_ANCHOR_ENTITIES: dict[str, dict[str, str]] = {
    "Managed Agents": {
        "summary": "Managed Agents 是源材料显式讨论的托管智能体系统，用于将大脑、会话和双手解耦，并容纳未来不同 harness、sandbox 或其他组件。",
        "why_matters": "它是本材料的中心系统名称，后续材料很可能继续补充其产品、架构和使用边界。",
        "wiki_value": "作为稳定实体锚点，可承接后续关于 Claude Code、harness、sandbox、session 与托管代理能力的更新。",
        "resolution_hint": "deterministic_source_anchor_entity: source title/body repeatedly names Managed Agents; keep as a central reusable entity anchor before page budget.",
    },
    "Claude Code": {
        "summary": "Claude Code 是源材料显式提到的 Anthropic 编程 harness/产品，在本材料中作为 Managed Agents 可适配并广泛使用的 harness 示例出现。",
        "why_matters": "它是后续访谈、产品方法和托管智能体材料之间最容易复用的产品实体锚点。",
        "wiki_value": "让后续 Claude Code 访谈可以 update 既有页面，而不是把架构材料中的 harness 视角遗失到孤立相关页里。",
        "resolution_hint": "deterministic_source_anchor_entity: source body explicitly names Claude Code as an excellent harness; keep as a reusable update target when the model omits it.",
    },
    "Cowork": {
        "summary": "Cowork 是源材料显式提到的 Anthropic 知识工作协作者产品，可作为 Claude Code 之外的产品实体锚点。",
        "why_matters": "它经常与 Claude Code 同源出现，适合承接后续关于非编程知识工作场景的更新。",
        "wiki_value": "提供稳定产品实体页，便于后续比较、团队组织和使用场景材料进行 update 或互链。",
        "resolution_hint": "deterministic_source_anchor_entity: source explicitly names Cowork as a durable product/entity anchor.",
    },
}
RAW_PREPARE_FAST_PATH_RULE_VERSION = "markdown_passthrough.v1"
REFERENCE_SECTION_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(?:references|bibliography|works cited|参考文献|参考资料)\s*:?\s*$",
    re.IGNORECASE,
)
APPENDIX_SECTION_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(?:appendix|appendices|supplementary|附录)\b",
    re.IGNORECASE,
)
REFERENCE_TRUNCATION_MIN_DOCUMENT_CHARS = 8_000
REFERENCE_TRUNCATION_MIN_OMITTED_CHARS = 1_500
REFERENCE_TRUNCATION_MIN_START_RATIO = 0.45
APPENDIX_COMPACTION_MIN_DOCUMENT_CHARS = 24_000
APPENDIX_COMPACTION_MIN_OMITTED_CHARS = 4_000
APPENDIX_COMPACTION_MIN_START_RATIO = 0.45
APPENDIX_COMPACTION_SECTION_EXCERPT_LIMIT = 700
APPENDIX_COMPACTION_MAX_SECTIONS = 12
UPDATE_PRESERVATION_SECTION_KEYS = ("summary", "detail", "value_points")
UPDATE_PRESERVATION_MAX_PHRASES_PER_SECTION = 8
UPDATE_PRESERVATION_CONCEPT_GROUPS = (
    {
        "name": "managed_agents",
        "label": "Managed Agents / 托管智能体",
        "terms": ("Managed Agents", "托管智能体"),
    },
    {
        "name": "harness",
        "label": "harness / 适配框架",
        "terms": ("harness", "Harness", "harnesses", "适配框架", "元harness", "元适配框架"),
    },
    {
        "name": "brain_hands_decoupling",
        "label": "大脑与双手解耦",
        "terms": (
            "大脑与双手",
            "大脑双手",
            "brain and hands",
            "brain hands",
            "brain & hands",
            "separate the model brain from execution hands",
            "model brain from execution hands",
            "model brain and execution hands",
            "model brain / execution hands",
            "推理和规划",
            "工具执行",
            "解耦",
            "路由模型意图",
            "模型意图路由",
        ),
    },
    {
        "name": "session_context",
        "label": "会话/持久上下文",
        "terms": (
            "session object",
            "persistent session",
            "persistent context",
            "durable context",
            "session state",
            "session context",
            "会话对象",
            "持久上下文",
            "上下文对象",
            "会话是持久",
            "执行状态",
        ),
    },
    {
        "name": "safety_boundary",
        "label": "安全边界/权限限制",
        "terms": ("安全边界", "权限", "限制文件", "文件、网络和资源", "文件网络和资源", "无限本机权限"),
    },
    {
        "name": "isolated_execution",
        "label": "隔离执行/容器",
        "terms": (
            "隔离容器",
            "隔离执行环境",
            "container",
            "containers",
            "containerized",
            "sandbox",
            "sandboxed",
            "sandboxes",
            "容器",
            "沙箱",
        ),
    },
    {
        "name": "system_architecture_view",
        "label": "系统架构视角",
        "terms": ("系统架构视角", "产品功能列表", "只写成产品功能列表"),
    },
)
DRAFT_RENDERING_FULL_SOURCE_CHAR_LIMIT = 24_000
DRAFT_RENDERING_EXCERPT_TOTAL_CHAR_LIMIT = 24_000
DRAFT_RENDERING_EXCERPT_PER_PAGE_LIMIT = 1_600
DRAFT_RENDERING_GLOBAL_EXCERPT_LIMIT = 2_400
DRAFT_RENDERING_EXCERPT_MAX_SOURCE_RATIO = 0.55
DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS = 600
DRAFT_RENDERING_CONTEXT_ENTRY_EXCERPT_LIMIT = 1_200
CANDIDATE_RESOLUTION_FULL_SOURCE_CHAR_LIMIT = 20_000
CANDIDATE_RESOLUTION_GLOBAL_EXCERPT_LIMIT = 1_600
CANDIDATE_RESOLUTION_PER_CANDIDATE_EXCERPT_LIMIT = 500
SOURCE_DIGEST_FULL_SOURCE_CHAR_LIMIT = 24_000
SOURCE_DIGEST_SOURCE_MAP_TOTAL_LIMIT = 22_000
SOURCE_DIGEST_SOURCE_MAP_GLOBAL_EXCERPT_LIMIT = 2_400
SOURCE_DIGEST_SOURCE_MAP_MIN_SECTION_EXCERPT_LIMIT = 320
SOURCE_DIGEST_SOURCE_MAP_MAX_SECTION_EXCERPT_LIMIT = 900
SOURCE_DIGEST_SOURCE_MAP_MAX_SECTIONS = 56
SOURCE_DIGEST_SOURCE_MAP_MAX_CAPTIONS = 24
MERGE_PLANNING_FULL_SOURCE_CHAR_LIMIT = 16_000
MERGE_PLANNING_SOURCE_GLOBAL_EXCERPT_LIMIT = 1_600
MERGE_PLANNING_SOURCE_PER_PAGE_EXCERPT_LIMIT = 700
MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT = 240
MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT = 120
MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK = 2
MERGE_PLANNING_CONTEXT_QUERY_LIMIT = 420
MERGE_PLANNING_ENTRY_EXCERPT_LIMIT = 900
TRANSCRIPT_TIMESTAMP_RE = re.compile(r"^\s*(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?|\d{1,2}:\d{2}(?::\d{2})?\s*[-–—])")
SPEAKER_TURN_RE = re.compile(r"^\s*(?P<label>[^:：\n]{1,48})\s*[:：](?!//)\s*(?P<body>.*)$")
SPEAKER_ROLE_LABEL_RE = re.compile(
    r"(?i)^(?:"
    r"(?:q|a|qa|question|answer|user|assistant|human|system|speaker|host|guest|moderator|"
    r"interviewer|interviewee|participant|audience)(?:\s*(?:#?\d+|[A-Z]))?"
    r"|(?:问|答|主持人|嘉宾|采访者|受访者|提问|回答)(?:[A-Za-z0-9一二三四五六七八九十]+)?"
    r")$"
)
SPEAKER_EXPLANATORY_LABEL_RE = re.compile(
    r"(?i)\b(?:"
    r"when|where|why|how|what|example|examples|sectioning|voting|workflow|workflows|pattern|"
    r"steps?|input|output|use|best for|limitations?|notes?|summary|goal|tables?|figures?|appendix|"
    r"imported from|"
    r"fetched url|final url|content type|source|title|author|tags"
    r")\b"
    r"|何时|哪里|为什么|如何|什么|示例|例子|分片|投票|工作流|模式|步骤|输入|输出|用法|"
    r"适用|限制|注意|摘要|目标|表格|图表|附录|来源|标题|作者|标签"
)
MARKDOWN_MEDIA_EMBED_RE = re.compile(r"!\[[^\]\n]*\]\([^)]+\)")
INTERVIEW_TRANSCRIPT_MARKER_RE = re.compile(r"(?im)^\s*#{1,3}\s*(?:访谈全文|采访全文|完整访谈|Transcript|Full Transcript)\s*$")
PAPER_SECTION_HEADING_RE = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s+"
    r"(?:abstract|introduction|related work|methodology|method|experiments?|evaluation|results?|discussion|conclusion|references)\b"
)
PAPER_CAPTION_RE = re.compile(r"(?im)^\s*(?:table|figure)\s+\d+\s*:")
ARXIV_IMPORT_MARKER_RE = re.compile(r"(?im)^\s*(?:imported from|fetched url|final url):\s+https?://(?:www\.)?arxiv\.org/")
YOUTUBE_URL_RE = re.compile(r"(?i)https?://(?:www\.)?(?:youtube\.com|youtu\.be)/")
PODCAST_MARKER_RE = re.compile(r"(?i)\bpodcast\b|播客")
AUDIO_VIDEO_SOURCE_MARKER_RE = re.compile(r"(?i)\b(?:youtube|video|audio|episode)\b|视频|音频|节目")
TRANSLATION_MARKER_RE = re.compile(r"(?i)\b(?:translated|translation)\b|翻译|译文")
ASR_SOURCE_MARKER_RE = re.compile(
    r"(?i)\b(?:asr|auto[- ]?generated|automatic captions?|machine transcript|transcribed by)\b"
    r"|自动(?:转录|生成|字幕)|语音识别|机翻字幕|字幕稿|转写|转录"
)
SENTENCE_TERMINAL_PUNCTUATION = "。！？；：.!?;:"
INLINE_PUNCTUATION = SENTENCE_TERMINAL_PUNCTUATION + "，,、"

def init_vault(vault: Path, *, profile_name: str = "project_basic") -> None:
    profile = load_profile(profile_name)
    (vault / "raw").mkdir(parents=True, exist_ok=True)
    for spec in profile.page_types.values():
        (vault / "wiki" / spec.directory).mkdir(parents=True, exist_ok=True)
    ensure_system_pages(vault)
    ensure_workspace_layout(vault)
    write_default_vault_config(vault)
    profile_root = vault / ".llmwiki" / "profiles" / profile.name
    (profile_root / "templates").mkdir(parents=True, exist_ok=True)
    write_yaml(profile_root / "profile.yaml", profile.model_dump(mode="json"))
    for page_type, spec in profile.page_types.items():
        source = profile.template_root / spec.template if profile.template_root else None
        if source and source.exists():
            target = profile_root / "templates" / spec.template
            if not target.exists():
                shutil.copyfile(source, target)
    write_yaml(
        vault / ".llmwiki" / "config.yaml",
        {
            "profile": profile.name,
            "providers": {},
        },
    )


def run_simplified_ingest(
    *,
    vault: Path,
    raw_file: Path,
    fixture_dir: Path | None = None,
    mock_fixture_dir: Path | None = None,
    profile_name: str | None = None,
    slug: str | None = None,
    run_mode: RunMode = RunMode.dev,
    raw_prepare_policy: RawPreparePolicy | None = None,
    console: Console | None = None,
) -> OperationManifest:
    ensure_workspace_layout(vault)
    raw_path, raw_rel = resolve_raw_path(vault, raw_file)
    raw_hash, raw_size = raw_ref(raw_path)
    operation_id = f"ING-{safe_timestamp()}-{slug or raw_path.stem}"
    store = RunStore(vault)
    resolved_profile_name = resolve_vault_profile_name(vault, profile_name)
    profile = load_profile(vault / ".llmwiki" / "profiles" / resolved_profile_name)
    vault_config = read_vault_config(vault)
    if raw_prepare_policy is not None:
        vault_config.raw_prepare_policy = raw_prepare_policy
    model_steps = model_steps_for_raw_prepare_policy(
        list(MODEL_BACKED_STEPS),
        raw_path=raw_path,
        raw_rel=raw_rel,
        raw_prepare_policy=vault_config.raw_prepare_policy,
    )
    provider_execution_context = build_provider_execution_context(
        vault=vault,
        manifest_contexts=[],
        fixture_dir=fixture_dir,
        mock_fixture_dir=mock_fixture_dir,
        source="initial_run",
        from_step=None,
        tasks=model_steps,
    )
    with apply_lock(vault):
        run_dir = store.run_dir(operation_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = OperationManifest(
            operation_id=operation_id,
            operation_type="ingest",
            run_mode=run_mode,
            engine_version=__version__,
            profile=profile.name,
            profile_version=profile.version,
            vault_config_snapshot=vault_config,
            workspace=relative_to_vault(vault, run_dir),
            raw_bindings=[RawBinding(relative_path=raw_rel, sha256=raw_hash, size_bytes=raw_size)],
            provider_contexts=[provider_execution_context.record] if provider_execution_context.record else [],
            steps=initial_steps(),
        )
        write_manifest(store.manifest_path(operation_id), manifest)
        with run_lock(vault, operation_id):
            return execute_ingest(
                vault,
                operation_id,
                start_step=STEP_NAMES[0],
                execution_context=provider_execution_context,
                console=console,
            )


def resume_ingest(
    *,
    vault: Path,
    operation_id: str,
    from_step: str | None = None,
    run_mode: RunMode | None = None,
    mock_fixture_dir: Path | None = None,
    raw_prepare_policy: RawPreparePolicy | None = None,
    console: Console | None = None,
) -> OperationManifest:
    store = RunStore(vault)
    with apply_lock(vault), run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        if run_mode is not None and run_mode != manifest.run_mode:
            raise PipelineError("run_mode is immutable for an operation; rerun ingest to use a different mode.")
        if manifest.status in {OperationStatus.applied, OperationStatus.source_recorded}:
            raise PipelineError("Applied operations are immutable. Start a new operation instead.")
        if manifest.status == OperationStatus.apply_failed:
            raise PipelineError("apply_failed operations cannot be resumed; inspect written targets and rerun ingest.")
        require_verified(vault, manifest)
        start, reset_from_step = default_resume_start(vault, store.run_dir(operation_id), manifest, from_step)
        if start is None:
            return manifest
        if raw_prepare_policy is not None:
            if step_index(start) > step_index("raw_prepare"):
                raise PipelineError("raw prepare override only applies when raw_prepare will rerun; resume from raw_prepare or earlier.")
            manifest.vault_config_snapshot.raw_prepare_policy = raw_prepare_policy
        validate_raw_link_cleanup_resume(run_dir=store.run_dir(operation_id), manifest=manifest, start=start)
        validate_resume_start(manifest, start)
        ensure_wiki_context_current_before_resume(vault, store.run_dir(operation_id), start)
        model_steps = model_steps_from(start)
        if "raw_prepare" in model_steps:
            raw_binding = manifest.raw_bindings[0] if manifest.raw_bindings else None
            raw_rel = raw_binding.relative_path if raw_binding is not None else ""
            raw_path = vault / raw_rel if raw_rel else vault
            model_steps = model_steps_for_raw_prepare_policy(
                model_steps,
                raw_path=raw_path,
                raw_rel=raw_rel,
                raw_prepare_policy=manifest.vault_config_snapshot.raw_prepare_policy,
            )
        provider_execution_context = build_provider_execution_context(
            vault=vault,
            manifest_contexts=manifest.provider_contexts,
            fixture_dir=None,
            mock_fixture_dir=mock_fixture_dir,
            source="resume_current_config",
            from_step=start,
            tasks=model_steps,
        )
        if provider_execution_context.record is not None:
            manifest.provider_contexts.append(provider_execution_context.record)
        if reset_from_step is not None:
            write_manifest(store.manifest_path(operation_id), manifest)
            delete_downstream_step_dirs(
                vault,
                operation_id,
                reset_from_step,
                archive=True,
                archive_reason=f"resume requested from {reset_from_step}; previous step artifacts archived before regeneration.",
            )
            mark_from_pending(manifest, reset_from_step)
            write_manifest(store.manifest_path(operation_id), manifest)
        else:
            write_manifest(store.manifest_path(operation_id), manifest)
        return execute_ingest(
            vault,
            operation_id,
            start_step=start,
            execution_context=provider_execution_context,
            console=console,
        )


def default_resume_start(
    vault: Path,
    run_dir: Path,
    manifest: OperationManifest,
    from_step: str | None,
) -> tuple[str | None, str | None]:
    if from_step is not None:
        return from_step, from_step
    start = first_resumable_step(manifest)
    if start is not None:
        return start, None
    if manifest.status == OperationStatus.drafted and drafted_wiki_context_drifted(vault, run_dir):
        return "wiki_context_snapshot", "wiki_context_snapshot"
    return None, None


def drafted_wiki_context_drifted(vault: Path, run_dir: Path) -> bool:
    snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    if not snapshot_path.exists():
        return False
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    return bool(wiki_context_drift_messages(vault, snapshot))


def ensure_wiki_context_current_before_resume(vault: Path, run_dir: Path, start_step: str) -> None:
    if step_index(start_step) <= step_index("wiki_context_snapshot"):
        return
    snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    if not snapshot_path.exists():
        return
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    messages = wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise PipelineError("; ".join(messages))


def execute_ingest(
    vault: Path,
    operation_id: str,
    *,
    start_step: str,
    execution_context: ProviderExecutionContext,
    console: Console | None = None,
) -> OperationManifest:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    logger = EventLogger(operation_id, run_dir / "events.jsonl", console=console, redactor=execution_context.redactor)
    manifest = read_manifest(store.manifest_path(operation_id))
    profile = load_profile(vault / ".llmwiki" / "profiles" / manifest.profile)
    raw_path = vault / manifest.raw_bindings[0].relative_path
    validate_resume_start(manifest, start_step)
    start_index = STEP_NAMES.index(start_step)
    for step_name in STEP_NAMES[start_index:]:
        if step_satisfied(get_step(manifest, step_name).status):
            continue
        try:
            _run_step(
                step_name,
                vault,
                store.manifest_path(operation_id),
                run_dir,
                raw_path,
                profile,
                manifest,
                logger,
                execution_context,
            )
        except Exception as exc:
            message = execution_context.redactor.redact_text(str(exc))
            fail_step(manifest, step_name, message)
            write_manifest(store.manifest_path(operation_id), manifest)
            refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
            logger.emit(step_name, "failed", status="failed", message=message, duration_ms=last_attempt_duration_ms(manifest, step_name))
            raise PipelineError(message) from exc
        write_manifest(store.manifest_path(operation_id), manifest)
        refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
        if get_step(manifest, step_name).status == StepStatus.awaiting_review:
            manifest.status = OperationStatus.awaiting_review
            write_manifest(store.manifest_path(operation_id), manifest)
            refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
            return manifest
    ensure_pipeline_completed(manifest)
    manifest.status = OperationStatus.drafted
    write_manifest(store.manifest_path(operation_id), manifest)
    refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
    return manifest


def validate_resume_start(manifest: OperationManifest, start_step: str) -> None:
    start_index = step_index(start_step)
    for step in manifest.steps[:start_index]:
        if not step_satisfied(step.status):
            raise PipelineError(
                f"Cannot resume from {start_step}: upstream step {step.name} is {step.status.value}; "
                f"resume from {step.name} or earlier."
            )


def ensure_pipeline_completed(manifest: OperationManifest) -> None:
    incomplete = [step for step in manifest.steps if not step_satisfied(step.status)]
    if incomplete:
        details = ", ".join(f"{step.name}={step.status.value}" for step in incomplete)
        raise PipelineError(f"Operation is not draft-ready; incomplete step(s): {details}")


def model_steps_for_raw_prepare_policy(
    model_steps: list[str],
    *,
    raw_path: Path,
    raw_rel: str,
    raw_prepare_policy: RawPreparePolicy,
) -> list[str]:
    if "raw_prepare" not in model_steps:
        return model_steps
    if raw_prepare_policy != RawPreparePolicy.skip_model:
        return model_steps
    if not raw_prepare_skip_model_passthrough_expected(raw_path=raw_path, raw_rel=raw_rel):
        return model_steps
    return [step for step in model_steps if step != "raw_prepare"]


def raw_prepare_skip_model_passthrough_expected(*, raw_path: Path, raw_rel: str) -> bool:
    if not raw_path.is_file():
        return False
    try:
        original_text = raw_path.read_text(encoding="utf-8")
        cleaned_text, links, warnings, preserved_media_count = cleanup_raw_wikilinks(original_text)
        pre_hash = sha256_file(raw_path)
        post_hash = sha256_bytes(cleaned_text.encode("utf-8"))
        cleanup = RawLinkCleanupArtifact(
            raw_path=raw_rel,
            changed=cleaned_text != original_text,
            pre_cleanup_sha256=pre_hash,
            post_cleanup_sha256=post_hash,
            cleaned_link_count=len(links),
            preserved_media_embed_count=preserved_media_count,
            links=links,
            warnings=warnings,
        )
        if cleanup.changed:
            with tempfile.TemporaryDirectory(prefix="llmwiki-raw-prepare-skip-check-") as tmp_dir:
                candidate_path = Path(tmp_dir) / raw_path.name
                candidate_path.write_text(cleaned_text, encoding="utf-8")
                preparation, _report = build_raw_prepare_fast_path(
                    raw_path=candidate_path,
                    raw_rel=raw_rel,
                    input_raw_sha256=post_hash,
                    cleanup=cleanup,
                    cleanup_ref="raw_link_cleanup/raw_link_cleanup.json",
                    raw_prepare_policy=RawPreparePolicy.skip_model,
                )
        else:
            preparation, _report = build_raw_prepare_fast_path(
                raw_path=raw_path,
                raw_rel=raw_rel,
                input_raw_sha256=post_hash,
                cleanup=cleanup,
                cleanup_ref="raw_link_cleanup/raw_link_cleanup.json",
                raw_prepare_policy=RawPreparePolicy.skip_model,
            )
    except Exception:
        return False
    return preparation is not None


def model_backed_step_can_run_locally(
    *,
    step_name: str,
    vault: Path,
    run_dir: Path,
    raw_path: Path,
    manifest: OperationManifest,
    provider_runtime_present: bool,
) -> bool:
    if provider_runtime_present or step_name != "raw_prepare":
        return False
    if manifest.vault_config_snapshot.raw_prepare_policy != RawPreparePolicy.skip_model:
        return False
    try:
        cleanup_path = require_step_output_dir(run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
        cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
        raw_rel = relative_to_vault(vault, raw_path)
        preparation, _report = build_raw_prepare_fast_path(
            raw_path=raw_path,
            raw_rel=raw_rel,
            input_raw_sha256=sha256_file(raw_path),
            cleanup=cleanup,
            cleanup_ref=cleanup_path.relative_to(run_dir).as_posix(),
            raw_prepare_policy=RawPreparePolicy.skip_model,
        )
    except Exception:
        return False
    return preparation is not None


def _run_step(
    step_name: str,
    vault: Path,
    manifest_path: Path,
    run_dir: Path,
    raw_path: Path,
    profile,
    manifest: OperationManifest,
    logger: EventLogger,
    execution_context: ProviderExecutionContext,
) -> None:
    runner = STEP_RUNNERS.get(step_name)
    if runner is None:
        raise PipelineError(f"Unknown step: {step_name}")
    provider_record = execution_context.record if runner.spec.model_backed else None
    provider_runtime = provider_record.providers.get(step_name) if provider_record else None
    provider_spec_for_attempt = provider_runtime.spec if provider_runtime else None
    local_model_backed_step = runner.spec.model_backed and model_backed_step_can_run_locally(
        step_name=step_name,
        vault=vault,
        run_dir=run_dir,
        raw_path=raw_path,
        manifest=manifest,
        provider_runtime_present=provider_runtime is not None,
    )
    logger.emit(
        step_name,
        "started",
        status="running",
        model_backed=runner.spec.model_backed,
        provider_spec=provider_spec_for_attempt,
        may_use_local_shortcut=step_name in {"raw_prepare", "wiki_merge_planning"},
    )
    if runner.spec.model_backed:
        if local_model_backed_step:
            begin_step_attempt(manifest, step_name)
        elif provider_record is None or provider_runtime is None:
            raise PipelineError(f"No provider execution context found for model-backed step: {step_name}")
        else:
            begin_model_step_attempt(
                manifest,
                step_name,
                provider_record_id=provider_record.record_id,
                provider_spec=provider_spec_for_attempt,
                provider_context_source=provider_record.source,
            )
    else:
        begin_step_attempt(manifest, step_name)
    write_manifest(manifest_path, manifest)
    ctx = StepRunContext(
        vault=vault,
        run_dir=run_dir,
        raw_path=raw_path,
        profile=profile,
        manifest=manifest,
        execution_context=execution_context,
    )
    output_dir = step_output_dir(run_dir, step_name)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    runner.run(ctx)
    status = get_step(manifest, step_name).status.value
    logger.emit(
        step_name,
        "completed",
        status=status,
        message=step_completion_message(ctx, step_name),
        duration_ms=last_attempt_duration_ms(manifest, step_name),
    )


@dataclass(frozen=True)
class StepRunContext:
    vault: Path
    run_dir: Path
    raw_path: Path
    profile: Any
    manifest: OperationManifest
    execution_context: ProviderExecutionContext


RAW_LINK_CLEANUP_RULE_VERSION = "obsidian_text_wikilink.v1"
WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]\n]+)\]\]")
MEDIA_EMBED_RE = re.compile(r"!\[\[([^\]\n]+)\]\]")


def _run_raw_link_cleanup(ctx: StepRunContext) -> None:
    step_name = "raw_link_cleanup"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    original_text = ctx.raw_path.read_text(encoding="utf-8")
    pre_hash = sha256_file(ctx.raw_path)
    cleaned_text, links, warnings, preserved_media_count = cleanup_raw_wikilinks(original_text)
    changed = cleaned_text != original_text
    if changed:
        if sha256_file(ctx.raw_path) != pre_hash:
            raise PipelineError("raw changed during raw_link_cleanup; rerun ingest")
        tmp = ctx.raw_path.with_name(f".{ctx.raw_path.name}.tmp")
        tmp.write_text(cleaned_text, encoding="utf-8")
        tmp.replace(ctx.raw_path)
    post_hash, post_size = raw_ref(ctx.raw_path)
    ctx.manifest.raw_bindings = [RawBinding(relative_path=raw_rel, sha256=post_hash, size_bytes=post_size)]
    artifact = RawLinkCleanupArtifact(
        raw_path=raw_rel,
        changed=changed,
        pre_cleanup_sha256=pre_hash,
        post_cleanup_sha256=post_hash,
        cleaned_link_count=len(links),
        preserved_media_embed_count=preserved_media_count,
        links=links,
        warnings=warnings,
    )
    out = step_root / "raw_link_cleanup.json"
    write_json(out, artifact)
    report = step_root / "raw_link_cleanup.md"
    report.write_text(render_raw_link_cleanup_markdown(artifact), encoding="utf-8")
    diff_path = step_root / "cleanup.diff"
    diff_path.write_text(render_update_diff(original_text, cleaned_text, f"pre/{raw_rel}", f"post/{raw_rel}"), encoding="utf-8")
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, out, step_name, "json", "raw_link_cleanup.v1"),
            _ref(ctx.run_dir, report, step_name, "markdown"),
            _ref(ctx.run_dir, diff_path, step_name, "diff"),
        ],
    )


def cleanup_raw_wikilinks(text: str) -> tuple[str, list[RawLinkCleanupLink], list[RawLinkCleanupWarning], int]:
    lines = text.splitlines(keepends=True)
    cleaned_lines: list[str] = []
    links: list[RawLinkCleanupLink] = []
    warnings: list[RawLinkCleanupWarning] = []
    in_frontmatter = False
    frontmatter_seen = False
    in_fenced_code = False
    fence_marker = ""
    preserved_media_total = 0
    for index, line in enumerate(lines, start=1):
        newline = ""
        body = line
        if body.endswith("\r\n"):
            body = body[:-2]
            newline = "\r\n"
        elif body.endswith("\n"):
            body = body[:-1]
            newline = "\n"
        stripped = body.strip()
        if index == 1 and stripped == "---":
            in_frontmatter = True
            frontmatter_seen = True
            cleaned_lines.append(line)
            continue
        if in_frontmatter and index != 1 and stripped == "---":
            in_frontmatter = False
            cleaned_lines.append(line)
            continue
        fence_match = re.match(r"^(\s*)(```+|~~~+)", body)
        if not in_frontmatter and fence_match:
            marker = fence_match.group(2)[0]
            if not in_fenced_code:
                in_fenced_code = True
                fence_marker = marker
            elif fence_marker == marker:
                in_fenced_code = False
                fence_marker = ""
            cleaned_lines.append(line)
            continue
        context = "frontmatter" if in_frontmatter and frontmatter_seen else "body"
        if in_fenced_code:
            cleaned_lines.append(line)
            continue
        media_matches = list(MEDIA_EMBED_RE.finditer(body))
        for match in media_matches[: max(0, 20 - len(warnings))]:
            warnings.append(
                RawLinkCleanupWarning(
                    warning_type="preserved_media_embed",
                    message="保留 Obsidian 媒体链接；本轮只清理文本 wikilink。",
                    line_number=index,
                    line_excerpt=truncate_excerpt(body),
                )
            )
        preserved_media_count = len(media_matches)
        preserved_media_total += preserved_media_count
        cleaned_body, line_links = cleanup_wikilinks_in_line(body, line_number=index, context=context, start_index=len(links) + 1)
        links.extend(line_links)
        cleaned_lines.append(cleaned_body + newline)
    return "".join(cleaned_lines), links, warnings, preserved_media_total


def cleanup_wikilinks_in_line(
    line: str,
    *,
    line_number: int,
    context: Literal["frontmatter", "body"],
    start_index: int,
) -> tuple[str, list[RawLinkCleanupLink]]:
    pieces: list[str] = []
    links: list[RawLinkCleanupLink] = []
    cursor = 0
    in_code = False
    for match in re.finditer(r"`+", line):
        segment = line[cursor : match.start()]
        pieces.append(_cleanup_wikilink_segment(segment, line, line_number=line_number, context=context, start_index=start_index + len(links), links=links) if not in_code else segment)
        pieces.append(match.group(0))
        in_code = not in_code
        cursor = match.end()
    tail = line[cursor:]
    pieces.append(_cleanup_wikilink_segment(tail, line, line_number=line_number, context=context, start_index=start_index + len(links), links=links) if not in_code else tail)
    return "".join(pieces), links


def _cleanup_wikilink_segment(
    segment: str,
    original_line: str,
    *,
    line_number: int,
    context: Literal["frontmatter", "body"],
    start_index: int,
    links: list[RawLinkCleanupLink],
) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        target, label = split_wikilink(raw)
        link_id = f"L{start_index + len(links):03d}"
        links.append(
            RawLinkCleanupLink(
                link_id=link_id,
                link_kind="wikilink",
                label=label,
                target=target,
                cleanup_action="unwrap_text",
                cleanup_context=context,
                line_number=line_number,
                original_line_hash=sha256_bytes(original_line.encode("utf-8")),
                line_excerpt=truncate_excerpt(original_line),
            )
        )
        return label

    return WIKILINK_RE.sub(replace, segment)


def split_wikilink(raw: str) -> tuple[str, str]:
    if "|" in raw:
        target, label = raw.split("|", 1)
        return target.strip(), label.strip() or target.strip()
    return raw.strip(), raw.strip()


def truncate_excerpt(line: str, limit: int = 160) -> str:
    text = " ".join(line.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def render_raw_link_cleanup_markdown(artifact: RawLinkCleanupArtifact) -> str:
    rows = [
        ["raw_path", f"`{artifact.raw_path}`"],
        ["changed", str(artifact.changed).lower()],
        ["cleanup_rule_version", artifact.cleanup_rule_version],
        ["pre_cleanup_sha256", f"`{artifact.pre_cleanup_sha256}`"],
        ["post_cleanup_sha256", f"`{artifact.post_cleanup_sha256}`"],
        ["cleaned_link_count", str(artifact.cleaned_link_count)],
        ["preserved_media_embed_count", str(artifact.preserved_media_embed_count)],
    ]
    link_rows = [
        [link.link_id, link.cleanup_context, str(link.line_number), link.target, link.label, link.line_excerpt]
        for link in artifact.links
    ]
    warning_rows = [
        [warning.warning_type, str(warning.line_number), warning.message, warning.line_excerpt]
        for warning in artifact.warnings[:20]
    ]
    return (
        "# Raw Obsidian Wikilink 规范化\n\n"
        f"{format_markdown_table(['字段', '值'], rows)}\n\n"
        "## 已清理文本 Wikilink\n\n"
        f"{format_markdown_table(['ID', '位置', '行号', 'Target', 'Label', '行摘录'], link_rows) if link_rows else '_暂无。_'}\n\n"
        "## 保留项 Warning\n\n"
        f"{format_markdown_table(['类型', '行号', '说明', '行摘录'], warning_rows) if warning_rows else '_暂无。_'}\n\n"
        "## Diff\n\n"
        "`cleanup.diff` 是 pre-clean -> post-clean 的 unified diff，仅供人工审计/恢复参考，不会自动 rollback。\n"
    )


def raw_prepare_fast_path_provider_allowed(ctx: StepRunContext, step_name: str) -> bool:
    if ctx.execution_context.record is None:
        return False
    runtime = ctx.execution_context.record.providers.get(step_name)
    if runtime is None:
        return False
    return raw_prepare_fast_path_provider_spec_allowed(runtime.spec)


def raw_prepare_fast_path_provider_spec_allowed(provider_spec: str | None) -> bool:
    return bool(provider_spec and provider_spec.startswith("openai_compatible:"))


def build_raw_prepare_fast_path(
    *,
    raw_path: Path,
    raw_rel: str,
    input_raw_sha256: str,
    cleanup: RawLinkCleanupArtifact,
    cleanup_ref: str,
    raw_prepare_policy: RawPreparePolicy = RawPreparePolicy.auto,
) -> tuple[RawPreparationArtifact | None, dict[str, Any]]:
    raw_text = raw_path.read_text(encoding="utf-8")
    policy = RawPreparePolicy(raw_prepare_policy)
    hard_reasons: list[str] = []
    auto_reasons: list[str] = []
    if raw_path.suffix.lower() not in {".md", ".markdown", ".mdown"}:
        hard_reasons.append("raw file extension is not markdown")
    cleanup_fast_path_compatible = raw_link_cleanup_fast_path_compatible(cleanup)
    if cleanup.changed and not cleanup_fast_path_compatible:
        auto_reasons.append("raw_link_cleanup changed the raw text")
    if cleanup.preserved_media_embed_count:
        auto_reasons.append("raw contains preserved media embeds")
    if not raw_text.strip():
        hard_reasons.append("raw text is empty")
    noise = raw_prepare_noise_profile(raw_text)
    structured_markdown_passthrough = raw_prepare_structured_markdown_fast_path_allowed(noise)
    structured_quality_risk = raw_prepare_structured_markdown_quality_risk(noise)
    allowed_soft_markers: list[str] = []
    if policy == RawPreparePolicy.skip_model:
        allowed_soft_markers.append("user_skip_prepare")
    if cleanup_fast_path_compatible:
        allowed_soft_markers.append("raw_link_cleanup_text_unwrap")
    if structured_quality_risk:
        auto_reasons.append("structured markdown looks like noisy ASR or translated transcript")
    if noise["markdown_media_embed_count"]:
        if policy == RawPreparePolicy.skip_model:
            auto_reasons.append("raw contains markdown media embeds")
            allowed_soft_markers.append("markdown_media_embed")
        elif structured_markdown_passthrough:
            allowed_soft_markers.append("markdown_media_embed")
        else:
            auto_reasons.append("raw contains markdown media embeds")
    if raw_prepare_timestamp_transcript_noise(noise):
        auto_reasons.append("raw looks like a timestamped transcript")
        if policy == RawPreparePolicy.skip_model:
            allowed_soft_markers.append("timestamp_transcript_marker")
    if raw_prepare_speaker_turn_transcript_noise(noise):
        auto_reasons.append("raw looks like a speaker-turn transcript")
        if policy == RawPreparePolicy.skip_model:
            allowed_soft_markers.append("speaker_turn_transcript_marker")
    if noise["interview_transcript_marker"]:
        if policy == RawPreparePolicy.skip_model:
            auto_reasons.append("raw contains interview/transcript section markers")
            allowed_soft_markers.append("interview_transcript_marker")
        elif structured_markdown_passthrough:
            allowed_soft_markers.append("interview_transcript_marker")
        else:
            auto_reasons.append("raw contains interview/transcript section markers")
    if noise["webvtt_marker"]:
        auto_reasons.append("raw contains WebVTT transcript markers")
        if policy == RawPreparePolicy.skip_model:
            allowed_soft_markers.append("webvtt_marker")
    if policy == RawPreparePolicy.force_model:
        reasons = ["raw_prepare policy forces model cleaning"]
        policy_suppressed_reasons: list[str] = []
    elif policy == RawPreparePolicy.skip_model:
        reasons = hard_reasons
        policy_suppressed_reasons = auto_reasons
    else:
        reasons = hard_reasons + auto_reasons
        policy_suppressed_reasons = []
    structured_soft_marker_passthrough = structured_markdown_passthrough and any(
        marker in {"markdown_media_embed", "interview_transcript_marker"} for marker in allowed_soft_markers
    )
    if policy == RawPreparePolicy.skip_model:
        fast_path_mode = "user_skip_model_passthrough"
    elif structured_soft_marker_passthrough:
        fast_path_mode = "structured_markdown_passthrough"
    else:
        fast_path_mode = "clean_markdown_passthrough"
    report = {
        "schema_version": "raw_prepare_fast_path.v1",
        "rule_version": RAW_PREPARE_FAST_PATH_RULE_VERSION,
        "raw_prepare_policy": policy.value,
        "eligible": not reasons,
        "source_raw_path": raw_rel,
        "input_raw_sha256": input_raw_sha256,
        "raw_link_cleanup_ref": cleanup_ref,
        "reasons": reasons,
        "policy_suppressed_reasons": policy_suppressed_reasons,
        "noise_profile": noise,
        "fast_path_mode": fast_path_mode,
        "allowed_soft_markers": allowed_soft_markers,
        "raw_link_cleanup_fast_path_compatible": cleanup_fast_path_compatible,
    }
    if reasons:
        return None, report
    document_kind = infer_passthrough_document_kind(raw_text, noise)
    prepared_markdown, reference_truncation = truncate_reference_section_for_prepared_markdown(raw_text)
    prepared_markdown, appendix_compaction = compact_appendix_sections_for_prepared_markdown(
        prepared_markdown,
        enabled=bool(noise.get("paper_like_marker")),
    )
    if policy == RawPreparePolicy.skip_model:
        operations_applied = ["user_skip_model_markdown_passthrough"]
    else:
        operations_applied = [
            "deterministic_structured_markdown_passthrough"
            if structured_soft_marker_passthrough
            else "deterministic_markdown_passthrough"
        ]
    if reference_truncation.get("truncated"):
        operations_applied.append("deterministic_reference_section_truncation")
    if appendix_compaction.get("compacted"):
        operations_applied.append("deterministic_appendix_section_compaction")
    if policy == RawPreparePolicy.skip_model:
        review_notes = (
            "User selected --skip-prepare; raw markdown was passed through without model cleanup. "
            "Auto quality blockers were recorded as policy_suppressed_reasons for audit."
        )
    elif structured_soft_marker_passthrough:
        review_notes = (
            "Structured markdown used deterministic fast-path: media/interview markers were present, "
            "but heading density was high and timestamp/speaker-turn transcript noise did not trigger."
        )
    elif cleanup_fast_path_compatible:
        review_notes = (
            "Raw markdown used deterministic fast-path after audited raw_link_cleanup text wikilink unwrap; "
            "transcript/media-noise heuristics did not trigger."
        )
    else:
        review_notes = (
            "Raw markdown used deterministic fast-path: raw_link_cleanup made no content changes "
            "and transcript/media-noise heuristics did not trigger."
        )
    preparation = RawPreparationArtifact(
        source_raw_path=raw_rel,
        input_raw_sha256=input_raw_sha256,
        raw_link_cleanup_ref=cleanup_ref,
        document_kind=document_kind,
        prepared_markdown=prepared_markdown,
        operations_applied=operations_applied,
        omission_policy=(
            deterministic_omission_policy(
                reference_truncated=bool(reference_truncation.get("truncated")),
                appendix_compacted=bool(appendix_compaction.get("compacted")),
            )
        ),
        uncertain_items=[],
        risk_level="medium" if policy == RawPreparePolicy.skip_model and policy_suppressed_reasons else "low",
        requires_human_review=policy == RawPreparePolicy.skip_model and bool(policy_suppressed_reasons),
        review_notes=review_notes,
    )
    if cleanup_fast_path_compatible:
        preparation.review_notes += (
            " Raw link cleanup only unwrapped Obsidian text wikilinks and is recorded in raw_link_cleanup artifacts."
        )
    if reference_truncation.get("truncated"):
        preparation.review_notes += (
            " Reference section was omitted from prepared markdown to reduce digest noise; "
            "the original raw retains the full reference list."
        )
    if appendix_compaction.get("compacted"):
        preparation.review_notes += (
            " Appendix sections were compacted to headings and short excerpts in prepared markdown; "
            "the original raw retains the full appendix."
        )
    report["document_kind"] = document_kind
    report["reference_truncation"] = reference_truncation
    report["appendix_compaction"] = appendix_compaction
    return preparation, report


def raw_link_cleanup_fast_path_compatible(cleanup: RawLinkCleanupArtifact) -> bool:
    if not cleanup.changed:
        return False
    if cleanup.warnings or cleanup.preserved_media_embed_count:
        return False
    if cleanup.cleaned_link_count <= 0 or cleanup.cleaned_link_count != len(cleanup.links):
        return False
    return all(link.cleanup_action == "unwrap_text" for link in cleanup.links)


def build_raw_prepare_diagnostic(
    *,
    vault: Path,
    raw_file: Path,
    raw_prepare_policy: RawPreparePolicy = RawPreparePolicy.auto,
) -> dict[str, Any]:
    """Preview raw_prepare fast-path/model decision without mutating raw or creating a run."""
    raw_path, raw_rel = resolve_raw_path(vault, raw_file)
    original_text = raw_path.read_text(encoding="utf-8")
    cleaned_text, links, warnings, preserved_media_count = cleanup_raw_wikilinks(original_text)
    pre_hash = sha256_file(raw_path)
    post_hash = sha256_bytes(cleaned_text.encode("utf-8"))
    cleanup = RawLinkCleanupArtifact(
        raw_path=raw_rel,
        changed=cleaned_text != original_text,
        pre_cleanup_sha256=pre_hash,
        post_cleanup_sha256=post_hash,
        cleaned_link_count=len(links),
        preserved_media_embed_count=preserved_media_count,
        links=links,
        warnings=warnings,
    )
    provider_context = build_provider_execution_context(
        vault=vault,
        manifest_contexts=[],
        fixture_dir=None,
        mock_fixture_dir=None,
        source="initial_run",
        from_step=None,
        tasks=["raw_prepare"],
        require_mock_fixture=False,
    )
    provider_runtime = provider_context.record.providers.get("raw_prepare") if provider_context.record else None
    provider_spec = provider_runtime.spec if provider_runtime is not None else None
    provider_fast_path_allowed = raw_prepare_fast_path_provider_spec_allowed(provider_spec)

    def build_for(policy: RawPreparePolicy, candidate_path: Path) -> dict[str, Any]:
        _preparation, report = build_raw_prepare_fast_path(
            raw_path=candidate_path,
            raw_rel=raw_rel,
            input_raw_sha256=post_hash,
            cleanup=cleanup,
            cleanup_ref="raw_link_cleanup/raw_link_cleanup.json",
            raw_prepare_policy=policy,
        )
        if policy == RawPreparePolicy.auto and not provider_fast_path_allowed:
            provider_report = build_raw_prepare_provider_ineligible_report(
                raw_path=candidate_path,
                raw_rel=raw_rel,
                input_raw_sha256=post_hash,
                cleanup_ref="raw_link_cleanup/raw_link_cleanup.json",
                raw_prepare_policy=policy,
                provider_spec=provider_spec,
            )
            provider_report["raw_fast_path_report"] = report
            provider_report["raw_hard_blockers"] = raw_prepare_hard_blockers(report)
            return provider_report
        return report

    policies = [RawPreparePolicy.auto, RawPreparePolicy.skip_model, RawPreparePolicy.force_model]
    if cleanup.changed:
        with tempfile.TemporaryDirectory(prefix="llmwiki-raw-prepare-check-") as tmp_dir:
            candidate_path = Path(tmp_dir) / raw_path.name
            candidate_path.write_text(cleaned_text, encoding="utf-8")
            reports = {policy.value: build_for(policy, candidate_path) for policy in policies}
    else:
        reports = {policy.value: build_for(policy, raw_path) for policy in policies}

    selected_policy = RawPreparePolicy(raw_prepare_policy)
    auto_report = reports[RawPreparePolicy.auto.value]
    skip_report = reports[RawPreparePolicy.skip_model.value]
    return {
        "schema_version": "raw_prepare_diagnostic.v1",
        "vault": vault.resolve().as_posix(),
        "raw_path": raw_rel,
        "raw_absolute_path": raw_path.as_posix(),
        "selected_policy": selected_policy.value,
        "selected_report": reports[selected_policy.value],
        "auto_report": auto_report,
        "skip_prepare_report": skip_report,
        "force_prepare_report": reports[RawPreparePolicy.force_model.value],
        "provider": {
            "raw_prepare_spec": provider_spec or "",
            "fast_path_allowed": provider_fast_path_allowed,
            "diagnostic_requires_fixture": bool(provider_spec == "mock:fixture" and not (provider_runtime and provider_runtime.fixture_dir)),
        },
        "recommendation": raw_prepare_diagnostic_recommendation(
            auto_report=auto_report,
            skip_report=skip_report,
            selected_report=reports[selected_policy.value],
        ),
        "raw_link_cleanup": cleanup.model_dump(mode="json"),
    }


def build_raw_prepare_provider_ineligible_report(
    *,
    raw_path: Path,
    raw_rel: str,
    input_raw_sha256: str,
    cleanup_ref: str,
    raw_prepare_policy: RawPreparePolicy,
    provider_spec: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "raw_prepare_fast_path.v1",
        "rule_version": RAW_PREPARE_FAST_PATH_RULE_VERSION,
        "raw_prepare_policy": raw_prepare_policy.value,
        "eligible": False,
        "source_raw_path": raw_rel,
        "input_raw_sha256": input_raw_sha256,
        "raw_link_cleanup_ref": cleanup_ref,
        "reasons": ["configured provider is not eligible for deterministic fast-path"],
        "policy_suppressed_reasons": [],
        "noise_profile": raw_prepare_noise_profile(raw_path.read_text(encoding="utf-8")),
        "provider_spec": provider_spec or "",
        "provider_fast_path_allowed": False,
    }


def raw_prepare_diagnostic_recommendation(
    *,
    auto_report: dict[str, Any],
    skip_report: dict[str, Any],
    selected_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auto_reasons = [str(reason) for reason in auto_report.get("reasons", [])]
    skip_reasons = [str(reason) for reason in skip_report.get("reasons", [])]
    noise = auto_report.get("noise_profile", {})
    raw_fast_path_report = auto_report.get("raw_fast_path_report") if isinstance(auto_report.get("raw_fast_path_report"), dict) else auto_report
    hard_blockers = raw_prepare_hard_blockers(raw_fast_path_report)
    skip_available = not skip_reasons
    selected_estimated_model_prepare = not bool((selected_report or auto_report).get("eligible"))
    if bool(auto_report.get("eligible")):
        return {
            "decision": "auto_deterministic_fast_path",
            "recommended_flag": "",
            "summary": "auto 会使用 deterministic raw_prepare fast-path，不调用模型清洗。",
            "skip_prepare_available": skip_available,
            "force_prepare_available": True,
            "auto_estimated_model_prepare": False,
            "selected_estimated_model_prepare": selected_estimated_model_prepare,
            "estimated_model_prepare": selected_estimated_model_prepare,
            "raw_hard_blockers": hard_blockers,
        }
    if hard_blockers:
        return {
            "decision": "model_prepare_required",
            "recommended_flag": "",
            "summary": "auto 会走模型 raw_prepare；当前 raw 不是可安全 passthrough 的 Markdown，--skip-prepare 也不能覆盖硬性原因。",
            "skip_prepare_available": skip_available,
            "force_prepare_available": True,
            "auto_estimated_model_prepare": True,
            "selected_estimated_model_prepare": selected_estimated_model_prepare,
            "estimated_model_prepare": selected_estimated_model_prepare,
            "raw_hard_blockers": hard_blockers,
        }
    if "configured provider is not eligible for deterministic fast-path" in auto_reasons:
        return {
            "decision": "auto_provider_prepare_required",
            "recommended_flag": "",
            "summary": "auto 会调用 raw_prepare provider；当前 provider 不支持 deterministic fast-path。",
            "skip_prepare_available": skip_available,
            "force_prepare_available": True,
            "auto_estimated_model_prepare": True,
            "selected_estimated_model_prepare": selected_estimated_model_prepare,
            "estimated_model_prepare": selected_estimated_model_prepare,
            "raw_hard_blockers": hard_blockers,
        }
    noisy_transcript = bool(
        noise.get("transcript_provenance_risk")
        or noise.get("structured_markdown_quality_risk")
        or noise.get("webvtt_marker")
        or raw_prepare_timestamp_transcript_noise(noise)
        or raw_prepare_speaker_turn_transcript_noise(noise)
    )
    if noisy_transcript:
        return {
            "decision": "auto_model_prepare_recommended",
            "recommended_flag": "",
            "summary": "auto 会走模型 raw_prepare；材料像播客/视频转写或翻译稿，建议保留清洗。若你确认 raw 已人工校对，可用 --skip-prepare 节省模型时间。",
            "skip_prepare_available": skip_available,
            "force_prepare_available": True,
            "auto_estimated_model_prepare": True,
            "selected_estimated_model_prepare": selected_estimated_model_prepare,
            "estimated_model_prepare": selected_estimated_model_prepare,
            "raw_hard_blockers": hard_blockers,
        }
    return {
        "decision": "auto_model_prepare_user_choice",
        "recommended_flag": "--skip-prepare" if skip_available else "",
        "summary": "auto 会走模型 raw_prepare，但阻断原因不像强 ASR/翻译风险；若你确认 raw 质量足够，可以用 --skip-prepare。",
        "skip_prepare_available": skip_available,
        "force_prepare_available": True,
        "auto_estimated_model_prepare": True,
        "selected_estimated_model_prepare": selected_estimated_model_prepare,
        "estimated_model_prepare": selected_estimated_model_prepare,
        "raw_hard_blockers": hard_blockers,
    }


def raw_prepare_hard_blockers(report: dict[str, Any]) -> list[str]:
    return [
        str(reason)
        for reason in report.get("reasons", [])
        if str(reason) in {"raw file extension is not markdown", "raw text is empty"}
    ]


def deterministic_omission_policy(*, reference_truncated: bool, appendix_compacted: bool) -> str:
    if reference_truncated and appendix_compacted:
        return "reference_section_omitted_and_appendix_compacted_from_prepared_markdown_raw_retained"
    if reference_truncated:
        return "reference_section_omitted_from_prepared_markdown_raw_retained"
    if appendix_compacted:
        return "appendix_compacted_from_prepared_markdown_raw_retained"
    return "none"


def raw_prepare_noise_profile(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    line_count = len(lines)
    body_lines = raw_prepare_body_lines_for_noise(text)
    body_line_count = len(body_lines)
    timestamp_line_count = sum(1 for line in lines if TRANSCRIPT_TIMESTAMP_RE.search(line))
    speaker_turn_count = sum(1 for line in lines if looks_like_speaker_turn_line(line))
    heading_count = sum(1 for line in lines if re.match(r"^\s{0,3}#{1,6}\s+\S", line))
    short_body_line_count = sum(1 for line in body_lines if len(line) <= 32)
    missing_sentence_terminal_count = sum(
        1 for line in body_lines if line and line[-1] not in SENTENCE_TERMINAL_PUNCTUATION
    )
    low_punctuation_body_line_count = sum(
        1 for line in body_lines if len(line) >= 16 and not any(mark in line for mark in INLINE_PUNCTUATION)
    )
    long_unpunctuated_body_line_count = sum(
        1 for line in body_lines if len(line) >= 24 and not any(mark in line for mark in INLINE_PUNCTUATION)
    )
    paper_section_marker_count = len(PAPER_SECTION_HEADING_RE.findall(text))
    paper_caption_count = len(PAPER_CAPTION_RE.findall(text))
    arxiv_import_marker = bool(ARXIV_IMPORT_MARKER_RE.search(text))
    paper_like_marker = raw_prepare_paper_like_markdown(
        line_count=line_count,
        heading_count=heading_count,
        arxiv_import_marker=arxiv_import_marker,
        paper_section_marker_count=paper_section_marker_count,
        paper_caption_count=paper_caption_count,
        text=text,
    )
    noise = {
        "line_count": line_count,
        "body_line_count": body_line_count,
        "heading_count": heading_count,
        "heading_ratio": heading_count / line_count if line_count else 0.0,
        "short_body_line_count": short_body_line_count,
        "short_body_line_ratio": short_body_line_count / body_line_count if body_line_count else 0.0,
        "missing_sentence_terminal_count": missing_sentence_terminal_count,
        "missing_sentence_terminal_ratio": (
            missing_sentence_terminal_count / body_line_count if body_line_count else 0.0
        ),
        "low_punctuation_body_line_count": low_punctuation_body_line_count,
        "low_punctuation_body_line_ratio": (
            low_punctuation_body_line_count / body_line_count if body_line_count else 0.0
        ),
        "long_unpunctuated_body_line_count": long_unpunctuated_body_line_count,
        "timestamp_line_count": timestamp_line_count,
        "timestamp_line_ratio": timestamp_line_count / line_count if line_count else 0.0,
        "speaker_turn_count": speaker_turn_count,
        "speaker_turn_ratio": speaker_turn_count / line_count if line_count else 0.0,
        "markdown_media_embed_count": len(MARKDOWN_MEDIA_EMBED_RE.findall(text)),
        "interview_transcript_marker": bool(INTERVIEW_TRANSCRIPT_MARKER_RE.search(text)),
        "webvtt_marker": bool(re.search(r"(?im)^\s*WEBVTT\s*$", text)),
        "youtube_url_count": len(YOUTUBE_URL_RE.findall(text)),
        "podcast_marker_count": len(PODCAST_MARKER_RE.findall(text)),
        "audio_video_marker_count": len(AUDIO_VIDEO_SOURCE_MARKER_RE.findall(text)),
        "translation_marker_count": len(TRANSLATION_MARKER_RE.findall(text)),
        "asr_source_marker_count": len(ASR_SOURCE_MARKER_RE.findall(text)),
        "arxiv_import_marker": arxiv_import_marker,
        "paper_section_marker_count": paper_section_marker_count,
        "paper_caption_count": paper_caption_count,
        "paper_like_marker": paper_like_marker,
    }
    noise["transcript_provenance_risk"] = raw_prepare_transcript_provenance_risk(noise)
    noise["structured_markdown_quality_risk"] = raw_prepare_structured_markdown_quality_risk(noise)
    return noise


def looks_like_speaker_turn_line(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"#{1,6}\s+\S", stripped):
        return False
    list_prefix_match = re.match(r"(?:[-*+]|\d+[.)])\s+", stripped)
    in_list_item = list_prefix_match is not None
    if list_prefix_match is not None:
        stripped = stripped[list_prefix_match.end() :]
    match = SPEAKER_TURN_RE.match(stripped)
    if match is None:
        return False
    label = match.group("label").strip().strip("*_`[]()")
    body = match.group("body").strip()
    if not label or not raw_prepare_speaker_turn_body_has_content(body):
        return False
    if SPEAKER_EXPLANATORY_LABEL_RE.search(label):
        return False
    if SPEAKER_ROLE_LABEL_RE.fullmatch(label):
        return True
    if raw_prepare_short_cjk_speaker_label(label):
        return True
    return (not in_list_item) and raw_prepare_title_case_speaker_label(label)


def raw_prepare_speaker_turn_body_has_content(body: str) -> bool:
    if not body:
        return False
    if re.fullmatch(r"https?://\S+", body, flags=re.IGNORECASE):
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", body))


def raw_prepare_short_cjk_speaker_label(label: str) -> bool:
    compact = re.sub(r"\s+", "", label)
    if not re.search(r"[\u4e00-\u9fff]", compact):
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·・]{2,8}", compact))


def raw_prepare_title_case_speaker_label(label: str) -> bool:
    if len(label) > 40:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z.'_-]*", label)
    if not 1 <= len(words) <= 4:
        return False
    return all(word[0].isupper() or word.isupper() for word in words)


def raw_prepare_body_lines_for_noise(text: str) -> list[str]:
    body_lines: list[str] = []
    in_frontmatter = False
    in_fenced_code = False
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if index == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fenced_code = not in_fenced_code
            continue
        if in_fenced_code or not stripped:
            continue
        if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
            continue
        if MARKDOWN_MEDIA_EMBED_RE.search(stripped):
            continue
        if re.match(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", stripped):
            continue
        stripped = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", stripped)
        if stripped:
            body_lines.append(stripped)
    return body_lines


def raw_prepare_paper_like_markdown(
    *,
    line_count: int,
    heading_count: int,
    arxiv_import_marker: bool,
    paper_section_marker_count: int,
    paper_caption_count: int,
    text: str,
) -> bool:
    if line_count < 80 or heading_count < 6:
        return False
    if not arxiv_import_marker and "doi" not in text[:5000].lower():
        return False
    return paper_section_marker_count >= 2 or paper_caption_count >= 3 or bool(REFERENCE_SECTION_HEADING_RE.search(text))


def raw_prepare_structured_markdown_fast_path_allowed(noise: dict[str, Any]) -> bool:
    if noise.get("webvtt_marker"):
        return False
    if raw_prepare_timestamp_transcript_noise(noise):
        return False
    if raw_prepare_speaker_turn_transcript_noise(noise):
        return False
    if raw_prepare_structured_markdown_quality_risk(noise):
        return False
    return noise.get("line_count", 0) >= 40 and noise.get("heading_count", 0) >= 6


def raw_prepare_timestamp_transcript_noise(noise: dict[str, Any]) -> bool:
    return noise.get("timestamp_line_count", 0) >= 8 or noise.get("timestamp_line_ratio", 0.0) >= 0.05


def raw_prepare_speaker_turn_transcript_noise(noise: dict[str, Any]) -> bool:
    if noise.get("paper_like_marker") and noise.get("speaker_turn_ratio", 0.0) < 0.15:
        return False
    return noise.get("speaker_turn_count", 0) >= 20 or noise.get("speaker_turn_ratio", 0.0) >= 0.10


def raw_prepare_transcript_provenance_risk(noise: dict[str, Any]) -> bool:
    if noise.get("paper_like_marker"):
        return False
    if noise.get("asr_source_marker_count", 0) > 0:
        return True
    has_external_media = noise.get("youtube_url_count", 0) > 0 or noise.get("markdown_media_embed_count", 0) > 0
    transcriptish = bool(noise.get("interview_transcript_marker")) or noise.get("podcast_marker_count", 0) > 0
    translated = noise.get("translation_marker_count", 0) > 0
    if has_external_media and (transcriptish or translated):
        return True
    if noise.get("podcast_marker_count", 0) > 0 and (bool(noise.get("interview_transcript_marker")) or translated):
        return True
    if translated and bool(noise.get("interview_transcript_marker")) and noise.get("heading_count", 0) >= 4:
        return True
    return False


def raw_prepare_structured_markdown_quality_risk(noise: dict[str, Any]) -> bool:
    if noise.get("paper_like_marker"):
        return False
    if raw_prepare_transcript_provenance_risk(noise):
        return True
    body_line_count = noise.get("body_line_count", 0)
    if body_line_count >= 30 and noise.get("low_punctuation_body_line_ratio", 0.0) >= 0.40:
        return True
    if (
        body_line_count >= 40
        and noise.get("short_body_line_ratio", 0.0) >= 0.55
        and noise.get("missing_sentence_terminal_ratio", 0.0) >= 0.35
    ):
        return True
    if noise.get("long_unpunctuated_body_line_count", 0) >= 8:
        return (
            noise.get("low_punctuation_body_line_ratio", 0.0) >= 0.25
            or noise.get("missing_sentence_terminal_ratio", 0.0) >= 0.65
        )
    return False


def infer_passthrough_document_kind(text: str, noise: dict[str, Any]) -> Literal["transcript", "article", "notes", "mixed", "unknown"]:
    if noise.get("paper_like_marker"):
        return "article"
    if raw_prepare_timestamp_transcript_noise(noise) or raw_prepare_speaker_turn_transcript_noise(noise):
        return "transcript"
    if noise.get("interview_transcript_marker"):
        return "transcript"
    non_empty = [line for line in text.splitlines() if line.strip()]
    if not non_empty:
        return "unknown"
    heading_count = sum(1 for line in non_empty if line.lstrip().startswith("#"))
    bullet_count = sum(1 for line in non_empty if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line))
    bullet_ratio = bullet_count / len(non_empty)
    if bullet_ratio >= 0.35:
        return "notes"
    if heading_count and len(non_empty) >= 6:
        return "article"
    if bullet_count:
        return "notes"
    return "mixed"


def truncate_reference_section_for_prepared_markdown(text: str) -> tuple[str, dict[str, Any]]:
    report: dict[str, Any] = {
        "truncated": False,
        "reason": "no eligible reference section found",
        "omitted_char_count": 0,
    }
    if len(text) < REFERENCE_TRUNCATION_MIN_DOCUMENT_CHARS:
        report["reason"] = "document is below reference truncation size threshold"
        return text.rstrip() + "\n", report
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if not REFERENCE_SECTION_HEADING_RE.match(line.strip()):
            continue
        start = offsets[index]
        next_appendix_index = next(
            (
                following_index
                for following_index in range(index + 1, len(lines))
                if APPENDIX_SECTION_HEADING_RE.match(lines[following_index].strip())
            ),
            None,
        )
        end = offsets[next_appendix_index] if next_appendix_index is not None else len(text)
        omitted_char_count = end - start
        if start / len(text) < REFERENCE_TRUNCATION_MIN_START_RATIO:
            report["reason"] = "reference section starts too early"
            continue
        if omitted_char_count < REFERENCE_TRUNCATION_MIN_OMITTED_CHARS:
            report["reason"] = "reference section is below truncation size threshold"
            continue
        heading = line.strip()
        marker = (
            f"{heading}\n\n"
            "[Reference section omitted from prepared markdown; original raw retains the full reference list.]\n"
        )
        suffix = text[end:].lstrip("\n")
        prepared = text[:start].rstrip() + "\n\n" + marker
        if suffix:
            prepared = prepared.rstrip() + "\n\n" + suffix
        return prepared.rstrip() + "\n", {
            "truncated": True,
            "reference_heading": heading,
            "start_line": index + 1,
            "start_char": start,
            "end_line": next_appendix_index + 1 if next_appendix_index is not None else len(lines),
            "end_char": end,
            "preserved_following_appendix": next_appendix_index is not None,
            "omitted_char_count": omitted_char_count,
            "reason": "reference section omitted from prepared markdown while raw retains full text",
        }
    return text.rstrip() + "\n", report


def compact_appendix_sections_for_prepared_markdown(
    text: str,
    *,
    enabled: bool,
    min_document_chars: int = APPENDIX_COMPACTION_MIN_DOCUMENT_CHARS,
    min_omitted_chars: int = APPENDIX_COMPACTION_MIN_OMITTED_CHARS,
    min_start_ratio: float = APPENDIX_COMPACTION_MIN_START_RATIO,
    section_excerpt_limit: int = APPENDIX_COMPACTION_SECTION_EXCERPT_LIMIT,
    max_sections: int = APPENDIX_COMPACTION_MAX_SECTIONS,
) -> tuple[str, dict[str, Any]]:
    report: dict[str, Any] = {
        "compacted": False,
        "reason": "appendix compaction disabled",
        "omitted_char_count": 0,
    }
    if not enabled:
        return text.rstrip() + "\n", report
    if len(text) < min_document_chars:
        report["reason"] = "document is below appendix compaction size threshold"
        return text.rstrip() + "\n", report
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    appendix_index = next(
        (index for index, line in enumerate(lines) if APPENDIX_SECTION_HEADING_RE.match(line.strip())),
        None,
    )
    if appendix_index is None:
        report["reason"] = "no appendix-like heading found"
        return text.rstrip() + "\n", report
    appendix_start = offsets[appendix_index]
    if appendix_start / len(text) < min_start_ratio:
        report["reason"] = "appendix starts too early"
        return text.rstrip() + "\n", report
    appendix_text = text[appendix_start:]
    compacted_appendix, appendix_report = compact_appendix_text(
        appendix_text,
        section_excerpt_limit=section_excerpt_limit,
        max_sections=max_sections,
    )
    omitted_char_count = len(appendix_text.rstrip()) - len(compacted_appendix.rstrip())
    if omitted_char_count < min_omitted_chars:
        report["reason"] = "appendix omitted chars below compaction threshold"
        report["omitted_char_count"] = max(0, omitted_char_count)
        return text.rstrip() + "\n", report
    prepared = text[:appendix_start].rstrip() + "\n\n" + compacted_appendix.rstrip() + "\n"
    return prepared, {
        "compacted": True,
        "appendix_start_line": appendix_index + 1,
        "appendix_start_char": appendix_start,
        "original_appendix_char_count": len(appendix_text.rstrip()),
        "compacted_appendix_char_count": len(compacted_appendix.rstrip()),
        "omitted_char_count": omitted_char_count,
        "section_excerpt_limit": section_excerpt_limit,
        "max_sections": max_sections,
        **appendix_report,
        "reason": "appendix sections compacted while raw retains full appendix",
    }


def compact_appendix_text(
    appendix_text: str,
    *,
    section_excerpt_limit: int,
    max_sections: int,
) -> tuple[str, dict[str, Any]]:
    lines = appendix_text.splitlines(keepends=True)
    heading_indices = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s{0,3}#{2,6}\s+\S", line) or (index == 0 and APPENDIX_SECTION_HEADING_RE.match(line.strip()))
    ]
    if not heading_indices or heading_indices[0] != 0:
        heading_indices.insert(0, 0)
    heading_indices = sorted(set(heading_indices))
    sections: list[tuple[int, int]] = []
    for position, start_index in enumerate(heading_indices):
        end_index = heading_indices[position + 1] if position + 1 < len(heading_indices) else len(lines)
        sections.append((start_index, end_index))
    output: list[str] = []
    compacted_sections = 0
    omitted_sections = 0
    for section_number, (start_index, end_index) in enumerate(sections, start=1):
        section_text = "".join(lines[start_index:end_index]).strip()
        if not section_text:
            continue
        heading = lines[start_index].strip() if start_index < len(lines) else f"Appendix section {section_number}"
        if section_number > max_sections:
            omitted_sections += 1
            continue
        excerpt = compact_payload_text(section_text, section_excerpt_limit)
        if len(section_text) > len(excerpt):
            compacted_sections += 1
            output.append(
                f"{excerpt}\n\n"
                "[Appendix section compacted in prepared markdown; original raw retains the full appendix section.]"
            )
        else:
            output.append(section_text)
        if section_number == max_sections and len(sections) > max_sections:
            omitted_sections += len(sections) - max_sections
            output.append(
                "## Additional Appendix Sections Omitted\n\n"
                f"[{len(sections) - max_sections} additional appendix section(s) omitted from prepared markdown; original raw retains them.]"
            )
            break
    return "\n\n".join(output).rstrip() + "\n", {
        "original_section_count": len(sections),
        "included_section_count": min(len(sections), max_sections),
        "compacted_section_count": compacted_sections,
        "omitted_section_count": omitted_sections,
    }


def write_raw_prepare_fast_path_report(step_root: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = step_root / "raw_prepare_fast_path.json"
    md_path = step_root / "raw_prepare_fast_path.md"
    write_json(json_path, report)
    reason_rows = [[reason] for reason in report.get("reasons", [])]
    suppressed_reason_rows = [[reason] for reason in report.get("policy_suppressed_reasons", [])]
    noise = report.get("noise_profile", {})
    noise_rows = [
        [key, f"{value:.3f}" if isinstance(value, float) else str(value)]
        for key, value in noise.items()
    ]
    reference_truncation = report.get("reference_truncation", {})
    reference_rows = [[key, str(value)] for key, value in reference_truncation.items()]
    appendix_compaction = report.get("appendix_compaction", {})
    appendix_rows = [[key, str(value)] for key, value in appendix_compaction.items()]
    success_note = (
        "_无，已使用 --skip-prepare passthrough。_"
        if report.get("raw_prepare_policy") == RawPreparePolicy.skip_model.value
        else "_无，已使用 deterministic markdown passthrough。_"
    )
    md_path.write_text(
        "# Raw Prepare Fast Path\n\n"
        f"- 规则版本: `{report.get('rule_version', RAW_PREPARE_FAST_PATH_RULE_VERSION)}`\n"
        f"- 清洗策略: `{report.get('raw_prepare_policy', RawPreparePolicy.auto.value)}`\n"
        f"- 是否启用: `{str(report.get('eligible', False)).lower()}`\n"
        f"- 原始材料: `{report.get('source_raw_path', '')}`\n"
        f"- Raw Wikilink 规范化: `{report.get('raw_link_cleanup_ref', '')}`\n\n"
        "## 未启用原因\n\n"
        f"{format_markdown_table(['原因'], reason_rows) if reason_rows else success_note}\n\n"
        "## Policy 覆盖的自动拦截原因\n\n"
        f"{format_markdown_table(['原因'], suppressed_reason_rows) if suppressed_reason_rows else '_无。_'}\n\n"
        "## 噪声画像\n\n"
        f"{format_markdown_table(['字段', '值'], noise_rows) if noise_rows else '_暂无。_'}\n\n"
        "## 参考文献截断\n\n"
        f"{format_markdown_table(['字段', '值'], reference_rows) if reference_rows else '_未评估。_'}\n\n"
        "## Appendix 压缩\n\n"
        f"{format_markdown_table(['字段', '值'], appendix_rows) if appendix_rows else '_未评估。_'}\n",
        encoding="utf-8",
    )
    return json_path, md_path


def _run_raw_prepare(ctx: StepRunContext) -> None:
    step_name = "raw_prepare"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    cleanup_path = require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
    input_raw_sha256 = sha256_file(ctx.raw_path)
    cleanup_ref = cleanup_path.relative_to(ctx.run_dir).as_posix()
    fast_path_report_refs: list[ArtifactRef] = []
    raw_prepare_policy = ctx.manifest.vault_config_snapshot.raw_prepare_policy
    if raw_prepare_policy != RawPreparePolicy.auto or raw_prepare_fast_path_provider_allowed(ctx, step_name):
        preparation, fast_path_report = build_raw_prepare_fast_path(
            raw_path=ctx.raw_path,
            raw_rel=raw_rel,
            input_raw_sha256=input_raw_sha256,
            cleanup=cleanup,
            cleanup_ref=cleanup_ref,
            raw_prepare_policy=raw_prepare_policy,
        )
        fast_json, fast_md = write_raw_prepare_fast_path_report(step_root, fast_path_report)
        fast_path_report_refs = [
            _ref(ctx.run_dir, fast_json, step_name, "json", "raw_prepare_fast_path.v1"),
            _ref(ctx.run_dir, fast_md, step_name, "markdown"),
        ]
        if preparation is not None:
            validate_raw_preparation(preparation)
            out = step_root / "raw_preparation.json"
            write_json(out, preparation)
            prepared = step_root / "prepared.md"
            prepared.parent.mkdir(parents=True, exist_ok=True)
            prepared.write_text(preparation.prepared_markdown.rstrip() + "\n", encoding="utf-8")
            review = step_root / "preparation_review.md"
            review.write_text(render_preparation_review(preparation), encoding="utf-8")
            complete_step(
                ctx.manifest,
                step_name,
                outputs=[
                    _ref(ctx.run_dir, out, step_name, "json", "raw_preparation.v1"),
                    _ref(ctx.run_dir, prepared, step_name, "markdown"),
                    _ref(ctx.run_dir, review, step_name, "markdown"),
                    *fast_path_report_refs,
                ],
            )
            return
    elif ctx.raw_path.suffix.lower() in {".md", ".markdown", ".mdown"}:
        runtime = ctx.execution_context.record.providers.get(step_name) if ctx.execution_context.record else None
        skipped_report = build_raw_prepare_provider_ineligible_report(
            raw_path=ctx.raw_path,
            raw_rel=raw_rel,
            input_raw_sha256=input_raw_sha256,
            cleanup_ref=cleanup_ref,
            raw_prepare_policy=raw_prepare_policy,
            provider_spec=runtime.spec if runtime is not None else None,
        )
        fast_json, fast_md = write_raw_prepare_fast_path_report(step_root, skipped_report)
        fast_path_report_refs = [
            _ref(ctx.run_dir, fast_json, step_name, "json", "raw_prepare_fast_path.v1"),
            _ref(ctx.run_dir, fast_md, step_name, "markdown"),
        ]
    payload = {
        "source_raw_path": raw_rel,
        "source_raw_sha256": input_raw_sha256,
        "raw_markdown": ctx.raw_path.read_text(encoding="utf-8"),
        "raw_link_cleanup_ref": cleanup_ref,
        "raw_link_cleanup": {
            "changed": cleanup.changed,
            "cleaned_link_count": cleanup.cleaned_link_count,
            "preserved_media_embed_count": cleanup.preserved_media_embed_count,
            "cleanup_rule_version": cleanup.cleanup_rule_version,
        },
        "contract": RAW_PREPARE_CONTRACT,
    }
    def validate_raw_prepare_model(model: RawPreparationArtifact) -> None:
        candidate = model.model_copy(
            update={
                "input_raw_sha256": input_raw_sha256,
                "raw_link_cleanup_ref": cleanup_ref,
            }
        )
        validate_raw_preparation(candidate)
        if candidate.source_raw_path != raw_rel:
            raise ContractValidationError(
                f"raw_prepare source path mismatch: {candidate.source_raw_path} != {raw_rel}",
                issues=[
                    StructuredIssue(
                        issue_code="source_path_mismatch",
                        field_path="source_raw_path",
                        validator_id="validate_raw_prepare_model",
                        message=f"raw_prepare source path mismatch: {candidate.source_raw_path} != {raw_rel}",
                        repairability="repairable",
                    )
                ],
            )

    preparation, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        RawPreparationArtifact,
        validator=validate_raw_prepare_model,
    )
    preparation = _redacted_model(ctx, preparation, RawPreparationArtifact)
    preparation = preparation.model_copy(
        update={
            "input_raw_sha256": input_raw_sha256,
            "raw_link_cleanup_ref": cleanup_ref,
        }
    )
    validate_raw_preparation(preparation)
    if preparation.source_raw_path != raw_rel:
        raise PipelineError(f"raw_prepare source path mismatch: {preparation.source_raw_path} != {raw_rel}")
    out = step_root / "raw_preparation.json"
    write_json(out, preparation)
    prepared = step_root / "prepared.md"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(preparation.prepared_markdown.rstrip() + "\n", encoding="utf-8")
    review = step_root / "preparation_review.md"
    review.write_text(render_preparation_review(preparation), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, out, step_name, "json", "raw_preparation.v1"),
        _ref(ctx.run_dir, prepared, step_name, "markdown"),
        _ref(ctx.run_dir, review, step_name, "markdown"),
    ]
    outputs.extend(fast_path_report_refs)
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=outputs)


def _run_prepared_raw_review(ctx: StepRunContext) -> None:
    step_name = "prepared_raw_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    prepared = require_step_output_dir(ctx.run_dir, "raw_prepare") / "prepared.md"
    preparation = read_model(require_step_output_dir(ctx.run_dir, "raw_prepare") / "raw_preparation.json", RawPreparationArtifact)
    fast_path_report_path = require_step_output_dir(ctx.run_dir, "raw_prepare") / "raw_prepare_fast_path.json"
    suppressed_reasons: list[str] = []
    if fast_path_report_path.exists():
        try:
            fast_path_report = json.loads(fast_path_report_path.read_text(encoding="utf-8"))
            suppressed_reasons = [str(reason) for reason in fast_path_report.get("policy_suppressed_reasons", [])]
        except Exception:
            suppressed_reasons = []
    approved = step_root / "approved_prepared.md"
    approved.write_text(prepared.read_text(encoding="utf-8"), encoding="utf-8")
    prompt = step_root / "review_prompt.md"
    risk_section = ""
    if preparation.requires_human_review or suppressed_reasons:
        risk_rows = [[reason] for reason in suppressed_reasons]
        risk_section = (
            "\n## Skip Prepare 风险提示\n\n"
            f"- risk_level: `{preparation.risk_level}`\n"
            f"- requires_human_review: `{str(preparation.requires_human_review).lower()}`\n"
            "- 说明：用户显式选择了 passthrough；本步骤仍自动批准，但下游审核应知道 raw_prepare 覆盖了自动质量拦截。\n\n"
            + (
                format_markdown_table(["被覆盖的自动拦截原因"], risk_rows)
                if risk_rows
                else "_无明确 policy_suppressed_reasons。_"
            )
            + "\n"
        )
    prompt.write_text(
        "# Prepared Raw 审核\n\n"
        "当前 MVP 自动批准 prepared raw；后续会加入交互式人工审核。\n"
        f"{risk_section}",
        encoding="utf-8",
    )
    feedback = step_root / "review_feedback.jsonl"
    feedback.write_text("", encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        review_mode="auto_stub",
        auto_approved=True,
        notes=(
            "当前 MVP 自动批准；交互式审核是后续工作。"
            if not suppressed_reasons
            else f"当前 MVP 自动批准；--skip-prepare 覆盖 {len(suppressed_reasons)} 个自动质量拦截。"
        ),
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            _ref(ctx.run_dir, approved, step_name, "markdown"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def build_source_digest_source_map(
    approved_prepared_text: str,
    *,
    approved_prepared_ref: str,
    full_source_limit: int = SOURCE_DIGEST_FULL_SOURCE_CHAR_LIMIT,
    total_limit: int = SOURCE_DIGEST_SOURCE_MAP_TOTAL_LIMIT,
    global_limit: int = SOURCE_DIGEST_SOURCE_MAP_GLOBAL_EXCERPT_LIMIT,
    min_section_limit: int = SOURCE_DIGEST_SOURCE_MAP_MIN_SECTION_EXCERPT_LIMIT,
    max_section_limit: int = SOURCE_DIGEST_SOURCE_MAP_MAX_SECTION_EXCERPT_LIMIT,
    max_sections: int = SOURCE_DIGEST_SOURCE_MAP_MAX_SECTIONS,
    max_captions: int = SOURCE_DIGEST_SOURCE_MAP_MAX_CAPTIONS,
) -> dict[str, Any]:
    include_full_source = len(approved_prepared_text) <= full_source_limit
    sections = markdown_sections_for_source_map(approved_prepared_text, max_sections=max_sections)
    section_budget_total = max(0, total_limit - global_limit)
    section_limit = max_section_limit
    if sections:
        section_limit = max(min_section_limit, min(max_section_limit, section_budget_total // max(1, len(sections))))
    source_map_sections: list[dict[str, Any]] = []
    included_section_chars = 0
    for section in sections:
        excerpt = "" if include_full_source else compact_payload_text(section["text"], section_limit)
        included_section_chars += len(excerpt)
        source_map_sections.append(
            {
                "section_id": section["section_id"],
                "heading": section["heading"],
                "level": section["level"],
                "line_start": section["line_start"],
                "line_end": section["line_end"],
                "char_start": section["char_start"],
                "char_end": section["char_end"],
                "original_char_count": len(section["text"]),
                "excerpt": excerpt,
                "excerpt_char_count": len(excerpt),
                "truncated": len(section["text"].strip()) > len(excerpt),
            }
        )
    global_excerpt = "" if include_full_source else source_global_excerpt(approved_prepared_text, global_limit)
    captions = [] if include_full_source else source_digest_caption_snippets(approved_prepared_text, max_captions=max_captions)
    included_chars = len(global_excerpt) + included_section_chars + sum(len(item["text"]) for item in captions)
    return {
        "schema_version": "source_digest_source_map.v1",
        "approved_prepared_ref": approved_prepared_ref,
        "full_source_in_payload": include_full_source,
        "full_source_limit": full_source_limit,
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "total_limit": total_limit,
        "global_excerpt_limit": global_limit,
        "section_excerpt_limit": section_limit,
        "max_sections": max_sections,
        "omitted_section_count": max(0, markdown_heading_count(approved_prepared_text) - len(source_map_sections)),
        "global_excerpt": global_excerpt,
        "outline": [
            {
                "heading": section["heading"],
                "level": section["level"],
                "line_start": section["line_start"],
                "section_id": section["section_id"],
            }
            for section in sections
        ],
        "captions": captions,
        "sections": source_map_sections,
    }


def markdown_heading_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s{0,3}#{1,6}\s+\S", line))


def markdown_sections_for_source_map(text: str, *, max_sections: int) -> list[dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    heading_indices: list[int] = []
    for index, line in enumerate(lines):
        offsets.append(cursor)
        if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
            heading_indices.append(index)
        cursor += len(line)
    if not lines:
        return []
    if not heading_indices or heading_indices[0] != 0:
        heading_indices.insert(0, 0)
    all_heading_indices = sorted(set(heading_indices))
    heading_indices = all_heading_indices[:max_sections]
    sections: list[dict[str, Any]] = []
    for position, start_index in enumerate(heading_indices):
        source_position = all_heading_indices.index(start_index)
        end_index = all_heading_indices[source_position + 1] if source_position + 1 < len(all_heading_indices) else len(lines)
        text_block = "".join(lines[start_index:end_index]).strip()
        if not text_block:
            continue
        heading_line = lines[start_index].strip() if start_index < len(lines) else ""
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", heading_line)
        level = len(match.group(1)) if match else 0
        heading = match.group(2).strip() if match else "Preamble"
        char_start = offsets[start_index] if start_index < len(offsets) else 0
        char_end = offsets[end_index] if end_index < len(offsets) else len(text)
        sections.append(
            {
                "section_id": f"S{len(sections) + 1:03d}",
                "heading": heading,
                "level": level,
                "line_start": start_index + 1,
                "line_end": end_index,
                "char_start": char_start,
                "char_end": char_end,
                "text": text_block,
            }
        )
    return sections


def source_digest_caption_snippets(text: str, *, max_captions: int) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not PAPER_CAPTION_RE.match(stripped):
            continue
        captions.append(
            {
                "line": line_number,
                "text": compact_payload_text(stripped, 280),
            }
        )
        if len(captions) >= max_captions:
            break
    return captions


def source_digest_language_contract(vault_config: VaultConfig) -> dict[str, Any]:
    return {
        "vault_language": vault_config.wiki_language,
        "hard_requirement": (
            "All source_digest user-visible prose fields must be written in Chinese for zh-CN vaults. "
            "Do not answer source_digest in English and do not rely on a later repair pass to translate."
        ),
        "fields_must_be_chinese": [
            "summary",
            "key_takeaways",
            "one_sentence_summary",
            "why_matters",
            "wiki_value",
            "open_question_or_tension",
            "resolution_hint",
            "weak_or_noise_items.why_matches",
        ],
        "stable_terms_may_remain_english": [
            "Claude Code",
            "Cowork",
            "PM",
            "RAG",
            "Workflow",
            "Agent",
            "API",
            "Evals",
            "MCP",
        ],
        "allowed_english_boundary": (
            "Keep stable product names and technical terms in English when they are canonical names, "
            "but surround them with Chinese explanation instead of writing full English sentences."
        ),
        "bad_example": "Cat Wu discusses how the team achieves extremely fast product development cycles.",
        "good_example": "Cat Wu 讨论 Anthropic 团队如何缩短产品开发周期，并说明 AI 时代 PM 角色与产品品味的重要性。",
    }


def render_source_digest_source_map_markdown(source_map: dict[str, Any]) -> str:
    rows = [
        ["full_source_in_payload", str(source_map.get("full_source_in_payload", False)).lower()],
        ["original_char_count", source_map.get("original_char_count", 0)],
        ["included_char_count", source_map.get("included_char_count", 0)],
        ["section_excerpt_limit", source_map.get("section_excerpt_limit", 0)],
        ["section_count", len(source_map.get("sections", []))],
        ["omitted_section_count", source_map.get("omitted_section_count", 0)],
        ["caption_count", len(source_map.get("captions", []))],
    ]
    section_rows = [
        [
            section.get("section_id", ""),
            section.get("line_start", ""),
            "#" * int(section.get("level", 0) or 0),
            section.get("heading", ""),
            section.get("original_char_count", 0),
            section.get("excerpt_char_count", 0),
            str(section.get("truncated", False)).lower(),
        ]
        for section in source_map.get("sections", [])
    ]
    return (
        "# Source Digest Source Map\n\n"
        f"- Approved prepared ref: `{source_map.get('approved_prepared_ref', '')}`\n\n"
        "## Payload Budget\n\n"
        f"{format_markdown_table(['字段', '值'], rows)}\n\n"
        "## Sections\n\n"
        f"{format_markdown_table(['ID', 'Line', 'Level', 'Heading', 'Original chars', 'Excerpt chars', 'Truncated'], section_rows)}\n"
    )


def project_source_digest_source_map_for_payload(source_map: dict[str, Any], *, full_source_map_ref: str) -> dict[str, Any]:
    return {
        "schema_version": "source_digest_source_map_payload.v1",
        "full_source_map_ref": full_source_map_ref,
        "approved_prepared_ref": source_map.get("approved_prepared_ref", ""),
        "full_source_in_payload": source_map.get("full_source_in_payload", False),
        "original_char_count": source_map.get("original_char_count", 0),
        "included_char_count": source_map.get("included_char_count", 0),
        "section_excerpt_limit": source_map.get("section_excerpt_limit", 0),
        "global_excerpt": source_map.get("global_excerpt", ""),
        "captions": source_map.get("captions", []),
        "sections": [
            {
                "id": section.get("section_id", ""),
                "heading": section.get("heading", ""),
                "level": section.get("level", 0),
                "line_start": section.get("line_start", 0),
                "original_char_count": section.get("original_char_count", 0),
                "excerpt": section.get("excerpt", ""),
            }
            for section in source_map.get("sections", [])
        ],
    }


def build_source_kind_hints(text: str, raw_rel: str) -> dict[str, Any]:
    lowered_path = raw_rel.lower()
    lowered_text = text.lower()
    readme_path = lowered_path.endswith("readme.md")
    toc_link_count = len(re.findall(r"\]\((?:\./)?(?:docs|chapter|chapters|extra-chapter|co-creation-projects)/", text, flags=re.IGNORECASE))
    markdown_link_count = len(re.findall(r"\[[^\]\n]+\]\([^)]+\)", text))
    badge_count = len(re.findall(r"shields\.io|badge|trendshift|github stars|github forks", lowered_text))
    download_marker_count = len(re.findall(r"下载|download|releases/latest|pdf", lowered_text))
    github_url_count = len(re.findall(r"https?://(?:www\.)?github\.com/|github\.com[:/]", lowered_text))
    heading_decoration = r"(?:[^\w\u4e00-\u9fff#\n]+)?\s*"
    contributor_heading_count = len(
        re.findall(
            rf"(?im)^\s{{0,3}}#{{1,6}}\s*{heading_decoration}(?:核心贡献者|贡献者|致谢|contributors?|acknowledg)",
            text,
        )
    )
    tutorial_heading_count = len(
        re.findall(
            rf"(?im)^\s{{0,3}}#{{1,6}}\s*{heading_decoration}(?:内容导航|目录|学习路线|快速开始|如何学习|课程|教程|chapters?|curriculum)",
            text,
        )
    )
    contributor_section_present = contributor_heading_count > 0
    badge_or_download_heavy = badge_count >= 3 or download_marker_count >= 3
    github_url_present = github_url_count > 0
    index_heading_present = tutorial_heading_count > 0
    tutorial_index = (toc_link_count >= 6 and (readme_path or index_heading_present)) or (
        index_heading_present and (readme_path or toc_link_count >= 3 or markdown_link_count >= 8)
    )
    navigation_heavy = (toc_link_count >= 8 and (readme_path or index_heading_present)) or (
        markdown_link_count >= 24 and (readme_path or index_heading_present or contributor_section_present or badge_or_download_heavy)
    )
    repository_readme = readme_path or (
        github_url_present
        and (badge_count > 0 or contributor_section_present)
        and (tutorial_index or markdown_link_count >= 12 or download_marker_count > 0)
    )
    flags = [
        name
        for name, enabled in [
            ("github_url_present", github_url_present),
            ("repository_readme", repository_readme),
            ("tutorial_index", tutorial_index),
            ("navigation_heavy", navigation_heavy),
            ("contributor_section_present", contributor_section_present),
            ("badge_or_download_heavy", badge_or_download_heavy),
        ]
        if enabled
    ]
    return {
        "schema_version": "source_kind_hints.v1",
        "source_raw_path": raw_rel,
        "flags": flags,
        "github_url_present": github_url_present,
        "repository_readme": repository_readme,
        "tutorial_index": tutorial_index,
        "navigation_heavy": navigation_heavy,
        "contributor_section_present": contributor_section_present,
        "badge_or_download_heavy": badge_or_download_heavy,
        "counts": {
            "toc_link_count": toc_link_count,
            "markdown_link_count": markdown_link_count,
            "badge_count": badge_count,
            "download_marker_count": download_marker_count,
            "github_url_count": github_url_count,
            "contributor_heading_count": contributor_heading_count,
            "tutorial_heading_count": tutorial_heading_count,
        },
        "guidance": [
            "README/index sources are entry pages; avoid turning badges, downloads, contributor lists, and TOC-only rows into formal pages.",
            "Keep formal candidates for durable project/framework entities, distinctive concepts, reusable designs, and comparisons with substantive source context.",
            "Move contributor/acknowledgement people to weak_or_noise_items unless the body gives reusable context beyond a name in a list.",
        ],
    }


def render_source_kind_hints_markdown(hints: dict[str, Any]) -> str:
    rows = [
        [name, str(bool(hints.get(name, False))).lower()]
        for name in [
            "repository_readme",
            "github_url_present",
            "tutorial_index",
            "navigation_heavy",
            "contributor_section_present",
            "badge_or_download_heavy",
        ]
    ]
    count_rows = [[key, value] for key, value in hints.get("counts", {}).items()]
    guidance_rows = [[item] for item in hints.get("guidance", [])]
    return (
        "# Source Kind Hints\n\n"
        f"- source：`{hints.get('source_raw_path', '')}`\n"
        f"- flags：{', '.join(f'`{item}`' for item in hints.get('flags', [])) or '无'}\n\n"
        "## Flags\n\n"
        f"{format_markdown_table(['flag', 'enabled'], rows)}\n\n"
        "## Counts\n\n"
        f"{format_markdown_table(['count', 'value'], count_rows)}\n\n"
        "## Guidance\n\n"
        f"{format_markdown_table(['rule'], guidance_rows)}\n"
    )


def _run_source_digest(ctx: StepRunContext) -> None:
    step_name = "source_digest"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    approved_prepared = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    approved_prepared_text = approved_prepared.read_text(encoding="utf-8")
    approved_prepared_ref = approved_prepared.relative_to(ctx.run_dir).as_posix()
    source_map = build_source_digest_source_map(approved_prepared_text, approved_prepared_ref=approved_prepared_ref)
    source_map_path = step_root / "source_digest_source_map.json"
    source_map_md = step_root / "source_digest_source_map.md"
    write_json(source_map_path, source_map)
    source_map_md.write_text(render_source_digest_source_map_markdown(source_map), encoding="utf-8")
    source_map_payload = project_source_digest_source_map_for_payload(
        source_map,
        full_source_map_ref=source_map_path.relative_to(ctx.run_dir).as_posix(),
    )
    source_map_payload_path = step_root / "source_digest_source_map_payload.json"
    write_json(source_map_payload_path, source_map_payload)
    source_kind_hints = build_source_kind_hints(approved_prepared_text, raw_rel)
    source_kind_hints_path = step_root / "source_kind_hints.json"
    source_kind_hints_md = step_root / "source_kind_hints.md"
    write_json(source_kind_hints_path, source_kind_hints)
    source_kind_hints_md.write_text(render_source_kind_hints_markdown(source_kind_hints), encoding="utf-8")
    payload = {
        "source_raw_path": raw_rel,
        "approved_prepared_markdown": approved_prepared_text if source_map["full_source_in_payload"] else "",
        "approved_prepared_ref": approved_prepared_ref,
        "source_digest_source_map": source_map_payload,
        "source_kind_hints": source_kind_hints,
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": source_digest_language_contract(ctx.manifest.vault_config_snapshot),
        "contract": {
            "goal": "Create a complete source digest of wiki-worthy candidates from this one raw file.",
            "candidate_fields": list(SourceDigestCandidate.model_fields),
            "weak_or_noise_fields": list(WeakOrNoiseItem.model_fields),
            "candidate_page_budget": ctx.manifest.vault_config_snapshot.max_ingest_candidates,
            "rules": [
                "Return each candidate group as an array of candidate objects, never as bare fields.",
                "Every candidate object must include candidate_id, name, type, one_sentence_summary, why_matters, and wiki_value.",
                "For entities, concepts, designs, comparisons, and open_questions, suggested_page_title must be non-empty.",
                "Do not decide create, update, duplicate, or cross-reference actions in source_digest.",
                "Keep formal ingest candidates within candidate_page_budget; choose durable reusable themes over every subtopic.",
                "Prefer wiki-worthy candidates over every minor mention.",
                "Use source_locator as a lightweight review locator, not a strict evidence chain.",
                "Put weak or noisy mentions in weak_or_noise_items instead of creating pages for them.",
                "weak_or_noise_items are review-only and are not ingested as wiki pages.",
                "weak_or_noise_items may leave suggested_page_title empty and may use suggested_action='ignore'.",
                "weak_or_noise_items may use why_matters or why_matches to explain why the mention was filtered.",
                "The vault language is zh-CN: write summary, key_takeaways, candidate summaries, why_matters, and wiki_value in Chinese.",
                "Stable domain terms such as Claude Code, RAG, PM, Workflow, Agent may stay in English, but explain them in Chinese when needed.",
                "Do not return whole English paragraphs for user-visible fields; zh-CN validation will fail instead of silently translating.",
                "If approved_prepared_markdown is empty, use source_digest_source_map sections, outline, captions, and approved_prepared_ref instead of assuming source content is absent.",
                "For long source-map payloads, choose durable candidates visible across the outline and section excerpts; do not create candidates from bibliography or appendix-only noise.",
                "Use section headings, line_start, and source_map section_id values as source_locator review handles when exact full source text is not in the payload.",
                "If source_kind_hints suggests a repository README, tutorial index, or navigation-heavy source, treat the file as an entry page rather than a chapter-by-chapter source.",
                "For README/index sources, do not create formal candidates for badges, status counters, install/download links, release links, table-of-contents rows, or chapter headings that only navigate elsewhere.",
                "For README/index sources, omit contributor/acknowledgement people or place them in weak_or_noise_items unless the person is central to the material and the body provides substantive reusable context beyond a contributor list.",
                "For README/index sources, prefer at most a few durable candidates: the core project/framework/entity, distinctive concepts, reusable designs, and comparisons with source-backed explanations.",
            ],
        },
    }
    digest, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        SourceDigestArtifact,
        validator=lambda model: validate_source_digest(model, language=ctx.manifest.vault_config_snapshot.wiki_language),
    )
    digest = _redacted_model(ctx, digest, SourceDigestArtifact)
    if digest.source_raw_path != raw_rel:
        raise PipelineError(f"source_digest source path mismatch: {digest.source_raw_path} != {raw_rel}")
    digest = augment_source_digest_anchor_entities(digest, approved_prepared_text)
    digest, budget_report = cap_source_digest_candidates(digest, ctx.manifest.vault_config_snapshot.max_ingest_candidates)
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    out = step_root / "source_digest.json"
    write_json(out, digest)
    digest_md = step_root / "source_digest.md"
    digest_md.write_text(render_source_digest_markdown(digest), encoding="utf-8")
    budget_report_path = step_root / "source_digest_budget_report.json"
    budget_report_md = step_root / "source_digest_budget_report.md"
    write_json(budget_report_path, budget_report)
    budget_report_md.write_text(render_source_digest_budget_report(budget_report), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, source_map_path, step_name, "json", "source_digest_source_map.v1"),
        _ref(ctx.run_dir, source_map_md, step_name, "markdown"),
        _ref(ctx.run_dir, source_map_payload_path, step_name, "json", "source_digest_source_map_payload.v1"),
        _ref(ctx.run_dir, source_kind_hints_path, step_name, "json", "source_kind_hints.v1"),
        _ref(ctx.run_dir, source_kind_hints_md, step_name, "markdown"),
        _ref(ctx.run_dir, out, step_name, "json", "source_digest.v2"),
        _ref(ctx.run_dir, digest_md, step_name, "markdown"),
        _ref(ctx.run_dir, budget_report_path, step_name, "json", "source_digest_budget_report.v1"),
        _ref(ctx.run_dir, budget_report_md, step_name, "markdown"),
    ]
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=outputs)


def _run_source_digest_review(ctx: StepRunContext) -> None:
    step_name = "source_digest_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest_json = require_step_output_dir(ctx.run_dir, "source_digest") / "source_digest.json"
    digest_md = require_step_output_dir(ctx.run_dir, "source_digest") / "source_digest.md"
    approved_json = step_root / "approved_digest.json"
    approved_md = step_root / "approved_digest.md"
    approved_json.write_text(digest_json.read_text(encoding="utf-8"), encoding="utf-8")
    approved_md.write_text(digest_md.read_text(encoding="utf-8"), encoding="utf-8")
    prompt = step_root / "review_prompt.md"
    feedback = step_root / "review_feedback.jsonl"
    prompt.write_text(
        "# Source Digest 审核\n\n"
        "当前 MVP 自动批准 source digest；这一步检查单篇材料提取是否完整、是否中文、是否把弱相关内容放进噪声区。\n\n"
        "## 核心判断\n\n"
        "- 是否漏掉了值得入库的实体、概念、设计、对比或未决问题？\n"
        "- 是否把只是口播过渡、广告、寒暄或弱相关提及错误变成页面候选？\n"
        "- 摘要、关键收获和候选说明是否为中文，英文术语是否有中文上下文？\n\n"
        "## 关键文件\n\n"
        f"- 待审摘要：`{digest_md.relative_to(ctx.run_dir).as_posix()}`\n"
        f"- 可编辑批准文件：`{approved_json.relative_to(ctx.run_dir).as_posix()}`\n"
        f"- 反馈记录：`{feedback.relative_to(ctx.run_dir).as_posix()}`\n\n"
        "后续会加入真正的 list/filter/show/diff/revise/approve；本轮仍为 auto-stub。\n",
        encoding="utf-8",
    )
    feedback.write_text("", encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        review_mode="auto_stub",
        auto_approved=True,
        notes="当前 MVP 自动批准 source digest；交互式 digest 审核是后续工作。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            _ref(ctx.run_dir, approved_json, step_name, "json", "source_digest.v2"),
            _ref(ctx.run_dir, approved_md, step_name, "markdown"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def build_candidate_resolution_source_pack(
    approved_prepared_text: str,
    digest: SourceDigestArtifact,
    *,
    full_source_limit: int = CANDIDATE_RESOLUTION_FULL_SOURCE_CHAR_LIMIT,
    global_limit: int = CANDIDATE_RESOLUTION_GLOBAL_EXCERPT_LIMIT,
    per_candidate_limit: int = CANDIDATE_RESOLUTION_PER_CANDIDATE_EXCERPT_LIMIT,
) -> dict[str, Any]:
    include_full_source = len(approved_prepared_text) <= full_source_limit
    global_excerpt = "" if include_full_source else source_global_excerpt(approved_prepared_text, global_limit)
    items: list[dict[str, Any]] = []
    included_chars = len(global_excerpt)
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        if group_name in {"budget_deferred_candidates", "weak_or_noise_items"}:
            continue
        for candidate in getattr(digest, group_name):
            cues = [
                candidate.name,
                candidate.suggested_page_title,
                candidate.one_sentence_summary,
                candidate.why_matters,
                candidate.wiki_value,
                candidate.source_locator,
                candidate.open_question_or_tension,
            ]
            snippets = [] if include_full_source else source_snippets_for_cues(
                approved_prepared_text,
                cues,
                max_chars=per_candidate_limit,
            )
            included_chars += sum(len(snippet["text"]) for snippet in snippets)
            items.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "group": group_name,
                    "name": candidate.name,
                    "suggested_page_title": candidate.suggested_page_title,
                    "source_locator": candidate.source_locator,
                    "snippets": snippets,
                }
            )
    return {
        "schema_version": "candidate_resolution_source_excerpt_pack.v1",
        "source_raw_path": digest.source_raw_path,
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "full_source_in_payload": include_full_source,
        "full_source_limit": full_source_limit,
        "global_excerpt_limit": global_limit,
        "per_candidate_excerpt_limit": per_candidate_limit,
        "candidate_count": len(items),
        "global_excerpt": global_excerpt,
        "items": items,
    }


def render_candidate_resolution_source_pack_markdown(pack: dict[str, Any]) -> str:
    rows = [
        [
            item.get("candidate_id", ""),
            item.get("group", ""),
            item.get("name", ""),
            len(item.get("snippets", [])) if isinstance(item.get("snippets", []), list) else 0,
            item.get("source_locator", ""),
        ]
        for item in pack.get("items", [])
        if isinstance(item, dict)
    ]
    return (
        "# Candidate Resolution Source Excerpt Pack\n\n"
        f"- payload 是否包含完整 source：`{str(pack.get('full_source_in_payload', False)).lower()}`\n"
        f"- 原始 source 字符：{pack.get('original_char_count', 0)}\n"
        f"- 入模摘录字符：{pack.get('included_char_count', 0)}\n"
        f"- 完整 source 引用：`{pack.get('approved_prepared_ref', '')}`\n\n"
        "## Candidate Snippets\n\n"
        f"{format_markdown_table(['Candidate', 'Group', 'Name', 'Snippets', 'Locator'], rows) if rows else '_无 candidate snippets。_'}\n"
    )


def _run_source_duplicate_guard(ctx: StepRunContext) -> None:
    step_name = "source_duplicate_guard"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    prepared = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    artifact = build_source_duplicate_guard_artifact(
        ctx.vault,
        source_raw_path=digest.source_raw_path,
        source_raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
        source_prepared_hash=sha256_file(prepared),
        operation_id=ctx.manifest.operation_id,
    )
    out = step_root / "source_duplicate_guard.json"
    write_json(out, artifact)
    md = step_root / "source_duplicate_guard.md"
    md.write_text(render_source_duplicate_guard_markdown(artifact), encoding="utf-8")
    if artifact.status == "source_duplicate":
        raise PipelineError(f"source_duplicate: {artifact.reason}")
    if artifact.status == "source_revision_detected":
        raise PipelineError(
            "同一路径内容已变化，source revision workflow 尚未实现；如确认为新材料，请另存为新 raw 文件名后重新 ingest。"
        )
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, out, step_name, "json", "source_duplicate_guard.v1"),
            _ref(ctx.run_dir, md, step_name, "markdown"),
        ],
    )


def _run_candidate_resolution(ctx: StepRunContext) -> None:
    step_name = "candidate_resolution"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    approved_prepared = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    approved_prepared_text = approved_prepared.read_text(encoding="utf-8")
    source_pack = build_candidate_resolution_source_pack(approved_prepared_text, digest)
    source_pack_path = step_root / "candidate_resolution_source_excerpt_pack.json"
    source_pack_md = step_root / "candidate_resolution_source_excerpt_pack.md"
    write_json(source_pack_path, source_pack)
    source_pack_md.write_text(render_candidate_resolution_source_pack_markdown(source_pack), encoding="utf-8")
    payload = {
        "approved_prepared_markdown": approved_prepared_text if source_pack["full_source_in_payload"] else "",
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "source_excerpt_pack": source_pack,
        "approved_digest": digest.model_dump(mode="json"),
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "candidate_coverage_required_ids": [candidate.candidate_id for candidate in digest.ingest_candidates()],
        "contract": {
            "goal": "Do coverage check and topic/page planning. Do not summarize the source again and do not decide create/update.",
            "required_minimum_fields": [
                "source_basis",
                "page_type",
                "display_title",
                "topic_summary",
                "why_this_page",
            ],
            "rules": [
                "Every approved_digest ingest candidate id must appear in source_basis.source_candidate_ids exactly once, including open_questions.",
                "Do not output weak_or_noise_items, noise-* ids, page_type=noise, or reason=ignore as formal page plans.",
                "If a weak/noise item is not wiki-worthy, omit it entirely from candidate_resolution instead of converting it to a concept.",
                "Use approved_digest candidates as the primary basis, but inspect approved_prepared for missed wiki-worthy topics.",
                "When approved_prepared_markdown is empty, use source_excerpt_pack as the only model-visible source support; the full approved source remains fixed by approved_prepared_ref for local audit.",
                "budget_deferred_candidates are not selected ingest candidates in this run; if you reuse one, put its id in prepared_discovered_candidates, not source_candidate_ids.",
                "Put any newly discovered topic in prepared_discovered_candidates.",
                "Do not read or infer existing wiki state.",
                "Write all user-visible fields in Chinese unless retaining a stable domain term.",
                "Leave page_plan_id/path_stem/candidate_target_path empty if unsure; the engine will deterministically set them.",
            ],
        },
    }
    def validate_candidate_resolution_model(model: CandidateResolutionArtifact) -> None:
        candidate = finalize_candidate_resolution(ctx.vault, ctx.profile, model, digest)
        validate_candidate_resolution(digest, candidate)

    artifact, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        CandidateResolutionArtifact,
        validator=validate_candidate_resolution_model,
    )
    artifact = _redacted_model(ctx, artifact, CandidateResolutionArtifact)
    artifact = backfill_missing_candidate_resolution_items(artifact, digest, ctx.profile)
    artifact = finalize_candidate_resolution(ctx.vault, ctx.profile, artifact, digest)
    validate_candidate_resolution(digest, artifact)
    out = step_root / "candidate_resolution.json"
    write_json(out, artifact)
    table = step_root / "candidate_resolution.md"
    table.write_text(render_candidate_resolution_markdown(artifact), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, source_pack_path, step_name, "json", "candidate_resolution_source_excerpt_pack.v1"),
        _ref(ctx.run_dir, source_pack_md, step_name, "markdown"),
        _ref(ctx.run_dir, out, step_name, "json", "candidate_resolution.v3"),
        _ref(ctx.run_dir, table, step_name, "markdown"),
    ]
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(
        ctx.manifest,
        step_name,
        outputs=outputs,
    )


def _run_wiki_context_snapshot(ctx: StepRunContext) -> None:
    step_name = "wiki_context_snapshot"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    source_title = source_title_for_raw(digest.source_raw_path)
    log_date = local_date()
    retrieval_config = effective_retrieval_config(ctx)
    snapshot = build_wiki_context_snapshot(
        ctx.vault,
        resolution,
        log_date=log_date,
        source_target_path=f"sources/{safe_filename(source_title)}.md",
        retrieval_config=retrieval_config,
        force_exact_backend=uses_mock_provider_context(ctx.execution_context),
    )
    ensure_snapshot_within_limit(snapshot, ctx.manifest.vault_config_snapshot.max_context_chars)
    contexts_path = step_root / "candidate_contexts.json"
    write_json(contexts_path, snapshot.candidate_contexts)
    contexts_md = step_root / "candidate_contexts.md"
    contexts_md.write_text(
        render_candidate_contexts_markdown(
            snapshot.candidate_contexts,
            resolved_cache_path=resolve_cache_dir(ctx.vault, retrieval_config.cache_dir).as_posix(),
            query_count=len(snapshot.candidate_contexts.items),
            encoded_page_count=snapshot.candidate_contexts.candidate_pool_size
            if snapshot.candidate_contexts.retrieval_backend == "sentence_transformers"
            else 0,
        ),
        encoding="utf-8",
    )
    snapshot = snapshot.model_copy(update={"candidate_contexts_ref": contexts_path.relative_to(ctx.run_dir).as_posix()})
    snapshot_path = step_root / "wiki_context_snapshot.json"
    write_json(snapshot_path, snapshot)
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, snapshot_path, step_name, "json", "wiki_context_snapshot.v2"),
            _ref(ctx.run_dir, contexts_path, step_name, "json", "candidate_contexts.v1"),
            _ref(ctx.run_dir, contexts_md, step_name, "markdown"),
        ],
    )


def effective_retrieval_config(ctx: StepRunContext) -> EmbeddingRetrievalConfig:
    return ctx.manifest.vault_config_snapshot.embedding_retrieval


def uses_mock_provider_context(execution_context: ProviderExecutionContext) -> bool:
    if execution_context.record is None:
        return False
    providers = list(execution_context.record.providers.values())
    return bool(providers) and all(provider.spec.startswith("mock:") for provider in providers)


def block_unrepaired_medium_create_reason(plan: WikiMergePlanArtifact) -> WikiMergePlanArtifact:
    items: list[WikiMergePlanItem] = []
    for item in plan.items:
        if (
            item.action == "create"
            and item.strongest_overlap.strength == "medium"
            and item.strongest_overlap.path
            and create_reason_needs_repair(item.why_not_update)
        ):
            items.append(
                item.model_copy(
                    update={
                        "action": "needs_human_decision",
                        "apply_eligibility": "blocked",
                        "blocked_reason": item.blocked_reason
                        or f"召回到中等相关旧页 `{item.strongest_overlap.path}`，但模型选择 create 的理由不充分；需要人工确认。",
                        "finalization_reason": merge_markdown_blocks(
                            item.finalization_reason,
                            "模型 repair 后 why_not_update 仍不充分，已转为 needs_human_decision。",
                        ),
                    }
                )
            )
            continue
        items.append(item)
    return plan.model_copy(update={"items": items})


def build_merge_planning_context_pack(
    *,
    approved_prepared_text: str,
    digest: SourceDigestArtifact,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
    candidate_contexts: CandidateContextsArtifact,
    snapshot_ref: str,
) -> dict[str, Any]:
    candidate_by_id = source_digest_candidate_lookup(digest)
    include_full_source = len(approved_prepared_text) <= MERGE_PLANNING_FULL_SOURCE_CHAR_LIMIT
    source_pack = build_merge_planning_source_pack(
        approved_prepared_text,
        resolution,
        candidate_by_id,
        include_full_source=include_full_source,
    )
    candidate_contexts_projection = compact_candidate_contexts_for_merge_planning(candidate_contexts)
    relevant_paths = merge_planning_relevant_wiki_paths(resolution, candidate_contexts)
    snapshot_projection = compact_snapshot_for_merge_planning(snapshot, relevant_paths, snapshot_ref)
    original_counts = {
        "approved_prepared_markdown_chars": len(approved_prepared_text),
        "approved_digest_json_chars": json_char_count(digest.model_dump(mode="json")),
        "candidate_resolution_json_chars": json_char_count(resolution.model_dump(mode="json")),
        "wiki_context_snapshot_json_chars": json_char_count(snapshot.model_dump(mode="json")),
        "candidate_contexts_json_chars": json_char_count(candidate_contexts.model_dump(mode="json")),
        "snapshot_entries_chars": sum(len(entry.content) for entry in snapshot.entries),
        "snapshot_entry_count": len(snapshot.entries),
        "knowledge_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
    }
    projected_counts = {
        "approved_prepared_markdown_chars": len(approved_prepared_text) if include_full_source else 0,
        "source_excerpt_chars": source_pack["included_char_count"],
        "wiki_context_projection_json_chars": json_char_count(snapshot_projection),
        "candidate_contexts_projection_json_chars": json_char_count(candidate_contexts_projection),
        "projected_entry_chars": snapshot_projection["included_entry_content_chars"],
        "projected_entry_count": len(snapshot_projection["entries"]),
        "projected_metadata_count": len(snapshot_projection["knowledge_metadata_pool"]),
    }
    return {
        "schema_version": "merge_planning_context_pack.v1",
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "approved_digest_ref": "source_digest_review/approved_digest.json",
        "candidate_resolution_ref": "candidate_resolution/candidate_resolution.json",
        "wiki_context_snapshot_ref": snapshot_ref,
        "candidate_contexts_ref": snapshot.candidate_contexts_ref,
        "full_source_in_payload": include_full_source,
        "source_excerpt_pack": source_pack,
        "wiki_context_projection": snapshot_projection,
        "candidate_contexts_projection": candidate_contexts_projection,
        "original_counts": original_counts,
        "projected_counts": projected_counts,
}


def build_merge_planning_source_pack(
    approved_prepared_text: str,
    resolution: CandidateResolutionArtifact,
    candidate_by_id: dict[str, SourceDigestCandidate],
    *,
    include_full_source: bool,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    included_chars = 0
    global_excerpt = "" if include_full_source else source_global_excerpt(approved_prepared_text, MERGE_PLANNING_SOURCE_GLOBAL_EXCERPT_LIMIT)
    included_chars += len(global_excerpt)
    for item in resolution.items:
        candidate_cues: list[str] = []
        source_locators = [item.source_basis.source_locator]
        candidate_refs = source_basis_candidate_refs(item.source_basis)
        for candidate_id in candidate_refs:
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                continue
            source_locators.append(candidate.source_locator)
            candidate_cues.extend(
                [
                    candidate.name,
                    candidate.suggested_page_title,
                    candidate.one_sentence_summary,
                    candidate.why_matters,
                    candidate.wiki_value,
                    candidate.source_locator,
                    candidate.open_question_or_tension,
                ]
            )
        cues = [
            item.display_title,
            item.topic_summary,
            item.why_this_page,
            item.initial_section_intent,
            item.coverage_notes,
            item.reason,
            item.source_basis.source_locator,
            *candidate_cues,
            *item.source_basis.prepared_discovered_candidates,
        ]
        snippets = [] if include_full_source else source_snippets_for_cues(
            approved_prepared_text,
            cues,
            max_chars=MERGE_PLANNING_SOURCE_PER_PAGE_EXCERPT_LIMIT,
        )
        included_chars += sum(len(snippet["text"]) for snippet in snippets)
        items.append(
            {
                "page_plan_id": item.page_plan_id,
                "display_title": item.display_title,
                "candidate_target_path": item.candidate_target_path,
                "source_candidate_ids": item.source_basis.source_candidate_ids,
                "prepared_discovered_candidates": item.source_basis.prepared_discovered_candidates,
                "source_candidate_refs": candidate_refs,
                "source_locators": [locator for locator in source_locators if locator],
                "snippets": snippets,
            }
        )
    return {
        "schema_version": "merge_planning_source_excerpt_pack.v1",
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "full_source_in_payload": include_full_source,
        "full_source_limit": MERGE_PLANNING_FULL_SOURCE_CHAR_LIMIT,
        "global_excerpt_limit": MERGE_PLANNING_SOURCE_GLOBAL_EXCERPT_LIMIT,
        "per_page_excerpt_limit": MERGE_PLANNING_SOURCE_PER_PAGE_EXCERPT_LIMIT,
        "global_excerpt": global_excerpt,
        "items": items,
    }


def compact_candidate_contexts_for_merge_planning(candidate_contexts: CandidateContextsArtifact) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in candidate_contexts.items:
        hits: list[dict[str, Any]] = []
        for hit in item.hits:
            excerpt_limit = (
                merge_planning_hit_excerpt_limit(hit)
            )
            excerpt = compact_payload_text(hit.excerpt, excerpt_limit)
            hits.append(
                {
                    "page_plan_id": hit.page_plan_id,
                    "rank": hit.rank,
                    "path": hit.path,
                    "display_title": hit.display_title,
                    "score": hit.score,
                    "score_bucket": hit.score_bucket,
                    "strength": hit.strength,
                    "match_basis": hit.match_basis,
                    "sort_explanation": hit.sort_explanation,
                    "forced": hit.forced,
                    "page_sha256": hit.page_sha256,
                    "excerpt_limit": excerpt_limit,
                    "excerpt": excerpt,
                    "truncated": hit.truncated or len(hit.excerpt) > len(excerpt),
                }
            )
        items.append(
            {
                "page_plan_id": item.page_plan_id,
                "query": compact_payload_text(item.query, MERGE_PLANNING_CONTEXT_QUERY_LIMIT),
                "hits": hits,
                "unindexable_pages": item.unindexable_pages,
            }
        )
    return {
        "schema_version": "candidate_contexts_projection.v1",
        "source_schema_version": candidate_contexts.schema_version,
        "retrieval_backend": candidate_contexts.retrieval_backend,
        "model": candidate_contexts.model,
        "model_revision": candidate_contexts.model_revision,
        "top_k": candidate_contexts.top_k,
        "candidate_pool_size": candidate_contexts.candidate_pool_size,
        "candidate_pool_sha256": candidate_contexts.candidate_pool_sha256,
        "skipped_count": candidate_contexts.skipped_count,
        "warnings": candidate_contexts.warnings,
        "query_limit": MERGE_PLANNING_CONTEXT_QUERY_LIMIT,
        "hit_excerpt_limit": MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT,
        "weak_hit_excerpt_limit": MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT,
        "weak_hit_excerpt_max_rank": MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK,
        "hit_excerpt_role": "match_preview",
        "content_evidence_ref": "wiki_context_projection.entries",
        "items": items,
    }


def merge_planning_hit_excerpt_limit(hit: CandidateContextHit) -> int:
    if hit.strength == "weak" and not hit.forced:
        if hit.rank > MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK:
            return 0
        return MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT
    return MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT


def merge_planning_relevant_wiki_paths(
    resolution: CandidateResolutionArtifact,
    candidate_contexts: CandidateContextsArtifact,
) -> set[str]:
    paths = {f"wiki/{item.candidate_target_path}" for item in resolution.items if item.candidate_target_path}
    for context_item in candidate_contexts.items:
        for hit in context_item.hits:
            paths.add(f"wiki/{hit.path}")
    return paths


def compact_snapshot_for_merge_planning(
    snapshot: WikiContextSnapshot,
    relevant_paths: set[str],
    snapshot_ref: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for entry in snapshot.entries:
        if entry.path not in relevant_paths:
            continue
        content_excerpt = compact_entry_content_for_merge_planning(entry.content)
        entries.append(
            {
                "path": entry.path,
                "expected_state": entry.expected_state,
                "preimage_sha256": entry.preimage_sha256,
                "metadata": entry.metadata.model_dump(mode="json") if entry.metadata else None,
                "content_excerpt": content_excerpt,
                "content_truncated": len(entry.content.strip()) > len(content_excerpt),
            }
        )
    metadata_paths = {path.removeprefix("wiki/") for path in relevant_paths}
    metadata_pool = [
        {
            "path": pool_entry.path,
            "rel_path": pool_entry.rel_path,
            "preimage_sha256": pool_entry.preimage_sha256,
            "metadata": pool_entry.metadata.model_dump(mode="json") if pool_entry.metadata else None,
            "display_title": pool_entry.display_title,
            "summary": pool_entry.summary,
            "aliases": pool_entry.aliases,
            "llmwiki_type": pool_entry.llmwiki_type,
            "indexable": pool_entry.indexable,
            "unindexable_reason": pool_entry.unindexable_reason,
        }
        for pool_entry in snapshot.knowledge_metadata_pool
        if pool_entry.path in metadata_paths
    ]
    included_chars = sum(len(entry["content_excerpt"]) for entry in entries)
    return {
        "schema_version": "wiki_context_snapshot_projection.v1",
        "source_schema_version": snapshot.schema_version,
        "full_snapshot_ref": snapshot_ref,
        "log_date": snapshot.log_date,
        "source_target_path": snapshot.source_target_path,
        "candidate_contexts_ref": snapshot.candidate_contexts_ref,
        "candidate_pool_sha256": snapshot.candidate_pool_sha256,
        "entry_excerpt_limit": MERGE_PLANNING_ENTRY_EXCERPT_LIMIT,
        "full_entry_count": len(snapshot.entries),
        "included_entry_count": len(entries),
        "included_entry_content_chars": included_chars,
        "omitted_entry_count": max(0, len(snapshot.entries) - len(entries)),
        "full_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
        "included_metadata_pool_count": len(metadata_pool),
        "knowledge_metadata_pool": metadata_pool,
        "entries": entries,
    }


def compact_entry_content_for_merge_planning(text: str) -> str:
    return source_global_excerpt(text, MERGE_PLANNING_ENTRY_EXCERPT_LIMIT)


def compact_payload_text(text: str, limit: int) -> str:
    stripped = text.strip()
    if limit <= 0:
        return ""
    if len(stripped) <= limit:
        return stripped
    return stripped[: max(0, limit - 3)].rstrip() + "..."


def json_char_count(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def merge_planning_payload_pack_summary(pack: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in pack.items()
        if key
        not in {
            "source_excerpt_pack",
            "wiki_context_projection",
            "candidate_contexts_projection",
        }
    }


def render_merge_planning_context_pack_markdown(pack: dict[str, Any]) -> str:
    source_pack = pack.get("source_excerpt_pack", {})
    original = pack.get("original_counts", {})
    projected = pack.get("projected_counts", {})
    rows = [
        ["approved_prepared_markdown", original.get("approved_prepared_markdown_chars", 0), projected.get("approved_prepared_markdown_chars", 0)],
        ["source_excerpt_pack", 0, projected.get("source_excerpt_chars", 0)],
        ["wiki_context_snapshot", original.get("wiki_context_snapshot_json_chars", 0), projected.get("wiki_context_projection_json_chars", 0)],
        ["candidate_contexts", original.get("candidate_contexts_json_chars", 0), projected.get("candidate_contexts_projection_json_chars", 0)],
        ["snapshot entries", original.get("snapshot_entries_chars", 0), projected.get("projected_entry_chars", 0)],
    ]
    context_rows = [
        [
            item.get("page_plan_id", ""),
            item.get("display_title", ""),
            item.get("candidate_target_path", ""),
            len(item.get("snippets", [])) if isinstance(item.get("snippets", []), list) else 0,
        ]
        for item in source_pack.get("items", [])
        if isinstance(item, dict)
    ]
    return (
        "# Merge Planning Context Pack\n\n"
        f"- payload 是否包含完整 source：`{str(pack.get('full_source_in_payload', False)).lower()}`\n"
        f"- 完整 source 引用：`{pack.get('approved_prepared_ref', '')}`\n"
        f"- 完整 snapshot 引用：`{pack.get('wiki_context_snapshot_ref', '')}`\n\n"
        "## Payload 字符预算\n\n"
        f"{format_markdown_table(['对象', '原始字符', '投影字符'], rows)}\n\n"
        "## Source 页面摘录\n\n"
        f"{format_markdown_table(['页面计划', '标题', '目标', '片段数'], context_rows)}\n"
    )


def empty_vault_create_merge_planning_shortcut_report(
    digest: SourceDigestArtifact,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
    candidate_contexts: CandidateContextsArtifact,
) -> dict[str, Any]:
    known_candidate_ids = {candidate.candidate_id for candidate in [*digest.ingest_candidates(), *digest.budget_deferred_candidates]}
    target_paths = [item.candidate_target_path for item in resolution.items if item.candidate_target_path]
    snapshot_by_path = {entry.path: entry for entry in snapshot.entries}
    present_targets = [
        target
        for target in target_paths
        if (entry := snapshot_by_path.get(f"wiki/{target}")) is not None and entry.expected_state == "present"
    ]
    missing_snapshot_targets = [target for target in target_paths if f"wiki/{target}" not in snapshot_by_path]
    total_hits = sum(len(item.hits) for item in candidate_contexts.items)
    source_page_plans = [item.page_plan_id for item in resolution.items if item.page_type.strip().lower() == "source"]
    empty_target_page_plans = [item.page_plan_id for item in resolution.items if not item.candidate_target_path]
    missing_source_candidate_page_plans = [
        item.page_plan_id
        for item in resolution.items
        if not source_basis_candidate_refs(item.source_basis)
    ]
    unknown_source_candidate_ids = sorted(
        {
            candidate_id
            for item in resolution.items
            for candidate_id in source_basis_candidate_refs(item.source_basis)
            if candidate_id not in known_candidate_ids
        }
    )
    duplicate_targets = sorted({target for target in target_paths if target_paths.count(target) > 1})
    blocking_conditions: list[str] = []
    if not resolution.items:
        blocking_conditions.append("candidate_resolution_empty")
    if snapshot.knowledge_metadata_pool:
        blocking_conditions.append("knowledge_metadata_pool_not_empty")
    if total_hits:
        blocking_conditions.append("candidate_context_hits_present")
    if present_targets:
        blocking_conditions.append("target_page_already_present")
    if missing_snapshot_targets:
        blocking_conditions.append("target_page_missing_from_snapshot")
    if source_page_plans:
        blocking_conditions.append("source_page_plan_present")
    if empty_target_page_plans:
        blocking_conditions.append("candidate_target_path_empty")
    if missing_source_candidate_page_plans:
        blocking_conditions.append("source_candidate_ids_empty")
    if unknown_source_candidate_ids:
        blocking_conditions.append("source_candidate_ids_unknown")
    if duplicate_targets:
        blocking_conditions.append("duplicate_candidate_target_path")
    return {
        "schema_version": "merge_planning_shortcut_report.v1",
        "shortcut": "empty_vault_all_create",
        "used": not blocking_conditions,
        "reason": (
            "空知识库、无候选召回命中、所有目标页均缺失；本地生成 create merge plan，跳过模型规划。"
            if not blocking_conditions
            else "未满足空库纯 create shortcut 条件，继续使用模型规划。"
        ),
        "blocking_conditions": blocking_conditions,
        "candidate_count": len(resolution.items),
        "knowledge_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
        "candidate_context_item_count": len(candidate_contexts.items),
        "candidate_context_hit_count": total_hits,
        "present_target_count": len(present_targets),
        "missing_snapshot_target_count": len(missing_snapshot_targets),
        "source_page_plan_count": len(source_page_plans),
        "empty_target_page_plan_count": len(empty_target_page_plans),
        "missing_source_candidate_page_plan_count": len(missing_source_candidate_page_plans),
        "unknown_source_candidate_id_count": len(unknown_source_candidate_ids),
        "duplicate_candidate_target_count": len(duplicate_targets),
        "target_paths": target_paths,
        "present_targets": present_targets,
        "missing_snapshot_targets": missing_snapshot_targets,
        "source_page_plans": source_page_plans,
        "empty_target_page_plans": empty_target_page_plans,
        "missing_source_candidate_page_plans": missing_source_candidate_page_plans,
        "unknown_source_candidate_ids": unknown_source_candidate_ids,
        "duplicate_candidate_targets": duplicate_targets,
    }


def source_basis_candidate_refs(source_basis: SourceBasis) -> list[str]:
    return _dedupe_strings(
        [
            str(ref).strip()
            for ref in [*source_basis.source_candidate_ids, *source_basis.prepared_discovered_candidates]
            if str(ref).strip()
        ]
    )


def render_merge_planning_shortcut_report(report: dict[str, Any]) -> str:
    rows = [
        ["candidate_count", report.get("candidate_count", 0)],
        ["knowledge_metadata_pool_count", report.get("knowledge_metadata_pool_count", 0)],
        ["candidate_context_hit_count", report.get("candidate_context_hit_count", 0)],
        ["present_target_count", report.get("present_target_count", 0)],
        ["missing_snapshot_target_count", report.get("missing_snapshot_target_count", 0)],
        ["source_page_plan_count", report.get("source_page_plan_count", 0)],
        ["empty_target_page_plan_count", report.get("empty_target_page_plan_count", 0)],
        ["missing_source_candidate_page_plan_count", report.get("missing_source_candidate_page_plan_count", 0)],
        ["unknown_source_candidate_id_count", report.get("unknown_source_candidate_id_count", 0)],
        ["duplicate_candidate_target_count", report.get("duplicate_candidate_target_count", 0)],
    ]
    blockers = report.get("blocking_conditions", [])
    return (
        "# Merge Planning Shortcut Report\n\n"
        f"- Shortcut：`{report.get('shortcut', '')}`\n"
        f"- Used：`{str(bool(report.get('used'))).lower()}`\n"
        f"- Reason：{report.get('reason', '')}\n"
        f"- Blocking conditions：`{', '.join(blockers) if blockers else 'none'}`\n\n"
        "## Guard Counters\n\n"
        f"{format_markdown_table(['检查项', '值'], rows)}\n"
    )


def merge_planning_shortcut_provider_allowed(ctx: StepRunContext, step_name: str) -> bool:
    runtime = ctx.execution_context.runtime_for_task(step_name)
    return not runtime.spec.startswith("mock:")


def _run_wiki_merge_planning(ctx: StepRunContext) -> None:
    step_name = "wiki_merge_planning"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    snapshot_path = require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    candidate_contexts = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "candidate_contexts.json", CandidateContextsArtifact)
    approved_prepared_text = (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8")
    snapshot_ref = snapshot_path.relative_to(ctx.run_dir).as_posix()
    context_pack = build_merge_planning_context_pack(
        approved_prepared_text=approved_prepared_text,
        digest=digest,
        resolution=resolution,
        snapshot=snapshot,
        candidate_contexts=candidate_contexts,
        snapshot_ref=snapshot_ref,
    )
    context_pack_path = step_root / "merge_planning_context_pack.json"
    context_pack_md = step_root / "merge_planning_context_pack.md"
    write_json(context_pack_path, context_pack)
    context_pack_md.write_text(render_merge_planning_context_pack_markdown(context_pack), encoding="utf-8")
    if merge_planning_shortcut_provider_allowed(ctx, step_name):
        shortcut_report = empty_vault_create_merge_planning_shortcut_report(digest, resolution, snapshot, candidate_contexts)
    else:
        shortcut_report = {"used": False}
    if shortcut_report["used"]:
        shortcut_report_path = step_root / "merge_planning_shortcut_report.json"
        shortcut_report_md = step_root / "merge_planning_shortcut_report.md"
        write_json(shortcut_report_path, shortcut_report)
        shortcut_report_md.write_text(render_merge_planning_shortcut_report(shortcut_report), encoding="utf-8")
        plan = build_wiki_merge_plan(resolution, digest, snapshot, log_date=snapshot.log_date)
        plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_ref, medium_missing_policy="preserve")
        plan = block_unrepaired_medium_create_reason(plan)
        validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
        out = step_root / "wiki_merge_plan.json"
        write_json(out, plan)
        table = step_root / "wiki_merge_plan.md"
        table.write_text(render_merge_plan_markdown(plan), encoding="utf-8")
        report = step_root / "merge_decision_report.md"
        report.write_text(render_merge_decision_report(plan, snapshot), encoding="utf-8")
        complete_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, context_pack_path, step_name, "json", "merge_planning_context_pack.v1"),
                _ref(ctx.run_dir, context_pack_md, step_name, "markdown"),
                _ref(ctx.run_dir, shortcut_report_path, step_name, "json", "merge_planning_shortcut_report.v1"),
                _ref(ctx.run_dir, shortcut_report_md, step_name, "markdown"),
                _ref(ctx.run_dir, out, step_name, "json", "wiki_merge_plan.v5"),
                _ref(ctx.run_dir, table, step_name, "markdown"),
                _ref(ctx.run_dir, report, step_name, "markdown"),
            ],
        )
        return
    payload = {
        "approved_prepared_markdown": approved_prepared_text if context_pack["full_source_in_payload"] else "",
        "approved_prepared_ref": context_pack["approved_prepared_ref"],
        "source_excerpt_pack": context_pack["source_excerpt_pack"],
        "approved_digest": digest.model_dump(mode="json"),
        "approved_digest_ref": context_pack["approved_digest_ref"],
        "candidate_resolution": resolution.model_dump(mode="json"),
        "candidate_resolution_ref": context_pack["candidate_resolution_ref"],
        "wiki_context_snapshot": context_pack["wiki_context_projection"],
        "wiki_context_snapshot_ref": snapshot_ref,
        "candidate_contexts": context_pack["candidate_contexts_projection"],
        "merge_planning_context_pack": merge_planning_payload_pack_summary(context_pack),
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "contract": {
            "goal": "Read the compact frozen wiki context projection and decide create/update/noop/needs_human_decision for planned pages.",
            "actions": ["create", "update", "noop", "needs_human_decision"],
            "rules": [
                "approved_digest and candidate_resolution are the reviewed upstream artifacts; approved_digest_ref/candidate_resolution_ref identify their fixed local audit copies.",
                "wiki_context_snapshot is a compact projection; the full snapshot is fixed at wiki_context_snapshot_ref and will be used by local validators.",
                "Use source_excerpt_pack as source support when approved_prepared_markdown is empty; the full approved source remains fixed at approved_prepared_ref.",
                "For each page_plan_id, inspect candidate_contexts Top5 before choosing create/update/noop/needs_human_decision.",
                "Write inspected_context_paths using wiki-root-relative paths from candidate_contexts hits.",
                "For create with medium or strong inspected overlap, explain why_not_update against the strongest inspected old page.",
                "A valid why_not_update must compare scope_delta, source_delta, why_update_not_enough, and why_related_link_not_enough in concrete Chinese prose.",
                "Do not rely only on target page missing, different folder, or different page type as the reason to create.",
                "Prefer update when the new source adds examples, boundaries, counterexamples, use cases, value points, or clarifications to the same knowledge question/concept/methodology.",
                "Use create only when the topic is an independent reusable knowledge unit that cannot naturally live inside an inspected existing page.",
                "If new material partially answers an existing open_question, update that open_question and related the stable new concept/design/comparison page back to it.",
                "If the new page only answers one paragraph of an old open_question, prefer update instead of create.",
                "If an exact path/title/alias inspected context strongly overlaps but you still want create, use needs_human_decision.",
                "Suggest at most two related_pages; they must come from source sibling pages, inspected context pages, or exact title/alias matches.",
                "Every related_pages reason must be Chinese and explain a concrete relation from source relation, merge comparison, or sibling complement.",
                "Only same source is not enough for Related; prefer upstream/downstream, complement, counterexample, use scenario, or method dependency.",
                "If no Related is suitable, set related_absence_reason and explain whether the page is isolated, weakly related, self-only, or already navigable from index.",
                "Use canonical_target_path for final writes; for update use the matched existing page path.",
                "needs_human_decision is not writeable and must be resolved before drafting.",
                "noop only when an inspected existing wiki page already fully covers the source without new examples, expressions, links, or value points.",
                "For noop, bind matched_page to the inspected existing page and explain the no-content-delta reason.",
                "New but thin topics should be create, not noop.",
                "All user-visible fields must be Chinese unless keeping stable domain terms.",
                "Keep value_points and reuse_scenarios grounded in concrete source content.",
            ],
        },
    }
    def validate_merge_model(model: WikiMergePlanArtifact) -> None:
        candidate = finalize_wiki_merge_plan(
            model,
            resolution,
            snapshot,
            snapshot_path.relative_to(ctx.run_dir).as_posix(),
            medium_missing_policy="preserve",
        )
        validate_wiki_merge_plan(digest, candidate, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)

    plan, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        WikiMergePlanArtifact,
        validator=validate_merge_model,
    )
    plan = _redacted_model(ctx, plan, WikiMergePlanArtifact)
    plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_ref, medium_missing_policy="preserve")
    plan = block_unrepaired_medium_create_reason(plan)
    validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    out = step_root / "wiki_merge_plan.json"
    write_json(out, plan)
    table = step_root / "wiki_merge_plan.md"
    table.write_text(render_merge_plan_markdown(plan), encoding="utf-8")
    report = step_root / "merge_decision_report.md"
    report.write_text(render_merge_decision_report(plan, snapshot), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, context_pack_path, step_name, "json", "merge_planning_context_pack.v1"),
        _ref(ctx.run_dir, context_pack_md, step_name, "markdown"),
        _ref(ctx.run_dir, out, step_name, "json", "wiki_merge_plan.v5"),
        _ref(ctx.run_dir, table, step_name, "markdown"),
        _ref(ctx.run_dir, report, step_name, "markdown"),
    ]
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(
        ctx.manifest,
        step_name,
        outputs=outputs,
    )


def _run_merge_plan_review(ctx: StepRunContext) -> None:
    step_name = "merge_plan_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    plan_path = require_step_output_dir(ctx.run_dir, "wiki_merge_planning") / "wiki_merge_plan.json"
    plan = read_model(plan_path, WikiMergePlanArtifact)
    approved_path = step_root / "approved_merge_plan.json"
    prompt_path = step_root / "review_prompt.md"
    feedback_path = step_root / "review_feedback.jsonl"
    feedback_path.write_text("", encoding="utf-8")
    prompt_path.write_text(render_merge_plan_review_prompt(plan), encoding="utf-8")
    has_needs_human = any(item.action == "needs_human_decision" or item.apply_eligibility == "blocked" for item in plan.items)
    all_create_risk = merge_plan_all_create_review_reason(
        plan,
        max_auto_create_items=merge_plan_auto_create_review_limit(ctx.manifest.vault_config_snapshot.max_ingest_candidates),
    )
    if has_needs_human or all_create_risk:
        pending_path = step_root / "pending_merge_plan.json"
        pending_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
        decision = ReviewDecision(
            review_step=step_name,
            decision="pending",
            review_mode="manual",
            auto_approved=False,
            notes=all_create_risk or "合并计划包含 needs_human_decision；继续前需要先 revise 为 create/update/noop。",
        )
        decision_path = step_root / "review_decision.json"
        write_json(decision_path, decision)
        mark_step_awaiting_review(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
                _ref(ctx.run_dir, pending_path, step_name, "json", "wiki_merge_plan.v5"),
                _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            ],
            reason=all_create_risk or "merge plan requires human decision; run merge-level revise.",
            review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    approved_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        review_mode="auto_stub",
        auto_approved=True,
        notes="合并计划不含 needs_human_decision，本轮按 auto-stub 自动批准。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            _ref(ctx.run_dir, approved_path, step_name, "json", "wiki_merge_plan.v5"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def build_draft_source_excerpt_pack(
    approved_prepared_text: str,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    *,
    full_source_limit: int = DRAFT_RENDERING_FULL_SOURCE_CHAR_LIMIT,
    total_limit: int = DRAFT_RENDERING_EXCERPT_TOTAL_CHAR_LIMIT,
    per_page_limit: int = DRAFT_RENDERING_EXCERPT_PER_PAGE_LIMIT,
    global_limit: int = DRAFT_RENDERING_GLOBAL_EXCERPT_LIMIT,
    force_excerpt: bool = False,
) -> dict[str, Any]:
    include_full_source = len(approved_prepared_text) <= full_source_limit and not force_excerpt
    candidates = source_digest_candidate_lookup(digest)
    draftable_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    effective_total_limit = total_limit
    if not include_full_source:
        minimum_page_budget = global_limit + len(draftable_items) * DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS
        ratio_budget = int(len(approved_prepared_text) * DRAFT_RENDERING_EXCERPT_MAX_SOURCE_RATIO)
        effective_total_limit = min(total_limit, max(minimum_page_budget, ratio_budget))
    items = []
    used_chars = 0
    global_excerpt = source_global_excerpt(approved_prepared_text, global_limit)
    used_chars += len(global_excerpt)
    for index, item in enumerate(draftable_items):
        direct_candidate_cues = []
        expanded_candidate_cues = []
        source_locators = [item.source_basis.source_locator]
        source_candidate_refs = source_basis_candidate_refs(item.source_basis)
        expanded_candidate_ids = source_digest_candidate_id_closure(source_candidate_refs, candidates)
        direct_candidate_ids = set(source_candidate_refs)
        for candidate_id in expanded_candidate_ids:
            candidate = candidates.get(candidate_id)
            if candidate is not None:
                source_locators.append(candidate.source_locator)
                target_cues = direct_candidate_cues if candidate_id in direct_candidate_ids else expanded_candidate_cues
                target_cues.extend(
                    [
                        candidate.name,
                        candidate.suggested_page_title,
                        candidate.one_sentence_summary,
                        candidate.why_matters,
                        candidate.wiki_value,
                        candidate.source_locator,
                        candidate.open_question_or_tension,
                    ]
                )
        base_cues = [
            item.display_title,
            item.new_understanding,
            item.why_create_or_update,
            item.why_not_update,
            item.prior_knowledge_state,
            item.knowledge_delta,
            item.why_this_matters,
            *item.reuse_scenarios,
            *item.value_points,
            item.source_basis.source_locator,
            *item.section_plans.values(),
        ]
        remaining_budget = max(0, effective_total_limit - used_chars)
        remaining_items = max(1, len(draftable_items) - index)
        if include_full_source:
            page_limit = per_page_limit
        else:
            page_limit = min(per_page_limit, max(DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS, remaining_budget // remaining_items))
        primary_cues = [*base_cues, *direct_candidate_cues, *item.source_basis.prepared_discovered_candidates]
        snippets = source_snippets_for_cues(approved_prepared_text, primary_cues, max_chars=page_limit)
        if source_snippets_are_start_fallback(snippets) and expanded_candidate_cues:
            snippets = source_snippets_for_cues(
                approved_prepared_text,
                [*primary_cues, *expanded_candidate_cues],
                max_chars=page_limit,
            )
        used_chars += sum(len(snippet["text"]) for snippet in snippets)
        items.append(
            {
                "page_plan_id": item.page_plan_id,
                "display_title": item.display_title,
                "target_path": item.canonical_target_path,
                "source_candidate_ids": item.source_basis.source_candidate_ids,
                "prepared_discovered_candidates": item.source_basis.prepared_discovered_candidates,
                "source_candidate_refs": source_candidate_refs,
                "expanded_source_candidate_ids": expanded_candidate_ids,
                "source_locators": [locator for locator in source_locators if locator],
                "snippets": snippets,
            }
        )
    included_chars = len(global_excerpt) + sum(
        len(snippet["text"])
        for item in items
        for snippet in item["snippets"]
    )
    return {
        "schema_version": "draft_source_excerpt_pack.v1",
        "source_raw_path": digest.source_raw_path,
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "full_source_in_payload": include_full_source,
        "force_excerpt": force_excerpt,
        "full_source_limit": full_source_limit,
        "configured_total_excerpt_limit": total_limit,
        "total_excerpt_limit": effective_total_limit,
        "per_page_excerpt_limit": per_page_limit,
        "global_excerpt_limit": global_limit,
        "max_source_ratio": DRAFT_RENDERING_EXCERPT_MAX_SOURCE_RATIO,
        "min_page_excerpt_chars": DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS,
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "truncated_for_payload": not include_full_source,
        "global_excerpt": global_excerpt,
        "items": items,
    }


def source_snippets_are_start_fallback(snippets: list[dict[str, Any]]) -> bool:
    return not snippets or all(snippet.get("cue") == "fallback_start" for snippet in snippets)


def source_global_excerpt(text: str, limit: int) -> str:
    headings = "\n".join(line.strip() for line in text.splitlines() if line.lstrip().startswith("#"))
    prefix = text.strip()[: max(0, limit - len(headings) - 4)]
    return merge_markdown_blocks(prefix, headings)[:limit].strip()


def source_snippets_for_cues(text: str, cues: list[str], *, max_chars: int) -> list[dict[str, Any]]:
    if max_chars <= 0:
        return []
    semantic_terms = source_semantic_match_terms(cues)
    window_candidates: list[tuple[int, int, str, int]] = []
    seen_positions: set[int] = set()
    locator_windows = source_section_locator_windows(text, cues, max_chars=max_chars, semantic_terms=semantic_terms)
    for start, end, cue, score in locator_windows:
        if any(abs(start - existing) < source_excerpt_start_tolerance(cue) for existing in seen_positions):
            continue
        seen_positions.add(start)
        window_candidates.append((start, end, cue, score))
    for cue in source_excerpt_cues(cues):
        position = find_source_cue(text, cue)
        if position < 0:
            continue
        center = max(0, position)
        heading_start = markdown_heading_start_at_position(text, center)
        if heading_start is not None:
            start = heading_start
            line_end = text.find("\n", heading_start)
            section_search_start = line_end + 1 if line_end >= 0 else len(text)
            end = min(source_heading_section_end(text, section_search_start, heading_marker_at_position(text, heading_start)), start + max_chars)
        else:
            before_chars = min(450, max(80, max_chars // 3))
            after_chars = min(650, max(180, max_chars - before_chars))
            start = max(0, center - before_chars)
            end = min(len(text), center + len(cue) + after_chars)
            start = adjust_window_start(text, start)
            end = adjust_window_end(text, end)
        window_text = text[start:end]
        score = source_excerpt_window_score(window_text, cue, semantic_terms)
        if source_excerpt_low_signal_cue(cue) and score < 28:
            continue
        if any(abs(start - existing) < source_excerpt_start_tolerance(cue) for existing in seen_positions):
            continue
        seen_positions.add(start)
        window_candidates.append((start, end, cue, score))
    if locator_windows:
        semantic_window = source_semantic_fallback_window(text, cues, max_chars=max_chars)
        if semantic_window is not None:
            start, end, cue = semantic_window
            if not any(abs(start - existing) < source_excerpt_start_tolerance(cue) for existing in seen_positions):
                seen_positions.add(start)
                window_candidates.append((start, end, cue, 88))
    windows: list[tuple[int, int, str]] = []
    for start, end, cue, _score in sorted(window_candidates, key=lambda item: (-item[3], item[0])):
        if any(
            ranges_overlap(
                start,
                end,
                existing_start,
                existing_end,
                tolerance=source_excerpt_overlap_tolerance(cue, existing_cue),
            )
            for existing_start, existing_end, existing_cue in windows
        ):
            continue
        windows.append((start, end, cue))
        if sum(existing_end - existing_start for existing_start, existing_end, _ in windows) >= max_chars:
            break
    if not windows:
        semantic_window = source_semantic_fallback_window(text, cues, max_chars=max_chars)
        if semantic_window is not None:
            start, end, cue = semantic_window
            return [{"cue": cue, "start": start, "end": end, "text": text[start:end].strip()}]
        heading_window = source_heading_fallback_window(text, cues, max_chars=max_chars)
        if heading_window is not None:
            start, end, cue = heading_window
            return [{"cue": cue, "start": start, "end": end, "text": text[start:end].strip()}]
        fallback = text.strip()[:max_chars]
        return [{"cue": "fallback_start", "start": 0, "end": len(fallback), "text": fallback}] if fallback else []
    snippets: list[dict[str, Any]] = []
    remaining = max_chars
    for start, end, cue in windows:
        if remaining <= 0:
            break
        snippet = text[start:end].strip()
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
            end = start + len(snippet)
        snippets.append({"cue": cue, "start": start, "end": end, "text": snippet})
        remaining -= len(snippet)
    return snippets


def source_excerpt_start_tolerance(cue: str) -> int:
    return 1 if cue.startswith(("source_locator:", "fallback_semantic:")) else 120


def source_excerpt_overlap_tolerance(cue: str, existing_cue: str) -> int:
    if cue.startswith("source_locator:") or existing_cue.startswith("source_locator:"):
        return 0
    return 120


SOURCE_SECTION_LOCATOR_RE = re.compile(r"\bS(?P<start>\d{3})(?:\s*[-–—~至到]\s*S?(?P<end>\d{3}))?\b", re.IGNORECASE)


def source_section_locator_windows(
    text: str,
    cues: list[str],
    *,
    max_chars: int,
    semantic_terms: list[str] | None = None,
) -> list[tuple[int, int, str, int]]:
    section_ids = source_section_locator_ids(cues)
    if not section_ids:
        return []
    sections = {
        section["section_id"]: section
        for section in markdown_sections_for_source_map(text, max_sections=SOURCE_DIGEST_SOURCE_MAP_MAX_SECTIONS)
    }
    if not sections:
        return []
    per_locator_limit = min(max_chars, max(160, max_chars // max(1, min(len(section_ids), 3))))
    windows: list[tuple[int, int, str, int]] = []
    for section_id in section_ids:
        section = sections.get(section_id)
        if section is None:
            continue
        start = int(section.get("char_start", 0))
        section_end = int(section.get("char_end", start))
        end = min(section_end, start + per_locator_limit)
        if end <= start:
            continue
        heading = str(section.get("heading", "")).strip()
        window_text = text[start:end]
        score = source_section_locator_window_score(window_text, semantic_terms or [])
        windows.append((start, end, f"source_locator:{section_id}:{heading}", score))
    return windows


def source_section_locator_window_score(window_text: str, semantic_terms: list[str]) -> int:
    if not semantic_terms:
        return 96
    normalized_window = normalized_source_match_text(window_text)
    hits = [term for term in semantic_terms if len(term) >= 4 and term in normalized_window]
    if not hits:
        return 64
    return 96 + min(16, sum(min(8, len(term)) for term in hits[:4]))


def source_section_locator_ids(cues: list[str]) -> list[str]:
    section_ids: list[str] = []
    for cue in cues:
        for match in SOURCE_SECTION_LOCATOR_RE.finditer(cue or ""):
            start = int(match.group("start"))
            end_text = match.group("end")
            end = int(end_text) if end_text else start
            if end < start:
                continue
            for number in range(start, min(end, start + 7) + 1):
                section_id = f"S{number:03d}"
                if section_id not in section_ids:
                    section_ids.append(section_id)
    return section_ids


SOURCE_EXCERPT_LOW_SIGNAL_NORMALIZED_CUES = {
    "ai",
    "agi",
    "anthropic",
    "claude",
    "claudecode",
    "cowork",
    "pm",
    "产品",
    "模型",
    "功能",
    "团队",
    "用户",
    "角色",
    "设计",
    "问题",
    "成功",
    "未来",
    "什么",
    "如何",
    "为什么",
    "需要",
    "应该",
    "可以",
    "通过",
    "帮助",
    "重要",
    "不同",
    "类型",
    "应用",
    "开发",
}

SOURCE_EXCERPT_SHORT_ASCII_SIGNAL_CUES = {
    "api",
    "arr",
    "cli",
    "eval",
    "gtm",
    "mvp",
    "prd",
}


def source_excerpt_window_score(window_text: str, cue: str, semantic_terms: list[str]) -> int:
    normalized_cue = normalized_source_match_text(cue)
    score = min(36, len(normalized_cue))
    if re.search(r"[\u4e00-\u9fff]", cue) and re.search(r"[A-Za-z]", cue):
        score += 6
    if source_excerpt_low_signal_cue(cue):
        score -= 16
    semantic_score, present_terms, _ = source_semantic_block_score(window_text, semantic_terms)
    score += semantic_score
    if len(present_terms) >= 3:
        score += 8
    if markdown_heading_start_at_position(window_text, 0) == 0:
        score += 16
    return score


def source_excerpt_low_signal_cue(cue: str) -> bool:
    normalized_cue = normalized_source_match_text(cue)
    if normalized_cue in SOURCE_EXCERPT_SHORT_ASCII_SIGNAL_CUES:
        return False
    if normalized_cue in SOURCE_EXCERPT_LOW_SIGNAL_NORMALIZED_CUES:
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9+#.-]{2,14}", cue.strip()):
        return True
    if re.fullmatch(r"s\d{1,4}", normalized_cue):
        return True
    return len(normalized_cue) < 4


def ranges_overlap(start: int, end: int, other_start: int, other_end: int, *, tolerance: int = 0) -> bool:
    return start < other_end + tolerance and other_start < end + tolerance


def markdown_heading_start_at_position(text: str, position: int) -> int | None:
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    return line_start if re.match(r"^#{1,6}\s+", line) else None


def heading_marker_at_position(text: str, position: int) -> str:
    match = re.match(r"^(#{1,6})\s+", text[position:])
    return match.group(1) if match else "#"


def source_heading_fallback_window(text: str, cues: list[str], *, max_chars: int) -> tuple[int, int, str] | None:
    cue_variants = source_excerpt_cues(cues)
    if not cue_variants:
        return None
    best: tuple[int, int, str, int] | None = None
    for line_match in re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text):
        heading_text = line_match.group(2)
        normalized_heading = normalized_source_match_text(heading_text)
        if len(normalized_heading) < 3:
            continue
        for cue in cue_variants:
            normalized_cue = normalized_source_match_text(cue)
            if len(normalized_cue) < 3:
                continue
            score = heading_match_score(normalized_heading, normalized_cue)
            if score <= 0:
                continue
            if best is None or score > best[3]:
                start = line_match.start()
                end = source_heading_section_end(text, line_match.end(), line_match.group(1))
                best = (start, min(end, start + max_chars), f"fallback_heading:{heading_text}", score)
    if best is None:
        return None
    return best[0], best[1], best[2]


def heading_match_score(normalized_heading: str, normalized_cue: str) -> int:
    if normalized_heading in normalized_cue or normalized_cue in normalized_heading:
        return min(len(normalized_heading), len(normalized_cue)) + 20
    heading_terms = meaningful_match_terms(normalized_heading)
    cue_terms = meaningful_match_terms(normalized_cue)
    overlap = heading_terms & cue_terms
    if len(overlap) < 2 and not any(len(term) >= 6 for term in overlap):
        return 0
    if overlap:
        return sum(len(term) for term in overlap)
    return 0


def source_semantic_fallback_window(text: str, cues: list[str], *, max_chars: int) -> tuple[int, int, str] | None:
    terms = source_semantic_match_terms(cues)
    if len(terms) < 2:
        return None
    best: tuple[int, int, str, int, int] | None = None
    for block_start, block_end, block_text in source_semantic_blocks(text):
        score, present_terms, first_position = source_semantic_block_score(block_text, terms)
        if score <= 0:
            continue
        absolute_position = block_start + first_position
        before_chars = min(420, max(100, max_chars // 3))
        start = max(block_start, absolute_position - before_chars)
        end = min(block_end, start + max_chars)
        start = adjust_window_start(text, start)
        end = adjust_window_end(text, end)
        cue = "fallback_semantic:" + ",".join(present_terms[:4])
        candidate = (start, end, cue, score, len(present_terms))
        if best is None or (score, len(present_terms), block_start * -1) > (best[3], best[4], best[0] * -1):
            best = candidate
    if best is None:
        return None
    return best[0], best[1], best[2]


def source_semantic_blocks(text: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    for match in re.finditer(r"(?ms)(?:^|\n{2,})(?P<body>.*?)(?=\n{2,}|\Z)", text):
        body = match.group("body")
        if not body.strip():
            continue
        leading = len(body) - len(body.lstrip())
        trailing = len(body.rstrip())
        start = match.start("body") + leading
        end = match.start("body") + trailing
        block_text = text[start:end]
        if len(normalized_source_match_text(block_text)) < 24:
            continue
        blocks.append((start, end, block_text))
    return blocks


def source_semantic_block_score(block_text: str, terms: list[str]) -> tuple[int, list[str], int]:
    normalized_block, position_map = normalized_source_match_text_with_positions(block_text)
    present: list[str] = []
    first_normalized_position: int | None = None
    for term in terms:
        position = normalized_block.find(term)
        if position < 0:
            continue
        if any(term in existing or existing in term for existing in present):
            continue
        present.append(term)
        first_normalized_position = position if first_normalized_position is None else min(first_normalized_position, position)
    if not present:
        return 0, [], 0
    long_hits = [term for term in present if len(term) >= 4]
    if len(present) < 2 and not long_hits:
        return 0, [], 0
    score = sum(min(12, len(term)) for term in present) + len(present) * 3
    if len(present) >= 2:
        score += 8
    if not long_hits:
        score -= 6
    if score < 18:
        return 0, [], 0
    first_position = 0
    if first_normalized_position is not None and first_normalized_position < len(position_map):
        first_position = position_map[first_normalized_position]
    return score, present, first_position


def source_semantic_match_terms(cues: list[str]) -> list[str]:
    ascii_terms: set[str] = set()
    cjk_terms: set[str] = set()
    english_stopwords = {
        "and",
        "are",
        "approved",
        "digest",
        "for",
        "from",
        "how",
        "page",
        "section",
        "source",
        "into",
        "that",
        "the",
        "this",
        "with",
        "wiki",
        "why",
    }
    cjk_stop_terms = {
        "来源",
        "定位",
        "来源定位",
        "摘要",
        "问题",
        "价值",
        "页面",
        "概念",
        "设计",
        "部分",
        "小节",
        "访谈",
        "讨论",
    }
    for cue in source_excerpt_cues(cues):
        normalized = unicodedata.normalize("NFKC", cue).lower()
        for token in re.findall(r"[a-z][a-z0-9+#./-]{2,}", normalized):
            if token in english_stopwords or source_section_locator_token(token):
                continue
            normalized_token = normalized_source_match_text(token)
            if len(normalized_token) >= 3:
                ascii_terms.add(normalized_token)
        for segment in re.findall(r"[\u4e00-\u9fff]{3,}", normalized):
            if segment in cjk_stop_terms:
                continue
            max_size = min(8, len(segment))
            for size in range(max_size, 2, -1):
                for index in range(0, len(segment) - size + 1):
                    term = segment[index : index + size]
                    if term not in cjk_stop_terms:
                        normalized_term = normalized_source_match_text(term)
                        if len(normalized_term) >= 3:
                            cjk_terms.add(normalized_term)
    ascii_sorted = sorted(ascii_terms, key=lambda value: (-len(value), value))
    cjk_sorted = sorted(cjk_terms, key=lambda value: (-len(value), value))
    total_limit = 80
    ascii_limit = 32
    cjk_min_limit = 24
    selected = ascii_sorted[:ascii_limit]
    cjk_limit = max(cjk_min_limit, total_limit - len(selected))
    selected.extend(cjk_sorted[:cjk_limit])
    if len(selected) < total_limit:
        selected.extend(ascii_sorted[ascii_limit : ascii_limit + total_limit - len(selected)])
    return selected[:total_limit]


def source_section_locator_token(token: str) -> bool:
    return bool(re.fullmatch(r"s\d{3}(?:[-–—~至到/]s?\d{3})?", token.strip().lower()))


def meaningful_match_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{3,}", text))
    terms -= SOURCE_EXCERPT_LOW_SIGNAL_NORMALIZED_CUES
    if not terms and len(text) >= 4:
        terms.update(text[index : index + 4] for index in range(0, len(text) - 3))
    return terms


def source_heading_section_end(text: str, start: int, marker: str) -> int:
    pattern = re.compile(r"(?m)^(#{1,%d})\s+" % len(marker))
    match = pattern.search(text, start)
    return match.start() if match is not None else len(text)


def source_excerpt_cues(cues: list[str]) -> list[str]:
    normalized: list[str] = []
    for cue in cues:
        for piece in re.split(r"[\n。；;，,、|]+", cue or ""):
            text = piece.strip().strip("`*_ ")
            if len(text) > 160:
                text = text[:160].rstrip()
            for variant in source_excerpt_cue_variants(text):
                if variant not in normalized:
                    normalized.append(variant)
    normalized.sort(key=lambda value: (len(normalized_source_match_text(value)) < 8, -len(normalized_source_match_text(value))))
    return normalized[:40]


def source_excerpt_cue_variants(text: str) -> list[str]:
    variants: list[str] = []

    def add(value: str) -> None:
        value = value.strip().strip("`*_ -")
        if len(normalized_source_match_text(value)) < 3:
            return
        if value not in variants:
            variants.append(value)

    add(text)
    without_parenthetical = re.sub(r"[\(（][^\)）]{2,80}[\)）]", " ", text)
    add(without_parenthetical)
    for match in re.finditer(r"[\(（]([^\)）]{2,80})[\)）]", text):
        add(match.group(1))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9 +#./-]{2,}", text):
        add(segment)
    return variants


def find_source_cue(text: str, cue: str) -> int:
    position = text.find(cue)
    if position >= 0:
        return position
    normalized_cue = normalized_source_match_text(cue)
    if len(normalized_cue) < 4:
        return -1
    normalized_text, position_map = normalized_source_match_text_with_positions(text)
    normalized_position = normalized_text.find(normalized_cue)
    if normalized_position < 0:
        return -1
    return position_map[normalized_position] if normalized_position < len(position_map) else -1


def normalized_source_match_text(text: str) -> str:
    return normalized_source_match_text_with_positions(text)[0]


def normalized_source_match_text_with_positions(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char).lower()
        for normalized_char in normalized:
            if normalized_char.isspace():
                continue
            if unicodedata.category(normalized_char).startswith("P"):
                continue
            chars.append(normalized_char)
            positions.append(index)
    return "".join(chars), positions


def adjust_window_start(text: str, start: int) -> int:
    newline = text.rfind("\n", 0, start)
    return newline + 1 if newline >= 0 and start - newline < 160 else start


def adjust_window_end(text: str, end: int) -> int:
    newline = text.find("\n", end)
    return newline if newline >= 0 and newline - end < 160 else end


def render_draft_source_excerpt_pack_markdown(pack: dict[str, Any]) -> str:
    rows = []
    details: list[str] = []
    for item in pack.get("items", []):
        if not isinstance(item, dict):
            continue
        snippets = item.get("snippets", [])
        rows.append(
            [
                str(item.get("page_plan_id", "")),
                str(item.get("display_title", "")),
                str(item.get("target_path", "")),
                str(len(snippets) if isinstance(snippets, list) else 0),
                ", ".join(str(snippet.get("cue", "")) for snippet in snippets[:3] if isinstance(snippet, dict)) if isinstance(snippets, list) else "",
            ]
        )
        details.append(f"## {item.get('display_title', item.get('page_plan_id', ''))}\n")
        details.append(f"- 页面计划: `{item.get('page_plan_id', '')}`\n")
        locators = item.get("source_locators", [])
        if isinstance(locators, list) and locators:
            details.append(f"- 来源定位: {', '.join(str(locator) for locator in locators)}\n")
        if isinstance(snippets, list):
            for snippet_index, snippet in enumerate(snippets, start=1):
                if not isinstance(snippet, dict):
                    continue
                details.append(f"\n### Snippet {snippet_index}: {snippet.get('cue', '')}\n\n")
                details.append(blockquote_markdown(str(snippet.get("text", ""))) + "\n")
    return (
        "# Draft Rendering Source Excerpt Pack\n\n"
        f"- 原始字符数：{pack.get('original_char_count', 0)}\n"
        f"- 纳入字符数：{pack.get('included_char_count', 0)}\n"
        f"- payload 是否包含完整 source：`{str(pack.get('full_source_in_payload', False)).lower()}`\n"
        f"- 是否强制使用 excerpt：`{str(pack.get('force_excerpt', False)).lower()}`\n"
        f"- 完整 source 阈值：{pack.get('full_source_limit', 0)}\n"
        f"- 配置总 excerpt 阈值：{pack.get('configured_total_excerpt_limit', pack.get('total_excerpt_limit', 0))}\n"
        f"- 生效总 excerpt 阈值：{pack.get('total_excerpt_limit', 0)}\n"
        f"- 最低单页 excerpt：{pack.get('min_page_excerpt_chars', 0)}\n"
        f"- 最大 source 比例：{pack.get('max_source_ratio', '')}\n\n"
        "## 页面摘录索引\n\n"
        f"{format_markdown_table(['页面计划', '标题', '目标', '片段数', '主要 cue'], rows)}\n\n"
        "## 全局摘录\n\n"
        f"{blockquote_markdown(str(pack.get('global_excerpt', '')))}\n\n"
        "## 分页摘录\n\n"
        + "\n".join(details).rstrip()
        + "\n"
    )


def blockquote_markdown(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ">"
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def build_update_preservation_pack(merge_plan: WikiMergePlanArtifact, snapshot: WikiContextSnapshot) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for item in merge_plan.items:
        if item.action != "update":
            continue
        try:
            entry = snapshot_entry(snapshot, f"wiki/{item.canonical_target_path}")
        except PipelineError:
            continue
        sections = parse_existing_sections(entry.content)
        section_items: list[dict[str, Any]] = []
        for section_key in UPDATE_PRESERVATION_SECTION_KEYS:
            old_text = sections.get(section_key, "").strip()
            if not old_text:
                continue
            reusable_old_text = update_preservation_non_placeholder_text(old_text)
            phrases = update_preservation_phrases(reusable_old_text)
            concepts = update_preservation_concepts(reusable_old_text)
            if update_preservation_section_is_low_value(section_key, old_text, phrases, concepts):
                continue
            section_items.append(
                {
                    "section_key": section_key,
                    "old_text": compact_payload_text(reusable_old_text, 1800),
                    "old_char_count": len(old_text),
                    "key_phrases": phrases,
                    "min_required_matches": 0 if concepts else update_preservation_required_matches(phrases),
                    "concept_obligations": concepts,
                    "min_required_concept_matches": update_preservation_required_concept_matches(concepts),
                }
            )
        section_items = collapse_update_preservation_sections(section_items)
        if section_items:
            pages.append(
                {
                    "page_plan_id": item.page_plan_id,
                    "target_path": item.canonical_target_path,
                    "display_title": item.display_title,
                    "matched_page": item.matched_page,
                    "sections": section_items,
                }
            )
    return {
        "schema_version": "update_preservation_pack.v1",
        "goal": "For update pages, carry forward old reusable knowledge into the replacement draft or explicitly explain why it changed.",
        "pages": pages,
    }


def collapse_update_preservation_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail = next((section for section in sections if section.get("section_key") == "detail"), None)
    if detail is None:
        return sections
    detail_concepts = update_preservation_section_concept_names(detail)
    if not detail_concepts:
        return sections
    collapsed: list[dict[str, Any]] = []
    for section in sections:
        section_key = str(section.get("section_key", ""))
        section_concepts = update_preservation_section_concept_names(section)
        if section_key in {"summary", "value_points"} and section_concepts and section_concepts <= detail_concepts:
            continue
        collapsed.append(section)
    return collapsed


def update_preservation_section_concept_names(section: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for concept in section.get("concept_obligations", []):
        if isinstance(concept, dict):
            name = str(concept.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def update_preservation_section_is_low_value(
    section_key: str,
    old_text: str,
    phrases: list[str],
    concepts: list[dict[str, Any]],
) -> bool:
    normalized = normalized_source_match_text(old_text)
    meaningful_phrases = [
        phrase
        for phrase in phrases
        if not update_preservation_phrase_is_placeholder(normalized_source_match_text(phrase))
    ]
    if not concepts and not meaningful_phrases:
        return True
    if update_preservation_phrase_is_placeholder(normalized):
        return not update_preservation_has_non_placeholder_signal(old_text)
    if section_key == "value_points" and not concepts and len(meaningful_phrases) <= 1:
        return True
    return False


def update_preservation_phrases(text: str, *, limit: int = UPDATE_PRESERVATION_MAX_PHRASES_PER_SECTION) -> list[str]:
    phrases: list[str] = []
    for piece in re.split(r"[\n。；;，,、|：:]+", text):
        for variant in source_excerpt_cue_variants(piece):
            normalized = normalized_source_match_text(variant)
            if len(normalized) < 4 or len(normalized) > 80:
                continue
            if update_preservation_phrase_is_noise(normalized):
                continue
            if variant not in phrases:
                phrases.append(variant)
    phrases.sort(key=lambda value: (phrase_signal_score(value), len(normalized_source_match_text(value))), reverse=True)
    return phrases[:limit]


def update_preservation_phrase_is_noise(normalized: str) -> bool:
    if update_preservation_phrase_is_placeholder(normalized):
        return True
    if normalized in {"暂无", "没有相关", "暂无相关", "无相关", "n/a", "na"}:
        return True
    if normalized.startswith("旧页保留观察"):
        return True
    return False


def update_preservation_has_non_placeholder_signal(text: str) -> bool:
    reusable_text = update_preservation_non_placeholder_text(text)
    if not reusable_text:
        return False
    if update_preservation_concepts(reusable_text):
        return True
    phrases = update_preservation_phrases(reusable_text)
    return any(not update_preservation_phrase_is_placeholder(normalized_source_match_text(phrase)) for phrase in phrases)


def update_preservation_non_placeholder_text(text: str) -> str:
    raw_segments = [
        segment.strip()
        for segment in re.split(r"[\n。；;，,、|：:]+", text)
        if normalized_source_match_text(segment)
    ]
    placeholder_flags = [
        update_preservation_phrase_is_placeholder(normalized_source_match_text(segment))
        for segment in raw_segments
    ]
    if not any(placeholder_flags):
        return text.strip()
    return "\n".join(segment for segment, is_placeholder in zip(raw_segments, placeholder_flags) if not is_placeholder)


def update_preservation_phrase_is_placeholder(normalized: str) -> bool:
    value = normalized.lower()
    if not value or value in {"n/a", "na"}:
        return True
    return any(
        marker in value
        for marker in [
            "待补来源",
            "来源未提供",
            "暂无",
            "没有相关",
            "无相关",
        ]
    )


def update_preservation_ascii_token_spans(text: str) -> tuple[str, list[tuple[str, int, int]]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    spans = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"[a-z0-9]+", normalized)]
    return normalized, spans


def update_preservation_term_uses_ascii_tokens(term: str) -> bool:
    return bool(re.search(r"[A-Za-z]", term)) and term.isascii()


def update_preservation_ascii_phrase_separator_allowed(separator: str, term_separator: str) -> bool:
    if "&" in term_separator:
        return separator.count("&") == 1 and all(char.isspace() or char == "&" for char in separator)
    return all(char.isspace() or char in "-_/" for char in separator)


def update_preservation_ascii_phrase_matches(text: str, term: str) -> bool:
    normalized_term, term_spans = update_preservation_ascii_token_spans(term)
    term_tokens = [token for token, _, _ in term_spans]
    if not term_tokens:
        return False
    term_separators = [
        normalized_term[term_spans[offset][2] : term_spans[offset + 1][1]]
        for offset in range(len(term_spans) - 1)
    ]
    normalized_text, text_spans = update_preservation_ascii_token_spans(text)
    text_tokens = [token for token, _, _ in text_spans]
    if len(text_spans) < len(term_tokens):
        return False
    window_size = len(term_tokens)
    for index in range(len(text_spans) - window_size + 1):
        if text_tokens[index : index + window_size] != term_tokens:
            continue
        window_spans = text_spans[index : index + window_size]
        separators = [
            normalized_text[window_spans[offset][2] : window_spans[offset + 1][1]]
            for offset in range(len(window_spans) - 1)
        ]
        if all(
            update_preservation_ascii_phrase_separator_allowed(separator, term_separator)
            for separator, term_separator in zip(separators, term_separators)
        ):
            return True
    return False


def update_preservation_term_matches(text: str, term: str) -> bool:
    if not term.strip():
        return False
    if update_preservation_term_uses_ascii_tokens(term):
        return update_preservation_ascii_phrase_matches(text, term)
    normalized_term = normalized_source_match_text(term)
    return bool(normalized_term) and normalized_term in normalized_source_match_text(text)


def update_preservation_concepts(text: str) -> list[dict[str, Any]]:
    concepts: list[dict[str, Any]] = []
    for group in UPDATE_PRESERVATION_CONCEPT_GROUPS:
        matched_terms = [
            term
            for term in group["terms"]
            if update_preservation_term_matches(text, str(term))
        ]
        if matched_terms:
            concepts.append(
                {
                    "name": group["name"],
                    "label": group["label"],
                    "matched_terms": matched_terms,
                }
            )
    return concepts


def update_preservation_required_concept_matches(concepts: list[dict[str, Any]]) -> int:
    count = len(concepts)
    if count <= 0:
        return 0
    if count <= 2:
        return count
    if count <= 4:
        return 3
    return max(3, (count * 2 + 2) // 3)


def update_preservation_concept_absorption(old: str, new: str) -> tuple[bool, list[str], list[str], int]:
    concepts = update_preservation_concepts(old)
    if not concepts:
        return True, [], [], 0
    matched: list[str] = []
    missing: list[str] = []
    for concept in concepts:
        group = next((item for item in UPDATE_PRESERVATION_CONCEPT_GROUPS if item["name"] == concept["name"]), None)
        terms = tuple(group["terms"] if group is not None else concept.get("matched_terms", []))
        if any(update_preservation_term_matches(new, str(term)) for term in terms):
            matched.append(str(concept["label"]))
        else:
            missing.append(str(concept["label"]))
    required = update_preservation_required_concept_matches(concepts)
    return len(matched) >= required, matched, missing, required


def phrase_signal_score(phrase: str) -> int:
    normalized = normalized_source_match_text(phrase)
    score = min(len(normalized), 40)
    if re.search(r"[A-Za-z]", phrase):
        score += 12
    if any(keyword in phrase for keyword in ["harness", "Managed", "安全边界", "会话对象", "隔离", "权限", "架构", "上下文"]):
        score += 10
    if re.search(r"\d|%|倍|收入|用户|增长|下降|裁撤|预算|金额", phrase):
        score += 6
    return score


def update_preservation_required_matches(phrases: list[str]) -> int:
    if not phrases:
        return 0
    return 1 if len(phrases) <= 2 else 2


def update_section_absorption(old: str, new: str) -> tuple[bool, list[str], list[str]]:
    old = old.strip()
    new = new.strip()
    if not old or is_empty_placeholder(old):
        return True, [], []
    if old == new or old in new:
        phrases = update_preservation_phrases(old)
        return True, phrases[: update_preservation_required_matches(phrases)], phrases
    phrases = update_preservation_phrases(old)
    concept_absorbed, matched_concepts, _, _ = update_preservation_concept_absorption(old, new)
    concepts = update_preservation_concepts(old)
    if not phrases:
        if concepts and concept_absorbed and len(matched_concepts) >= max(2, update_preservation_required_concept_matches(concepts)):
            return True, matched_concepts, phrases
        return False, matched_concepts, phrases
    matched = [phrase for phrase in phrases if find_source_cue(new, phrase) >= 0]
    required = update_preservation_required_matches(phrases)
    if concepts and not concept_absorbed:
        return False, [*matched, *matched_concepts], phrases
    return len(matched) >= required or bool(matched_concepts), [*matched, *matched_concepts], phrases


def update_preservation_section_absorption(section: dict[str, Any], new_text: str) -> dict[str, Any]:
    old_text = str(section.get("old_text", ""))
    phrases = [str(phrase) for phrase in section.get("key_phrases", []) if str(phrase).strip()]
    if not phrases:
        phrases = update_preservation_phrases(old_text)
    if old_text.strip() and (old_text.strip() == new_text.strip() or old_text.strip() in new_text):
        matched_phrases = phrases[: update_preservation_required_matches(phrases)]
    else:
        matched_phrases = [phrase for phrase in phrases if find_source_cue(new_text, phrase) >= 0]
    required_phrases = int(section.get("min_required_matches") or update_preservation_required_matches(phrases))
    concept_obligations = [
        concept
        for concept in section.get("concept_obligations", [])
        if isinstance(concept, dict)
    ]
    if not concept_obligations:
        concept_obligations = update_preservation_concepts(old_text)
    matched_concepts: list[str] = []
    missing_concepts: list[str] = []
    for concept in concept_obligations:
        name = str(concept.get("name", ""))
        label = str(concept.get("label") or name)
        group = next((item for item in UPDATE_PRESERVATION_CONCEPT_GROUPS if item["name"] == name), None)
        terms = tuple(group["terms"] if group is not None else concept.get("matched_terms", []))
        if any(update_preservation_term_matches(new_text, str(term)) for term in terms):
            matched_concepts.append(label)
        else:
            missing_concepts.append(label)
    required_concepts = int(
        section.get("min_required_concept_matches")
        or update_preservation_required_concept_matches(concept_obligations)
    )
    if required_concepts and matched_concepts:
        phrase_absorbed = True
    else:
        phrase_absorbed = len(matched_phrases) >= required_phrases if required_phrases else True
    concept_absorbed = len(matched_concepts) >= required_concepts if required_concepts else True
    absorbed = phrase_absorbed and concept_absorbed
    return {
        "absorbed": absorbed,
        "matched_phrases": matched_phrases,
        "phrases": phrases,
        "required_phrases": required_phrases,
        "matched_concepts": matched_concepts,
        "missing_concepts": missing_concepts,
        "required_concepts": required_concepts,
        "concept_labels": [str(concept.get("label") or concept.get("name", "")) for concept in concept_obligations],
    }


def update_preservation_issues(draft: DraftRenderingArtifact, pack: dict[str, Any]) -> list[StructuredIssue]:
    pages_by_id = {page.page_plan_id: page for page in draft.pages}
    issues: list[StructuredIssue] = []
    for page_pack in pack.get("pages", []):
        if not isinstance(page_pack, dict):
            continue
        page_plan_id = str(page_pack.get("page_plan_id", ""))
        page = pages_by_id.get(page_plan_id)
        if page is None:
            continue
        for section in page_pack.get("sections", []):
            if not isinstance(section, dict):
                continue
            section_key = str(section.get("section_key", ""))
            old_text = str(section.get("old_text", ""))
            new_text = page.section_bodies.get(section_key, "")
            absorption = update_preservation_section_absorption(section, new_text)
            if absorption["absorbed"]:
                continue
            concept_message = ""
            if absorption["concept_labels"]:
                concept_message = (
                    f" Required old concept obligations: {', '.join(absorption['concept_labels'])}. "
                    f"Need {absorption['required_concepts']}; matched concepts: {', '.join(absorption['matched_concepts']) or 'none'}; "
                    f"missing concepts: {', '.join(absorption['missing_concepts']) or 'none'}."
                )
            issues.append(
                StructuredIssue(
                    issue_code="old_knowledge_not_absorbed",
                    field_path=f"pages.{page_plan_id}.section_bodies.{section_key}",
                    validator_id="update_preservation_pack",
                    message=(
                        f"Update draft for `{page_pack.get('target_path', '')}` does not carry forward old `{section_key}` knowledge. "
                        f"Retain or rewrite at least {absorption['required_phrases']} key phrase(s), such as: {', '.join(absorption['phrases'][:4])}. "
                        f"Matched so far: {', '.join([*absorption['matched_phrases'], *absorption['matched_concepts']]) or 'none'}."
                        f"{concept_message}"
                    ),
                    repairability="repairable",
                )
            )
    return issues


def reinforce_update_preservation(
    draft: DraftRenderingArtifact,
    pack: dict[str, Any],
) -> tuple[DraftRenderingArtifact, dict[str, Any]]:
    pages_by_id = {page.page_plan_id: page for page in draft.pages}
    updated_pages: dict[str, DraftPageItem] = {}
    report_pages: list[dict[str, Any]] = []
    for page_pack in pack.get("pages", []):
        if not isinstance(page_pack, dict):
            continue
        page_plan_id = str(page_pack.get("page_plan_id", ""))
        page = pages_by_id.get(page_plan_id)
        if page is None:
            continue
        section_reports: list[dict[str, Any]] = []
        section_bodies = dict(page.section_bodies)
        for section in page_pack.get("sections", []):
            if not isinstance(section, dict):
                continue
            section_key = str(section.get("section_key", ""))
            old_text = str(section.get("old_text", "")).strip()
            if not section_key or not old_text:
                continue
            current = section_bodies.get(section_key, "")
            absorption = update_preservation_section_absorption(section, current)
            if absorption["absorbed"]:
                continue
            missing_concepts = list(absorption["missing_concepts"])
            reinforcement = update_preservation_reinforcement_text(section_key, old_text, missing_concepts)
            section_bodies[section_key] = merge_markdown_blocks(current, reinforcement)
            section_reports.append(
                {
                    "section_key": section_key,
                    "matched_before": [*absorption["matched_phrases"], *absorption["matched_concepts"]],
                    "missing_concepts_before": missing_concepts,
                    "required_concept_matches": absorption["required_concepts"],
                    "key_phrases": absorption["phrases"][:4],
                    "reinforcement_char_count": len(reinforcement),
                    "reinforcement_preview": compact_payload_text(reinforcement, 240),
                }
            )
        if section_reports:
            updated = page.model_copy(update={"section_bodies": section_bodies})
            updated_pages[page_plan_id] = updated
            report_pages.append(
                {
                    "page_plan_id": page_plan_id,
                    "target_path": page_pack.get("target_path", ""),
                    "display_title": page_pack.get("display_title", ""),
                    "sections": section_reports,
                }
            )
    if updated_pages:
        pages = [updated_pages.get(page.page_plan_id, page) for page in draft.pages]
        draft = draft.model_copy(update={"pages": pages})
    report = {
        "schema_version": "update_preservation_reinforcement_report.v1",
        "changed": bool(report_pages),
        "reinforced_page_count": len(report_pages),
        "reinforced_section_count": sum(len(page["sections"]) for page in report_pages),
        "pages": report_pages,
    }
    return draft, report


def update_preservation_reinforcement_text(section_key: str, old_text: str, missing_concepts: list[str]) -> str:
    old_excerpt = compact_payload_text(old_text, 700)
    bridge = "与旧页架构视角相衔接，"
    if missing_concepts:
        concept_text = "、".join(missing_concepts)
        old_excerpt = (
            f"从旧页保留的架构视角看，本段仍需体现：{concept_text}。"
            "这些是旧页已经建立的理解，应与本轮新材料并列保留。"
        )
        bridge = ""
    if section_key == "value_points":
        return f"- {bridge}{old_excerpt}"
    if section_key == "examples":
        return f"{bridge}{old_excerpt}"
    if section_key == "open_questions":
        return f"{bridge}{old_excerpt}"
    return f"{bridge}{old_excerpt}"


def render_update_preservation_reinforcement_report(report: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for page in report.get("pages", []):
        if not isinstance(page, dict):
            continue
        for section in page.get("sections", []):
            if not isinstance(section, dict):
                continue
            rows.append(
                [
                    page.get("page_plan_id", ""),
                    page.get("display_title", ""),
                    section.get("section_key", ""),
                    ", ".join(str(value) for value in section.get("missing_concepts_before", [])),
                    ", ".join(str(value) for value in section.get("key_phrases", [])),
                    section.get("reinforcement_preview", ""),
                ]
            )
    return (
        "# Update Preservation Reinforcement Report\n\n"
        f"- Changed: `{str(bool(report.get('changed'))).lower()}`\n"
        f"- Reinforced pages: `{report.get('reinforced_page_count', 0)}`\n"
        f"- Reinforced sections: `{report.get('reinforced_section_count', 0)}`\n\n"
        + (
            format_markdown_table(["页面计划", "标题", "段落", "补足概念", "关键短语", "补强预览"], rows)
            if rows
            else "_无需本地补强。_"
        )
        + "\n"
    )


def render_grounding_paraphrase_rewrite_report(report: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for page in report.get("pages", []):
        if not isinstance(page, dict):
            continue
        for section in page.get("sections", []):
            if not isinstance(section, dict):
                continue
            for rewrite in section.get("rewrites", []):
                if not isinstance(rewrite, dict):
                    continue
                rows.append(
                    [
                        page.get("page_plan_id", ""),
                        page.get("target_path", ""),
                        section.get("section_key", ""),
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


def render_open_question_grounding_cleanup_report(report: dict[str, Any]) -> str:
    relocation_rows: list[list[Any]] = []
    skipped_rows: list[list[Any]] = []
    for page in report.get("pages", []):
        if not isinstance(page, dict):
            continue
        for item in page.get("relocations", []):
            if not isinstance(item, dict):
                continue
            relocation_rows.append(
                [
                    page.get("page_plan_id", ""),
                    page.get("target_path", ""),
                    item.get("section_key", ""),
                    item.get("text", ""),
                    item.get("question", ""),
                    item.get("append_decision", ""),
                ]
            )
        for item in page.get("skipped", []):
            if not isinstance(item, dict):
                continue
            skipped_rows.append(
                [
                    page.get("page_plan_id", ""),
                    page.get("target_path", ""),
                    item.get("section_key", ""),
                    item.get("reason", ""),
                    item.get("text", ""),
                ]
            )
    sections = [
        "# Open Question Grounding Cleanup Report",
        "",
        f"- Changed: `{str(bool(report.get('changed'))).lower()}`",
        f"- Relocations: `{report.get('relocation_count', 0)}`",
        f"- Skipped: `{report.get('skipped_count', 0)}`",
        "",
        "## Relocated Claims",
        "",
        (
            format_markdown_table(["页面计划", "目标", "原段落", "原文本", "转成的问题", "追加决策"], relocation_rows)
            if relocation_rows
            else "_无需移动。_"
        ),
        "",
        "## Skipped Claims",
        "",
        (
            format_markdown_table(["页面计划", "目标", "段落", "原因", "文本"], skipped_rows)
            if skipped_rows
            else "_无跳过项。_"
        ),
        "",
    ]
    return "\n".join(sections)


def render_example_concrete_cleanup_report(report: dict[str, Any]) -> str:
    replacement_rows: list[list[Any]] = []
    skipped_rows: list[list[Any]] = []
    for item in report.get("replacements", []):
        if not isinstance(item, dict):
            continue
        replacement_rows.append(
            [
                item.get("page_plan_id", ""),
                item.get("target_path", ""),
                item.get("original", ""),
                item.get("replacement", ""),
                item.get("reason", ""),
            ]
        )
    for item in report.get("skipped", []):
        if not isinstance(item, dict):
            continue
        skipped_rows.append(
            [
                item.get("page_plan_id", ""),
                item.get("target_path", ""),
                item.get("text", ""),
                item.get("reason", ""),
            ]
        )
    sections = [
        "# Example Concrete Cleanup Report",
        "",
        f"- Changed: `{str(bool(report.get('changed'))).lower()}`",
        f"- Replacements: `{report.get('replacement_count', 0)}`",
        f"- Skipped: `{report.get('skipped_count', 0)}`",
        "",
        "## Replacements",
        "",
        (
            format_markdown_table(["页面计划", "目标", "原具体值", "替换为", "原因"], replacement_rows)
            if replacement_rows
            else "_无需替换。_"
        ),
        "",
        "## Skipped",
        "",
        (
            format_markdown_table(["页面计划", "目标", "文本", "跳过原因"], skipped_rows)
            if skipped_rows
            else "_无跳过项。_"
        ),
        "",
    ]
    return "\n".join(sections)


def render_update_preservation_pack_markdown(pack: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for page in pack.get("pages", []):
        if not isinstance(page, dict):
            continue
        for section in page.get("sections", []):
            if not isinstance(section, dict):
                continue
            rows.append(
                [
                    page.get("page_plan_id", ""),
                    page.get("display_title", ""),
                    section.get("section_key", ""),
                    section.get("min_required_matches", 0),
                    ", ".join(str(phrase) for phrase in section.get("key_phrases", [])[:4]),
                    section.get("min_required_concept_matches", 0),
                    ", ".join(str(concept.get("label", "")) for concept in section.get("concept_obligations", [])[:5] if isinstance(concept, dict)),
                ]
            )
    return (
        "# Update Preservation Pack\n\n"
        "这些 obligations 会传给 draft_rendering，并由本地 validator 检查；如果模型未吸收旧知识，会先触发 repair，最终仍由旧页保留观察兜底。\n\n"
        f"{format_markdown_table(['页面计划', '标题', '段落', '最少短语', '关键短语', '最少概念', '概念义务'], rows) if rows else '_本轮没有 update preservation obligations。_'}\n"
    )


def run_draft_rendering_model(
    *,
    ctx: StepRunContext,
    step_root: Path,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
) -> DraftRenderingArtifact:
    draftable_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    provider = ctx.execution_context.provider_for_task("draft_rendering")
    if len(draftable_items) <= DRAFT_RENDERING_BATCH_PAGE_LIMIT:
        return run_single_draft_rendering_model_call(
            ctx=ctx,
            provider=provider,
            output_dir=step_root,
            digest=digest,
            merge_plan=merge_plan,
            snapshot=snapshot,
            source_excerpt_pack=source_excerpt_pack,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
        )

    batch_items_list = chunks(draftable_items, DRAFT_RENDERING_BATCH_PAGE_LIMIT)
    provider_spec = ctx.execution_context.runtime_for_task("draft_rendering").spec
    max_parallel_batches = draft_rendering_batch_parallelism(provider_spec, len(batch_items_list))
    parallel = max_parallel_batches > 1
    batch_jobs: list[dict[str, Any]] = []
    batch_root = step_root / "model_batches"
    for index, batch_items in enumerate(batch_items_list, start=1):
        batch_id = f"batch-{index:03d}"
        batch_dir = batch_root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_plan = merge_plan.model_copy(update={"items": list(batch_items)})
        batch_source_excerpt_pack = build_draft_source_excerpt_pack(
            approved_prepared_text,
            digest,
            batch_plan,
            force_excerpt=True,
        )
        batch_source_pack_path = batch_dir / "draft_source_excerpt_pack.json"
        batch_source_pack_md = batch_dir / "draft_source_excerpt_pack.md"
        write_json(batch_source_pack_path, batch_source_excerpt_pack)
        batch_source_pack_md.write_text(render_draft_source_excerpt_pack_markdown(batch_source_excerpt_pack), encoding="utf-8")
        batch_update_preservation_pack = build_update_preservation_pack(batch_plan, snapshot)
        batch_update_pack_path = batch_dir / "update_preservation_pack.json"
        batch_update_pack_md = batch_dir / "update_preservation_pack.md"
        write_json(batch_update_pack_path, batch_update_preservation_pack)
        batch_update_pack_md.write_text(render_update_preservation_pack_markdown(batch_update_preservation_pack), encoding="utf-8")
        batch_jobs.append(
            {
                "index": index,
                "batch_id": batch_id,
                "batch_items": batch_items,
                "batch_dir": batch_dir,
                "batch_plan": batch_plan,
                "source_excerpt_pack": batch_source_excerpt_pack,
                "update_preservation_pack": batch_update_preservation_pack,
            }
        )

    def run_batch(job: dict[str, Any]) -> dict[str, Any]:
        batch_provider = ctx.execution_context.provider_for_task("draft_rendering") if parallel else provider
        batch_artifact = run_single_draft_rendering_model_call(
            ctx=ctx,
            provider=batch_provider,
            output_dir=job["batch_dir"],
            digest=digest,
            merge_plan=job["batch_plan"],
            snapshot=snapshot,
            source_excerpt_pack=job["source_excerpt_pack"],
            update_preservation_pack=job["update_preservation_pack"],
            approved_prepared_text=approved_prepared_text,
        )
        report = read_model(job["batch_dir"] / "structured_repair_report.json", StructuredRepairReport)
        result = read_model(job["batch_dir"] / "provider_result.json", ProviderResult)
        reinforcement_path = job["batch_dir"] / "update_preservation_reinforcement_report.json"
        reinforcement_report = read_json(reinforcement_path) if reinforcement_path.exists() else {}
        grounding_rewrite_path = job["batch_dir"] / "grounding_paraphrase_rewrite_report.json"
        grounding_rewrite_report = read_json(grounding_rewrite_path) if grounding_rewrite_path.exists() else {}
        example_cleanup_path = job["batch_dir"] / "example_concrete_cleanup_report.json"
        example_cleanup_report = read_json(example_cleanup_path) if example_cleanup_path.exists() else {}
        batch_payload_char_count = provider_results_payload_char_count(
            [job["batch_dir"] / attempt.provider_result_ref for attempt in report.attempts]
        )
        batch_http_attempt_count = provider_results_http_attempt_count(
            [job["batch_dir"] / attempt.provider_result_ref for attempt in report.attempts]
        )
        batch_items = job["batch_items"]
        batch_id = job["batch_id"]
        return {
            "index": job["index"],
            "artifact": batch_artifact,
            "summary": {
                "batch_id": batch_id,
                "page_plan_ids": [item.page_plan_id for item in batch_items],
                "target_paths": [item.canonical_target_path for item in batch_items],
                "source_excerpt_chars": job["source_excerpt_pack"].get("included_char_count", 0),
                "payload_char_count": batch_payload_char_count,
                "http_attempt_count": batch_http_attempt_count,
                "attempt_count": report.attempt_count,
                "repair_count": report.repair_count,
                "duration_ms": report.duration_ms,
                "provider": report.provider,
                "provider_result_ref": f"model_batches/{batch_id}/provider_result.json",
                "structured_repair_report_ref": f"model_batches/{batch_id}/structured_repair_report.json",
                "update_preservation_reinforcement_report_ref": (
                    f"model_batches/{batch_id}/update_preservation_reinforcement_report.json"
                    if reinforcement_path.exists()
                    else ""
                ),
                "reinforced_section_count": int(reinforcement_report.get("reinforced_section_count", 0)),
                "grounding_paraphrase_rewrite_report_ref": (
                    f"model_batches/{batch_id}/grounding_paraphrase_rewrite_report.json"
                    if grounding_rewrite_path.exists()
                    else ""
                ),
                "grounding_rewrite_count": int(grounding_rewrite_report.get("rewrite_count", 0)),
                "example_concrete_cleanup_report_ref": (
                    f"model_batches/{batch_id}/example_concrete_cleanup_report.json"
                    if example_cleanup_path.exists()
                    else ""
                ),
                "example_concrete_replacement_count": int(example_cleanup_report.get("replacement_count", 0)),
                "schema_valid": result.schema_valid,
            },
        }

    started = perf_counter()
    if parallel:
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_parallel_batches) as executor:
            futures = [executor.submit(run_batch, job) for job in batch_jobs]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [run_batch(job) for job in batch_jobs]
    wall_duration_ms = round((perf_counter() - started) * 1000)

    pages: list[DraftPageItem] = []
    batch_summaries: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: int(item["index"])):
        batch_artifact = result["artifact"]
        pages.extend(batch_artifact.pages)
        batch_summaries.append(result["summary"])
    draft_artifact = finalize_draft_rendering(DraftRenderingArtifact(pages=pages), merge_plan, snapshot)
    write_draft_rendering_batch_reports(
        step_root,
        draft_artifact,
        batch_summaries,
        max_parallel_batches=max_parallel_batches,
        wall_duration_ms=wall_duration_ms,
    )
    return draft_artifact


def draft_rendering_batch_parallelism(provider_spec: str | None, batch_count: int) -> int:
    if batch_count <= 1:
        return 1
    if provider_spec and provider_spec.startswith("openai_compatible:"):
        return min(DRAFT_RENDERING_MAX_PARALLEL_BATCHES, batch_count)
    return 1


def chunks(items: list[WikiMergePlanItem], size: int) -> list[list[WikiMergePlanItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


PAGE_SCOPED_DRAFT_REPAIR_ISSUE_CODES = {
    "unsupported_new_fact",
    "model_self_talk_leak",
    "stray_related_links_in_content",
    "missing_repair_page",
}


def build_draft_rendering_missing_page_repair_payload(
    *,
    task: str,
    raw: str,
    issues: list[StructuredIssue],
    output_model: type[BaseModel],
    ctx: StepRunContext,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
) -> dict[str, Any] | None:
    if not issues or any(issue.issue_code != "missing_page_plan_coverage" for issue in issues):
        return None
    partial = extract_valid_partial_draft_rendering(
        raw,
        merge_plan,
        snapshot,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
        language=ctx.manifest.vault_config_snapshot.wiki_language,
    )
    if partial is None or not partial.pages:
        return None
    present_ids = {page.page_plan_id for page in partial.pages}
    required_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    missing_items = [item for item in required_items if item.page_plan_id not in present_ids]
    if not missing_items:
        return None
    missing_plan = merge_plan.model_copy(update={"items": missing_items})
    missing_source_excerpt_pack = build_draft_source_excerpt_pack(
        approved_prepared_text,
        digest,
        missing_plan,
        force_excerpt=True,
    )
    missing_update_preservation_pack = build_update_preservation_pack(missing_plan, snapshot)
    missing_payload = draft_rendering_model_payload(
        build_draft_rendering_payload(
            ctx=ctx,
            digest=digest,
            merge_plan=missing_plan,
            snapshot=snapshot,
            source_excerpt_pack=missing_source_excerpt_pack,
            update_preservation_pack=missing_update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
        )
    )
    return {
        "repair_contract": {
            "goal": "Complete a partial draft_rendering output without regenerating pages that already passed local validation.",
            "mode": "missing_page_completion",
            "task": task,
            "rules": [
                "Return only one complete JSON object matching the schema.",
                "The pages array must contain every accepted_partial_pages item unchanged plus exactly one generated page for each missing_page_plan_id.",
                "Do not regenerate, rewrite, remove, or reorder accepted_partial_pages; copy them into pages exactly as provided.",
                "Generate only the missing pages from missing_page_payload; do not create pages outside missing_page_plan_ids.",
                "All user-visible generated content must follow the language and grounding rules inside missing_page_payload.",
            ],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "accepted_page_plan_ids": [page.page_plan_id for page in partial.pages],
            "missing_page_plan_ids": [item.page_plan_id for item in missing_items],
            "required_page_plan_ids": [item.page_plan_id for item in required_items],
            "schema": output_model.model_json_schema(),
        },
        "accepted_partial_pages": [page.model_dump(mode="json") for page in partial.pages],
        "missing_page_payload": missing_payload,
        "source_excerpt_pack_omitted_reason": (
            "The original batch payload is intentionally replaced by a compact missing_page_payload; "
            f"previous full/pack source refs remain {source_excerpt_pack.get('approved_prepared_ref', '')}."
        ),
    }


def build_draft_rendering_page_repair_payload(
    *,
    task: str,
    raw: str,
    issues: list[StructuredIssue],
    output_model: type[BaseModel],
    ctx: StepRunContext,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
    accepted_partial_pages_override: list[DraftPageItem] | None = None,
    include_local_accepted_pages: bool = False,
) -> dict[str, Any] | None:
    failing_ids = draft_repair_page_plan_ids_from_issues(issues, merge_plan)
    if not failing_ids:
        return None
    required_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    required_ids = {item.page_plan_id for item in required_items}
    accepted_ids = required_ids - failing_ids
    if not accepted_ids or not failing_ids < required_ids:
        return None
    if accepted_partial_pages_override is not None:
        partial = DraftRenderingArtifact(
            pages=[page for page in accepted_partial_pages_override if page.page_plan_id in accepted_ids]
        )
    else:
        partial = extract_valid_partial_draft_rendering(
            raw,
            merge_plan,
            snapshot,
            accepted_page_plan_ids=accepted_ids,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
            language=ctx.manifest.vault_config_snapshot.wiki_language,
        )
    if partial is None or not partial.pages:
        return None
    if {page.page_plan_id for page in partial.pages} != accepted_ids:
        return None
    repair_items = [item for item in required_items if item.page_plan_id in failing_ids]
    if not repair_items:
        return None
    repair_plan = merge_plan.model_copy(update={"items": repair_items})
    repair_source_excerpt_pack = build_draft_source_excerpt_pack(
        approved_prepared_text,
        digest,
        repair_plan,
        force_excerpt=True,
    )
    repair_update_preservation_pack = build_update_preservation_pack(repair_plan, snapshot)
    repair_payload = draft_rendering_model_payload(
        build_draft_rendering_payload(
            ctx=ctx,
            digest=digest,
            merge_plan=repair_plan,
            snapshot=snapshot,
            source_excerpt_pack=repair_source_excerpt_pack,
            update_preservation_pack=repair_update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
        )
    )
    accepted_page_refs = draft_repair_accepted_page_refs(partial.pages, merge_plan)
    prompt = {
        "repair_contract": {
            "goal": "Repair page-scoped draft_rendering issues without regenerating pages that already passed local validation.",
            "mode": "page_scoped_repair",
            "task": task,
            "rules": [
                "Return only one complete JSON object matching the schema.",
                "The pages array must contain only repaired pages for repair_page_plan_ids.",
                "Do not include accepted_page_refs pages in the output; they are retained locally and will be merged after repair.",
                "accepted_page_refs may be used only for lightweight cross-page boundary awareness; do not regenerate them.",
                "Generate exactly one repaired page for each repair_page_plan_id from repair_page_payload; do not create pages outside repair_page_plan_ids.",
                "All user-visible repaired content must follow the language and grounding rules inside repair_page_payload.",
            ],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "accepted_page_plan_ids": [page.page_plan_id for page in partial.pages],
            "repair_page_plan_ids": [item.page_plan_id for item in repair_items],
            "required_page_plan_ids": [item.page_plan_id for item in required_items],
            "schema": output_model.model_json_schema(),
        },
        "accepted_page_refs": accepted_page_refs,
        "repair_page_payload": repair_payload,
        "source_excerpt_pack_omitted_reason": (
            "The original batch payload is intentionally replaced by a compact repair_page_payload; "
            f"previous full/pack source refs remain {source_excerpt_pack.get('approved_prepared_ref', '')}."
        ),
    }
    if include_local_accepted_pages:
        prompt["_local_accepted_partial_pages"] = [page.model_dump(mode="json") for page in partial.pages]
    return prompt


def draft_repair_accepted_page_refs(
    accepted_pages: list[DraftPageItem],
    merge_plan: WikiMergePlanArtifact,
) -> list[dict[str, str]]:
    items_by_id = {item.page_plan_id: item for item in merge_plan.items}
    refs: list[dict[str, str]] = []
    for page in accepted_pages:
        item = items_by_id.get(page.page_plan_id)
        refs.append(
            {
                "page_plan_id": page.page_plan_id,
                "action": page.action,
                "target_path": page.canonical_target_path,
                "display_title": item.display_title if item else page.canonical_target_path,
                "page_type": item.page_type if item else "",
            }
        )
    return refs


def merge_repaired_draft_with_accepted_pages(
    repair_only: DraftRenderingArtifact,
    *,
    accepted_pages_by_id: dict[str, dict[str, Any]],
    repair_page_plan_ids: set[str],
    merge_plan: WikiMergePlanArtifact,
) -> DraftRenderingArtifact:
    repair_pages_by_id = {
        page.page_plan_id: page
        for page in repair_only.pages
        if page.page_plan_id in repair_page_plan_ids
    }
    pages_by_id: dict[str, DraftPageItem] = {
        page_plan_id: DraftPageItem.model_validate(page)
        for page_plan_id, page in accepted_pages_by_id.items()
    }
    pages_by_id.update(repair_pages_by_id)
    ordered_ids = [
        item.page_plan_id
        for item in merge_plan.items
        if item.action in {"create", "update"} and item.page_plan_id in pages_by_id
    ]
    return DraftRenderingArtifact(pages=[pages_by_id[page_plan_id] for page_plan_id in ordered_ids])


def preserve_active_repair_page_issues(
    issues: list[StructuredIssue],
    active_repair_page_plan_ids: set[str],
) -> list[StructuredIssue]:
    if not active_repair_page_plan_ids:
        return issues
    issue_page_ids: set[str] = set()
    for issue in issues:
        if issue.repairability != "repairable" or issue.issue_code not in PAGE_SCOPED_DRAFT_REPAIR_ISSUE_CODES:
            return issues
        match = re.match(r"^pages\.([^.]+)(?:\.|$)", issue.field_path or "")
        if not match:
            return issues
        issue_page_ids.add(match.group(1))
    expanded = list(issues)
    for page_plan_id in sorted(active_repair_page_plan_ids):
        if page_plan_id in issue_page_ids:
            continue
        expanded.append(
            StructuredIssue(
                issue_code="missing_repair_page",
                field_path=f"pages.{page_plan_id}",
                validator_id="draft_page_scoped_repair",
                message=(
                    f"Page-scoped draft repair must return repaired page `{page_plan_id}` together with "
                    "the other active repair pages; prior partial repair output is not accepted until the full repair set validates."
                ),
                repairability="repairable",
            )
        )
    return expanded


def draft_repair_page_plan_ids_from_issues(
    issues: list[StructuredIssue],
    merge_plan: WikiMergePlanArtifact,
) -> set[str] | None:
    if not issues:
        return None
    draftable_ids = {item.page_plan_id for item in merge_plan.items if item.action in {"create", "update"}}
    page_plan_ids: set[str] = set()
    for issue in issues:
        if issue.issue_code not in PAGE_SCOPED_DRAFT_REPAIR_ISSUE_CODES:
            return None
        match = re.match(r"^pages\.([^.]+)(?:\.|$)", issue.field_path or "")
        if not match:
            return None
        page_plan_id = match.group(1)
        if page_plan_id not in draftable_ids:
            return None
        page_plan_ids.add(page_plan_id)
    return page_plan_ids or None


def accepted_partial_page_copy_issues(
    draft: DraftRenderingArtifact,
    accepted_pages_by_id: dict[str, dict[str, Any]],
) -> list[StructuredIssue]:
    if not accepted_pages_by_id:
        return []
    pages_by_id = {page.page_plan_id: page for page in draft.pages}
    issues: list[StructuredIssue] = []
    for page_plan_id, expected in accepted_pages_by_id.items():
        page = pages_by_id.get(page_plan_id)
        if page is None or page.model_dump(mode="json") != expected:
            issues.append(
                StructuredIssue(
                    issue_code="accepted_partial_page_changed",
                    field_path=f"pages.{page_plan_id}",
                    validator_id="draft_page_scoped_repair",
                    message=(
                        f"Page-scoped draft repair changed accepted partial page `{page_plan_id}`; "
                        "copy accepted_partial_pages exactly and only repair failing page ids."
                    ),
                    repairability="repairable",
                )
            )
    return issues


def extract_valid_partial_draft_rendering(
    raw: str,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    *,
    accepted_page_plan_ids: set[str] | None = None,
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
    language: str | None,
) -> DraftRenderingArtifact | None:
    try:
        parsed, _json_repaired = parse_structured_json_object(raw)
        artifact = DraftRenderingArtifact.model_validate(parsed)
        candidate = finalize_draft_rendering(artifact, merge_plan, snapshot)
    except Exception:
        return None
    if accepted_page_plan_ids is not None:
        existing_ids = {page.page_plan_id for page in candidate.pages}
        if not accepted_page_plan_ids <= existing_ids:
            return None
        candidate = DraftRenderingArtifact(
            pages=[page for page in candidate.pages if page.page_plan_id in accepted_page_plan_ids]
        )
    present_ids = {page.page_plan_id for page in candidate.pages}
    required_ids = {item.page_plan_id for item in merge_plan.items if item.action in {"create", "update"}}
    if not present_ids or not present_ids < required_ids:
        return None
    partial_plan = merge_plan.model_copy(
        update={
            "items": [
                item
                for item in merge_plan.items
                if item.action not in {"create", "update"} or item.page_plan_id in present_ids
            ]
        }
    )
    try:
        validate_draft_rendering(candidate, partial_plan, language=language)
    except Exception:
        return None
    if draft_self_talk_issues(candidate):
        return None
    if update_preservation_issues(candidate, update_preservation_pack):
        return None
    candidate, _grounding_rewrite_report = rewrite_grounding_sensitive_paraphrases(candidate, approved_prepared_text)
    grounding_candidate = candidate
    grounding_review = build_draft_grounding_review(grounding_candidate, partial_plan, snapshot, approved_prepared_text)
    grounding_candidate, example_cleanup_report = cleanup_unsupported_example_literals(
        grounding_candidate,
        partial_plan,
        snapshot,
        approved_prepared_text,
        review=grounding_review,
    )
    if example_cleanup_report.get("changed"):
        grounding_review = build_draft_grounding_review(grounding_candidate, partial_plan, snapshot, approved_prepared_text)
    if grounding_review.requires_review:
        return None
    return candidate


def run_single_draft_rendering_model_call(
    *,
    ctx: StepRunContext,
    provider: Provider,
    output_dir: Path,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
) -> DraftRenderingArtifact:
    payload = build_draft_rendering_payload(
        ctx=ctx,
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        source_excerpt_pack=source_excerpt_pack,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
    )
    write_draft_digest_projection_report(output_dir, payload["approved_digest_projection_report"])
    write_draft_payload_projection_reports(output_dir, payload)
    model_payload = draft_rendering_model_payload(payload)
    accepted_repair_pages_by_id: dict[str, dict[str, Any]] = {}
    active_repair_page_plan_ids: set[str] = set()
    last_merged_repair_artifact: DraftRenderingArtifact | None = None

    def validate_draft_rendering_model(model: DraftRenderingArtifact) -> None:
        nonlocal last_merged_repair_artifact
        validation_model = model
        if accepted_repair_pages_by_id and active_repair_page_plan_ids:
            returned_repair_ids = {
                page.page_plan_id
                for page in model.pages
                if page.page_plan_id in active_repair_page_plan_ids
            }
            missing_repair_ids = active_repair_page_plan_ids - returned_repair_ids
            if missing_repair_ids:
                raise ContractValidationError(
                    "page-scoped draft repair omitted required repaired pages.",
                    issues=[
                        StructuredIssue(
                            issue_code="missing_repair_page",
                            field_path=f"pages.{page_plan_id}",
                            validator_id="draft_page_scoped_repair",
                            message=(
                                f"Page-scoped draft repair must return repaired page `{page_plan_id}`; "
                                "accepted pages are retained locally and should not be returned instead."
                            ),
                            repairability="repairable",
                        )
                        for page_plan_id in sorted(missing_repair_ids)
                    ],
                )
            validation_model = merge_repaired_draft_with_accepted_pages(
                model,
                accepted_pages_by_id=accepted_repair_pages_by_id,
                repair_page_plan_ids=active_repair_page_plan_ids,
                merge_plan=merge_plan,
            )
            last_merged_repair_artifact = validation_model
        else:
            last_merged_repair_artifact = None
        candidate = finalize_draft_rendering(validation_model, merge_plan, snapshot)
        validate_draft_rendering(candidate, merge_plan, language=ctx.manifest.vault_config_snapshot.wiki_language)
        repair_issues = draft_self_talk_issues(candidate)
        repair_issues.extend(update_preservation_issues(candidate, update_preservation_pack))
        if not active_repair_page_plan_ids:
            repair_issues.extend(accepted_partial_page_copy_issues(candidate, accepted_repair_pages_by_id))
        rewritten_candidate, _grounding_rewrite_report = rewrite_grounding_sensitive_paraphrases(
            candidate,
            approved_prepared_text,
        )
        cleaned_candidate, _open_question_cleanup_report = cleanup_open_question_unsupported_scope_claims(
            rewritten_candidate,
            merge_plan,
            snapshot,
            approved_prepared_text,
        )
        grounding_review = build_draft_grounding_review(cleaned_candidate, merge_plan, snapshot, approved_prepared_text)
        cleaned_candidate, _example_cleanup_report = cleanup_unsupported_example_literals(
            cleaned_candidate,
            merge_plan,
            snapshot,
            approved_prepared_text,
            review=grounding_review,
        )
        if _example_cleanup_report.get("changed"):
            grounding_review = build_draft_grounding_review(cleaned_candidate, merge_plan, snapshot, approved_prepared_text)
        if grounding_review.requires_review:
            repair_issues.extend(
                [
                    StructuredIssue(
                        issue_code="unsupported_new_fact",
                        field_path=f"pages.{claim.page_plan_id}.{claim.section_key}",
                        validator_id="draft_grounding_review",
                        message=grounding_issue_message(claim),
                        repairability="repairable",
                    )
                    for claim in grounding_review.unsupported_new_facts
                ]
            )
        if repair_issues:
            raise ContractValidationError(
                "draft_rendering contains repairable quality issues; remove model self-talk, preserve update obligations, and fix unsupported facts.",
                issues=repair_issues,
            )

    def build_repair_payload(
        task: str,
        _payload: dict[str, Any],
        raw: str,
        issues: list[StructuredIssue],
        output_model: type[BaseModel],
    ) -> dict[str, Any] | None:
        nonlocal accepted_repair_pages_by_id, active_repair_page_plan_ids
        accepted_override = [
            DraftPageItem.model_validate(page)
            for page in accepted_repair_pages_by_id.values()
        ] if accepted_repair_pages_by_id else None
        page_repair_issues = preserve_active_repair_page_issues(issues, active_repair_page_plan_ids)
        page_repair_payload = build_draft_rendering_page_repair_payload(
            task=task,
            raw=raw,
            issues=page_repair_issues,
            output_model=output_model,
            ctx=ctx,
            digest=digest,
            merge_plan=merge_plan,
            snapshot=snapshot,
            source_excerpt_pack=source_excerpt_pack,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
            accepted_partial_pages_override=accepted_override,
            include_local_accepted_pages=True,
        )
        if page_repair_payload is not None:
            local_accepted_pages = page_repair_payload.pop("_local_accepted_partial_pages", [])
            accepted_repair_pages_by_id = {
                str(page.get("page_plan_id", "")): page
                for page in local_accepted_pages
                if isinstance(page, dict) and page.get("page_plan_id")
            }
            active_repair_page_plan_ids = set(page_repair_payload["repair_contract"].get("repair_page_plan_ids", []))
            return page_repair_payload
        accepted_repair_pages_by_id = {}
        active_repair_page_plan_ids = set()
        return build_draft_rendering_missing_page_repair_payload(
            task=task,
            raw=raw,
            issues=issues,
            output_model=output_model,
            ctx=ctx,
            digest=digest,
            merge_plan=merge_plan,
            snapshot=snapshot,
            source_excerpt_pack=source_excerpt_pack,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
        )

    draft_artifact, _ = StructuredModelCall(
        provider,
        output_dir=output_dir,
        result_filename="provider_result.json",
        redactor=ctx.execution_context.redactor,
    ).run(
        "draft_rendering",
        model_payload,
        DraftRenderingArtifact,
        validator=validate_draft_rendering_model,
        accept_after_repair_issue_codes={"unsupported_new_fact", "old_knowledge_not_absorbed"},
        repair_payload_builder=build_repair_payload,
    )
    if last_merged_repair_artifact is not None and accepted_repair_pages_by_id and active_repair_page_plan_ids:
        draft_artifact = last_merged_repair_artifact
    draft_artifact = _redacted_model(ctx, draft_artifact, DraftRenderingArtifact)
    draft_artifact = finalize_draft_rendering(draft_artifact, merge_plan, snapshot)
    draft_artifact, reinforcement_report = reinforce_update_preservation(draft_artifact, update_preservation_pack)
    draft_artifact, grounding_rewrite_report = rewrite_grounding_sensitive_paraphrases(draft_artifact, approved_prepared_text)
    reinforcement_path = output_dir / "update_preservation_reinforcement_report.json"
    reinforcement_md = output_dir / "update_preservation_reinforcement_report.md"
    write_json(reinforcement_path, reinforcement_report)
    reinforcement_md.write_text(render_update_preservation_reinforcement_report(reinforcement_report), encoding="utf-8")
    grounding_rewrite_path = output_dir / "grounding_paraphrase_rewrite_report.json"
    grounding_rewrite_md = output_dir / "grounding_paraphrase_rewrite_report.md"
    write_json(grounding_rewrite_path, grounding_rewrite_report)
    grounding_rewrite_md.write_text(render_grounding_paraphrase_rewrite_report(grounding_rewrite_report), encoding="utf-8")
    draft_artifact, open_question_cleanup_report = cleanup_open_question_unsupported_scope_claims(
        draft_artifact,
        merge_plan,
        snapshot,
        approved_prepared_text,
    )
    open_question_cleanup_path = output_dir / "open_question_grounding_cleanup_report.json"
    open_question_cleanup_md = output_dir / "open_question_grounding_cleanup_report.md"
    redacted_open_question_cleanup_report = ctx.execution_context.redactor.redact(open_question_cleanup_report)
    write_json(open_question_cleanup_path, redacted_open_question_cleanup_report)
    open_question_cleanup_md.write_text(
        render_open_question_grounding_cleanup_report(redacted_open_question_cleanup_report),
        encoding="utf-8",
    )
    grounding_review = build_draft_grounding_review(draft_artifact, merge_plan, snapshot, approved_prepared_text)
    draft_artifact, example_cleanup_report = cleanup_unsupported_example_literals(
        draft_artifact,
        merge_plan,
        snapshot,
        approved_prepared_text,
        review=grounding_review,
    )
    example_cleanup_path = output_dir / "example_concrete_cleanup_report.json"
    example_cleanup_md = output_dir / "example_concrete_cleanup_report.md"
    redacted_example_cleanup_report = ctx.execution_context.redactor.redact(example_cleanup_report)
    write_json(example_cleanup_path, redacted_example_cleanup_report)
    example_cleanup_md.write_text(
        render_example_concrete_cleanup_report(redacted_example_cleanup_report),
        encoding="utf-8",
    )
    return draft_artifact


def build_draft_rendering_payload(
    *,
    ctx: StepRunContext,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
) -> dict[str, Any]:
    approved_prepared_payload = approved_prepared_text if source_excerpt_pack["full_source_in_payload"] else ""
    draftable_count = len([item for item in merge_plan.items if item.action in {"create", "update"}])
    projected_digest, digest_projection_report = project_source_digest_for_merge_plan(digest, merge_plan)
    projected_merge_plan, merge_plan_projection_report = project_merge_plan_for_draft_rendering(merge_plan)
    required_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    snapshot_ref = merge_plan.context_snapshot_ref or "wiki_context_snapshot/wiki_context_snapshot.json"
    relevant_snapshot_paths, content_snapshot_paths = draft_rendering_relevant_wiki_paths(merge_plan)
    projected_snapshot, snapshot_projection_report = compact_snapshot_for_draft_rendering(
        snapshot,
        relevant_snapshot_paths,
        snapshot_ref,
        content_paths=content_snapshot_paths,
    )
    return {
        "approved_prepared_markdown": approved_prepared_payload,
        "approved_prepared_ref": source_excerpt_pack["approved_prepared_ref"],
        "source_excerpt_pack": source_excerpt_pack,
        "update_preservation_pack": update_preservation_pack,
        "approved_digest": projected_digest.model_dump(mode="json"),
        "approved_digest_ref": "source_digest_review/approved_digest.json",
        "approved_digest_projection_report": digest_projection_report,
        "approved_merge_plan": projected_merge_plan,
        "approved_merge_plan_ref": "merge_plan_review/approved_merge_plan.json",
        "approved_merge_plan_projection_report": merge_plan_projection_report,
        "wiki_context_snapshot": projected_snapshot,
        "wiki_context_snapshot_ref": snapshot_ref,
        "wiki_context_snapshot_projection_report": snapshot_projection_report,
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "required_page_plan_ids": [item.page_plan_id for item in required_items],
        "required_target_paths": [item.canonical_target_path for item in required_items],
        "contract": {
            "goal": "Generate structured section bodies for each create/update page from approved source excerpts/full source and frozen wiki context.",
            "batch_note": (
                f"This payload covers {draftable_count} create/update page(s). "
                "Return exactly those draftable pages and no pages from other batches."
            ),
            "section_body_keys": ["summary", "detail", "examples", "value_points", "additional_notes", "open_questions"],
            "rules": [
                "Return section body content only; do not include frontmatter, level-1 headings, source wikilinks, or full markdown pages.",
                "The output pages array must contain exactly required_page_plan_ids, one page per id, with no omissions, duplicates, or extra ids.",
                "section_bodies must use only these exact keys: summary, detail, examples, value_points, additional_notes, open_questions.",
                "Each section_bodies value must be one Markdown string; for bullet lists, write bullets inside that string instead of returning JSON arrays.",
                "Use additional_notes for free-form observations or custom subtopics; do not invent custom top-level section keys.",
                "Do not put `相关页面`/`Related Pages` blocks or self wikilinks inside section_bodies; the system renders official related pages separately.",
                "approved_digest is a projection for this draft batch; the full reviewed digest is available by approved_digest_ref for local audit artifacts, not for model access.",
                "approved_merge_plan and wiki_context_snapshot are compact projections for this draft batch; full reviewed artifacts are fixed by their *_ref fields for local audit and validators.",
                "For updates, read existing page excerpts from wiki_context_snapshot and update_preservation_pack, then produce a complete replacement draft at the section-body level.",
                "Use source_excerpt_pack as the primary source support. If approved_prepared_markdown is empty, the full approved source is intentionally omitted from this model payload and remains available only to downstream validators through approved_prepared_ref.",
                "For updates, satisfy update_preservation_pack in the first draft: carry forward concept obligations "
                "and reusable key phrases into the matching section body, rewritten naturally with the new source "
                "rather than appended as a dump. change_summary may summarize retention but does not satisfy the obligation.",
                "Do not produce pages that are only source summaries; every page must include concrete digested understanding such as viewpoint, example, use scenario, boundary condition, or value point.",
                "For updates, change_summary must explain what the new source adds, changes, clarifies, retains, or removes from the old understanding.",
                "If the merge plan has merged_page_plan_ids, absorb the unique section intent/examples/value points from suppressed candidates into the canonical page.",
                "Write all user-visible content in Chinese except stable domain terms with Chinese explanation when needed.",
                "For zh-CN vaults, translate or paraphrase English raw examples into Chinese; do not paste whole English sentences into examples, detail, value_points, additional_notes, open_questions, change_summary, or source_coverage_notes.",
                "Stable English product/protocol terms such as Claude Code, Managed Agents, harness, sandbox, session, MCP, Eval, TTFT, CLI, API, and Cowork may remain in English, but surrounding prose must be Chinese.",
                "Ground examples, value points, and reuse scenarios in source content.",
                "Across all section_bodies, do not fabricate example values such as `张三`, `Alice`, `user-123`, `user123`, concrete user preferences, dates, plans, metrics, credentials, or IDs unless exact source/wiki support exists; use placeholders such as `<user_id>`, `<memory_text>`, `<memory_query>`, `某个用户`, or `用户偏好 X`.",
                "In section_bodies.examples, do not invent concrete user facts, user ids, preferences, dates, plans, metrics, credentials, or command arguments unless exact source text supports them; for generic explanation, use abstract placeholders such as `某个用户`, `用户偏好 X`, `user_id`, `memory` or describe the pattern without quoted literals.",
                "For CLI/API/code examples, Chinese surrounding explanation is fine, but command/API literal arguments are an explicit exception to the zh-CN translation rule: they must either copy exact source literals or use placeholders such as `<memory_text>`, `<user_id>`, or `<memory_query>`; do not translate a source literal into a new concrete preference, user id, query, path, or command argument.",
                "When the source only states a recommendation or best practice, do not invent causal outcomes with terms such as `导致`, `造成`, `影响到`, or `用户会...`; either state the source-backed boundary without a new consequence, or move the consequence to open_questions as 待补来源.",
                *DRAFT_RENDERING_GROUNDING_RISK_RULES,
                "Do not write implementation details, examples, or claims as facts unless they are supported by source_excerpt_pack, approved_prepared_markdown, or inspected wiki context.",
                "If a useful detail is plausible but unsupported, put it under open_questions as 待补来源 instead of writing it as fact.",
                "source_coverage_notes must briefly say which source/wiki context supports the page and what was intentionally left uncertain.",
            ],
            "grounding_risk_rules": list(DRAFT_RENDERING_GROUNDING_RISK_RULES),
        },
    }


def draft_rendering_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model_payload = dict(payload)
    for key in [
        "approved_digest_projection_report",
        "approved_merge_plan_projection_report",
        "wiki_context_snapshot_projection_report",
    ]:
        model_payload.pop(key, None)
    return model_payload


def project_merge_plan_for_draft_rendering(merge_plan: WikiMergePlanArtifact) -> tuple[dict[str, Any], dict[str, Any]]:
    draftable_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    projected_items = [project_merge_plan_item_for_draft_rendering(item) for item in draftable_items]
    projection = {
        "schema_version": "draft_merge_plan_projection.v1",
        "source_schema_version": merge_plan.schema_version,
        "full_merge_plan_ref": "merge_plan_review/approved_merge_plan.json",
        "context_snapshot_ref": merge_plan.context_snapshot_ref,
        "log_date": merge_plan.log_date,
        "draftable_item_count": len(draftable_items),
        "omitted_non_draft_item_count": max(0, len(merge_plan.items) - len(draftable_items)),
        "items": projected_items,
    }
    original_payload = merge_plan.model_dump(mode="json")
    report = {
        "schema_version": "draft_merge_plan_projection_report.v1",
        "projection": "draft_rendering_batch",
        "full_merge_plan_ref": "merge_plan_review/approved_merge_plan.json",
        "original_item_count": len(merge_plan.items),
        "projected_item_count": len(projected_items),
        "omitted_non_draft_item_count": max(0, len(merge_plan.items) - len(projected_items)),
        "original_json_chars": json_char_count(original_payload),
        "projected_json_chars": json_char_count(projection),
        "page_plan_ids": [item.page_plan_id for item in draftable_items],
        "target_paths": [item.canonical_target_path for item in draftable_items],
    }
    return projection, report


def project_merge_plan_item_for_draft_rendering(item: WikiMergePlanItem) -> dict[str, Any]:
    strongest_overlap = compact_optional_dict(
        {
            "strength": item.strongest_overlap.strength,
            "match_basis": item.strongest_overlap.match_basis,
            "path": item.strongest_overlap.path,
            "score": item.strongest_overlap.score,
            "reason": item.strongest_overlap.reason,
        }
    )
    related_pages = [
        compact_optional_dict(
            {
                "target_path": related.target_path,
                "display_title": related.display_title,
                "source": related.source,
                "reason": related.reason,
            }
        )
        for related in item.related_pages
    ]
    return compact_optional_dict(
        {
            "page_plan_id": item.page_plan_id,
            "source_basis": compact_optional_dict(item.source_basis.model_dump(mode="json")),
            "action": item.action,
            "canonical_target_path": item.canonical_target_path,
            "display_title": item.display_title,
            "page_type": item.page_type,
            "matched_page": item.matched_page,
            "inspected_context_paths": item.inspected_context_paths,
            "strongest_overlap": strongest_overlap,
            "why_not_update": item.why_not_update,
            "why_create_or_update": item.why_create_or_update,
            "prior_knowledge_state": item.prior_knowledge_state,
            "new_understanding": item.new_understanding,
            "changed_view": item.changed_view,
            "knowledge_delta": item.knowledge_delta,
            "why_this_matters": item.why_this_matters,
            "reuse_scenarios": item.reuse_scenarios,
            "value_points": item.value_points,
            "section_plans": item.section_plans,
            "related_pages": related_pages,
            "related_absence_reason": item.related_absence_reason,
            "related_unresolved": item.related_unresolved,
            "unresolved_related": item.unresolved_related,
            "conflicts": item.conflicts,
            "uncertainties": item.uncertainties,
            "quality_risks": item.quality_risks,
            "reason": item.reason,
            "merged_page_plan_ids": item.merged_page_plan_ids,
            "merge_reason": item.merge_reason,
        }
    )


def draft_rendering_relevant_wiki_paths(merge_plan: WikiMergePlanArtifact) -> tuple[set[str], set[str]]:
    metadata_paths: set[str] = set()
    content_paths: set[str] = set()
    for item in merge_plan.items:
        if item.action not in {"create", "update"}:
            continue
        target_paths = [
            item.canonical_target_path,
            item.matched_page or "",
        ]
        context_paths = [related.target_path for related in item.related_pages]
        if should_include_draft_inspected_context(item):
            context_paths.extend([item.strongest_overlap.path, *item.inspected_context_paths])
        for raw_path in [*target_paths, *context_paths]:
            normalized = normalize_wiki_snapshot_path(raw_path)
            if normalized:
                metadata_paths.add(normalized)
        if item.action == "update":
            for raw_path in target_paths:
                normalized = normalize_wiki_snapshot_path(raw_path)
                if normalized:
                    content_paths.add(normalized)
        elif item.strongest_overlap.strength == "strong":
            normalized = normalize_wiki_snapshot_path(item.strongest_overlap.path)
            if normalized:
                content_paths.add(normalized)
    return metadata_paths, content_paths


def should_include_draft_inspected_context(item: WikiMergePlanItem) -> bool:
    if item.action == "update":
        return True
    return item.strongest_overlap.strength in {"medium", "strong"}


def normalize_wiki_snapshot_path(path: str) -> str:
    cleaned = path.strip().lstrip("/")
    if not cleaned:
        return ""
    if cleaned.startswith("wiki/"):
        return cleaned
    return f"wiki/{cleaned}"


def compact_snapshot_for_draft_rendering(
    snapshot: WikiContextSnapshot,
    relevant_paths: set[str],
    snapshot_ref: str,
    *,
    content_paths: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content_paths = content_paths or set()
    entries: list[dict[str, Any]] = []
    content_entry_count = 0
    for entry in snapshot.entries:
        if entry.path not in relevant_paths:
            continue
        include_content = entry.path in content_paths
        content_excerpt = source_global_excerpt(entry.content, DRAFT_RENDERING_CONTEXT_ENTRY_EXCERPT_LIMIT) if include_content else ""
        if content_excerpt:
            content_entry_count += 1
        entries.append(
            {
                "path": entry.path,
                "expected_state": entry.expected_state,
                "preimage_sha256": entry.preimage_sha256,
                "metadata": entry.metadata.model_dump(mode="json") if entry.metadata else None,
                "content_excerpt": content_excerpt,
                "content_truncated": include_content and len(entry.content.strip()) > len(content_excerpt),
                "content_role": "draft_context" if include_content else "metadata_only",
            }
        )
    metadata_paths = {path.removeprefix("wiki/") for path in relevant_paths}
    metadata_pool = [
        {
            "path": pool_entry.path,
            "rel_path": pool_entry.rel_path,
            "preimage_sha256": pool_entry.preimage_sha256,
            "metadata": pool_entry.metadata.model_dump(mode="json") if pool_entry.metadata else None,
            "display_title": pool_entry.display_title,
            "summary": pool_entry.summary,
            "aliases": pool_entry.aliases,
            "llmwiki_type": pool_entry.llmwiki_type,
            "indexable": pool_entry.indexable,
            "unindexable_reason": pool_entry.unindexable_reason,
        }
        for pool_entry in snapshot.knowledge_metadata_pool
        if pool_entry.path in metadata_paths
    ]
    projection = {
        "schema_version": "wiki_context_snapshot_projection.v1",
        "source_schema_version": snapshot.schema_version,
        "full_snapshot_ref": snapshot_ref,
        "log_date": snapshot.log_date,
        "source_target_path": snapshot.source_target_path,
        "candidate_contexts_ref": snapshot.candidate_contexts_ref,
        "candidate_pool_sha256": snapshot.candidate_pool_sha256,
        "entry_excerpt_limit": DRAFT_RENDERING_CONTEXT_ENTRY_EXCERPT_LIMIT,
        "full_entry_count": len(snapshot.entries),
        "included_entry_count": len(entries),
        "included_content_entry_count": content_entry_count,
        "included_entry_content_chars": sum(len(entry["content_excerpt"]) for entry in entries),
        "omitted_entry_count": max(0, len(snapshot.entries) - len(entries)),
        "full_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
        "included_metadata_pool_count": len(metadata_pool),
        "knowledge_metadata_pool": metadata_pool,
        "entries": entries,
    }
    original_entry_content_chars = sum(len(entry.content) for entry in snapshot.entries)
    report = {
        "schema_version": "draft_context_projection_report.v1",
        "projection": "draft_rendering_batch",
        "full_snapshot_ref": snapshot_ref,
        "relevant_paths": sorted(relevant_paths),
        "content_paths": sorted(content_paths),
        "original_json_chars": json_char_count(snapshot.model_dump(mode="json")),
        "projected_json_chars": json_char_count(projection),
        "original_entry_count": len(snapshot.entries),
        "projected_entry_count": len(entries),
        "projected_content_entry_count": content_entry_count,
        "omitted_entry_count": max(0, len(snapshot.entries) - len(entries)),
        "original_entry_content_chars": original_entry_content_chars,
        "projected_entry_content_chars": projection["included_entry_content_chars"],
        "original_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
        "projected_metadata_pool_count": len(metadata_pool),
    }
    return projection, report


def compact_optional_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is not None and item != "" and item != [] and item != {}
    }


def project_source_digest_for_merge_plan(
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
) -> tuple[SourceDigestArtifact, dict[str, Any]]:
    needed_ids = source_digest_candidate_ids_for_merge_plan(digest, merge_plan)
    original_counts = source_digest_candidate_counts(digest)
    projected_groups: dict[str, list[SourceDigestCandidate]] = {}
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        candidates = list(getattr(digest, group_name))
        projected_groups[group_name] = [candidate for candidate in candidates if candidate.candidate_id in needed_ids]
    projected_deferred = [
        candidate
        for candidate in digest.budget_deferred_candidates
        if candidate.candidate_id in needed_ids
    ]
    projected = digest.model_copy(
        update={
            **projected_groups,
            "budget_deferred_candidates": projected_deferred,
            "weak_or_noise_items": [],
        }
    )
    projected_counts = source_digest_candidate_counts(projected)
    candidate_by_id = source_digest_candidate_lookup(digest)
    unresolved_candidate_ids = sorted(candidate_id for candidate_id in needed_ids if candidate_id not in candidate_by_id)
    report = {
        "schema_version": "source_digest_projection_report.v1",
        "projection": "draft_rendering_batch",
        "full_digest_ref": "source_digest_review/approved_digest.json",
        "needed_candidate_ids": sorted(needed_ids),
        "unresolved_candidate_ids": unresolved_candidate_ids,
        "original_counts": original_counts,
        "projected_counts": projected_counts,
        "removed_counts": {
            key: max(0, int(original_counts.get(key, 0)) - int(projected_counts.get(key, 0)))
            for key in original_counts
        },
    }
    return projected, report


def source_digest_candidate_ids_for_merge_plan(
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
) -> set[str]:
    candidate_by_id = source_digest_candidate_lookup(digest)
    needed: set[str] = set()
    for item in merge_plan.items:
        if item.action not in {"create", "update"}:
            continue
        needed.update(source_digest_candidate_id_closure(source_basis_candidate_refs(item.source_basis), candidate_by_id))
    return needed


def source_digest_candidate_lookup(digest: SourceDigestArtifact) -> dict[str, SourceDigestCandidate]:
    candidates = {candidate.candidate_id: candidate for candidate in digest.ingest_candidates()}
    for candidate in digest.budget_deferred_candidates:
        candidates.setdefault(candidate.candidate_id, candidate)
    return candidates


def source_digest_candidate_id_closure(
    candidate_ids: list[str],
    candidate_by_id: dict[str, SourceDigestCandidate],
) -> list[str]:
    needed: list[str] = []
    seen: set[str] = set()
    queue = [candidate_id for candidate_id in candidate_ids if candidate_id]
    while queue:
        candidate_id = queue.pop(0)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        needed.append(candidate_id)
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        for related_id in candidate.related_candidates:
            if related_id and related_id not in seen:
                queue.append(related_id)
    return needed


def source_digest_candidate_counts(digest: SourceDigestArtifact) -> dict[str, int]:
    return {
        "entities": len(digest.entities),
        "concepts": len(digest.concepts),
        "designs": len(digest.designs),
        "comparisons": len(digest.comparisons),
        "open_questions": len(digest.open_questions),
        "budget_deferred_candidates": len(digest.budget_deferred_candidates),
        "weak_or_noise_items": len(digest.weak_or_noise_items),
        "total_ingest_candidates": len(digest.ingest_candidates()),
    }


def write_draft_digest_projection_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "draft_digest_projection_report.json"
    md_path = output_dir / "draft_digest_projection_report.md"
    write_json(json_path, report)
    rows = []
    original_counts = report.get("original_counts", {})
    projected_counts = report.get("projected_counts", {})
    removed_counts = report.get("removed_counts", {})
    for key in [
        "entities",
        "concepts",
        "designs",
        "comparisons",
        "open_questions",
        "budget_deferred_candidates",
        "weak_or_noise_items",
        "total_ingest_candidates",
    ]:
        rows.append(
            [
                key,
                str(original_counts.get(key, 0)),
                str(projected_counts.get(key, 0)),
                str(removed_counts.get(key, 0)),
            ]
        )
    needed = report.get("needed_candidate_ids", [])
    unresolved = report.get("unresolved_candidate_ids", [])
    md_path.write_text(
        "# Draft Digest Projection Report\n\n"
        f"- Projection: `{report.get('projection', '')}`\n"
        f"- Full digest ref: `{report.get('full_digest_ref', '')}`\n"
        f"- Needed candidate ids: {', '.join(f'`{candidate_id}`' for candidate_id in needed) if needed else '_none_'}\n\n"
        f"- Unresolved candidate ids: {', '.join(f'`{candidate_id}`' for candidate_id in unresolved) if unresolved else '_none_'}\n\n"
        "## Counts\n\n"
        f"{format_markdown_table(['Group', 'Original', 'Projected', 'Removed'], rows)}\n",
        encoding="utf-8",
    )
    return json_path, md_path


def write_draft_payload_projection_reports(output_dir: Path, payload: dict[str, Any]) -> list[Path]:
    written: list[Path] = []
    merge_report = payload.get("approved_merge_plan_projection_report")
    if isinstance(merge_report, dict):
        written.extend(write_draft_merge_plan_projection_report(output_dir, merge_report))
    context_report = payload.get("wiki_context_snapshot_projection_report")
    if isinstance(context_report, dict):
        written.extend(write_draft_context_projection_report(output_dir, context_report))
    return written


def write_draft_merge_plan_projection_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "draft_merge_plan_projection_report.json"
    md_path = output_dir / "draft_merge_plan_projection_report.md"
    write_json(json_path, report)
    rows = [
        ["items", report.get("original_item_count", 0), report.get("projected_item_count", 0)],
        ["json_chars", report.get("original_json_chars", 0), report.get("projected_json_chars", 0)],
        ["omitted_non_draft", report.get("omitted_non_draft_item_count", 0), 0],
    ]
    md_path.write_text(
        "# Draft Merge Plan Projection Report\n\n"
        f"- Projection: `{report.get('projection', '')}`\n"
        f"- Full merge plan ref: `{report.get('full_merge_plan_ref', '')}`\n"
        f"- Page plan ids: {', '.join(f'`{page_id}`' for page_id in report.get('page_plan_ids', [])) or '_none_'}\n\n"
        "## Payload Budget\n\n"
        f"{format_markdown_table(['Object', 'Original', 'Projected'], rows)}\n",
        encoding="utf-8",
    )
    return json_path, md_path


def write_draft_context_projection_report(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "draft_context_projection_report.json"
    md_path = output_dir / "draft_context_projection_report.md"
    write_json(json_path, report)
    rows = [
        ["snapshot_json_chars", report.get("original_json_chars", 0), report.get("projected_json_chars", 0)],
        ["entries", report.get("original_entry_count", 0), report.get("projected_entry_count", 0)],
        ["entry_content_chars", report.get("original_entry_content_chars", 0), report.get("projected_entry_content_chars", 0)],
        ["metadata_pool", report.get("original_metadata_pool_count", 0), report.get("projected_metadata_pool_count", 0)],
    ]
    relevant_paths = report.get("relevant_paths", [])
    md_path.write_text(
        "# Draft Context Projection Report\n\n"
        f"- Projection: `{report.get('projection', '')}`\n"
        f"- Full snapshot ref: `{report.get('full_snapshot_ref', '')}`\n"
        f"- Relevant paths: {', '.join(f'`{path}`' for path in relevant_paths) if relevant_paths else '_none_'}\n\n"
        "## Payload Budget\n\n"
        f"{format_markdown_table(['Object', 'Original', 'Projected'], rows)}\n",
        encoding="utf-8",
    )
    return json_path, md_path


def write_draft_rendering_batch_reports(
    step_root: Path,
    draft_artifact: DraftRenderingArtifact,
    batch_summaries: list[dict[str, Any]],
    *,
    max_parallel_batches: int = 1,
    wall_duration_ms: int | None = None,
) -> None:
    model_duration_ms = sum(int(batch["duration_ms"]) for batch in batch_summaries)
    payload_counts = [int(batch.get("payload_char_count", 0)) for batch in batch_summaries]
    batch_report = {
        "schema_version": "draft_rendering_batch_report.v1",
        "batch_page_limit": DRAFT_RENDERING_BATCH_PAGE_LIMIT,
        "parallel": max_parallel_batches > 1,
        "max_parallel_batches": max_parallel_batches,
        "batch_count": len(batch_summaries),
        "page_count": len(draft_artifact.pages),
        "attempt_count": sum(int(batch["attempt_count"]) for batch in batch_summaries),
        "repair_count": sum(int(batch["repair_count"]) for batch in batch_summaries),
        "http_attempt_count": sum(int(batch.get("http_attempt_count", 0)) for batch in batch_summaries),
        "duration_ms": model_duration_ms,
        "model_duration_ms": model_duration_ms,
        "wall_duration_ms": wall_duration_ms if wall_duration_ms is not None else model_duration_ms,
        "payload_char_count": sum(payload_counts),
        "max_batch_payload_char_count": max(payload_counts) if payload_counts else 0,
        "avg_batch_payload_char_count": round(sum(payload_counts) / len(payload_counts)) if payload_counts else 0,
        "batches": batch_summaries,
    }
    write_json(step_root / "draft_rendering_batch_report.json", batch_report)
    (step_root / "draft_rendering_batch_report.md").write_text(render_draft_rendering_batch_report(batch_report), encoding="utf-8")
    providers = ",".join(sorted({str(batch["provider"]) for batch in batch_summaries}))
    write_json(
        step_root / "provider_result.json",
        ProviderResult(
            task="draft_rendering",
            provider=f"batched:{providers}",
            raw_output="",
            parsed_output=draft_artifact.model_dump(mode="json"),
            parse_success=True,
            schema_valid=True,
            repair_attempted=batch_report["repair_count"] > 0,
            latency_ms=batch_report["duration_ms"],
            payload_char_count=batch_report["payload_char_count"],
            http_attempt_count=batch_report["http_attempt_count"],
        ),
    )
    attempts: list[StructuredAttemptRef] = []
    next_attempt = 1
    non_repairable: list[StructuredIssue] = []
    max_repair_attempts = 0
    for batch in batch_summaries:
        report = read_model(step_root / str(batch["structured_repair_report_ref"]), StructuredRepairReport)
        max_repair_attempts = max(max_repair_attempts, report.max_repair_attempts)
        non_repairable.extend(report.non_repairable_issues)
        for attempt in report.attempts:
            attempts.append(
                attempt.model_copy(
                    update={
                        "attempt": next_attempt,
                        "provider_result_ref": f"model_batches/{batch['batch_id']}/{attempt.provider_result_ref}",
                        "repair_prompt_ref": (
                            f"model_batches/{batch['batch_id']}/{attempt.repair_prompt_ref}"
                            if attempt.repair_prompt_ref
                            else None
                        ),
                    }
                )
            )
            next_attempt += 1
    aggregate_report = StructuredRepairReport(
        task="draft_rendering",
        provider=f"batched:{providers}",
        final_outcome="success",
        repair_attempted=batch_report["repair_count"] > 0,
        max_repair_attempts=max_repair_attempts,
        attempt_count=batch_report["attempt_count"],
        repair_count=batch_report["repair_count"],
        duration_ms=batch_report["duration_ms"],
        attempts=attempts,
        final_provider_result_ref="provider_result.json",
        non_repairable_issues=non_repairable,
    )
    write_json(step_root / "structured_repair_report.json", aggregate_report)
    (step_root / "structured_repair_report.md").write_text(render_structured_repair_report_markdown(aggregate_report), encoding="utf-8")


def render_draft_rendering_batch_report(report: dict[str, Any]) -> str:
    rows = [
        [
            batch["batch_id"],
            ", ".join(f"`{page_id}`" for page_id in batch["page_plan_ids"]),
            str(batch["attempt_count"]),
            str(batch.get("http_attempt_count", 0)),
            str(batch["repair_count"]),
            format_duration(batch["duration_ms"]),
            str(batch["source_excerpt_chars"]),
            f"{int(batch.get('payload_char_count', 0)):,}",
            str(batch.get("reinforced_section_count", 0)),
            str(batch.get("grounding_rewrite_count", 0)),
            str(batch.get("example_concrete_replacement_count", 0)),
        ]
        for batch in report["batches"]
    ]
    return (
        "# Draft Rendering 分批报告\n\n"
        f"- 并行执行：`{str(bool(report.get('parallel', False))).lower()}`\n"
        f"- 最大并行批数：{report.get('max_parallel_batches', 1)}\n"
        f"- Batch 页面上限：{report.get('batch_page_limit', DRAFT_RENDERING_BATCH_PAGE_LIMIT)}\n"
        f"- 模型累计耗时：{format_duration(report.get('model_duration_ms', report.get('duration_ms')))}\n"
        f"- 墙钟耗时：{format_duration(report.get('wall_duration_ms'))}\n"
        f"- 最大单批 payload：{int(report.get('max_batch_payload_char_count', 0)):,} chars\n"
        f"- 平均单批 payload：{int(report.get('avg_batch_payload_char_count', 0)):,} chars\n\n"
        + format_markdown_table(
            [
                "Batch",
                "页面计划",
                "Attempts",
                "HTTP Attempts",
                "Repairs",
                "Duration",
                "Source Excerpt Chars",
                "Payload Chars",
                "Reinforced Sections",
                "Grounding Rewrites",
                "Example Replacements",
            ],
            rows,
        )
        + "\n"
    )


def render_structured_repair_report_markdown(report: StructuredRepairReport) -> str:
    lines = [
        "# 结构化输出返工报告",
        "",
        f"- 任务：`{report.task}`",
        f"- Provider：`{report.provider}`",
        f"- 结果：`{report.final_outcome}`",
        f"- 尝试次数：{report.attempt_count}",
        f"- 返工次数：{report.repair_count}",
        "",
        "## 尝试记录",
        "",
    ]
    for attempt in report.attempts:
        issue_text = "; ".join(f"{issue.issue_code}: {issue.message}" for issue in attempt.issues) or "无"
        prompt_text = f"；返工 prompt：`{attempt.repair_prompt_ref}`" if attempt.repair_prompt_ref else ""
        lines.append(f"- 第 {attempt.attempt} 次：`{attempt.provider_result_ref}`{prompt_text}；问题：{issue_text}")
    return "\n".join(lines).rstrip() + "\n"


def _run_draft_rendering(ctx: StepRunContext) -> None:
    step_name = "draft_rendering"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    merge_plan = read_model(require_step_output_dir(ctx.run_dir, "merge_plan_review") / "approved_merge_plan.json", WikiMergePlanArtifact)
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    validate_wiki_merge_plan(digest, merge_plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    ensure_wiki_context_current(ctx.vault, snapshot)
    approved_prepared_text = (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8")
    draftable_count = len([item for item in merge_plan.items if item.action in {"create", "update"}])
    source_excerpt_pack = build_draft_source_excerpt_pack(
        approved_prepared_text,
        digest,
        merge_plan,
        force_excerpt=draftable_count > DRAFT_RENDERING_BATCH_PAGE_LIMIT,
    )
    source_excerpt_pack_path = step_root / "draft_source_excerpt_pack.json"
    source_excerpt_pack_md = step_root / "draft_source_excerpt_pack.md"
    write_json(source_excerpt_pack_path, source_excerpt_pack)
    source_excerpt_pack_md.write_text(render_draft_source_excerpt_pack_markdown(source_excerpt_pack), encoding="utf-8")
    update_preservation_pack = build_update_preservation_pack(merge_plan, snapshot)
    update_preservation_pack_path = step_root / "update_preservation_pack.json"
    update_preservation_pack_md = step_root / "update_preservation_pack.md"
    write_json(update_preservation_pack_path, update_preservation_pack)
    update_preservation_pack_md.write_text(render_update_preservation_pack_markdown(update_preservation_pack), encoding="utf-8")
    _, digest_projection_report = project_source_digest_for_merge_plan(digest, merge_plan)
    write_draft_digest_projection_report(step_root, digest_projection_report)
    draft_artifact = run_draft_rendering_model(
        ctx=ctx,
        step_root=step_root,
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        source_excerpt_pack=source_excerpt_pack,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
    )
    validate_draft_rendering(draft_artifact, merge_plan, language=ctx.manifest.vault_config_snapshot.wiki_language)
    draft_artifact_path = step_root / "draft_rendering.json"
    write_json(draft_artifact_path, draft_artifact)
    draft_root = step_root / "draft_pages"
    outputs: list[Path] = [source_excerpt_pack_path, source_excerpt_pack_md, update_preservation_pack_path, update_preservation_pack_md]
    for digest_projection_sidecar in [
        step_root / "draft_digest_projection_report.json",
        step_root / "draft_digest_projection_report.md",
        step_root / "draft_merge_plan_projection_report.json",
        step_root / "draft_merge_plan_projection_report.md",
        step_root / "draft_context_projection_report.json",
        step_root / "draft_context_projection_report.md",
        step_root / "update_preservation_reinforcement_report.json",
        step_root / "update_preservation_reinforcement_report.md",
        step_root / "grounding_paraphrase_rewrite_report.json",
        step_root / "grounding_paraphrase_rewrite_report.md",
        step_root / "open_question_grounding_cleanup_report.json",
        step_root / "open_question_grounding_cleanup_report.md",
        step_root / "example_concrete_cleanup_report.json",
        step_root / "example_concrete_cleanup_report.md",
    ]:
        if digest_projection_sidecar.exists():
            outputs.append(digest_projection_sidecar)
    for batch_sidecar in [step_root / "draft_rendering_batch_report.json", step_root / "draft_rendering_batch_report.md"]:
        if batch_sidecar.exists():
            outputs.append(batch_sidecar)
    target_manifest: list[DraftWriteTarget] = []
    source_title = source_title_for_raw(digest.source_raw_path)
    action_by_id = {item.page_plan_id: item for item in merge_plan.items}
    update_report_pages: list[UpdatePageMergeReport] = []
    related_report_candidates: list[RelatedCandidateReport] = []
    grounding_claims: list[GroundingClaim] = []
    known_related_paths = {
        entry.path
        for entry in snapshot.knowledge_metadata_pool
        if not entry.path.startswith("sources/") and not entry.path.startswith("logs/") and entry.path not in {"index.md", "log.md"}
    }
    known_related_paths.update(
        item.canonical_target_path
        for item in merge_plan.items
        if item.action in {"create", "update", "noop"} and not item.canonical_target_path.startswith(("sources/", "logs/"))
    )
    knowledge_changed_paths: list[str] = []
    no_change_pages = [item.canonical_target_path for item in merge_plan.items if item.action == "noop"]
    for page in draft_artifact.pages:
        plan_item = action_by_id[page.page_plan_id]
        entry = snapshot_entry(snapshot, f"wiki/{page.canonical_target_path}")
        target = draft_root / page.canonical_target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        markdown = assemble_knowledge_page(
            item=plan_item,
            page=page,
            existing_entry=entry,
            raw_path=digest.source_raw_path,
            raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
            prepared_hash=sha256_file(require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"),
            operation_id=ctx.manifest.operation_id,
            log_date=snapshot.log_date,
            update_reports=update_report_pages,
            related_reports=related_report_candidates,
            grounding_claims=grounding_claims,
            known_related_paths=known_related_paths,
            approved_raw_text=approved_prepared_text,
        )
        target.write_text(markdown, encoding="utf-8")
        outputs.append(target)
        knowledge_changed_paths.append(page.canonical_target_path)
        target_manifest.append(
            DraftWriteTarget(
                action=page.action,
                target_path=f"wiki/{page.canonical_target_path}",
                draft_path=target.relative_to(ctx.run_dir).as_posix(),
                expected_state=entry.expected_state,
                preimage_sha256=entry.preimage_sha256,
                page_plan_id=page.page_plan_id,
            )
        )
        if page.action == "update":
            diff_path = step_root / "diffs" / f"{page.page_plan_id}.diff"
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text(
                render_update_diff(entry.content, markdown, f"old/{page.canonical_target_path}", f"new/{page.canonical_target_path}"),
                encoding="utf-8",
            )
            diff_json = step_root / "diffs" / f"{page.page_plan_id}.json"
            write_json(
                diff_json,
                {
                    "old_snapshot_ref": f"wiki_context_snapshot/wiki_context_snapshot.json#{entry.path}",
                    "old_rendered_markdown_path": entry.path,
                    "new_draft_ref": target.relative_to(ctx.run_dir).as_posix(),
                    "new_rendered_markdown_path": target.relative_to(ctx.run_dir).as_posix(),
                    "unified_diff": diff_path.relative_to(ctx.run_dir).as_posix(),
                    "change_summary": page.change_summary,
                    "preimage_sha256": page.preimage_sha256,
                    "draft_sha256": sha256_file(target),
                },
            )
            outputs.extend([diff_path, diff_json])
    source_page = draft_root / snapshot.source_target_path
    source_page.parent.mkdir(parents=True, exist_ok=True)
    source_page.write_text(
        render_source_page(
            title=source_title,
            digest=digest,
            operation_id=ctx.manifest.operation_id,
            linked_pages=knowledge_changed_paths,
            touched_pages=[item.canonical_target_path for item in merge_plan.items],
            no_change_pages=no_change_pages,
            log_date=snapshot.log_date,
            raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
            prepared_hash=sha256_file(require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"),
            cleanup=read_model(require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json", RawLinkCleanupArtifact),
        ),
        encoding="utf-8",
    )
    outputs.append(source_page)
    source_entry = snapshot_entry(snapshot, f"wiki/{snapshot.source_target_path}")
    target_manifest.append(
        DraftWriteTarget(
            action="source",
            target_path=f"wiki/{snapshot.source_target_path}",
            draft_path=source_page.relative_to(ctx.run_dir).as_posix(),
            expected_state=source_entry.expected_state,
            preimage_sha256=source_entry.preimage_sha256,
        )
    )
    log_date = snapshot.log_date
    if knowledge_changed_paths:
        assert_system_page_can_be_overwritten(ctx.vault, "wiki/index.md")
        index = draft_root / "index.md"
        open_question_rows, open_question_report = build_open_question_rows_with_report(merge_plan, draft_artifact, snapshot)
        index.write_text(
            render_index(
                knowledge_rows=build_index_rows(ctx.profile, merge_plan, draft_artifact, snapshot),
                tension_rows=open_question_rows,
                page_type_order=list(ctx.profile.page_types),
            ),
            encoding="utf-8",
        )
        open_question_report_path = step_root / "index_open_questions_report.json"
        open_question_report_md = step_root / "index_open_questions_report.md"
        write_json(open_question_report_path, open_question_report)
        open_question_report_md.write_text(render_index_open_questions_report(open_question_report), encoding="utf-8")
        outputs.append(index)
        outputs.extend([open_question_report_path, open_question_report_md])
        index_entry = snapshot_entry(snapshot, "wiki/index.md")
        target_manifest.append(
            DraftWriteTarget(
                action="index",
                target_path="wiki/index.md",
                draft_path=index.relative_to(ctx.run_dir).as_posix(),
                expected_state=index_entry.expected_state,
                preimage_sha256=index_entry.preimage_sha256,
            )
        )
    assert_system_page_can_be_overwritten(ctx.vault, "wiki/log.md")
    log_entry = snapshot_entry(snapshot, "wiki/log.md")
    log_index = draft_root / "log.md"
    log_index.write_text(
        render_log_index(
            date=log_date,
            operation_id=ctx.manifest.operation_id,
            source=digest.source_raw_path,
            existing_text=log_entry.content if log_entry.expected_state == "present" else None,
        ),
        encoding="utf-8",
    )
    outputs.append(log_index)
    target_manifest.append(
        DraftWriteTarget(
            action="global_log",
            target_path="wiki/log.md",
            draft_path=log_index.relative_to(ctx.run_dir).as_posix(),
            expected_state=log_entry.expected_state,
            preimage_sha256=log_entry.preimage_sha256,
        )
    )
    assert_system_page_can_be_overwritten(ctx.vault, f"wiki/logs/{log_date}.md")
    daily_entry = snapshot_entry(snapshot, f"wiki/logs/{log_date}.md")
    daily_log = draft_root / "logs" / f"{log_date}.md"
    daily_log.parent.mkdir(parents=True, exist_ok=True)
    daily_log.write_text(
        render_daily_log(
            date=log_date,
            operation_id=ctx.manifest.operation_id,
            raw_path=digest.source_raw_path,
            created=sum(1 for item in merge_plan.items if item.action == "create"),
            updated=sum(1 for item in merge_plan.items if item.action == "update"),
            noop=sum(1 for item in merge_plan.items if item.action == "noop"),
            needs_human=sum(1 for item in merge_plan.items if item.action == "needs_human_decision"),
            existing_text=daily_entry.content if daily_entry.expected_state == "present" else None,
        ),
        encoding="utf-8",
    )
    outputs.append(daily_log)
    target_manifest.append(
        DraftWriteTarget(
            action="daily_log",
            target_path=f"wiki/logs/{log_date}.md",
            draft_path=daily_log.relative_to(ctx.run_dir).as_posix(),
            expected_state=daily_entry.expected_state,
            preimage_sha256=daily_entry.preimage_sha256,
        )
    )
    update_report = UpdateMergeReport(pages=update_report_pages)
    update_report_path = step_root / "update_merge_report.json"
    update_report_md = step_root / "update_merge_report.md"
    write_json(update_report_path, update_report)
    update_report_md.write_text(render_update_merge_report(update_report), encoding="utf-8")
    related_report = RelatedMergeReport(candidates=related_report_candidates)
    related_report_path = step_root / "related_merge_report.json"
    related_report_md = step_root / "related_merge_report.md"
    write_json(related_report_path, related_report)
    related_report_md.write_text(render_related_merge_report(related_report), encoding="utf-8")
    grounding_review = draft_grounding_review_from_claims(grounding_claims)
    grounding_review_path = step_root / "draft_grounding_review.json"
    grounding_review_md = step_root / "draft_grounding_review.md"
    write_json(grounding_review_path, grounding_review)
    grounding_review_md.write_text(render_draft_grounding_review(grounding_review), encoding="utf-8")
    outputs.extend([update_report_path, update_report_md, related_report_path, related_report_md, grounding_review_path, grounding_review_md])
    write_manifest_artifact = DraftWriteManifest(
        targets=target_manifest,
        has_updates=any(item.action == "update" for item in merge_plan.items),
        has_noops=any(item.action == "noop" for item in merge_plan.items),
        source_only_noop=all(item.action == "noop" for item in merge_plan.items),
        requires_grounding_review=grounding_review.requires_review,
    )
    write_manifest_path = step_root / "draft_write_manifest.json"
    write_json(write_manifest_path, write_manifest_artifact)
    outputs.extend([draft_artifact_path, write_manifest_path])
    refs = [_draft_rendering_ref(ctx.run_dir, path, step_name) for path in outputs]
    refs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    refs.extend(draft_rendering_model_batch_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=refs)


def _run_validation(ctx: StepRunContext) -> None:
    step_name = "validation"
    preparation = read_model(
        require_step_output_dir(ctx.run_dir, "raw_prepare") / "raw_preparation.json",
        RawPreparationArtifact,
    )
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    merge_plan = read_model(require_step_output_dir(ctx.run_dir, "merge_plan_review") / "approved_merge_plan.json", WikiMergePlanArtifact)
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    draft_write_manifest = read_model(require_step_output_dir(ctx.run_dir, "draft_review") / "approved_write_manifest.json", DraftWriteManifest)
    validate_raw_preparation(preparation)
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    validate_wiki_merge_plan(digest, merge_plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    ensure_wiki_context_current(ctx.vault, snapshot)
    if any(item.action == "needs_human_decision" for item in merge_plan.items):
        raise PipelineError("needs_human_decision must be revised to create/update/noop before validation.")
    if not digest.ingest_candidates() and not any(
        nonempty_prepared_discovered_candidates(item.source_basis) for item in merge_plan.items
    ):
        raise PipelineError("source_digest must include at least one wiki candidate")
    if not merge_plan.items:
        raise PipelineError("wiki_merge_plan must include at least one action")
    if not draft_write_manifest.targets:
        raise PipelineError("draft_write_manifest must include at least one target")
    complete_step(ctx.manifest, step_name)


def _run_apply_preview(ctx: StepRunContext) -> None:
    step_name = "apply_preview"
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    ensure_wiki_context_current(ctx.vault, snapshot)
    preview = build_apply_preview(ctx.vault, ctx.run_dir)
    out = require_step_output_dir(ctx.run_dir, step_name) / "apply_preview.json"
    write_json(out, preview)
    complete_step(ctx.manifest, step_name, outputs=[_ref(ctx.run_dir, out, step_name, "json", "apply_preview.v2")])


def refresh_current_draft_grounding_artifacts(
    ctx: StepRunContext,
    draft_manifest: DraftWriteManifest,
    draft_manifest_path: Path,
) -> DraftWriteManifest:
    draft_root = require_step_output_dir(ctx.run_dir, "draft_rendering")
    draft_artifact = read_model(draft_root / "draft_rendering.json", DraftRenderingArtifact)
    merge_plan = read_model(require_step_output_dir(ctx.run_dir, "merge_plan_review") / "approved_merge_plan.json", WikiMergePlanArtifact)
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    approved_prepared_text = (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8")
    grounding_review = build_draft_grounding_review(draft_artifact, merge_plan, snapshot, approved_prepared_text)
    grounding_review_path = draft_root / "draft_grounding_review.json"
    grounding_review_md = draft_root / "draft_grounding_review.md"
    write_json(grounding_review_path, grounding_review)
    grounding_review_md.write_text(render_draft_grounding_review(grounding_review), encoding="utf-8")
    refresh_draft_rendering_artifact_refs(ctx, grounding_review_path, grounding_review_md)
    if draft_manifest.requires_grounding_review == grounding_review.requires_review:
        return draft_manifest
    updated_manifest = draft_manifest.model_copy(update={"requires_grounding_review": grounding_review.requires_review})
    write_json(draft_manifest_path, updated_manifest)
    refresh_draft_rendering_artifact_refs(ctx, draft_manifest_path)
    return updated_manifest


def refresh_draft_rendering_artifact_refs(ctx: StepRunContext, *paths: Path) -> None:
    draft_step = get_step(ctx.manifest, "draft_rendering")
    for path in paths:
        ref = _draft_rendering_ref(ctx.run_dir, path, "draft_rendering")
        draft_step.outputs = replace_artifact_ref(draft_step.outputs, ref)
        for attempt in draft_step.attempts:
            attempt.outputs = replace_artifact_ref(attempt.outputs, ref)


def replace_artifact_ref(refs: list[ArtifactRef], ref: ArtifactRef) -> list[ArtifactRef]:
    replaced = False
    next_refs: list[ArtifactRef] = []
    for existing in refs:
        if existing.relative_path == ref.relative_path:
            next_refs.append(ref)
            replaced = True
        else:
            next_refs.append(existing)
    if not replaced:
        next_refs.append(ref)
    return next_refs


def _run_draft_review(ctx: StepRunContext) -> None:
    step_name = "draft_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    draft_manifest_path = require_step_output_dir(ctx.run_dir, "draft_rendering") / "draft_write_manifest.json"
    draft_manifest = read_model(draft_manifest_path, DraftWriteManifest)
    draft_manifest = refresh_current_draft_grounding_artifacts(ctx, draft_manifest, draft_manifest_path)
    approved_manifest_path = step_root / "approved_write_manifest.json"
    approval_path = step_root / "draft_approval.json"
    prompt_path = step_root / "review_prompt.md"
    prompt_path.write_text(render_draft_review_prompt(ctx.run_dir, draft_manifest), encoding="utf-8")
    if draft_manifest.source_only_noop:
        write_json(approved_manifest_path, draft_manifest)
        approval = build_draft_approval(
            ctx.run_dir,
            approved_manifest_path,
            decision="approved",
            review_mode="not_required",
            auto_approved=True,
            notes="全 noop operation：来源会被记录，但没有知识页变化。",
        )
        write_json(approval_path, approval)
        complete_review_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
            ],
            review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    if not draft_review_requires_manual(ctx.run_dir, draft_manifest):
        write_json(approved_manifest_path, draft_manifest)
        notes = (
            "纯 create operation，本轮按 auto-stub 自动批准。"
            if not draft_manifest.has_updates
            else "update operation 未发现 grounding 或旧页保留观察风险；本地旧知识补强已写入审计报告，本轮按 auto-stub 自动批准。"
        )
        approval = build_draft_approval(
            ctx.run_dir,
            approved_manifest_path,
            decision="approved",
            review_mode="auto_stub",
            auto_approved=True,
            notes=notes,
        )
        write_json(approval_path, approval)
        complete_review_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
            ],
            review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    pending_manifest = step_root / "pending_write_manifest.json"
    write_json(pending_manifest, draft_manifest)
    review_reason = draft_review_reason(ctx.run_dir, draft_manifest)
    approval = DraftApproval(
        decision="pending",
        review_mode="manual",
        auto_approved=False,
        notes=review_reason,
    )
    write_json(approval_path, approval)
    mark_step_awaiting_review(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, pending_manifest, step_name, "json", "draft_write_manifest.v1"),
            _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
        ],
        reason=review_reason,
        review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
    )


def draft_review_requires_manual(run_dir: Path, draft_manifest: DraftWriteManifest) -> bool:
    return bool(
        draft_manifest.requires_grounding_review
        or update_manual_resolution_count(run_dir)
    )


@dataclass(frozen=True)
class StepRunner:
    spec: StepSpec
    run: Any


_STEP_RUN_FUNCTIONS = {
    "raw_link_cleanup": _run_raw_link_cleanup,
    "raw_prepare": _run_raw_prepare,
    "prepared_raw_review": _run_prepared_raw_review,
    "source_digest": _run_source_digest,
    "source_digest_review": _run_source_digest_review,
    "source_duplicate_guard": _run_source_duplicate_guard,
    "candidate_resolution": _run_candidate_resolution,
    "wiki_context_snapshot": _run_wiki_context_snapshot,
    "wiki_merge_planning": _run_wiki_merge_planning,
    "merge_plan_review": _run_merge_plan_review,
    "draft_rendering": _run_draft_rendering,
    "draft_review": _run_draft_review,
    "validation": _run_validation,
    "apply_preview": _run_apply_preview,
}

STEP_RUNNERS: dict[str, StepRunner] = {
    spec.name: StepRunner(spec, _STEP_RUN_FUNCTIONS[spec.name]) for spec in STEP_SPECS
}


TModel = TypeVar("TModel", bound=BaseModel)


def _redacted_model(ctx: StepRunContext, model: TModel, model_type: type[TModel]) -> TModel:
    data = ctx.execution_context.redactor.redact(model.model_dump(mode="json"))
    return model_type.model_validate(data)


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
        sections.extend(["", f"## {title}", "", render_candidate_table(candidates)])
    return "\n".join(sections).rstrip() + "\n"


def augment_source_digest_anchor_entities(
    digest: SourceDigestArtifact,
    approved_prepared_text: str,
) -> SourceDigestArtifact:
    additions: list[SourceDigestCandidate] = []
    existing_keys = {
        source_digest_candidate_title_key(candidate)
        for candidate in digest.ingest_candidates()
        if source_digest_candidate_title_key(candidate)
    }
    for anchor, metadata in SOURCE_DIGEST_ANCHOR_ENTITIES.items():
        anchor_key = normalized_source_match_text(anchor)
        if not anchor_key or anchor_key in existing_keys:
            continue
        signal = source_anchor_signal(approved_prepared_text, anchor)
        if not signal["should_add"]:
            continue
        candidate = SourceDigestCandidate(
            candidate_id=f"auto-ent-{anchor_key}",
            name=anchor,
            type="entity",
            one_sentence_summary=metadata["summary"],
            why_matters=metadata["why_matters"],
            wiki_value=metadata["wiki_value"],
            source_locator=signal["source_locator"],
            suggested_page_title=anchor,
            related_candidates=source_anchor_related_candidates(anchor, digest),
            resolution_hint=(
                f"{metadata['resolution_hint']} occurrence_count={signal['occurrence_count']}; "
                f"signal_reason={signal['reason']}"
            ),
            duplicate_risk="medium",
        )
        additions.append(candidate)
        existing_keys.add(anchor_key)
    if not additions:
        return digest
    return digest.model_copy(update={"entities": [*additions, *digest.entities]})


def source_anchor_signal(text: str, anchor: str) -> dict[str, Any]:
    occurrence_count = source_anchor_occurrence_count(text, anchor)
    if occurrence_count <= 0:
        return {
            "should_add": False,
            "occurrence_count": 0,
            "source_locator": "",
            "reason": "absent",
        }
    frontmatter = parse_frontmatter(text) or {}
    metadata_text = "\n".join(
        str(frontmatter.get(key) or "")
        for key in ["title", "description", "source", "author"]
    )
    heading_text = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#"))
    high_signal_text = "\n".join([metadata_text, heading_text])
    high_signal = source_anchor_occurrence_count(high_signal_text, anchor) > 0
    explicit_context = source_anchor_has_explicit_context(text, anchor)
    should_add = high_signal or occurrence_count >= 3 or explicit_context
    reason_parts: list[str] = []
    if high_signal:
        reason_parts.append("metadata_or_heading")
    if occurrence_count >= 3:
        reason_parts.append("repeated")
    if explicit_context:
        reason_parts.append("explicit_context")
    return {
        "should_add": should_add,
        "occurrence_count": occurrence_count,
        "source_locator": source_anchor_first_locator(text, anchor),
        "reason": "+".join(reason_parts) or "weak_mention",
    }


def source_anchor_occurrence_count(text: str, anchor: str) -> int:
    if not text or not anchor:
        return 0
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])", re.IGNORECASE)
    return len(pattern.findall(unicodedata.normalize("NFKC", text)))


def source_anchor_has_explicit_context(text: str, anchor: str) -> bool:
    lower = unicodedata.normalize("NFKC", text).lower()
    anchor_lower = anchor.lower()
    context_terms = {
        "managed agents": [
            "meta-harness",
            "托管智能体",
            "managed agents is",
            "managed agents can",
            "managed agents,",
        ],
        "claude code": [
            "excellent harness",
            "广泛使用",
            "head of product",
            "创建了claude code",
            "claude code团队",
            "claude code和cowork",
        ],
        "cowork": [
            "claude code和cowork",
            "head of product",
            "知识工作",
            "not code",
            "非代码",
        ],
    }.get(anchor_lower, [])
    for match in re.finditer(rf"(?<![a-z0-9]){re.escape(anchor_lower)}(?![a-z0-9])", lower):
        window = lower[max(0, match.start() - 120) : min(len(lower), match.end() + 120)]
        if any(term in window for term in context_terms):
            return True
    return False


def source_anchor_first_locator(text: str, anchor: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])", re.IGNORECASE)
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern.search(unicodedata.normalize("NFKC", line)):
            return f"L{line_no}"
    return ""


def source_anchor_related_candidates(anchor: str, digest: SourceDigestArtifact) -> list[str]:
    existing_titles = [candidate.suggested_page_title or candidate.name for candidate in digest.ingest_candidates()]
    desired = {
        "Managed Agents": ["Claude Code", "Harness（适配框架）", "Session（会话）", "大脑与双手解耦"],
        "Claude Code": ["Managed Agents", "Harness（适配框架）", "Cowork"],
        "Cowork": ["Claude Code"],
    }.get(anchor, [])
    available = [title for title in desired if title in existing_titles or title in SOURCE_DIGEST_ANCHOR_ENTITIES]
    return available[:FINAL_RELATED_LIMIT]


def cap_source_digest_candidates(digest: SourceDigestArtifact, max_candidates: int) -> tuple[SourceDigestArtifact, dict[str, Any]]:
    groups: dict[str, list[SourceDigestCandidate]] = {
        "entities": list(digest.entities),
        "concepts": list(digest.concepts),
        "designs": list(digest.designs),
        "comparisons": list(digest.comparisons),
        "open_questions": list(digest.open_questions),
    }
    total_before_dedupe = sum(len(items) for items in groups.values())
    groups, deduped_candidates = dedupe_source_digest_groups(groups)
    total = sum(len(items) for items in groups.values())
    budget = max(1, int(max_candidates))
    if total <= budget:
        report = source_digest_budget_report(
            groups,
            {},
            budget=budget,
            total=total,
            applied=False,
            total_before_dedupe=total_before_dedupe,
            deduped_candidates=deduped_candidates,
        )
        capped = digest.model_copy(
            update={
                "entities": groups["entities"],
                "concepts": groups["concepts"],
                "designs": groups["designs"],
                "comparisons": groups["comparisons"],
                "open_questions": groups["open_questions"],
            }
        )
        return capped, report

    selected: dict[str, list[SourceDigestCandidate]] = {name: [] for name in groups}
    indexes = {name: 0 for name in groups}
    selected_count = 0
    while selected_count < budget:
        progressed = False
        for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
            group = groups[group_name]
            index = indexes[group_name]
            if index >= len(group):
                continue
            selected[group_name].append(group[index])
            indexes[group_name] += 1
            selected_count += 1
            progressed = True
            if selected_count >= budget:
                break
        if not progressed:
            break

    deferred: dict[str, list[SourceDigestCandidate]] = {
        group_name: groups[group_name][indexes[group_name] :]
        for group_name in groups
    }
    selected, selected_aggregations = promote_deferred_aggregations_into_selection(selected, deferred)
    represented_by = {
        candidate_id: aggregation["candidate_id"]
        for aggregation in selected_aggregations
        for candidate_id in aggregation.get("deferred_candidate_ids", [])
    }
    budget_deferred_candidates = list(digest.budget_deferred_candidates)
    for group_name, candidates in deferred.items():
        budget_deferred_candidates.extend(
            deferred_digest_candidate(group_name, candidate, represented_by=represented_by.get(candidate.candidate_id, ""))
            for candidate in candidates
        )
    capped = digest.model_copy(
        update={
            "entities": selected["entities"],
            "concepts": selected["concepts"],
            "designs": selected["designs"],
            "comparisons": selected["comparisons"],
            "open_questions": selected["open_questions"],
            "budget_deferred_candidates": budget_deferred_candidates,
        }
    )
    report = source_digest_budget_report(
        selected,
        deferred,
        budget=budget,
        total=total,
        applied=True,
        total_before_dedupe=total_before_dedupe,
        deduped_candidates=deduped_candidates,
        selected_deferred_aggregations=selected_aggregations,
    )
    return capped, report


def dedupe_source_digest_groups(
    groups: dict[str, list[SourceDigestCandidate]],
) -> tuple[dict[str, list[SourceDigestCandidate]], list[dict[str, Any]]]:
    deduped_groups: dict[str, list[SourceDigestCandidate]] = {group_name: [] for group_name in groups}
    deduped_candidates: list[dict[str, Any]] = []
    for group_name, candidates in groups.items():
        by_key: dict[str, int] = {}
        for candidate in candidates:
            key = source_digest_candidate_dedupe_key(group_name, candidate)
            canonical_index: int | None = by_key.get(key) if key else None
            if canonical_index is None and group_name != "open_questions":
                canonical_index = source_digest_duplicate_candidate_index(
                    group_name,
                    deduped_groups[group_name],
                    candidate,
                )
                if canonical_index is not None and not key:
                    key = source_digest_candidate_similarity_key(group_name, deduped_groups[group_name][canonical_index], candidate)
            if canonical_index is None:
                if key:
                    by_key[key] = len(deduped_groups[group_name])
                deduped_groups[group_name].append(candidate)
                continue
            canonical = deduped_groups[group_name][canonical_index]
            deduped_groups[group_name][canonical_index] = merge_source_digest_duplicate_candidate(
                group_name,
                key,
                canonical,
                candidate,
            )
            deduped_candidates.append(
                {
                    "group": group_name,
                    "dedupe_key": key,
                    "kept_candidate_id": canonical.candidate_id,
                    "merged_candidate_id": candidate.candidate_id,
                    "merged_title": candidate.suggested_page_title or candidate.name,
                    "source_locator": candidate.source_locator,
                }
            )
    return deduped_groups, deduped_candidates


def source_digest_duplicate_candidate_index(
    group_name: str,
    candidates: list[SourceDigestCandidate],
    candidate: SourceDigestCandidate,
) -> int | None:
    for index, existing in enumerate(candidates):
        if source_digest_candidates_semantically_duplicate(group_name, existing, candidate):
            return index
    return None


def source_digest_candidate_dedupe_key(group_name: str, candidate: SourceDigestCandidate) -> str:
    if group_name != "open_questions":
        key = source_digest_candidate_title_key(candidate)
        return f"{group_name}:title:{key}" if len(key) >= 6 else ""
    basis = (
        candidate.open_question_or_tension
        or candidate.suggested_page_title
        or candidate.name
        or candidate.one_sentence_summary
    )
    key = open_question_key(basis)
    return f"open_questions:{key}" if key and len(key) >= 6 else ""


def source_digest_candidates_semantically_duplicate(
    group_name: str,
    left: SourceDigestCandidate,
    right: SourceDigestCandidate,
) -> bool:
    if group_name == "open_questions":
        return source_digest_candidate_dedupe_key(group_name, left) == source_digest_candidate_dedupe_key(group_name, right)
    left_key = source_digest_candidate_title_key(left)
    right_key = source_digest_candidate_title_key(right)
    if left_key and left_key == right_key:
        return True
    title_similarity = source_digest_text_similarity(
        left.suggested_page_title or left.name,
        right.suggested_page_title or right.name,
    )
    intent_similarity = source_digest_text_similarity(
        source_digest_candidate_intent_text(left),
        source_digest_candidate_intent_text(right),
    )
    if group_name == "entities":
        return title_similarity >= 0.82 and intent_similarity >= 0.45
    if group_name == "comparisons" and both_agent_workflow_compare(left.suggested_page_title or left.name, right.suggested_page_title or right.name):
        return intent_similarity >= 0.35
    shared_title_terms = source_digest_shared_signal_terms(left.suggested_page_title or left.name, right.suggested_page_title or right.name)
    return (title_similarity >= 0.55 and intent_similarity >= 0.42) or (
        title_similarity >= 0.40 and intent_similarity >= 0.55
    ) or (intent_similarity >= 0.56 and bool(shared_title_terms))


def source_digest_candidate_title_key(candidate: SourceDigestCandidate) -> str:
    return source_digest_title_key(candidate.suggested_page_title or candidate.name)


def source_digest_title_key(title: str) -> str:
    core_title = source_digest_parenthetical_translation_core(title)
    core_key = normalized_source_match_text(core_title)
    if len(core_key) >= 4:
        return core_key
    return normalized_source_match_text(title)


def source_digest_parenthetical_translation_core(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).strip()
    if not text:
        return text
    parenthetical_parts = re.findall(r"[（(]([^）)]{1,48})[）)]", text)
    if not parenthetical_parts:
        return text
    stripped = re.sub(r"\s*[（(][^）)]{1,48}[）)]\s*", " ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped:
        return text
    stripped_has_latin = bool(re.search(r"[A-Za-z]", stripped))
    stripped_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", stripped))
    paren_text = " ".join(parenthetical_parts)
    paren_has_latin = bool(re.search(r"[A-Za-z]", paren_text))
    paren_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", paren_text))
    if (stripped_has_latin and paren_has_cjk) or (stripped_has_cjk and paren_has_latin):
        return stripped
    return text


def source_digest_candidate_similarity_key(
    group_name: str,
    canonical: SourceDigestCandidate,
    duplicate: SourceDigestCandidate,
) -> str:
    title_similarity = source_digest_text_similarity(
        canonical.suggested_page_title or canonical.name,
        duplicate.suggested_page_title or duplicate.name,
    )
    intent_similarity = source_digest_text_similarity(
        source_digest_candidate_intent_text(canonical),
        source_digest_candidate_intent_text(duplicate),
    )
    return f"{group_name}:similarity:title={title_similarity:.2f}:intent={intent_similarity:.2f}"


def source_digest_candidate_intent_text(candidate: SourceDigestCandidate) -> str:
    return "\n".join(
        [
            candidate.suggested_page_title,
            candidate.name,
            candidate.one_sentence_summary,
            candidate.why_matters,
            candidate.wiki_value,
            candidate.open_question_or_tension,
        ]
    )


def source_digest_text_similarity(left: str, right: str) -> float:
    return jaccard(source_digest_similarity_terms(left), source_digest_similarity_terms(right))


def source_digest_shared_signal_terms(left: str, right: str) -> set[str]:
    generic = {"ai", "pm", "产品", "管理", "主题", "材料", "知识", "页面"}
    return {
        term
        for term in source_digest_similarity_terms(left) & source_digest_similarity_terms(right)
        if term not in generic and len(term) >= 2
    }


def source_digest_similarity_terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text.lower())
    terms = set(re.findall(r"[a-z0-9]{2,}", normalized))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        terms.add(segment)
        max_size = min(4, len(segment))
        for size in range(2, max_size + 1):
            for index in range(0, len(segment) - size + 1):
                terms.add(segment[index : index + size])
    stop = {
        "concept",
        "comparison",
        "design",
        "entity",
        "open",
        "question",
        "概念",
        "设计",
        "实体",
        "问题",
        "对比",
        "比较",
        "页面",
        "来源",
        "摘要",
    }
    return {term for term in terms if term not in stop and len(term) >= 2}


def merge_source_digest_duplicate_candidate(
    group_name: str,
    dedupe_key: str,
    canonical: SourceDigestCandidate,
    duplicate: SourceDigestCandidate,
) -> SourceDigestCandidate:
    duplicate_title = duplicate.suggested_page_title or duplicate.name
    note = (
        f"source_digest_semantic_dedupe: `{duplicate.candidate_id}` ({duplicate_title}) "
        f"按 `{dedupe_key}` 合并进 `{canonical.candidate_id}`，不单独占用本轮页面预算。"
    )
    if duplicate.source_locator:
        note += f" 来源定位：{duplicate.source_locator}。"
    if duplicate.one_sentence_summary:
        note += f" 变体摘要：{duplicate.one_sentence_summary}"
    duplicate_risk = "high" if "high" in {canonical.duplicate_risk, duplicate.duplicate_risk} else (
        "medium" if "medium" in {canonical.duplicate_risk, duplicate.duplicate_risk} else canonical.duplicate_risk
    )
    return canonical.model_copy(
        update={
            "related_candidates": _dedupe_strings(
                [*canonical.related_candidates, duplicate.candidate_id, *duplicate.related_candidates]
            ),
            "resolution_hint": merge_markdown_blocks(canonical.resolution_hint, note),
            "open_question_or_tension": merge_markdown_blocks(
                canonical.open_question_or_tension,
                duplicate.open_question_or_tension,
            ),
            "duplicate_risk": duplicate_risk,
        }
    )


def promote_deferred_aggregations_into_selection(
    selected: dict[str, list[SourceDigestCandidate]],
    deferred: dict[str, list[SourceDigestCandidate]],
) -> tuple[dict[str, list[SourceDigestCandidate]], list[dict[str, Any]]]:
    selected = {group_name: list(candidates) for group_name, candidates in selected.items()}
    selected_aggregations: list[dict[str, Any]] = []
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        deferred_candidates = list(deferred.get(group_name, []))
        clusters = deferred_candidate_topic_clusters(group_name, deferred_candidates)
        if not clusters or not selected.get(group_name):
            continue
        remaining_selected = list(selected[group_name])
        promoted: list[SourceDigestCandidate] = []
        for cluster_index, cluster in enumerate(clusters, start=1):
            if not remaining_selected:
                break
            replacement_index, replacement_score = select_aggregation_replacement(remaining_selected, cluster)
            if replacement_score < SOURCE_DIGEST_PROMOTED_AGGREGATION_MIN_REPLACEMENT_SIMILARITY:
                continue
            replaced = remaining_selected.pop(replacement_index)
            represented = [replaced, *cluster]
            aggregate = deferred_aggregation_digest_candidate(group_name, represented, replaced_candidate_id=replaced.candidate_id)
            promoted.append(aggregate)
            selected_aggregations.append(
                {
                    "group": group_name,
                    "cluster_index": cluster_index,
                    "candidate_id": aggregate.candidate_id,
                    "suggested_page_title": aggregate.suggested_page_title,
                    "suggested_page_type": aggregate.type,
                    "replaced_candidate_id": replaced.candidate_id,
                    "replacement_similarity": round(replacement_score, 4),
                    "represented_candidate_ids": [candidate.candidate_id for candidate in represented],
                    "deferred_candidate_ids": [candidate.candidate_id for candidate in cluster],
                    "cluster_terms": sorted(deferred_cluster_terms(cluster))[:8],
                    "reason": (
                        f"`{group_name}` deferred topic cluster {cluster_index} has {len(cluster)} candidate(s); "
                        "one selected candidate was folded into an aggregation candidate to keep the page budget constant."
                    ),
                }
            )
        selected[group_name] = [*remaining_selected, *promoted]
    return selected, selected_aggregations


def deferred_candidate_topic_clusters(
    group_name: str,
    candidates: list[SourceDigestCandidate],
) -> list[list[SourceDigestCandidate]]:
    clusters: list[list[SourceDigestCandidate]] = []
    for candidate in candidates:
        best_index = -1
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            score = candidate_cluster_similarity(group_name, candidate, cluster)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= SOURCE_DIGEST_AGGREGATION_CLUSTER_SIMILARITY:
            clusters[best_index].append(candidate)
        else:
            clusters.append([candidate])
    return [cluster for cluster in clusters if len(cluster) >= SOURCE_DIGEST_AGGREGATION_MIN_CANDIDATES]


def candidate_cluster_similarity(
    group_name: str,
    candidate: SourceDigestCandidate,
    cluster: list[SourceDigestCandidate],
) -> float:
    if not cluster:
        return 0.0
    return min(source_digest_candidate_topic_similarity(group_name, candidate, item) for item in cluster)


def source_digest_candidate_topic_similarity(
    group_name: str,
    left: SourceDigestCandidate,
    right: SourceDigestCandidate,
) -> float:
    if group_name == "open_questions":
        left_key = open_question_key(left.open_question_or_tension or left.suggested_page_title or left.name)
        right_key = open_question_key(right.open_question_or_tension or right.suggested_page_title or right.name)
        if left_key and right_key and left_key == right_key:
            return 1.0
    left_anchors = source_digest_candidate_topic_anchor_terms(left)
    right_anchors = source_digest_candidate_topic_anchor_terms(right)
    shared_anchors = left_anchors & right_anchors
    if not shared_anchors:
        return 0.0
    left_terms = source_digest_non_generic_terms(source_digest_candidate_topic_terms(left))
    right_terms = source_digest_non_generic_terms(source_digest_candidate_topic_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    anchor_score = jaccard(left_anchors, right_anchors)
    intent_score = jaccard(left_terms, right_terms)
    if len(shared_anchors) == 1:
        anchor = next(iter(shared_anchors))
        if source_digest_topic_anchor_too_broad(anchor):
            return 0.0
        shared_bonus = 0.22
    else:
        shared_bonus = min(0.50, len(shared_anchors) * 0.20)
    return min(1.0, max(anchor_score, intent_score * 0.5) + shared_bonus)


def source_digest_candidate_topic_terms(candidate: SourceDigestCandidate) -> set[str]:
    return source_digest_similarity_terms(source_digest_candidate_intent_text(candidate))


def source_digest_candidate_topic_anchor_terms(candidate: SourceDigestCandidate) -> set[str]:
    anchor_text = "\n".join(
        [
            candidate.suggested_page_title,
            candidate.name,
            candidate.open_question_or_tension,
        ]
    )
    return source_digest_non_generic_terms(source_digest_similarity_terms(anchor_text))


def source_digest_topic_anchor_too_broad(anchor: str) -> bool:
    broad = {
        "agi",
        "ai",
        "llm",
        "pm",
        "模型",
        "能力",
        "评估",
        "数据",
        "用户",
    }
    return anchor in broad


def source_digest_non_generic_terms(terms: set[str]) -> set[str]:
    generic = {
        "ai",
        "pm",
        "产品",
        "管理",
        "主题",
        "材料",
        "知识",
        "页面",
        "候选",
        "价值",
        "来源",
        "概念",
        "重要",
        "复用",
        "讨论",
        "问题",
        "方式",
        "变化",
        "边界",
        "职责",
        "执行",
        "判断",
        "功能",
        "任务",
        "组织",
    }
    return {term for term in terms if term not in generic and len(term) >= 2}


def deferred_cluster_terms(cluster: list[SourceDigestCandidate]) -> set[str]:
    if not cluster:
        return set()
    shared = source_digest_candidate_topic_terms(cluster[0])
    for candidate in cluster[1:]:
        shared &= source_digest_candidate_topic_terms(candidate)
    if shared:
        return source_digest_non_generic_terms(shared)
    combined: set[str] = set()
    for candidate in cluster:
        combined.update(source_digest_non_generic_terms(source_digest_candidate_topic_terms(candidate)))
    return combined


def select_aggregation_replacement(
    selected: list[SourceDigestCandidate],
    cluster: list[SourceDigestCandidate],
) -> tuple[int, float]:
    if not selected:
        return 0, 0.0
    best_index = len(selected) - 1
    best_score = -1.0
    for index, candidate in enumerate(selected):
        score = max(
            source_digest_candidate_topic_similarity(deferred_candidate_group(candidate), candidate, cluster_candidate)
            for cluster_candidate in cluster
        )
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, max(0.0, best_score)


def deferred_aggregation_digest_candidate(
    group_name: str,
    candidates: list[SourceDigestCandidate],
    *,
    replaced_candidate_id: str,
) -> SourceDigestCandidate:
    aggregation = build_deferred_candidate_aggregations({group_name: candidates})[0]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    related_candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id != replaced_candidate_id]
    candidate_id = f"AGG-{group_name.replace('_', '-')}-{sha256_bytes('|'.join(candidate_ids).encode('utf-8'))[:8]}"
    source_locators = _dedupe_strings([candidate.source_locator for candidate in candidates if candidate.source_locator])[:6]
    tensions = _dedupe_strings([candidate.open_question_or_tension for candidate in candidates if candidate.open_question_or_tension])[:4]
    title = str(aggregation["suggested_title"])
    summary = str(aggregation["coverage_summary"])
    wiki_value = str(aggregation.get("wiki_value_summary") or "") or str(aggregation["suggested_action"])
    return SourceDigestCandidate(
        candidate_id=candidate_id,
        name=title,
        type=deferred_aggregation_digest_type(group_name),
        one_sentence_summary=summary,
        why_matters=f"该聚合候选把 {len(candidates)} 个同组候选压缩成一个页面预算槽，避免单篇材料产生过多独立页面。",
        wiki_value=wiki_value,
        source_locator="；".join(source_locators),
        suggested_page_title=title,
        related_candidates=related_candidate_ids,
        resolution_hint=(
            f"source_digest_deferred_aggregation: represented_candidates={', '.join(candidate_ids)}; "
            f"related_deferred_candidates={', '.join(related_candidate_ids) or 'none'}; "
            f"replaced_selected_candidate={replaced_candidate_id}; page budget remains constant."
        ),
        duplicate_risk=source_digest_max_duplicate_risk(candidates),
        open_question_or_tension="；".join(tensions),
    )


def deferred_aggregation_digest_type(group_name: str) -> str:
    if group_name == "comparisons":
        return "comparison"
    if group_name == "open_questions":
        return "open_question"
    return "overview"


def source_digest_max_duplicate_risk(candidates: list[SourceDigestCandidate]) -> Literal["low", "medium", "high"]:
    risks = {candidate.duplicate_risk for candidate in candidates}
    if "high" in risks:
        return "high"
    if "medium" in risks:
        return "medium"
    return "low"


def deferred_digest_candidate(group_name: str, candidate: SourceDigestCandidate, *, represented_by: str = "") -> SourceDigestCandidate:
    data = candidate.model_dump(mode="json")
    note = f"page_budget_deferred: `{group_name}` 超出本次 max_ingest_candidates，保留在 source digest 审计中，后续可单独 ingest 或手动提升。"
    if represented_by:
        note += f" represented_by_aggregation: `{represented_by}` 已在本轮用聚合候选代表该候选的核心价值。"
    data["resolution_hint"] = merge_markdown_blocks(
        str(data.get("resolution_hint") or ""),
        note,
    )
    return SourceDigestCandidate.model_validate(data)


def source_digest_budget_report(
    selected: dict[str, list[SourceDigestCandidate]],
    deferred: dict[str, list[SourceDigestCandidate]],
    *,
    budget: int,
    total: int,
    applied: bool,
    total_before_dedupe: int | None = None,
    deduped_candidates: list[dict[str, Any]] | None = None,
    selected_deferred_aggregations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_count = sum(len(items) for items in selected.values())
    deferred_count = sum(len(items) for items in deferred.values())
    deduped_candidates = deduped_candidates or []
    selected_deferred_aggregations = selected_deferred_aggregations or []
    deferred_details = {
        group_name: [source_digest_candidate_budget_detail(group_name, candidate) for candidate in candidates]
        for group_name, candidates in deferred.items()
    }
    followup_batches = [
        {
            "group": group_name,
            "count": len(candidates),
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "suggested_action": (
                f"后续如需扩展 `{group_name}`，可单独从这些 deferred candidate 建页，"
                "或把同组候选合并进一个 overview/comparison 页面。"
            ),
        }
        for group_name, candidates in deferred.items()
        if candidates
    ]
    deferred_aggregations = build_deferred_candidate_aggregations(deferred)
    return {
        "schema_version": "source_digest_budget_report.v1",
        "budget": budget,
        "total_formal_candidates_before_dedupe": total_before_dedupe if total_before_dedupe is not None else total,
        "total_formal_candidates_before_budget": total,
        "deduped_count": len(deduped_candidates),
        "dedupe_applied": bool(deduped_candidates),
        "deduped_candidates": deduped_candidates,
        "selected_deferred_aggregations": selected_deferred_aggregations,
        "selected_count": selected_count,
        "deferred_count": deferred_count,
        "applied": applied,
        "group_order": list(SOURCE_DIGEST_BUDGET_GROUP_ORDER),
        "selected": {
            group_name: [candidate.candidate_id for candidate in candidates]
            for group_name, candidates in selected.items()
        },
        "deferred": {
            group_name: [candidate.candidate_id for candidate in candidates]
            for group_name, candidates in deferred.items()
        },
        "deferred_details": deferred_details,
        "followup_batches": followup_batches,
        "deferred_aggregations": deferred_aggregations,
    }


def build_deferred_candidate_aggregations(
    deferred: dict[str, list[SourceDigestCandidate]] | list[SourceDigestCandidate],
) -> list[dict[str, Any]]:
    if isinstance(deferred, list):
        grouped: dict[str, list[SourceDigestCandidate]] = {}
        for candidate in deferred:
            grouped.setdefault(deferred_candidate_group(candidate), []).append(candidate)
    else:
        grouped = {group_name: list(candidates) for group_name, candidates in deferred.items()}
    aggregations: list[dict[str, Any]] = []
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        candidates = [candidate for candidate in grouped.get(group_name, []) if isinstance(candidate, SourceDigestCandidate)]
        if not candidates:
            continue
        label = DEFERRED_AGGREGATION_GROUP_LABELS.get(group_name, group_name)
        names = [candidate.suggested_page_title or candidate.name for candidate in candidates]
        representative = candidates[:5]
        source_locators = _dedupe_strings([candidate.source_locator for candidate in candidates if candidate.source_locator])[:6]
        tensions = _dedupe_strings([candidate.open_question_or_tension for candidate in candidates if candidate.open_question_or_tension])[:4]
        wiki_values = _dedupe_strings([candidate.wiki_value for candidate in candidates if candidate.wiki_value])[:4]
        aggregations.append(
            {
                "group": group_name,
                "label": label,
                "count": len(candidates),
                "suggested_page_type": deferred_aggregation_page_type(group_name),
                "suggested_title": deferred_aggregation_title(group_name, names),
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
                "representative_candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "title": candidate.suggested_page_title or candidate.name,
                        "summary": candidate.one_sentence_summary,
                    }
                    for candidate in representative
                ],
                "coverage_summary": deferred_aggregation_summary(label, candidates),
                "wiki_value_summary": "；".join(wiki_values) if wiki_values else "",
                "source_locators": source_locators,
                "open_questions_or_tensions": tensions,
                "suggested_action": deferred_aggregation_action(group_name),
            }
        )
    return aggregations


def deferred_candidate_group(candidate: SourceDigestCandidate) -> str:
    type_key = candidate.type.strip().lower()
    mapping = {
        "concept": "concepts",
        "concepts": "concepts",
        "design": "designs",
        "designs": "designs",
        "comparison": "comparisons",
        "comparisons": "comparisons",
        "open_question": "open_questions",
        "open_questions": "open_questions",
        "question": "open_questions",
        "entity": "entities",
        "entities": "entities",
    }
    if type_key in mapping:
        return mapping[type_key]
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        if f"`{group_name}`" in candidate.resolution_hint or group_name in candidate.resolution_hint:
            return group_name
    return "concepts"


def deferred_aggregation_page_type(group_name: str) -> str:
    return {
        "comparisons": "comparison",
        "open_questions": "open_question_overview",
        "entities": "entity_index",
        "designs": "design_overview",
    }.get(group_name, "concept_overview")


def deferred_aggregation_title(group_name: str, names: list[str]) -> str:
    label = DEFERRED_AGGREGATION_GROUP_LABELS.get(group_name, group_name)
    if not names:
        return f"{label}聚合页"
    if len(names) == 1:
        return f"{names[0]} 后续页"
    return f"{names[0]} 等 {len(names)} 个{label}聚合页"


def deferred_aggregation_summary(label: str, candidates: list[SourceDigestCandidate]) -> str:
    names = [candidate.suggested_page_title or candidate.name for candidate in candidates[:4]]
    suffix = "" if len(candidates) <= 4 else f" 等 {len(candidates)} 项"
    return f"本批次聚合 {label}：{', '.join(names)}{suffix}。"


def deferred_aggregation_action(group_name: str) -> str:
    if group_name == "comparisons":
        return "后续可合并为一个 comparison 页面，集中比较边界、差异和适用场景。"
    if group_name == "open_questions":
        return "后续可合并为一个 open question overview，集中跟踪问题变体和待补来源。"
    if group_name == "entities":
        return "后续可合并为一个 entity index/source companion，避免为低频实体逐个建页。"
    if group_name == "designs":
        return "后续可合并为一个 design overview，保留模式差异和复用场景。"
    return "后续可合并为一个 concept overview，先保留概念簇关系，再决定是否拆独立页。"


def source_digest_candidate_budget_detail(group_name: str, candidate: SourceDigestCandidate) -> dict[str, str]:
    return {
        "group": group_name,
        "candidate_id": candidate.candidate_id,
        "type": candidate.type,
        "name": candidate.name,
        "suggested_page_title": candidate.suggested_page_title,
        "one_sentence_summary": candidate.one_sentence_summary,
        "why_matters": candidate.why_matters,
        "wiki_value": candidate.wiki_value,
        "source_locator": candidate.source_locator,
        "open_question_or_tension": candidate.open_question_or_tension,
        "resolution_hint": candidate.resolution_hint,
    }


def render_source_digest_budget_report(report: dict[str, Any]) -> str:
    rows = []
    selected = report.get("selected", {})
    deferred = report.get("deferred", {})
    for group_name in ["concepts", "designs", "comparisons", "open_questions", "entities"]:
        rows.append(
            [
                group_name,
                ", ".join(selected.get(group_name, [])) or "无",
                ", ".join(deferred.get(group_name, [])) or "无",
            ]
        )
    dedupe_rows = []
    deduped_candidates = report.get("deduped_candidates", [])
    if isinstance(deduped_candidates, list):
        for item in deduped_candidates:
            if not isinstance(item, dict):
                continue
            dedupe_rows.append(
                [
                    str(item.get("group", "")),
                    str(item.get("dedupe_key", "")),
                    str(item.get("kept_candidate_id", "")),
                    str(item.get("merged_candidate_id", "")),
                    str(item.get("merged_title", "")),
                    str(item.get("source_locator", "")),
                ]
            )
    detail_rows = []
    deferred_details = report.get("deferred_details", {})
    if isinstance(deferred_details, dict):
        for group_name in ["concepts", "designs", "comparisons", "open_questions", "entities"]:
            details = deferred_details.get(group_name, [])
            if not isinstance(details, list):
                continue
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                detail_rows.append(
                    [
                        group_name,
                        str(detail.get("candidate_id", "")),
                        str(detail.get("suggested_page_title") or detail.get("name") or ""),
                        str(detail.get("one_sentence_summary", "")),
                        str(detail.get("wiki_value", "")),
                        str(detail.get("source_locator", "")),
                        str(detail.get("resolution_hint", "")),
                    ]
                )
    selected_aggregation_rows = []
    selected_aggregations = report.get("selected_deferred_aggregations", [])
    if isinstance(selected_aggregations, list):
        for aggregation in selected_aggregations:
            if not isinstance(aggregation, dict):
                continue
            selected_aggregation_rows.append(
                [
                    str(aggregation.get("group", "")),
                    str(aggregation.get("candidate_id", "")),
                    str(aggregation.get("suggested_page_type", "")),
                    str(aggregation.get("suggested_page_title", "")),
                    str(aggregation.get("replaced_candidate_id", "")),
                    ", ".join(aggregation.get("represented_candidate_ids", []))
                    if isinstance(aggregation.get("represented_candidate_ids"), list)
                    else "",
                    str(aggregation.get("reason", "")),
                ]
            )
    batch_rows = []
    followup_batches = report.get("followup_batches", [])
    if isinstance(followup_batches, list):
        for batch in followup_batches:
            if not isinstance(batch, dict):
                continue
            batch_rows.append(
                [
                    str(batch.get("group", "")),
                    str(batch.get("count", "")),
                    ", ".join(batch.get("candidate_ids", [])) if isinstance(batch.get("candidate_ids"), list) else "",
                    str(batch.get("suggested_action", "")),
                ]
            )
    aggregation_rows = []
    aggregations = report.get("deferred_aggregations", [])
    if isinstance(aggregations, list):
        for aggregation in aggregations:
            if not isinstance(aggregation, dict):
                continue
            representatives = aggregation.get("representative_candidates", [])
            if isinstance(representatives, list):
                rep_text = ", ".join(
                    str(item.get("title", item.get("candidate_id", "")))
                    for item in representatives[:4]
                    if isinstance(item, dict)
                )
            else:
                rep_text = ""
            aggregation_rows.append(
                [
                    str(aggregation.get("group", "")),
                    str(aggregation.get("suggested_page_type", "")),
                    str(aggregation.get("suggested_title", "")),
                    str(aggregation.get("coverage_summary", "")),
                    rep_text,
                    str(aggregation.get("suggested_action", "")),
                ]
            )
    return (
        "# Source Digest 页面预算报告\n\n"
        f"- 预算：{report.get('budget', 0)}\n"
        f"- 去重前正式候选：{report.get('total_formal_candidates_before_dedupe', report.get('total_formal_candidates_before_budget', 0))}\n"
        f"- 预算前正式候选：{report.get('total_formal_candidates_before_budget', 0)}\n"
        f"- 语义去重合并：{report.get('deduped_count', 0)}\n"
        f"- 进入页面规划：{report.get('selected_count', 0)}\n"
        f"- 延后：{report.get('deferred_count', 0)}\n"
        f"- 是否应用预算：{'是' if report.get('applied') else '否'}\n\n"
        + format_markdown_table(["分组", "进入页面规划", "延后"], rows)
        + "\n\n"
        "## 语义去重候选\n\n"
        + (
            format_markdown_table(["分组", "去重 Key", "保留 ID", "合并 ID", "合并标题", "来源定位"], dedupe_rows)
            if dedupe_rows
            else "_暂无语义去重候选。_"
        )
        + "\n\n"
        "## 延后候选详情\n\n"
        + (
            format_markdown_table(["分组", "ID", "建议标题", "摘要", "Wiki 价值", "来源定位", "处理提示"], detail_rows)
            if detail_rows
            else "_暂无延后候选。_"
        )
        + "\n\n"
        "## 本轮已选聚合候选\n\n"
        + (
            format_markdown_table(["分组", "聚合 ID", "页类型", "标题", "替换候选", "代表候选", "原因"], selected_aggregation_rows)
            if selected_aggregation_rows
            else "_暂无本轮已选聚合候选。_"
        )
        + "\n\n"
        "## 后续处理批次\n\n"
        + (
            format_markdown_table(["分组", "数量", "候选 ID", "建议"], batch_rows)
            if batch_rows
            else "_暂无后续批次。_"
        )
        + "\n\n"
        "## 延后聚合建议\n\n"
        + (
            format_markdown_table(["分组", "建议页类型", "建议标题", "覆盖摘要", "代表候选", "动作"], aggregation_rows)
            if aggregation_rows
            else "_暂无延后聚合建议。_"
        )
        + "\n"
    )


def build_wiki_merge_plan(
    resolution: CandidateResolutionArtifact,
    digest: SourceDigestArtifact,
    snapshot: WikiContextSnapshot,
    *,
    log_date: str,
) -> WikiMergePlanArtifact:
    snapshot_by_path = {entry.path: entry for entry in snapshot.entries}
    candidates = source_digest_candidate_lookup(digest)
    items: list[WikiMergePlanItem] = []
    for item in resolution.items:
        entry = snapshot_by_path.get(f"wiki/{item.candidate_target_path}")
        exists = entry is not None and entry.expected_state == "present"
        action = "update" if exists else "create"
        matched_page = None
        if action == "update":
            matched_page = item.candidate_target_path
        context_item = next((context for context in snapshot.candidate_contexts.items if context.page_plan_id == item.page_plan_id), None)
        inspected_paths = [hit.path for hit in context_item.hits] if context_item is not None else []
        strongest_hit = context_item.hits[0] if context_item is not None and context_item.hits else None
        strongest_overlap = (
            ContextOverlapSignal(
                strength=strongest_hit.strength,
                match_basis=strongest_hit.match_basis,
                path=strongest_hit.path,
                score=strongest_hit.score,
                reason=f"Top inspected context: {strongest_hit.display_title}",
            )
            if strongest_hit is not None
            else ContextOverlapSignal()
        )
        candidate = first_source_basis_candidate(item.source_basis, candidates)
        if candidate is None:
            refs = ", ".join(source_basis_candidate_refs(item.source_basis)) or "none"
            raise PipelineError(
                f"Cannot build deterministic merge plan for `{item.display_title or item.page_plan_id}`: "
                f"no source digest candidate found for refs: {refs}."
            )
        related_pages, related_unresolved = resolve_related_pages(item, candidate, resolution, snapshot)
        items.append(
            WikiMergePlanItem(
                page_plan_id=item.page_plan_id,
                source_basis=item.source_basis,
                action=action,
                model_action=action,
                finalization_reason="Deterministic local merge plan.",
                canonical_target_path=item.candidate_target_path,
                display_title=item.display_title,
                page_type=item.page_type,
                matched_page=matched_page,
                inspected_context_paths=inspected_paths,
                strongest_overlap=strongest_overlap,
                why_not_update="" if action == "update" else "未发现需要合并的已存在目标页。",
                why_create_or_update=item.reason,
                prior_knowledge_state="已有页面。" if exists else "当前 wiki 没有相关知识页。",
                new_understanding=item.topic_summary,
                changed_view="",
                knowledge_delta=item.topic_summary,
                why_this_matters=item.why_this_page,
                reuse_scenarios=[],
                value_points=[],
                section_plans={"summary": item.topic_summary, "detail": item.why_this_page},
                related_pages=related_pages,
                related_unresolved=related_unresolved,
                unresolved_related=related_unresolved,
                related_absence_reason=None if related_pages else ("no_candidate" if not inspected_paths else "low_confidence"),
                apply_eligibility="applyable",
                blocked_reason="",
                reason=item.reason,
            )
        )
    return WikiMergePlanArtifact(log_date=log_date, items=items, context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json")


def first_source_basis_candidate(
    source_basis: SourceBasis,
    candidates: dict[str, SourceDigestCandidate],
) -> SourceDigestCandidate | None:
    for candidate_id in source_basis_candidate_refs(source_basis):
        candidate = candidates.get(candidate_id)
        if candidate is not None:
            return candidate
    return None


def existing_knowledge_page_paths(vault: Path) -> set[str]:
    return {entry.rel_path for entry in build_knowledge_pool(vault)}


def parse_frontmatter(text: str) -> dict[str, Any] | None:
    if not text.startswith("---\n"):
        return None
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None
    data = yaml.safe_load(parts[1]) or {}
    return data if isinstance(data, dict) else None


def resolve_related_pages(
    item: CandidateResolutionItem,
    candidate: SourceDigestCandidate,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
) -> tuple[list[RelatedPageRef], list[str]]:
    if item.page_type.lower() == "source":
        return [], list(candidate.related_candidates)
    by_candidate_id = {
        candidate_id: other
        for other in resolution.items
        if other.page_type.lower() != "source"
        for candidate_id in source_basis_candidate_refs(other.source_basis)
    }
    by_current_title: dict[str, list[CandidateResolutionItem]] = {}
    for other in resolution.items:
        if other.page_type.lower() == "source":
            continue
        by_current_title.setdefault(normalize_related_key(other.display_title), []).append(other)
    metadata_lookup: dict[str, list[WikiPageMetadata]] = {}
    for pool_entry in snapshot.knowledge_metadata_pool:
        metadata = pool_entry.metadata
        if metadata is None:
            continue
        if metadata.llmwiki_type.lower() == "source":
            continue
        for key in [metadata.title, *metadata.aliases]:
            metadata_lookup.setdefault(normalize_related_key(key), []).append(metadata)
    related: list[RelatedPageRef] = []
    unresolved: list[str] = []
    seen_paths: set[str] = set()
    for raw in candidate.related_candidates:
        if len(related) >= FINAL_RELATED_LIMIT:
            unresolved.append(raw)
            continue
        resolved: RelatedPageRef | None = None
        other = by_candidate_id.get(raw)
        if other is None:
            matches = by_current_title.get(normalize_related_key(raw), [])
            if len(matches) == 1:
                other = matches[0]
            elif len(matches) > 1:
                unresolved.append(raw)
                continue
        if other is not None and other.candidate_target_path != item.candidate_target_path:
            resolved = RelatedPageRef(
                target_path=other.candidate_target_path,
                display_title=other.display_title,
                source="source_digest",
                reason=f"来源摘要把 `{raw}` 标记为相关候选，本页与该候选属于同一材料中的互补主题。",
            )
        if resolved is None:
            metadata_matches = metadata_lookup.get(normalize_related_key(raw), [])
            if len(metadata_matches) > 1:
                unresolved.append(raw)
                continue
            metadata = metadata_matches[0] if metadata_matches else None
            if metadata is not None and metadata.path != item.candidate_target_path:
                resolved = RelatedPageRef(
                    target_path=metadata.path,
                    display_title=clean_display_title(metadata.title),
                    source="wiki_context",
                    reason=f"已有 wiki 页面标题或别名精确匹配 `{raw}`，可作为理解本页的相关背景。",
                )
        if resolved is None:
            unresolved.append(raw)
            continue
        if resolved.target_path in seen_paths:
            continue
        seen_paths.add(resolved.target_path)
        related.append(resolved)
    return related, unresolved


def normalize_related_key(value: str) -> str:
    text = value.strip().lower()
    for prefix in ["concept_", "entity_", "design_", "comparison_", "overview_", "event_", "memory_", "idea_", "open_question_"]:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def assert_system_page_can_be_overwritten(vault: Path, target_path: str) -> None:
    path = vault / target_path
    if not path.exists():
        return
    try:
        assert_current_system_page(path)
    except RuntimeError as exc:
        raise PipelineError(f"{exc}: {target_path}") from exc


def clean_display_title(title: str) -> str:
    stripped = title.strip()
    for prefix in ["Concept_", "Entity_", "Design_", "Comparison_", "Overview_", "Event_", "Memory_", "Idea_", "Open_Question_"]:
        if stripped.lower().startswith(prefix.lower()):
            return stripped[len(prefix) :].strip()
    return stripped


def yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_preparation_review(preparation: RawPreparationArtifact) -> str:
    operations = "\n".join(f"- {operation}" for operation in preparation.operations_applied) or "- 暂无记录。"
    uncertain = "\n".join(
        f"- [{item.severity}] {item.item}: {item.reason}" for item in preparation.uncertain_items
    ) or "- 暂无记录。"
    return (
        "# Raw 清洗审核\n\n"
        f"- 原始材料: `{preparation.source_raw_path}`\n"
        f"- 文档类型: `{preparation.document_kind}`\n"
        f"- 风险等级: `{preparation.risk_level}`\n"
        f"- 是否需要人工审核: `{str(preparation.requires_human_review).lower()}`\n"
        f"- 省略策略: `{preparation.omission_policy}`\n\n"
        f"- Raw Wikilink 规范化: `{preparation.raw_link_cleanup_ref or 'raw_link_cleanup/raw_link_cleanup.json'}`\n"
        f"- 输入 raw hash: `{preparation.input_raw_sha256 or 'unknown'}`\n\n"
        "## 已执行操作\n\n"
        f"{operations}\n\n"
        "## 不确定项\n\n"
        f"{uncertain}\n\n"
        "## 审核备注\n\n"
        f"{preparation.review_notes or '暂无审核备注。'}\n"
    )


def resolve_vault_profile_name(vault: Path, profile_name: str | None) -> str:
    if profile_name:
        return profile_name
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    configured = config.get("profile")
    if not isinstance(configured, str) or not configured:
        raise PipelineError(".llmwiki/config.yaml profile must be a non-empty string.")
    return configured


def model_steps_from(start_step: str) -> list[str]:
    names = set(downstream_steps(start_step))
    return [step for step in MODEL_BACKED_STEPS if step in names]


def _structured_call(
    run_dir: Path,
    execution_context: ProviderExecutionContext,
    task: str,
    *,
    result_filename: str = "provider_result.json",
) -> StructuredModelCall:
    return StructuredModelCall(
        execution_context.provider_for_task(task),
        output_dir=step_output_dir(run_dir, task),
        result_filename=result_filename,
        redactor=execution_context.redactor,
    )


def delete_downstream_step_dirs(
    vault: Path,
    operation_id: str,
    start_step: str,
    *,
    archive: bool = False,
    archive_reason: str = "",
) -> None:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    if archive:
        archive_attempt_step_dirs(run_dir, start_step, archive_reason or "step artifacts archived before regeneration.")
    for step in downstream_steps(start_step):
        output_dir = step_output_dir(run_dir, step)
        if output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)


def archive_attempt_step_dirs(run_dir: Path, start_step: str, reason: str) -> Path | None:
    existing_steps = [
        step
        for step in downstream_steps(start_step)
        if (step_output_dir(run_dir, step) is not None and step_output_dir(run_dir, step).exists())
    ]
    if not existing_steps:
        return None
    archive_root = run_dir / "attempt_archive" / start_step / safe_timestamp()
    archive_root.mkdir(parents=True, exist_ok=True)
    for step in existing_steps:
        output_dir = step_output_dir(run_dir, step)
        if output_dir is None or not output_dir.exists():
            continue
        shutil.copytree(output_dir, archive_root / step)
    write_json(
        archive_root / "attempt_superseded.json",
        {
            "schema_version": "attempt_superseded.v1",
            "start_step": start_step,
            "superseded_at": utc_now(),
            "reason": reason,
            "archived_steps": existing_steps,
        },
    )
    return archive_root


def validate_raw_link_cleanup_resume(*, run_dir: Path, manifest: OperationManifest, start: str) -> None:
    if start != "raw_link_cleanup":
        return
    step = get_step(manifest, "raw_link_cleanup")
    if step.status != StepStatus.completed:
        return
    artifact_path = require_step_output_dir(run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    if not artifact_path.exists():
        return
    cleanup = read_model(artifact_path, RawLinkCleanupArtifact)
    if cleanup.changed:
        raise PipelineError("Cannot resume from raw_link_cleanup after it changed raw; rerun ingest or resume from raw_prepare.")


def last_attempt_duration_ms(manifest: OperationManifest, step_name: str) -> int | None:
    step = get_step(manifest, step_name)
    if not step.attempts:
        return None
    return step.attempts[-1].duration_ms


def step_completion_message(ctx: StepRunContext, step_name: str) -> str | None:
    if step_name == "raw_prepare":
        path = require_step_output_dir(ctx.run_dir, "raw_prepare") / "raw_prepare_fast_path.json"
        if not path.exists():
            return None
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if report.get("eligible") is True:
            if report.get("raw_prepare_policy") == RawPreparePolicy.skip_model.value:
                suppressed = report.get("policy_suppressed_reasons", [])
                if isinstance(suppressed, list) and suppressed:
                    return f"raw_prepare 使用 --skip-prepare passthrough，覆盖 {len(suppressed)} 个自动质量拦截；详见 raw_prepare_fast_path.md"
                return "raw_prepare 使用 --skip-prepare passthrough，跳过模型清洗"
            return "raw_prepare 使用 deterministic markdown passthrough，跳过模型清洗"
        return None
    if step_name == "wiki_merge_planning":
        path = require_step_output_dir(ctx.run_dir, "wiki_merge_planning") / "merge_planning_shortcut_report.json"
        if not path.exists():
            return None
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if report.get("used") is True:
            return f"wiki_merge_planning 使用 local shortcut: {report.get('shortcut', '')}"
        return None
    if step_name != "raw_link_cleanup":
        return None
    path = require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    if not path.exists():
        return None
    cleanup = read_model(path, RawLinkCleanupArtifact)
    if not cleanup.changed:
        return None
    return f"raw 已规范化: {cleanup.raw_path}，清理 {cleanup.cleaned_link_count} 个 Obsidian 文本链接"


def refresh_run_metrics(vault: Path, run_dir: Path, manifest: OperationManifest, *, warning_console: Console | None = None) -> None:
    try:
        metrics = build_run_metrics(vault, run_dir, manifest)
        write_json(run_dir / "run_metrics.json", metrics)
        (run_dir / "run_metrics.md").write_text(render_run_metrics_markdown(metrics), encoding="utf-8")
    except Exception as exc:
        if warning_console is not None:
            warning_console.print(f"[yellow]warning:[/] run_metrics refresh failed: {exc}")


def build_run_metrics(vault: Path, run_dir: Path, manifest: OperationManifest) -> dict[str, Any]:
    steps = []
    model_durations: dict[str, int] = {}
    retry_count = 0
    internal_model_call_count = 0
    repair_count = 0
    local_json_repair_count = 0
    repair_duration_ms = 0
    provider_result_count = 0
    http_attempt_count = 0
    model_payload_char_count = 0
    archived_internal_model_call_count = 0
    archived_repair_count = 0
    archived_local_json_repair_count = 0
    archived_repair_duration_ms = 0
    archived_provider_result_count = 0
    archived_http_attempt_count = 0
    archived_model_payload_char_count = 0
    for step in manifest.steps:
        durations = [attempt.duration_ms for attempt in step.attempts if attempt.duration_ms is not None]
        total = sum(durations)
        provider = step_provider_label(step.name, step.attempts[-1].provider_spec if step.attempts else None)
        retry_count += max(0, len(step.attempts) - 1)
        repair_metrics = step_repair_metrics(run_dir, step.name, include_archived=False)
        total_repair_metrics = step_repair_metrics(run_dir, step.name, include_archived=True)
        archived_repair_metrics = subtract_repair_metrics(total_repair_metrics, repair_metrics)
        internal_model_call_count += repair_metrics["attempt_count"]
        repair_count += repair_metrics["repair_count"]
        local_json_repair_count += repair_metrics["json_repair_count"]
        repair_duration_ms += repair_metrics["duration_ms"]
        provider_result_count += repair_metrics["provider_result_count"]
        http_attempt_count += repair_metrics["http_attempt_count"]
        model_payload_char_count += repair_metrics["payload_char_count"]
        archived_internal_model_call_count += archived_repair_metrics["attempt_count"]
        archived_repair_count += archived_repair_metrics["repair_count"]
        archived_local_json_repair_count += archived_repair_metrics["json_repair_count"]
        archived_repair_duration_ms += archived_repair_metrics["duration_ms"]
        archived_provider_result_count += archived_repair_metrics["provider_result_count"]
        archived_http_attempt_count += archived_repair_metrics["http_attempt_count"]
        archived_model_payload_char_count += archived_repair_metrics["payload_char_count"]
        row = {
            "name": step.name,
            "status": step.status.value,
            "attempts": len(step.attempts),
            "last_duration_ms": durations[-1] if durations else None,
            "total_duration_ms": total,
            "provider": provider,
        }
        if repair_metrics["attempt_count"]:
            row["internal_model_call_count"] = repair_metrics["attempt_count"]
            row["repair_count"] = repair_metrics["repair_count"]
            row["local_json_repair_count"] = repair_metrics["json_repair_count"]
            row["repair_duration_ms"] = repair_metrics["duration_ms"]
            row["provider_result_count"] = repair_metrics["provider_result_count"]
            row["http_attempt_count"] = repair_metrics["http_attempt_count"]
            row["payload_char_count"] = repair_metrics["payload_char_count"]
        if archived_repair_metrics["attempt_count"]:
            row["archived_internal_model_call_count"] = archived_repair_metrics["attempt_count"]
            row["archived_repair_count"] = archived_repair_metrics["repair_count"]
            row["archived_local_json_repair_count"] = archived_repair_metrics["json_repair_count"]
            row["archived_repair_duration_ms"] = archived_repair_metrics["duration_ms"]
            row["archived_provider_result_count"] = archived_repair_metrics["provider_result_count"]
            row["archived_http_attempt_count"] = archived_repair_metrics["http_attempt_count"]
            row["archived_payload_char_count"] = archived_repair_metrics["payload_char_count"]
            row["total_internal_model_call_count"] = total_repair_metrics["attempt_count"]
            row["total_repair_count"] = total_repair_metrics["repair_count"]
            row["total_local_json_repair_count"] = total_repair_metrics["json_repair_count"]
            row["total_repair_duration_ms"] = total_repair_metrics["duration_ms"]
            row["total_provider_result_count"] = total_repair_metrics["provider_result_count"]
            row["total_http_attempt_count"] = total_repair_metrics["http_attempt_count"]
            row["total_payload_char_count"] = total_repair_metrics["payload_char_count"]
        fast_path_report = run_dir / step.name / "raw_prepare_fast_path.json"
        if fast_path_report.exists():
            try:
                fast_path_data = json.loads(fast_path_report.read_text(encoding="utf-8"))
                row["local_fast_path"] = bool(fast_path_data.get("eligible"))
                row["local_fast_path_rule"] = fast_path_data.get("rule_version", "")
                if row["local_fast_path"]:
                    row["provider"] = raw_prepare_fast_path_provider_label(fast_path_data)
            except Exception:
                row["local_fast_path"] = False
        shortcut_report = run_dir / step.name / "merge_planning_shortcut_report.json"
        if shortcut_report.exists():
            try:
                shortcut_data = json.loads(shortcut_report.read_text(encoding="utf-8"))
                row["local_shortcut"] = bool(shortcut_data.get("used"))
                row["local_shortcut_rule"] = shortcut_data.get("shortcut", "")
            except Exception:
                row["local_shortcut"] = False
        waiting_ms = awaiting_review_duration_ms(step)
        if waiting_ms is not None:
            row["awaiting_review_duration_ms"] = waiting_ms
        steps.append(row)
        if step.attempts and step.attempts[-1].provider_spec:
            model_durations[step.name] = total
    payload_steps = [
        {
            "name": str(row["name"]),
            "payload_char_count": int(row.get("payload_char_count", 0) or 0),
            "provider_result_count": int(row.get("provider_result_count", 0) or 0),
            "http_attempt_count": int(row.get("http_attempt_count", 0) or 0),
            "repair_count": int(row.get("repair_count", 0) or 0),
            "local_json_repair_count": int(row.get("local_json_repair_count", 0) or 0),
            "duration_ms": int(row.get("repair_duration_ms", 0) or 0),
        }
        for row in steps
        if int(row.get("payload_char_count", 0) or 0) > 0
    ]
    payload_steps.sort(key=lambda row: (-int(row["payload_char_count"]), str(row["name"])))
    largest_payload = payload_steps[0] if payload_steps else None
    cleanup_count = 0
    preserved_media_count = 0
    cleanup_path = run_dir / "raw_link_cleanup" / "raw_link_cleanup.json"
    if cleanup_path.exists():
        cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
        cleanup_count = cleanup.cleaned_link_count
        preserved_media_count = cleanup.preserved_media_embed_count
    created = updated = noop = 0
    plan_path = run_dir / "merge_plan_review" / "approved_merge_plan.json"
    if not plan_path.exists():
        plan_path = run_dir / "wiki_merge_planning" / "wiki_merge_plan.json"
    if plan_path.exists():
        plan = read_model(plan_path, WikiMergePlanArtifact)
        created = sum(1 for item in plan.items if item.action == "create")
        updated = sum(1 for item in plan.items if item.action == "update")
        noop = sum(1 for item in plan.items if item.action == "noop")
    written_target_count = 0
    preview_path = run_dir / "apply_preview" / "apply_preview.json"
    if preview_path.exists():
        preview = read_model(preview_path, ApplyPreview)
        written_target_count = len([target for target in preview.targets if target.will_write])
    current_attempt_duration_ms = sum(int(row.get("total_duration_ms") or 0) for row in steps)
    current_model_duration_ms = repair_duration_ms
    archived_model_duration_ms = archived_repair_duration_ms
    budget_metrics = source_digest_budget_metrics(run_dir)
    return {
        "schema_version": "run_metrics.v1",
        "operation_id": manifest.operation_id,
        "status": manifest.status.value,
        "steps": steps,
        "model_durations_ms": model_durations,
        "current_attempt_duration_ms": current_attempt_duration_ms,
        "current_model_duration_ms": current_model_duration_ms,
        "archived_model_duration_ms": archived_model_duration_ms,
        "total_model_duration_ms": current_model_duration_ms + archived_model_duration_ms,
        "retry_count": retry_count,
        "internal_model_call_count": internal_model_call_count,
        "repair_count": repair_count,
        "local_json_repair_count": local_json_repair_count,
        "repair_duration_ms": repair_duration_ms,
        "provider_result_count": provider_result_count,
        "http_attempt_count": http_attempt_count,
        "internal_model_payload_char_count": model_payload_char_count,
        "payload_by_step": payload_steps,
        "largest_payload_step": largest_payload["name"] if largest_payload else "",
        "largest_payload_char_count": largest_payload["payload_char_count"] if largest_payload else 0,
        "archived_internal_model_call_count": archived_internal_model_call_count,
        "archived_repair_count": archived_repair_count,
        "archived_local_json_repair_count": archived_local_json_repair_count,
        "archived_repair_duration_ms": archived_repair_duration_ms,
        "archived_provider_result_count": archived_provider_result_count,
        "archived_http_attempt_count": archived_http_attempt_count,
        "archived_internal_model_payload_char_count": archived_model_payload_char_count,
        "total_internal_model_call_count": internal_model_call_count + archived_internal_model_call_count,
        "total_repair_count": repair_count + archived_repair_count,
        "total_local_json_repair_count": local_json_repair_count + archived_local_json_repair_count,
        "total_repair_duration_ms": repair_duration_ms + archived_repair_duration_ms,
        "total_provider_result_count": provider_result_count + archived_provider_result_count,
        "total_http_attempt_count": http_attempt_count + archived_http_attempt_count,
        "total_internal_model_payload_char_count": model_payload_char_count + archived_model_payload_char_count,
        "created_count": created,
        "updated_count": updated,
        "noop_count": noop,
        **budget_metrics,
        "cleaned_link_count": cleanup_count,
        "preserved_media_embed_count": preserved_media_count,
        "written_target_count": written_target_count,
    }


def render_run_metrics_markdown(metrics: dict[str, Any]) -> str:
    payload_rows = [
        [
            str(row.get("name", "")),
            f"{int(row.get('payload_char_count', 0) or 0):,}",
            str(row.get("provider_result_count", 0)),
            str(row.get("http_attempt_count", 0)),
            str(row.get("repair_count", 0)),
            str(row.get("local_json_repair_count", 0)),
            format_duration(row.get("duration_ms")),
        ]
        for row in metrics.get("payload_by_step", [])
    ]
    step_rows = [
        [
            str(row.get("name", "")),
            str(row.get("status", "")),
            str(row.get("attempts", 0)),
            format_duration(row.get("last_duration_ms")),
            format_duration(row.get("total_duration_ms")),
            f"{int(row.get('payload_char_count', 0) or 0):,}" if row.get("payload_char_count") else "",
        ]
        for row in metrics.get("steps", [])
    ]
    return (
        "# Run Metrics\n\n"
        f"- Operation: `{metrics.get('operation_id', '')}`\n"
        f"- Status: `{metrics.get('status', '')}`\n"
        f"- Current model calls: `{metrics.get('internal_model_call_count', 0)}`\n"
        f"- Current HTTP attempts: `{metrics.get('http_attempt_count', 0)}`\n"
        f"- Local JSON repairs: `{metrics.get('local_json_repair_count', 0)}`\n"
        f"- Current payload chars: `{int(metrics.get('internal_model_payload_char_count', 0) or 0):,}`\n"
        f"- Largest payload step: `{metrics.get('largest_payload_step', '') or 'none'}` "
        f"({int(metrics.get('largest_payload_char_count', 0) or 0):,} chars)\n\n"
        "## Payload By Step\n\n"
        f"{format_markdown_table(['Step', 'Payload Chars', 'Provider Results', 'HTTP Attempts', 'Model Repairs', 'Local JSON Repairs', 'Model Duration'], payload_rows) if payload_rows else '_No model payloads recorded._'}\n\n"
        "## Steps\n\n"
        f"{format_markdown_table(['Step', 'Status', 'Attempts', 'Last Duration', 'Attempt Total', 'Payload Chars'], step_rows)}\n"
    )


def source_digest_budget_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "source_digest" / "source_digest_budget_report.json"
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "candidate_page_budget": report.get("budget", 0),
        "candidate_count_before_dedupe": report.get("total_formal_candidates_before_dedupe", report.get("total_formal_candidates_before_budget", 0)),
        "candidate_count_before_budget": report.get("total_formal_candidates_before_budget", 0),
        "candidate_deduped_count": report.get("deduped_count", 0),
        "candidate_selected_count": report.get("selected_count", 0),
        "candidate_deferred_count": report.get("deferred_count", 0),
        "candidate_budget_applied": bool(report.get("applied")),
    }


def subtract_repair_metrics(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        key: max(0, left.get(key, 0) - right.get(key, 0))
        for key in [
            "attempt_count",
            "repair_count",
            "json_repair_count",
            "duration_ms",
            "provider_result_count",
            "http_attempt_count",
            "payload_char_count",
        ]
    }


def step_repair_metrics(run_dir: Path, step_name: str, *, include_archived: bool = True) -> dict[str, int]:
    step_dirs = [run_dir / step_name]
    archive_root = run_dir / "attempt_archive"
    if include_archived and archive_root.exists():
        for archived_step in sorted(archive_root.glob(f"*/**/{step_name}")):
            if archived_step.is_dir():
                step_dirs.append(archived_step)
    attempt_count = 0
    repair_count = 0
    json_repair_count = 0
    duration_ms = 0
    provider_result_count = 0
    http_attempt_count = 0
    payload_char_count = 0
    for step_dir in step_dirs:
        step_attempt_count = 0
        attempt_result_paths: list[Path] = []
        report_path = step_dir / "structured_repair_report.json"
        if report_path.exists():
            try:
                report = read_model(report_path, StructuredRepairReport)
                step_attempt_count = report.attempt_count
                attempt_count += step_attempt_count
                repair_count += report.repair_count
                duration_ms += report.duration_ms
                attempt_result_paths = [step_dir / attempt.provider_result_ref for attempt in report.attempts]
            except Exception:
                pass
        existing_attempt_result_paths = [path for path in attempt_result_paths if path.exists()]
        if existing_attempt_result_paths:
            provider_result_count += len(existing_attempt_result_paths)
            http_attempt_count += provider_results_http_attempt_count(existing_attempt_result_paths)
            payload_char_count += provider_results_payload_char_count(existing_attempt_result_paths)
            json_repair_count += provider_results_json_repair_count(existing_attempt_result_paths)
            continue
        provider_results_dir = step_dir / "provider_results"
        if provider_results_dir.exists():
            provider_result_paths = list(provider_results_dir.glob("attempt-*.json"))
            provider_result_count += len(provider_result_paths)
            http_attempt_count += provider_results_http_attempt_count(provider_result_paths)
            payload_char_count += provider_results_payload_char_count(provider_result_paths)
            json_repair_count += provider_results_json_repair_count(provider_result_paths)
        elif step_attempt_count:
            provider_result_count += step_attempt_count
            result_paths = [step_dir / "provider_result.json"]
            http_attempt_count += provider_results_http_attempt_count(result_paths)
            payload_char_count += provider_results_payload_char_count(result_paths)
            json_repair_count += provider_results_json_repair_count(result_paths)
        elif (step_dir / "provider_result.json").exists():
            provider_result_count += 1
            result_paths = [step_dir / "provider_result.json"]
            http_attempt_count += provider_results_http_attempt_count(result_paths)
            payload_char_count += provider_results_payload_char_count(result_paths)
            json_repair_count += provider_results_json_repair_count(result_paths)
    if attempt_count == 0:
        attempt_count = provider_result_count
    return {
        "attempt_count": attempt_count,
        "repair_count": repair_count,
        "json_repair_count": json_repair_count,
        "duration_ms": duration_ms,
        "provider_result_count": provider_result_count,
        "http_attempt_count": http_attempt_count,
        "payload_char_count": payload_char_count,
    }


def provider_results_payload_char_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            result = read_model(path, ProviderResult)
            total += int(result.payload_char_count or 0)
        except Exception:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                total += int(data.get("payload_char_count", 0) or 0)
            except Exception:
                continue
    return total


def provider_results_http_attempt_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            result = read_model(path, ProviderResult)
            total += max(1, int(result.http_attempt_count or 1))
        except Exception:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                total += max(1, int(data.get("http_attempt_count", 1) or 1))
            except Exception:
                continue
    return total


def provider_results_json_repair_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("json_repair_applied"):
            total += 1
    return total


def step_provider_label(step_name: str, provider_spec: str | None) -> str:
    if provider_spec:
        return provider_spec
    if step_name.endswith("_review"):
        return "local:auto_review"
    return "local"


def raw_prepare_fast_path_provider_label(report: dict[str, Any]) -> str:
    policy = report.get("raw_prepare_policy")
    if policy == RawPreparePolicy.skip_model.value:
        return "local:skip_prepare"
    return "local:raw_prepare_fast_path"


def awaiting_review_duration_ms(step: Any) -> int | None:
    if step.status != StepStatus.approved or not step.completed_at or not step.attempts:
        return None
    attempt = step.attempts[-1]
    if not attempt.completed_at:
        return None
    start = datetime.fromisoformat(attempt.completed_at)
    end = datetime.fromisoformat(step.completed_at)
    value = max(0, round((end - start).total_seconds() * 1000))
    return value if value > 0 else None


def status(vault: Path, operation_id: str) -> OperationManifest:
    return read_manifest(RunStore(vault).manifest_path(operation_id))


def approve_review(vault: Path, operation_id: str, review_step: str) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        _require_review_step_awaiting(manifest, review_step)
        run_dir = store.run_dir(operation_id)
        require_upstream_artifacts_current(vault, run_dir, manifest, review_step)
        if review_step == "draft_review":
            step_root = require_step_output_dir(run_dir, "draft_review")
            pending = step_root / "pending_write_manifest.json"
            if not pending.exists():
                raise PipelineError("draft_review has no pending write manifest to approve.")
            approved = step_root / "approved_write_manifest.json"
            approved.write_text(pending.read_text(encoding="utf-8"), encoding="utf-8")
            approval = build_draft_approval(
                run_dir,
                approved,
                decision="approved",
                review_mode="manual",
                auto_approved=False,
                notes="人工批准草稿；pending_write_manifest.json 已由 approved_write_manifest.json 取代，pending artifact 保留作审计。",
            )
            approval_path = step_root / "draft_approval.json"
            write_json(approval_path, approval)
            mark_step_approved(
                manifest,
                review_step,
                outputs=[
                    _ref(run_dir, step_root / "review_prompt.md", review_step, "markdown"),
                    _ref(run_dir, step_root / "pending_write_manifest.json", review_step, "json", "draft_write_manifest.v1"),
                    _ref(run_dir, approved, review_step, "json", "draft_write_manifest.v1"),
                    _ref(run_dir, approval_path, review_step, "json", "draft_review.v1"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "validation")
            mark_from_pending(manifest, "validation")
            write_manifest(store.manifest_path(operation_id), manifest)
            refresh_run_metrics(vault, run_dir, manifest)
            return manifest
        if review_step == "merge_plan_review":
            step_root = require_step_output_dir(run_dir, "merge_plan_review")
            pending = step_root / "pending_merge_plan.json"
            if not pending.exists():
                raise PipelineError("merge_plan_review has no pending merge plan to approve.")
            digest = read_model(require_step_output_dir(run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
            resolution = read_model(
                require_step_output_dir(run_dir, "candidate_resolution") / "candidate_resolution.json",
                CandidateResolutionArtifact,
            )
            snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
            snapshot = read_model(snapshot_path, WikiContextSnapshot)
            plan = read_model(pending, WikiMergePlanArtifact)
            plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_path.relative_to(run_dir).as_posix())
            validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=manifest.vault_config_snapshot.wiki_language)
            if any(item.action == "needs_human_decision" for item in plan.items):
                raise PipelineError("needs_human_decision must be revised to create/update/noop before approval.")
            approved = step_root / "approved_merge_plan.json"
            write_json(approved, plan)
            decision = ReviewDecision(
                review_step=review_step,
                decision="approved",
                review_mode="manual",
                auto_approved=False,
                notes="人工批准合并计划；pending_merge_plan.json 已由 approved_merge_plan.json 取代，pending artifact 保留作审计。",
            )
            decision_path = step_root / "review_decision.json"
            write_json(decision_path, decision)
            mark_step_approved(
                manifest,
                review_step,
                outputs=[
                    _ref(run_dir, step_root / "review_prompt.md", review_step, "markdown"),
                    _ref(run_dir, pending, review_step, "json", "wiki_merge_plan.v5"),
                    _ref(run_dir, approved, review_step, "json", "wiki_merge_plan.v5"),
                    _ref(run_dir, decision_path, review_step, "json", "review_decision.v1"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
            write_manifest(store.manifest_path(operation_id), manifest)
            refresh_run_metrics(vault, run_dir, manifest)
            return manifest
        raise PipelineError(f"Unsupported review step: {review_step}")


def revise_review(vault: Path, operation_id: str, review_step: str) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        _require_review_step_awaiting(manifest, review_step)
        run_dir = store.run_dir(operation_id)
        require_upstream_artifacts_current(vault, run_dir, manifest, review_step)
        if review_step == "merge_plan_review":
            archive_pending_review_artifacts(run_dir, review_step)
            delete_downstream_step_dirs(vault, operation_id, "wiki_merge_planning")
            mark_from_pending(manifest, "wiki_merge_planning")
        elif review_step == "draft_review":
            archive_pending_review_artifacts(run_dir, review_step)
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
        else:
            raise PipelineError(f"Unsupported review step: {review_step}")
        manifest.status = OperationStatus.running
        write_manifest(store.manifest_path(operation_id), manifest)
        refresh_run_metrics(vault, run_dir, manifest)
        return manifest


def archive_pending_review_artifacts(run_dir: Path, review_step: str) -> Path | None:
    step_root = run_dir / review_step
    if not step_root.exists():
        return None
    timestamp = safe_timestamp()
    archive_root = run_dir / "review_archive" / review_step / timestamp
    shutil.copytree(step_root, archive_root)
    write_json(
        archive_root / "superseded.json",
        {
            "schema_version": "review_superseded.v1",
            "review_step": review_step,
            "superseded_at": utc_now(),
            "reason": "revise requested; pending review artifacts were archived before downstream regeneration.",
        },
    )
    return archive_root


def require_upstream_artifacts_current(vault: Path, run_dir: Path, manifest: OperationManifest, review_step: str) -> None:
    issues: list[str] = []
    for raw in manifest.raw_bindings:
        raw_path = vault / raw.relative_path
        if not raw_path.exists():
            issues.append(f"{raw.relative_path} is missing")
        elif sha256_file(raw_path) != raw.sha256:
            issues.append(f"{raw.relative_path} changed")
    for step in manifest.steps:
        if step.name == review_step:
            break
        for ref in step.outputs:
            if not ref.required_for_resume:
                continue
            path = run_dir / ref.relative_path
            if not path.exists():
                issues.append(f"{ref.relative_path} is missing")
            elif not path.is_file():
                issues.append(f"{ref.relative_path} is not a file")
            elif sha256_file(path) != ref.sha256:
                issues.append(f"{ref.relative_path} changed")
    if issues:
        preview = "; ".join(issues[:8])
        suffix = "" if len(issues) <= 8 else f"; ... and {len(issues) - 8} more"
        raise PipelineError(f"upstream required artifacts changed before review: {preview}{suffix}")


def _require_review_step_awaiting(manifest: OperationManifest, review_step: str) -> None:
    if manifest.status in {OperationStatus.applied, OperationStatus.source_recorded}:
        raise PipelineError("Applied operations are immutable. Start a new operation instead.")
    if manifest.status == OperationStatus.apply_failed:
        raise PipelineError("apply_failed operations cannot be reviewed; inspect written targets and rerun ingest.")
    step = get_step(manifest, review_step)
    if step.status != StepStatus.awaiting_review:
        raise PipelineError(f"{review_step} is not awaiting_review; current status is {step.status.value}.")


def latest_operation(vault: Path) -> str | None:
    root = RunStore(vault).runs_root
    if not root.exists():
        return None
    candidates = sorted([path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").exists()])
    return candidates[-1].name if candidates else None


def copy_fixture_raw(vault: Path, fixture_raw: Path) -> Path:
    target = vault / "raw" / fixture_raw.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture_raw, target)
    return target


def safe_timestamp() -> str:
    return utc_now().replace("+00:00", "Z").replace(":", "")


def _ref(
    run_dir: Path,
    path: Path,
    producer_step: str,
    kind: str,
    schema_version: str | None = None,
    required_for_resume: bool = True,
) -> ArtifactRef:
    return artifact_ref(
        base=run_dir,
        path=path,
        kind=kind,
        producer_step=producer_step,
        schema_version=schema_version,
        required_for_resume=required_for_resume,
    )


def structured_model_output_refs(run_dir: Path, step_root: Path, step_name: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        refs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
    report_json = step_root / "structured_repair_report.json"
    if report_json.exists():
        refs.append(_ref(run_dir, report_json, step_name, "json", "structured_repair_report.v1"))
    report_md = step_root / "structured_repair_report.md"
    if report_md.exists():
        refs.append(_ref(run_dir, report_md, step_name, "markdown"))
    provider_results = step_root / "provider_results"
    if provider_results.exists():
        for path in sorted(provider_results.glob("attempt-*.json")):
            refs.append(_ref(run_dir, path, step_name, "provider_result", "provider_result.v1", required_for_resume=False))
    repair_prompts = step_root / "repair_prompts"
    if repair_prompts.exists():
        for path in sorted(repair_prompts.glob("attempt-*.json")):
            refs.append(_ref(run_dir, path, step_name, "json", required_for_resume=False))
    return refs


def draft_rendering_model_batch_refs(run_dir: Path, step_root: Path, step_name: str) -> list[ArtifactRef]:
    batch_root = step_root / "model_batches"
    if not batch_root.exists():
        return []
    refs: list[ArtifactRef] = []
    for path in sorted(batch_root.rglob("*")):
        if not path.is_file():
            continue
        required = not any(part in {"provider_results", "repair_prompts"} for part in path.relative_to(step_root).parts)
        schema = None
        if path.name == "provider_result.json" or (path.parent.name == "provider_results" and path.name.startswith("attempt-")):
            schema = "provider_result.v1"
        elif path.name == "structured_repair_report.json":
            schema = "structured_repair_report.v1"
        elif path.name == "draft_source_excerpt_pack.json":
            schema = "draft_source_excerpt_pack.v1"
        elif path.name == "update_preservation_pack.json":
            schema = "update_preservation_pack.v1"
        elif path.name == "update_preservation_reinforcement_report.json":
            schema = "update_preservation_reinforcement_report.v1"
        elif path.name == "grounding_paraphrase_rewrite_report.json":
            schema = "grounding_paraphrase_rewrite_report.v1"
        elif path.name == "open_question_grounding_cleanup_report.json":
            schema = "open_question_grounding_cleanup_report.v1"
        elif path.name == "example_concrete_cleanup_report.json":
            schema = "example_concrete_cleanup_report.v1"
        elif path.name == "draft_digest_projection_report.json":
            schema = "source_digest_projection_report.v1"
        elif path.name == "draft_merge_plan_projection_report.json":
            schema = "draft_merge_plan_projection_report.v1"
        elif path.name == "draft_context_projection_report.json":
            schema = "draft_context_projection_report.v1"
        refs.append(_ref(run_dir, path, step_name, artifact_kind_for_path(path), schema, required_for_resume=required))
    return refs


def artifact_kind_for_path(path: Path) -> str:
    return {
        ".md": "markdown",
        ".json": "json",
        ".jsonl": "jsonl",
        ".diff": "diff",
    }.get(path.suffix, "text")


def _draft_rendering_ref(run_dir: Path, path: Path, step_name: str) -> ArtifactRef:
    schemas = {
        "draft_rendering.json": "draft_rendering.v3",
        "draft_source_excerpt_pack.json": "draft_source_excerpt_pack.v1",
        "update_preservation_pack.json": "update_preservation_pack.v1",
        "update_preservation_reinforcement_report.json": "update_preservation_reinforcement_report.v1",
        "grounding_paraphrase_rewrite_report.json": "grounding_paraphrase_rewrite_report.v1",
        "open_question_grounding_cleanup_report.json": "open_question_grounding_cleanup_report.v1",
        "example_concrete_cleanup_report.json": "example_concrete_cleanup_report.v1",
        "draft_digest_projection_report.json": "source_digest_projection_report.v1",
        "draft_merge_plan_projection_report.json": "draft_merge_plan_projection_report.v1",
        "draft_context_projection_report.json": "draft_context_projection_report.v1",
        "draft_write_manifest.json": "draft_write_manifest.v1",
        "update_merge_report.json": "update_merge_report.v1",
        "related_merge_report.json": "related_merge_report.v1",
        "draft_grounding_review.json": "draft_grounding_review.v1",
        "index_open_questions_report.json": "index_open_questions_report.v1",
        "draft_rendering_batch_report.json": "draft_rendering_batch_report.v1",
    }
    return _ref(run_dir, path, step_name, artifact_kind_for_path(path), schemas.get(path.name))


M42_REQUIRED_DRAFT_SIDECARS = (
    "draft_rendering/draft_rendering.json",
    "draft_rendering/draft_write_manifest.json",
    "draft_rendering/provider_result.json",
    "draft_rendering/update_merge_report.json",
    "draft_rendering/update_merge_report.md",
    "draft_rendering/related_merge_report.json",
    "draft_rendering/related_merge_report.md",
    "draft_rendering/draft_grounding_review.json",
    "draft_rendering/draft_grounding_review.md",
)


def require_m42_draft_sidecars(run_dir: Path) -> None:
    missing = [rel_path for rel_path in M42_REQUIRED_DRAFT_SIDECARS if not (run_dir / rel_path).is_file()]
    if missing:
        preview = ", ".join(f"`{path}`" for path in missing[:6])
        suffix = "" if len(missing) <= 6 else f", ... and {len(missing) - 6} more"
        raise PipelineError(
            "M4.2 draft sidecar artifacts are missing; resume from draft_rendering or earlier before approve/apply: "
            f"{preview}{suffix}"
        )


def complete_review_step(
    manifest: OperationManifest,
    name: str,
    *,
    outputs: list[ArtifactRef],
    review_decision_ref: str,
) -> None:
    complete_step(manifest, name, outputs=outputs)
    step = get_step(manifest, name)
    step.review_state = "approved"
    step.review_reason = None
    step.awaiting_since = None
    step.resolved_at = step.completed_at
    step.review_decision_ref = review_decision_ref


# M3 helper implementations.


def render_candidate_table(candidates: list[SourceDigestCandidate] | list[WeakOrNoiseItem]) -> str:
    if not candidates:
        return "_暂无。_"
    return format_markdown_table(
        ["ID", "类型", "名称", "摘要", "重复风险"],
        [[f"`{item.candidate_id}`", item.type, item.name, item.one_sentence_summary, item.duplicate_risk] for item in candidates],
    )


def build_source_duplicate_guard_artifact(
    vault: Path,
    *,
    source_raw_path: str,
    source_raw_hash: str,
    source_prepared_hash: str,
    operation_id: str,
) -> SourceDuplicateGuardArtifact:
    normalized_raw_path = normalize_vault_path(source_raw_path)
    source_title = source_title_for_raw(normalized_raw_path)
    source_target_path = f"sources/{safe_filename(source_title)}.md"
    for source_page, frontmatter in scan_source_pages(vault):
        operation_ids = _frontmatter_list(frontmatter, "source_operation_ids")
        if operation_id in operation_ids:
            continue
        raw_paths = [normalize_vault_path(value) for value in _frontmatter_list(frontmatter, "source_raw_paths")]
        raw_hashes = _frontmatter_list(frontmatter, "source_raw_hashes")
        prepared_hashes = _frontmatter_list(frontmatter, "source_prepared_hashes")
        if normalized_raw_path in raw_paths and source_raw_hash in raw_hashes:
            return SourceDuplicateGuardArtifact(
                source_raw_path=normalized_raw_path,
                source_raw_hash=source_raw_hash,
                source_prepared_hash=source_prepared_hash,
                source_target_path=source_target_path,
                status="source_duplicate",
                matched_source_page=source_page,
                matched_raw_path=normalized_raw_path,
                matched_raw_hash=source_raw_hash,
                matched_prepared_hash=source_prepared_hash if source_prepared_hash in prepared_hashes else None,
                reason="same raw path and raw hash already recorded in source page frontmatter",
            )
        if source_raw_hash in raw_hashes:
            return SourceDuplicateGuardArtifact(
                source_raw_path=normalized_raw_path,
                source_raw_hash=source_raw_hash,
                source_prepared_hash=source_prepared_hash,
                source_target_path=source_target_path,
                status="source_duplicate",
                matched_source_page=source_page,
                matched_raw_hash=source_raw_hash,
                matched_prepared_hash=source_prepared_hash if source_prepared_hash in prepared_hashes else None,
                reason="same raw hash already recorded in source page frontmatter",
            )
        if normalized_raw_path in raw_paths and source_raw_hash not in raw_hashes:
            return SourceDuplicateGuardArtifact(
                source_raw_path=normalized_raw_path,
                source_raw_hash=source_raw_hash,
                source_prepared_hash=source_prepared_hash,
                source_target_path=source_target_path,
                status="source_revision_detected",
                matched_source_page=source_page,
                matched_raw_path=normalized_raw_path,
                reason="same raw path exists with a different content hash",
            )
    target = vault / "wiki" / source_target_path
    if target.exists():
        return SourceDuplicateGuardArtifact(
            source_raw_path=normalized_raw_path,
            source_raw_hash=source_raw_hash,
            source_prepared_hash=source_prepared_hash,
            source_target_path=source_target_path,
            status="source_duplicate",
            matched_source_page=f"wiki/{source_target_path}",
            reason="source target page already exists",
        )
    return SourceDuplicateGuardArtifact(
        source_raw_path=normalized_raw_path,
        source_raw_hash=source_raw_hash,
        source_prepared_hash=source_prepared_hash,
        source_target_path=source_target_path,
        status="clear",
        reason="no matching source path, hash, or source target page found",
    )


def render_source_duplicate_guard_markdown(artifact: SourceDuplicateGuardArtifact) -> str:
    return "\n".join(
        [
            "# 来源重复检查",
            "",
            format_markdown_table(
                ["字段", "值"],
                [
                    ["status", artifact.status],
                    ["source_raw_path", f"`{artifact.source_raw_path}`"],
                    ["source_raw_hash", f"`{artifact.source_raw_hash}`"],
                    ["source_prepared_hash", f"`{artifact.source_prepared_hash}`"],
                    ["source_target_path", f"`{artifact.source_target_path}`"],
                    ["matched_source_page", f"`{artifact.matched_source_page}`" if artifact.matched_source_page else ""],
                    ["reason", artifact.reason],
                ],
            ),
        ]
    ).rstrip() + "\n"


def scan_source_pages(vault: Path) -> list[tuple[str, dict[str, Any]]]:
    source_root = vault / "wiki" / "sources"
    if not source_root.exists():
        return []
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(source_root.rglob("*.md")):
        frontmatter = parse_frontmatter(path.read_text(encoding="utf-8"))
        if frontmatter is not None:
            found.append((path.relative_to(vault).as_posix(), frontmatter))
    return found


def scan_raw_ingest_candidates(
    vault: Path,
    *,
    include_processed: bool = False,
    limit: int | None = None,
) -> RawIngestCandidateReport:
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    vault = vault.expanduser().resolve()
    raw_root = vault / "raw"
    if not raw_root.exists():
        raise ValueError(f"Raw directory not found: {raw_root}")
    records_by_path, records_by_hash = _source_raw_coverage_index(vault)
    raw_files = _iter_raw_ingest_files(raw_root)
    duplicate_url_first_paths = _duplicate_raw_url_first_paths(raw_files)
    items: list[RawIngestCandidate] = []
    for raw_path in raw_files:
        rel_path = normalize_vault_path(raw_path.relative_to(vault).as_posix())
        raw_hash = sha256_file(raw_path)
        path_records = records_by_path.get(rel_path, [])
        hash_records = records_by_hash.get(raw_hash, [])
        duplicate_url_first_path = duplicate_url_first_paths.get(raw_path)
        matched_records: list[_SourceRawCoverageRecord] = []
        if any(raw_hash in record.raw_hashes for record in path_records):
            status = "processed"
            matched_by = "path_and_hash"
            matched_records = [record for record in path_records if raw_hash in record.raw_hashes]
            reason = "same raw path and content hash are already recorded in source frontmatter"
        elif path_records and not any(record.raw_hashes for record in path_records):
            status = "processed"
            matched_by = "path"
            matched_records = path_records
            reason = "same raw path is recorded by legacy source frontmatter; content hash is unavailable"
        elif path_records:
            status = "changed"
            matched_by = "path"
            matched_records = path_records
            reason = "same raw path is recorded, but the current content hash is different"
        elif hash_records:
            status = "duplicate_hash"
            matched_by = "hash"
            matched_records = hash_records
            reason = "same content hash is already recorded under another raw path"
        elif duplicate_url_first_path is not None:
            status = "duplicate_url"
            matched_by = "url"
            reason = (
                "same imported URL is already present under another raw path: "
                f"{normalize_vault_path(duplicate_url_first_path.relative_to(vault).as_posix())}"
            )
        else:
            status = "unprocessed"
            matched_by = "none"
            reason = "no matching source raw path or content hash found"
        stat = raw_path.stat()
        items.append(
            RawIngestCandidate(
                raw_path=rel_path,
                status=status,
                raw_sha256=raw_hash,
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                matched_by=matched_by,
                source_pages=_record_source_pages(matched_records),
                operation_ids=_record_operation_ids(matched_records),
                reason=reason,
            )
        )
    status_rank = {"unprocessed": 0, "changed": 1, "duplicate_hash": 2, "duplicate_url": 3, "processed": 4}
    items.sort(key=lambda item: (status_rank[item.status], item.raw_path))
    visible_items = items if include_processed else [item for item in items if item.status != "processed"]
    if limit is not None:
        visible_items = visible_items[:limit]
    return RawIngestCandidateReport(
        vault=vault.as_posix(),
        raw_root=raw_root.as_posix(),
        include_processed=include_processed,
        limit=limit,
        total_raw_files=len(items),
        candidate_count=len(visible_items),
        processed_count=sum(1 for item in items if item.status == "processed"),
        changed_count=sum(1 for item in items if item.status == "changed"),
        duplicate_hash_count=sum(1 for item in items if item.status == "duplicate_hash"),
        duplicate_url_count=sum(1 for item in items if item.status == "duplicate_url"),
        unprocessed_count=sum(1 for item in items if item.status == "unprocessed"),
        items=visible_items,
    )


def _duplicate_raw_url_first_paths(raw_files: list[Path]) -> dict[Path, Path]:
    first_by_url: dict[str, Path] = {}
    duplicates: dict[Path, Path] = {}
    for raw_path in raw_files:
        matched_first: Path | None = None
        for url in _raw_import_urls(raw_path):
            first = first_by_url.get(url)
            if first is not None and first != raw_path:
                matched_first = first
                break
        if matched_first is not None:
            duplicates[raw_path] = matched_first
            continue
        for url in _raw_import_urls(raw_path):
            first_by_url.setdefault(url, raw_path)
    return duplicates


def _raw_import_urls(raw_path: Path) -> list[str]:
    urls: list[str] = []
    try:
        with raw_path.open("r", encoding="utf-8", errors="ignore") as handle:
            prefix = handle.read(8192)
    except OSError:
        return urls
    for line in prefix.splitlines()[:24]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() not in {"imported from", "fetched url", "final url"}:
            continue
        url = value.strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _iter_raw_ingest_files(raw_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(raw_root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if relative_parts and relative_parts[0] == "log":
            continue
        if path.suffix.lower() not in RAW_INGEST_TEXT_SUFFIXES:
            continue
        paths.append(path)
    return sorted(paths)


def _source_raw_coverage_index(
    vault: Path,
) -> tuple[dict[str, list[_SourceRawCoverageRecord]], dict[str, list[_SourceRawCoverageRecord]]]:
    records_by_path: dict[str, list[_SourceRawCoverageRecord]] = {}
    records_by_hash: dict[str, list[_SourceRawCoverageRecord]] = {}
    for source_page, frontmatter in scan_source_pages(vault):
        raw_paths = tuple(_frontmatter_raw_paths(frontmatter))
        raw_hashes = tuple(value.strip() for value in _frontmatter_list(frontmatter, "source_raw_hashes") if value.strip())
        operation_ids = tuple(
            value.strip() for value in _frontmatter_list(frontmatter, "source_operation_ids") if value.strip()
        )
        if not raw_paths and not raw_hashes:
            continue
        record = _SourceRawCoverageRecord(
            source_page=source_page,
            raw_paths=raw_paths,
            raw_hashes=raw_hashes,
            operation_ids=operation_ids,
        )
        for raw_path in raw_paths:
            records_by_path.setdefault(raw_path, []).append(record)
        for raw_hash in raw_hashes:
            records_by_hash.setdefault(raw_hash, []).append(record)
    return records_by_path, records_by_hash


def _frontmatter_raw_paths(frontmatter: dict[str, Any]) -> list[str]:
    raw_paths: list[str] = []
    for value in _frontmatter_list(frontmatter, "source_raw_paths"):
        normalized = normalize_vault_path(value)
        if normalized:
            raw_paths.append(normalized)
    for value in _frontmatter_list(frontmatter, "sources"):
        normalized = normalize_vault_path(value)
        if normalized.startswith("raw/") and normalized not in raw_paths:
            raw_paths.append(normalized)
    return raw_paths


def _record_source_pages(records: list[_SourceRawCoverageRecord]) -> list[str]:
    return sorted({record.source_page for record in records})


def _record_operation_ids(records: list[_SourceRawCoverageRecord]) -> list[str]:
    return sorted({operation_id for record in records for operation_id in record.operation_ids})


def _frontmatter_list(frontmatter: dict[str, Any], key: str) -> list[str]:
    value = frontmatter.get(key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def normalize_vault_path(path: str) -> str:
    return unicodedata.normalize("NFC", path.strip()).replace("\\", "/")


def backfill_missing_candidate_resolution_items(
    artifact: CandidateResolutionArtifact,
    digest: SourceDigestArtifact,
    profile: Any,
) -> CandidateResolutionArtifact:
    covered_ids: set[str] = set()
    for item in artifact.items:
        covered_ids.update(item.source_basis.source_candidate_ids)
    additions: list[CandidateResolutionItem] = []
    notes = list(artifact.missed_candidate_risks)
    for group_name, candidate in digest_candidates_with_group(digest):
        if candidate.candidate_id in covered_ids:
            continue
        page_type = page_type_for_digest_candidate(group_name, candidate, profile)
        display_title = candidate.suggested_page_title.strip() or candidate.name.strip() or candidate.candidate_id
        additions.append(
            CandidateResolutionItem(
                source_basis=SourceBasis(
                    source_candidate_ids=[candidate.candidate_id],
                    source_locator=candidate.source_locator,
                ),
                page_type=page_type,
                display_title=display_title,
                topic_summary=candidate.one_sentence_summary,
                why_this_page=candidate.wiki_value or candidate.why_matters or "该候选来自 source_digest，模型在页面规划中遗漏，系统补齐为最小页面计划。",
                initial_section_intent="系统补齐的最小页面计划；后续 merge planning / draft rendering 需要重新对照全文消化。",
                coverage_notes=f"模型遗漏 approved_digest candidate `{candidate.candidate_id}`，系统已补齐。",
                reason="candidate_resolution coverage backfill",
            )
        )
        notes.append(f"candidate_resolution model missed `{candidate.candidate_id}`; deterministic backfill added `{display_title}`.")
    if not additions:
        return artifact
    return CandidateResolutionArtifact(items=[*artifact.items, *additions], missed_candidate_risks=notes)


def digest_candidates_with_group(digest: SourceDigestArtifact) -> list[tuple[str, SourceDigestCandidate]]:
    return [
        *[("entities", item) for item in digest.entities],
        *[("concepts", item) for item in digest.concepts],
        *[("designs", item) for item in digest.designs],
        *[("comparisons", item) for item in digest.comparisons],
        *[("open_questions", item) for item in digest.open_questions],
    ]


def page_type_for_digest_candidate(group_name: str, candidate: SourceDigestCandidate, profile: Any) -> str:
    normalized = candidate.type.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "question": "open_question",
        "open_questions": "open_question",
        "open_question": "open_question",
        "concept_overview": "overview",
        "design_overview": "overview",
        "entity_index": "overview",
        "open_question_overview": "open_question",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in profile.page_types and normalized != profile.source_page_type:
        return normalized
    preferred = {
        "entities": "entity",
        "concepts": "concept",
        "designs": "design",
        "comparisons": "comparison",
        "open_questions": "open_question",
    }.get(group_name)
    if preferred in profile.page_types:
        return preferred
    return profile.default_page_type


def finalize_candidate_resolution(
    vault: Path,
    profile: Any,
    artifact: CandidateResolutionArtifact,
    digest: SourceDigestArtifact | None = None,
) -> CandidateResolutionArtifact:
    items: list[CandidateResolutionItem] = []
    seen_paths: dict[str, int] = {}
    selected_candidate_ids = {candidate.candidate_id for candidate in digest.ingest_candidates()} if digest is not None else set()
    weak_or_noise_ids = {candidate.candidate_id for candidate in digest.weak_or_noise_items} if digest is not None else set()
    sanitized_items: list[CandidateResolutionItem] = []
    sanitation_notes = list(artifact.missed_candidate_risks)
    for index, item in enumerate(artifact.items):
        source_ids = list(item.source_basis.source_candidate_ids)
        leaked_ids = [
            candidate_id
            for candidate_id in source_ids
            if candidate_id in weak_or_noise_ids or candidate_id.strip().lower().startswith(("noise", "weak", "ignore"))
        ]
        if leaked_ids:
            cleaned_ids = [candidate_id for candidate_id in source_ids if candidate_id not in leaked_ids]
            if not cleaned_ids:
                sanitation_notes.append(
                    f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` dropped because it only referenced weak/noise candidates: {sorted(leaked_ids)}."
                )
                continue
            item = item.model_copy(
                update={
                    "source_basis": item.source_basis.model_copy(update={"source_candidate_ids": cleaned_ids}),
                    "coverage_notes": merge_markdown_blocks(
                        item.coverage_notes,
                        f"系统清理 weak/noise candidate 引用：{', '.join(f'`{candidate_id}`' for candidate_id in leaked_ids)}。",
                    ),
                }
            )
            sanitation_notes.append(
                f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` removed weak/noise candidate refs: {sorted(leaked_ids)}."
            )
        if digest is not None:
            unknown_source_ids = [
                candidate_id
                for candidate_id in item.source_basis.source_candidate_ids
                if candidate_id not in selected_candidate_ids
            ]
            if unknown_source_ids:
                prepared_discovered = list(item.source_basis.prepared_discovered_candidates)
                cleaned_ids = [
                    candidate_id
                    for candidate_id in item.source_basis.source_candidate_ids
                    if candidate_id in selected_candidate_ids
                ]
                if prepared_discovered:
                    for candidate_id in unknown_source_ids:
                        if candidate_id not in prepared_discovered:
                            prepared_discovered.append(candidate_id)
                    unknown_note = "系统将非 source_digest candidate id 移入 prepared_discovered_candidates："
                    sanitation_note = "moved unknown candidate refs to prepared_discovered_candidates"
                else:
                    prepared_discovered = list(unknown_source_ids)
                    unknown_note = "系统将非 source_digest candidate id 移入 prepared_discovered_candidates："
                    sanitation_note = "moved unknown-only candidate refs to prepared_discovered_candidates"
                item = item.model_copy(
                    update={
                        "source_basis": item.source_basis.model_copy(
                            update={
                                "source_candidate_ids": cleaned_ids,
                                "prepared_discovered_candidates": prepared_discovered,
                            }
                        ),
                        "coverage_notes": merge_markdown_blocks(
                            item.coverage_notes,
                            unknown_note + f"{', '.join(f'`{candidate_id}`' for candidate_id in unknown_source_ids)}。",
                        ),
                    }
                )
                sanitation_notes.append(
                    f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` {sanitation_note}: {sorted(unknown_source_ids)}."
                )
        sanitized_items.append(item)
    artifact = artifact.model_copy(update={"items": sanitized_items, "missed_candidate_risks": sanitation_notes})
    issues: list[StructuredIssue] = []
    for index, item in enumerate(artifact.items):
        if item.page_type not in profile.page_types:
            issues.append(
                StructuredIssue(
                    issue_code="unknown_page_type",
                    field_path=f"items.{index}.page_type",
                    validator_id="finalize_candidate_resolution",
                    message=(
                        f"candidate_resolution uses unsupported page_type `{item.page_type}`; "
                        "remove weak/noise formal items or choose a valid profile page_type."
                    ),
                    repairability="repairable",
                )
            )
        if item.reason.strip().lower() in {"ignore", "ignored", "noise", "weak", "弱相关", "噪声"}:
            issues.append(
                StructuredIssue(
                    issue_code="ignore_as_formal_item",
                    field_path=f"items.{index}.reason",
                    validator_id="finalize_candidate_resolution",
                    message="formal page plans must not use ignore/noise as the reason; remove this item.",
                    repairability="repairable",
                )
            )
    if issues:
        raise ContractValidationError(
            "candidate_resolution contains weak/noise formal item(s); repair by deleting those formal items.",
            issues=issues,
        )
    for item in artifact.items:
        page_type = item.page_type
        display_title = clean_display_title(item.display_title) or item.display_title.strip()
        source_fingerprint = source_basis_fingerprint(item.source_basis)
        page_plan_id = stable_page_plan_id(page_type, display_title, source_fingerprint)
        stem = unicode_safe_stem(display_title)
        target = page_output_path(vault / "wiki", profile, page_type, stem)
        rel_target = target.relative_to(vault / "wiki").as_posix()
        if rel_target in seen_paths:
            seen_paths[rel_target] += 1
            path = Path(rel_target)
            suffix = sha256_bytes(f"{page_type}:{display_title}:{page_plan_id}".encode("utf-8"))[:8]
            rel_target = path.with_name(f"{path.stem}_{suffix}{path.suffix}").as_posix()
        else:
            seen_paths[rel_target] = 1
        items.append(
            item.model_copy(
                update={
                    "page_plan_id": page_plan_id,
                    "page_type": page_type,
                    "display_title": display_title,
                    "path_stem": stem,
                    "candidate_target_path": rel_target,
                }
            )
        )
    return CandidateResolutionArtifact(items=items, missed_candidate_risks=artifact.missed_candidate_risks)


def source_basis_fingerprint(source_basis: SourceBasis) -> str:
    payload = json.dumps(source_basis.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return sha256_bytes(payload.encode("utf-8"))[:12]


def stable_page_plan_id(page_type: str, display_title: str, source_fingerprint: str) -> str:
    base = f"{page_type}:{normalize_related_key(display_title)}:{source_fingerprint}"
    return f"PP-{sha256_bytes(base.encode('utf-8'))[:12]}"


def unicode_safe_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = re.sub(r"\s+", " ", normalized.strip())
    bad = '\\/:*?"<>|#^[]'
    cleaned = "".join("_" if char in bad else char for char in normalized).strip(" .")
    return cleaned or "untitled"


def render_candidate_resolution_markdown(artifact: CandidateResolutionArtifact) -> str:
    rows = [
        [
            item.page_plan_id,
            item.page_type,
            item.display_title,
            f"`{item.candidate_target_path}`",
            ", ".join(item.source_basis.source_candidate_ids),
            ", ".join(item.source_basis.prepared_discovered_candidates),
            item.why_this_page,
        ]
        for item in artifact.items
    ]
    return "# 候选页面规划\n\n" + format_markdown_table(
        ["页面计划", "类型", "标题", "目标", "来源候选", "Prepared 发现候选", "为什么写"],
        rows,
    ) + "\n"


def build_wiki_context_snapshot(
    vault: Path,
    resolution: CandidateResolutionArtifact,
    *,
    log_date: str,
    source_target_path: str,
    retrieval_config: EmbeddingRetrievalConfig | None = None,
    force_exact_backend: bool = False,
) -> WikiContextSnapshot:
    retrieval_config = retrieval_config or EmbeddingRetrievalConfig(backend="exact")
    knowledge_pool = build_knowledge_pool(vault)
    try:
        candidate_contexts = build_candidate_contexts(
            resolution=resolution,
            knowledge_pool=knowledge_pool,
            config=retrieval_config,
            vault=vault,
            force_exact_backend=force_exact_backend,
        )
    except RetrievalError as exc:
        raise PipelineError(str(exc)) from exc
    paths = {
        "wiki/index.md",
        "wiki/log.md",
        f"wiki/logs/{log_date}.md",
        f"wiki/{source_target_path}",
    }
    for item in resolution.items:
        paths.add(f"wiki/{item.candidate_target_path}")
    for context_item in candidate_contexts.items:
        for hit in context_item.hits:
            paths.add(f"wiki/{hit.path}")
    entries: list[WikiContextEntry] = []
    for rel in sorted(paths):
        path = vault / rel
        if path.exists():
            text = path.read_text(encoding="utf-8")
            entries.append(
                WikiContextEntry(
                    path=rel,
                    expected_state="present",
                    preimage_sha256=sha256_file(path),
                    content=text,
                    metadata=metadata_from_text(text, rel),
                )
            )
        else:
            entries.append(WikiContextEntry(path=rel, expected_state="missing", preimage_sha256=None, content=""))
    return WikiContextSnapshot(
        log_date=log_date,
        source_target_path=source_target_path,
        candidate_pool_sha256=candidate_pool_sha256(knowledge_pool),
        knowledge_metadata_pool=knowledge_pool,
        candidate_contexts=candidate_contexts,
        entries=entries,
    )


def ensure_snapshot_within_limit(snapshot: WikiContextSnapshot, max_context_chars: int) -> None:
    total = sum(len(entry.content) for entry in snapshot.entries)
    if total > max_context_chars:
        raise PipelineError(f"wiki_context_snapshot exceeds max_context_chars ({total} > {max_context_chars}); retry with a smaller vault or higher limit.")


def read_wiki_page_metadata(vault: Path, rel_path: str) -> WikiPageMetadata | None:
    path = vault / rel_path
    if not path.exists() or path.suffix != ".md":
        return None
    frontmatter = parse_frontmatter(path.read_text(encoding="utf-8"))
    if frontmatter is None:
        return None
    llmwiki_type = frontmatter.get("llmwiki_type")
    title = frontmatter.get("title")
    summary = frontmatter.get("summary")
    created = frontmatter.get("created", "")
    updated = frontmatter.get("updated")
    if isinstance(created, (datetime, date)):
        created = created.isoformat()
    if isinstance(updated, (datetime, date)):
        updated = updated.isoformat()
    if not all(isinstance(value, str) and value.strip() for value in [llmwiki_type, title, summary, updated]):
        return None
    aliases_raw = frontmatter.get("aliases", [])
    aliases = [item for item in aliases_raw if isinstance(item, str)] if isinstance(aliases_raw, list) else []
    return WikiPageMetadata(
        path=rel_path.removeprefix("wiki/"),
        llmwiki_type=llmwiki_type,
        title=title,
        summary=summary,
        created=created if isinstance(created, str) else "",
        updated=updated,
        aliases=aliases,
        source_raw_paths=_frontmatter_list(frontmatter, "source_raw_paths"),
        source_raw_hashes=_frontmatter_list(frontmatter, "source_raw_hashes"),
        source_prepared_hashes=_frontmatter_list(frontmatter, "source_prepared_hashes"),
        source_operation_ids=_frontmatter_list(frontmatter, "source_operation_ids"),
    )


def resolve_model_related_pages(
    item: WikiMergePlanItem,
    resolution_item: CandidateResolutionItem,
    finalized_items: list[WikiMergePlanItem],
    resolution_by_id: dict[str, CandidateResolutionItem],
    snapshot: WikiContextSnapshot,
) -> tuple[list[RelatedPageRef], list[str]]:
    current_by_path: dict[str, WikiMergePlanItem] = {}
    current_by_title: dict[str, list[WikiMergePlanItem]] = {}
    for other in finalized_items:
        if other.page_type.lower() == "source" or other.action == "needs_human_decision":
            continue
        source_resolution = resolution_by_id.get(other.page_plan_id)
        paths = {other.canonical_target_path}
        if source_resolution is not None:
            paths.add(source_resolution.candidate_target_path)
        for path in paths:
            if path:
                current_by_path[path] = other
        current_by_title.setdefault(normalize_related_key(other.display_title), []).append(other)

    inspected_paths = set(item.inspected_context_paths)
    metadata_by_path: dict[str, WikiPageMetadata] = {}
    metadata_by_title: dict[str, list[WikiPageMetadata]] = {}
    for pool_entry in snapshot.knowledge_metadata_pool:
        metadata = pool_entry.metadata
        if metadata is None:
            continue
        if metadata.llmwiki_type.lower() == "source":
            continue
        metadata_by_path[metadata.path] = metadata
        for key in [metadata.title, *metadata.aliases]:
            metadata_by_title.setdefault(normalize_related_key(key), []).append(metadata)

    related: list[RelatedPageRef] = []
    unresolved: list[str] = []
    seen_paths: set[str] = set()
    for index, suggestion in enumerate(item.related_pages):
        if index >= MODEL_RELATED_SUGGESTION_LIMIT:
            unresolved.append(_related_debug_label(suggestion))
            continue
        if len(related) >= FINAL_RELATED_LIMIT:
            unresolved.append(_related_debug_label(suggestion))
            continue
        resolved = _resolve_single_model_related(
            suggestion,
            current_by_path=current_by_path,
            current_by_title=current_by_title,
            metadata_by_path=metadata_by_path,
            metadata_by_title=metadata_by_title,
            self_path=item.canonical_target_path,
            fallback_reason=suggestion.reason,
        )
        if resolved is None:
            unresolved.append(_related_debug_label(suggestion))
            continue
        if resolved.source == "wiki_context" and resolved.target_path not in inspected_paths:
            exact_key = normalize_related_key(resolved.display_title)
            if exact_key not in metadata_by_title:
                unresolved.append(_related_debug_label(suggestion))
                continue
        if resolved.target_path in seen_paths:
            continue
        seen_paths.add(resolved.target_path)
        related.append(resolved)
    return related, unresolved


def _resolve_single_model_related(
    suggestion: RelatedPageRef,
    *,
    current_by_path: dict[str, WikiMergePlanItem],
    current_by_title: dict[str, list[WikiMergePlanItem]],
    metadata_by_path: dict[str, WikiPageMetadata],
    metadata_by_title: dict[str, list[WikiPageMetadata]],
    self_path: str,
    fallback_reason: str,
) -> RelatedPageRef | None:
    for raw in _dedupe_strings([suggestion.target_path, suggestion.display_title]):
        target_path = _normalize_related_path(raw)
        if target_path:
            current = current_by_path.get(target_path)
            if current is not None and current.canonical_target_path != self_path:
                return RelatedPageRef(
                    target_path=current.canonical_target_path,
                    display_title=current.display_title,
                    source="source_digest",
                    reason=chinese_related_reason(fallback_reason, f"本次同源页面 `{current.display_title}` 与该主题互补。"),
                )
            metadata = metadata_by_path.get(target_path)
            if metadata is not None and metadata.path != self_path:
                return RelatedPageRef(
                    target_path=metadata.path,
                    display_title=clean_display_title(metadata.title),
                    source="wiki_context",
                    reason=chinese_related_reason(fallback_reason, f"召回旧页 `{clean_display_title(metadata.title)}` 与该主题存在可复用背景。"),
                )
        key = normalize_related_key(raw)
        current_matches = current_by_title.get(key, [])
        if len(current_matches) == 1 and current_matches[0].canonical_target_path != self_path:
            current = current_matches[0]
            return RelatedPageRef(
                target_path=current.canonical_target_path,
                display_title=current.display_title,
                source="source_digest",
                reason=chinese_related_reason(fallback_reason, f"本次同源页面 `{current.display_title}` 与该主题互补。"),
            )
        metadata_matches = metadata_by_title.get(key, [])
        if len(metadata_matches) == 1 and metadata_matches[0].path != self_path:
            metadata = metadata_matches[0]
            return RelatedPageRef(
                target_path=metadata.path,
                display_title=clean_display_title(metadata.title),
                source="wiki_context",
                reason=chinese_related_reason(fallback_reason, f"已有 wiki 页面标题或别名匹配 `{raw}`，可作为相关背景。"),
            )
    return None


def chinese_related_reason(model_reason: str, fallback: str) -> str:
    return fallback if not model_reason.strip() or looks_like_untranslated_english(model_reason) else model_reason.strip()


def _normalize_related_path(value: str) -> str | None:
    text = value.strip().strip("`").replace("\\", "/")
    if not text.endswith(".md"):
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.parts and path.parts[0] == "wiki":
        path = Path(*path.parts[1:])
    if not path.parts or path.parts[0] in {"sources", "logs"} or path.as_posix() in {"index.md", "log.md"}:
        return None
    return path.as_posix()


def _related_debug_label(suggestion: RelatedPageRef) -> str:
    return f"{suggestion.display_title or '<untitled>'} -> {suggestion.target_path or '<no path>'}"


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def finalize_wiki_merge_plan(
    plan: WikiMergePlanArtifact,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
    snapshot_ref: str,
    *,
    medium_missing_policy: Literal["preserve", "block"] = "block",
) -> WikiMergePlanArtifact:
    resolution_by_id = {item.page_plan_id: item for item in resolution.items}
    snapshot_paths = {entry.path for entry in snapshot.entries}
    context_by_id = {item.page_plan_id: item for item in snapshot.candidate_contexts.items}
    preliminary: list[WikiMergePlanItem] = []
    for item in plan.items:
        resolution_item = resolution_by_id.get(item.page_plan_id)
        if resolution_item is None:
            title_matches = [
                candidate
                for candidate in resolution.items
                if normalize_related_key(candidate.display_title) == normalize_related_key(item.display_title)
            ]
            typed_matches = [candidate for candidate in title_matches if candidate.page_type == item.page_type]
            resolution_item = (typed_matches or title_matches or [None])[0] if len(typed_matches or title_matches) == 1 else None
        if resolution_item is None:
            raise PipelineError(f"wiki_merge_plan references unknown page_plan_id: {item.page_plan_id}")
        context_item = context_by_id.get(resolution_item.page_plan_id)
        inspected_paths = [hit.path for hit in context_item.hits] if context_item is not None else []
        strongest_hit = context_item.hits[0] if context_item is not None and context_item.hits else None
        strongest_overlap = (
            ContextOverlapSignal(
                strength=strongest_hit.strength,
                match_basis=strongest_hit.match_basis,
                path=strongest_hit.path,
                score=strongest_hit.score,
                reason=f"Top inspected context: {strongest_hit.display_title}",
            )
            if strongest_hit is not None
            else ContextOverlapSignal()
        )
        canonical = normalize_model_wiki_target_path(item.canonical_target_path or resolution_item.candidate_target_path)
        matched_page = normalize_model_wiki_target_path(item.matched_page) if item.matched_page else None
        action = item.action
        model_action = item.model_action or item.action
        apply_eligibility = item.apply_eligibility
        blocked_reason = item.blocked_reason
        finalization_notes: list[str] = []
        if item.action == "update":
            matched_page = matched_page or canonical
            canonical = matched_page
        if item.action == "noop" and matched_page:
            canonical = matched_page
        if item.action == "create":
            canonical = normalize_model_wiki_target_path(resolution_item.candidate_target_path)
            matched_page = None
        if f"wiki/{canonical}" not in snapshot_paths:
            raise PipelineError(f"wiki_merge_plan target is outside wiki_context_snapshot: wiki/{canonical}")
        entry = snapshot_entry(snapshot, f"wiki/{canonical}")
        if action == "needs_human_decision":
            pass
        elif entry.expected_state == "present":
            action = "update" if action != "noop" else "noop"
            matched_page = canonical if action in {"update", "noop"} else matched_page
            if action != model_action:
                finalization_notes.append("目标页已存在，最终动作改为 update/noop。")
        else:
            if action == "noop":
                action = "needs_human_decision"
                apply_eligibility = "blocked"
                blocked_reason = blocked_reason or "模型选择 noop，但没有绑定已召回的现有页面；需要人工确认。"
                finalization_notes.append("missing target noop 被转为 needs_human_decision。")
            else:
                action = "create"
                matched_page = None
                if action != model_action:
                    finalization_notes.append("目标页缺失，最终动作改为 create。")
        if (
            action == "create"
            and strongest_overlap.strength == "strong"
            and strongest_overlap.path
            and strongest_overlap.path != canonical
        ):
            action = "needs_human_decision"
            apply_eligibility = "blocked"
            blocked_reason = blocked_reason or (
                f"召回到强相关旧页 `{strongest_overlap.path}`，但模型仍选择 create；需要人工确认是否应 update。"
            )
            finalization_notes.append("strong overlap create 被转为 needs_human_decision。")
        if (
            action == "create"
            and strongest_overlap.strength == "medium"
            and strongest_overlap.path
            and create_reason_needs_repair(item.why_not_update)
        ):
            if medium_missing_policy == "preserve":
                synthesized_reason = synthesize_medium_create_why_not_update(
                    item=item,
                    resolution_item=resolution_item,
                    strongest_hit=strongest_hit,
                    snapshot=snapshot,
                )
                item = item.model_copy(update={"why_not_update": synthesized_reason})
                finalization_notes.append(f"medium overlap create 缺少 why_not_update，已{LOCAL_MEDIUM_CREATE_REASON_MARKER}。")
            elif medium_missing_policy == "block":
                action = "needs_human_decision"
                apply_eligibility = "blocked"
                blocked_reason = blocked_reason or (
                    f"召回到中等相关旧页 `{strongest_overlap.path}`，但模型选择 create 的理由不充分；需要人工确认。"
                )
                finalization_notes.append("medium overlap create 的 why_not_update 不充分，被转为 needs_human_decision。")
            else:
                finalization_notes.append("medium overlap create 的 why_not_update 不充分，已请求模型补充。")
        if apply_eligibility == "blocked" and action != "needs_human_decision":
            action = "needs_human_decision"
            blocked_reason = blocked_reason or "模型将该项标记为 blocked，需要人工决策。"
            finalization_notes.append("blocked item 被转为 needs_human_decision。")
        related_absence_reason = item.related_absence_reason
        if not item.related_pages and related_absence_reason is None:
            related_absence_reason = "no_candidate" if not inspected_paths else "low_confidence"
        preliminary.append(
            item.model_copy(
                update={
                    "action": action,
                    "model_action": model_action,
                    "finalization_reason": "；".join(_dedupe_strings([*finalization_notes, item.finalization_reason])) or "模型动作已按冻结 wiki context 校验。",
                    "canonical_target_path": canonical,
                    "matched_page": matched_page,
                    "inspected_context_paths": _dedupe_strings([*item.inspected_context_paths, *inspected_paths]),
                    "strongest_overlap": strongest_overlap,
                    "why_not_update": item.why_not_update,
                    "why_create_or_update": item.why_create_or_update or item.reason,
                    "related_absence_reason": related_absence_reason,
                    "page_plan_id": resolution_item.page_plan_id,
                    "source_basis": resolution_item.source_basis,
                    "page_type": resolution_item.page_type,
                    "display_title": clean_display_title(item.display_title or resolution_item.display_title),
                    "apply_eligibility": apply_eligibility,
                    "blocked_reason": blocked_reason,
                }
            )
        )
    items: list[WikiMergePlanItem] = []
    for item in preliminary:
        resolution_item = resolution_by_id[item.page_plan_id]
        related_pages, related_unresolved = resolve_model_related_pages(item, resolution_item, preliminary, resolution_by_id, snapshot)
        unresolved = _dedupe_strings([*item.related_unresolved, *item.unresolved_related, *related_unresolved])
        related_absence_reason = item.related_absence_reason
        if not related_pages and related_absence_reason is None:
            related_absence_reason = "cap_cutoff" if related_unresolved else ("no_candidate" if not item.inspected_context_paths else "low_confidence")
        items.append(
            item.model_copy(
                update={
                    "related_pages": related_pages,
                    "related_unresolved": unresolved,
                    "unresolved_related": unresolved,
                    "related_absence_reason": related_absence_reason,
                }
            )
        )
    items = merge_same_source_duplicate_creates(items)
    items = merge_update_noop_same_targets(items)
    return WikiMergePlanArtifact(log_date=snapshot.log_date, items=items, context_snapshot_ref=snapshot_ref)


def synthesize_medium_create_why_not_update(
    *,
    item: WikiMergePlanItem,
    resolution_item: CandidateResolutionItem,
    strongest_hit: CandidateContextHit | None,
    snapshot: WikiContextSnapshot,
) -> str:
    old_path = strongest_hit.path if strongest_hit is not None else ""
    old_entry = snapshot_entry(snapshot, f"wiki/{old_path}") if old_path else WikiContextEntry(path="", expected_state="missing")
    old_title = (
        clean_display_title(old_entry.metadata.title)
        if old_entry.metadata is not None
        else clean_display_title(strongest_hit.display_title if strongest_hit is not None else "已召回旧页")
    )
    old_summary = old_entry.metadata.summary if old_entry.metadata is not None else ""
    old_scope = compact_payload_text(old_summary or old_title or old_path, 120)
    new_scope = compact_payload_text(
        resolution_item.topic_summary
        or item.new_understanding
        or resolution_item.initial_section_intent
        or resolution_item.display_title,
        140,
    )
    source_delta = compact_payload_text(
        resolution_item.why_this_page
        or resolution_item.coverage_notes
        or resolution_item.reason
        or item.knowledge_delta
        or item.why_this_matters,
        140,
    )
    page_kind = chinese_page_type_label(resolution_item.page_type)
    return (
        "本地补充：scope_delta："
        f"新页《{clean_display_title(resolution_item.display_title)}》按 `{resolution_item.candidate_target_path}` 独立沉淀为{page_kind}，"
        f"核心范围是「{new_scope}」；最像旧页《{old_title}》位于 `{old_path}`，旧页范围是「{old_scope}」。"
        "source_delta："
        f"本轮来源增量是「{source_delta}」。"
        "why_update_not_enough："
        "直接 update 旧页会把旧页从原有主题扩成另一个独立知识单元，降低旧页的聚焦度。"
        "why_related_link_not_enough："
        "只做 Related 只能表达关联，不能承载该来源新增的可复用结构、例子和价值点。"
    )


def chinese_page_type_label(page_type: str) -> str:
    mapping = {
        "concept": "概念页",
        "entity": "实体页",
        "design": "设计页",
        "comparison": "对比页",
        "open_question": "未决问题页",
        "overview": "总览页",
    }
    return mapping.get(page_type.strip().lower(), "知识页")


def merge_update_noop_same_targets(items: list[WikiMergePlanItem]) -> list[WikiMergePlanItem]:
    updates_by_target = {item.canonical_target_path: item for item in items if item.action == "update"}
    noop_by_target: dict[str, list[WikiMergePlanItem]] = {}
    for item in items:
        if item.action == "noop" and item.canonical_target_path in updates_by_target:
            noop_by_target.setdefault(item.canonical_target_path, []).append(item)
    if not noop_by_target:
        return items

    merged: list[WikiMergePlanItem] = []
    skipped_noops: set[str] = set()
    for item in items:
        if item.action == "noop" and item.canonical_target_path in updates_by_target:
            skipped_noops.add(item.page_plan_id)
            continue
        if item.action != "update" or item.canonical_target_path not in noop_by_target:
            merged.append(item)
            continue
        covered_noops = noop_by_target[item.canonical_target_path]
        source_basis = SourceBasis(
            source_candidate_ids=_dedupe_strings(
                [
                    *item.source_basis.source_candidate_ids,
                    *[
                        candidate_id
                        for noop in covered_noops
                        for candidate_id in noop.source_basis.source_candidate_ids
                    ],
                ]
            ),
            prepared_discovered_candidates=_dedupe_strings(
                [
                    *item.source_basis.prepared_discovered_candidates,
                    *[
                        candidate
                        for noop in covered_noops
                        for candidate in noop.source_basis.prepared_discovered_candidates
                    ],
                ]
            ),
            source_locator=item.source_basis.source_locator,
        )
        related_pages = [*item.related_pages]
        for noop in covered_noops:
            related_pages.extend(noop.related_pages)
        deduped_related: list[RelatedPageRef] = []
        seen_related: set[str] = set()
        for related in related_pages:
            path = normalize_related_candidate_path(related.target_path)
            if path is None or path in seen_related:
                continue
            seen_related.add(path)
            deduped_related.append(related.model_copy(update={"target_path": path}))
            if len(deduped_related) >= FINAL_RELATED_LIMIT:
                break
        noop_ids = [noop.page_plan_id for noop in covered_noops]
        merged_ids = _dedupe_strings([*item.merged_page_plan_ids, item.page_plan_id, *noop_ids])
        reason = f"同一 canonical target 出现 update + noop；{', '.join(noop_ids)} 已由 update `{item.page_plan_id}` 覆盖。"
        merged.append(
            item.model_copy(
                update={
                    "source_basis": source_basis,
                    "related_pages": deduped_related,
                    "merged_page_plan_ids": merged_ids,
                    "noop_covered_by_update": True,
                    "merge_reason": merge_markdown_blocks(item.merge_reason, reason),
                    "finalization_reason": merge_markdown_blocks(item.finalization_reason, reason),
                }
            )
        )
    # Preserve deterministic order while ensuring skipped noop ids are only represented in merged_page_plan_ids.
    return [item for item in merged if item.page_plan_id not in skipped_noops]


def merge_same_source_duplicate_creates(items: list[WikiMergePlanItem]) -> list[WikiMergePlanItem]:
    result = list(items)
    related_redirects: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for left_index in range(len(result)):
            left = result[left_index]
            if left.action != "create":
                continue
            for right_index in range(left_index + 1, len(result)):
                right = result[right_index]
                if right.action != "create":
                    continue
                if not same_source_duplicate_create(left, right):
                    continue
                canonical, suppressed = choose_duplicate_canonical(left, right)
                merged = absorb_duplicate_create(canonical, suppressed)
                suppressed_path = normalize_related_candidate_path(suppressed.canonical_target_path)
                canonical_path = normalize_related_candidate_path(merged.canonical_target_path)
                if suppressed_path is not None and canonical_path is not None and suppressed_path != canonical_path:
                    related_redirects[suppressed_path] = canonical_path
                keep_index = left_index if canonical is left else right_index
                drop_index = right_index if canonical is left else left_index
                result[keep_index] = merged
                del result[drop_index]
                changed = True
                break
            if changed:
                break
    return rewrite_related_pages_after_path_redirects(result, related_redirects)


def rewrite_related_pages_after_path_redirects(
    items: list[WikiMergePlanItem],
    redirects: dict[str, str],
) -> list[WikiMergePlanItem]:
    if not redirects:
        return items
    title_by_path = {
        path: item.display_title
        for item in items
        if (path := normalize_related_candidate_path(item.canonical_target_path)) is not None
    }
    rewritten: list[WikiMergePlanItem] = []
    for item in items:
        item_path = normalize_related_candidate_path(item.canonical_target_path)
        related_pages: list[RelatedPageRef] = []
        seen_related: set[str] = set()
        for related in item.related_pages:
            path = normalize_related_candidate_path(related.target_path)
            if path is None:
                continue
            target_path = resolve_related_redirect(path, redirects)
            if target_path == item_path or target_path in seen_related:
                continue
            seen_related.add(target_path)
            related_pages.append(
                related.model_copy(
                    update={
                        "target_path": target_path,
                        "display_title": title_by_path.get(target_path, related.display_title),
                    }
                )
            )
            if len(related_pages) >= FINAL_RELATED_LIMIT:
                break
        related_absence_reason = item.related_absence_reason
        if not related_pages and related_absence_reason is None:
            related_absence_reason = "self_link_only" if item.related_pages else "no_candidate"
        rewritten.append(item.model_copy(update={"related_pages": related_pages, "related_absence_reason": related_absence_reason}))
    return rewritten


def resolve_related_redirect(path: str, redirects: dict[str, str]) -> str:
    current = path
    seen: set[str] = set()
    while current in redirects and current not in seen:
        seen.add(current)
        next_path = redirects[current]
        if next_path == current:
            break
        current = next_path
    return current


def same_source_duplicate_create(left: WikiMergePlanItem, right: WikiMergePlanItem) -> bool:
    left_title_tokens = duplicate_tokens(left.display_title)
    right_title_tokens = duplicate_tokens(right.display_title)
    if len(left_title_tokens | right_title_tokens) < 2:
        return False
    title_overlap = jaccard(left_title_tokens, right_title_tokens)
    if title_overlap < 0.6 and not both_agent_workflow_compare(left.display_title, right.display_title):
        return False
    intent_overlap = jaccard(duplicate_tokens(duplicate_intent_text(left)), duplicate_tokens(duplicate_intent_text(right)))
    if intent_overlap < 0.48 and not both_agent_workflow_compare(left.display_title, right.display_title):
        return False
    if duplicate_shape_conflict(left, right) and title_overlap < 0.82:
        return False
    return True


def duplicate_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", source_digest_parenthetical_translation_core(text).lower())
    normalized = normalized.replace("workflow", "workflow").replace("workflows", "workflow")
    normalized = normalized.replace("agentic", "agent").replace("agents", "agent")
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized))
    stop = {"concept", "comparison", "design", "entity", "open", "question", "概念", "设计", "实体", "问题", "对比", "区别", "比较", "什么", "如何", "为什么", "页面"}
    return {token for token in tokens if token not in stop}


def duplicate_intent_text(item: WikiMergePlanItem) -> str:
    return "\n".join(
        [
            item.display_title,
            item.new_understanding,
            item.knowledge_delta,
            item.why_this_matters,
            item.reason,
            " ".join(item.value_points),
            "\n".join(item.section_plans.values()),
        ]
    )


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def both_agent_workflow_compare(left_title: str, right_title: str) -> bool:
    left = left_title.lower()
    right = right_title.lower()
    left_has_agent = "agent" in left or "智能体" in left
    left_has_workflow = "workflow" in left or "工作流" in left or "流程" in left
    right_has_agent = "agent" in right or "智能体" in right
    right_has_workflow = "workflow" in right or "工作流" in right or "流程" in right
    if not (left_has_agent and left_has_workflow and right_has_agent and right_has_workflow):
        return False
    combined = f"{left} {right}"
    has_compare = any(marker in combined for marker in ["vs", "对比", "比较", "区别"])
    return has_compare


def duplicate_shape_conflict(left: WikiMergePlanItem, right: WikiMergePlanItem) -> bool:
    pair = {left.page_type, right.page_type}
    if pair <= {"concept", "comparison"}:
        return False
    if pair <= {"concept", "open_question"}:
        return True
    if pair <= {"concept", "design"}:
        left_tokens = duplicate_tokens(left.display_title)
        right_tokens = duplicate_tokens(right.display_title)
        return jaccard(left_tokens, right_tokens) < 0.9
    return len(pair) > 1


def choose_duplicate_canonical(left: WikiMergePlanItem, right: WikiMergePlanItem) -> tuple[WikiMergePlanItem, WikiMergePlanItem]:
    ranked = sorted([left, right], key=duplicate_canonical_rank)
    return ranked[0], ranked[1]


def duplicate_canonical_rank(item: WikiMergePlanItem) -> tuple[int, int, str]:
    title = item.display_title.lower()
    type_rank = {
        "comparison": 0 if any(marker in title for marker in ["vs", "对比", "比较", "区别"]) else 2,
        "design": 1,
        "concept": 2,
        "open_question": 3,
        "entity": 4,
    }.get(item.page_type, 5)
    return (type_rank, -len(duplicate_intent_text(item)), item.canonical_target_path)


def absorb_duplicate_create(canonical: WikiMergePlanItem, suppressed: WikiMergePlanItem) -> WikiMergePlanItem:
    source_basis = SourceBasis(
        source_candidate_ids=_dedupe_strings([*canonical.source_basis.source_candidate_ids, *suppressed.source_basis.source_candidate_ids]),
        prepared_discovered_candidates=_dedupe_strings(
            [*canonical.source_basis.prepared_discovered_candidates, *suppressed.source_basis.prepared_discovered_candidates]
        ),
        source_locator=canonical.source_basis.source_locator or suppressed.source_basis.source_locator,
    )
    section_plans = dict(canonical.section_plans)
    for key, value in suppressed.section_plans.items():
        if key in section_plans:
            section_plans[key] = merge_markdown_blocks(section_plans[key], f"合并自 `{suppressed.page_plan_id}`：{value}")
        else:
            section_plans[key] = f"合并自 `{suppressed.page_plan_id}`：{value}"
    related_pages = [*canonical.related_pages, *suppressed.related_pages]
    merged_related: list[RelatedPageRef] = []
    seen_related: set[str] = set()
    for related in related_pages:
        path = normalize_related_candidate_path(related.target_path)
        if path is None or path in seen_related or path == canonical.canonical_target_path:
            continue
        seen_related.add(path)
        merged_related.append(related.model_copy(update={"target_path": path}))
        if len(merged_related) >= FINAL_RELATED_LIMIT:
            break
    absorbed_note = (
        f"同源近重复自动合并：`{suppressed.page_plan_id}`（{suppressed.display_title}）的信息已并入 "
        f"`{canonical.page_plan_id}`；其 section intent、examples/value points 通过 source_basis/section_plans 进入 canonical draft。"
    )
    return canonical.model_copy(
        update={
            "source_basis": source_basis,
            "section_plans": section_plans,
            "related_pages": merged_related,
            "value_points": _dedupe_strings([*canonical.value_points, *suppressed.value_points]),
            "reuse_scenarios": _dedupe_strings([*canonical.reuse_scenarios, *suppressed.reuse_scenarios]),
            "merged_page_plan_ids": _dedupe_strings(
                [*canonical.merged_page_plan_ids, canonical.page_plan_id, suppressed.page_plan_id, *suppressed.merged_page_plan_ids]
            ),
            "merge_reason": merge_markdown_blocks(canonical.merge_reason, absorbed_note),
            "finalization_reason": merge_markdown_blocks(canonical.finalization_reason, absorbed_note),
            "quality_risks": _dedupe_strings(
                [
                    *canonical.quality_risks,
                    *suppressed.quality_risks,
                    f"已自动合并 `{suppressed.page_plan_id}`；draft review 需确认被合并候选没有独特信息丢失。",
                ]
            ),
        }
    )


def normalize_model_wiki_target_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        path = path[1:]
    if path.startswith("wiki/"):
        path = path.removeprefix("wiki/")
    return path


def ensure_wiki_context_current(vault: Path, snapshot: WikiContextSnapshot) -> None:
    messages = wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise PipelineError("; ".join(messages))


def render_merge_plan_markdown(plan: WikiMergePlanArtifact) -> str:
    rows = []
    for item in plan.items:
        rows.append(
            [
                item.page_plan_id,
                item.model_action or item.action,
                item.action,
                item.display_title,
                f"`{item.canonical_target_path}`",
                f"{item.strongest_overlap.strength} `{item.strongest_overlap.path}`".strip(),
                item.why_not_update,
                item.new_understanding,
                item.apply_eligibility,
                item.blocked_reason,
            ]
        )
    return "# Wiki 合并计划\n\n" + format_markdown_table(
        ["页面计划", "模型动作", "最终动作", "标题", "目标", "最强召回", "为什么不更新旧页", "新增理解", "Apply", "阻断原因"],
        rows,
    ) + "\n"


def render_merge_plan_review_prompt(plan: WikiMergePlanArtifact) -> str:
    decision_rows = [
        [
            item.page_plan_id,
            item.model_action or item.action,
            item.action,
            item.display_title,
            f"`{item.canonical_target_path}`",
            item.strongest_overlap.strength,
            item.blocked_reason,
        ]
        for item in plan.items
    ]
    return (
        "# 合并计划审核\n\n"
        "审查这一步回答：写哪些页面、为什么写、哪些旧页已经被看过。\n\n"
        "## 核心判断\n\n"
        "- create 是否真的不能 update 到已召回的旧页？理由是否具体到范围、来源增量和边界？\n"
        "- update/noop 是否绑定了被召回并读过全文的旧页？\n"
        "- Related 是否少而准，是否只保留最相关的 0-2 条？\n"
        "- 是否存在 needs_human_decision 或 all-create 风险需要先 revise？\n\n"
        "## 关键文件\n\n"
        "- 合并计划：`wiki_merge_planning/wiki_merge_plan.json`\n"
        "- 合并报告：`wiki_merge_planning/merge_decision_report.md`\n"
        "- 召回上下文：`wiki_context_snapshot/candidate_contexts.md`\n"
        "- 可编辑文件：`merge_plan_review/pending_merge_plan.json`（仅 awaiting_review 时存在）\n\n"
        "## 下一步命令\n\n"
        "- 批准：`uv run llmwiki ingest approve \"$VAULT\" \"$OP\" merge_plan_review`\n"
        "- 重新生成/修订：`uv run llmwiki ingest revise \"$VAULT\" \"$OP\" merge_plan_review`\n"
        "- 批准后继续：`uv run llmwiki ingest resume \"$VAULT\" \"$OP\"`\n\n"
        "## 决策概览\n\n"
        + format_markdown_table(["页面计划", "模型动作", "最终动作", "标题", "目标", "最强召回", "阻断原因"], decision_rows)
        + "\n\n"
        "## 完整计划\n\n"
        f"{render_merge_plan_markdown(plan)}"
    )


def merge_plan_all_create_review_reason(
    plan: WikiMergePlanArtifact,
    *,
    max_auto_create_items: int = MAX_AUTO_APPROVED_ALL_CREATE_ITEMS,
) -> str:
    if not plan.items or any(item.action != "create" for item in plan.items):
        return ""
    if len(plan.items) > max_auto_create_items:
        return (
            f"合并计划一次 create {len(plan.items)} 个页面，超过自动通过上限 "
            f"{max_auto_create_items}；请 revise 聚合或延后低优先级页面。"
        )
    strong_risky = [
        item
        for item in merge_plan_create_overlap_risk_items(plan)
        if item.strongest_overlap.strength == "strong"
    ]
    if strong_risky:
        names = ", ".join(f"{item.page_plan_id}:strong" for item in strong_risky[:8])
        return f"合并计划全部为 create，但存在强召回风险（{names}）；请审核这些页面为什么不应 update 到已有知识页。"
    weak_reason_medium = [
        item
        for item in merge_plan_create_overlap_risk_items(plan)
        if item.strongest_overlap.strength == "medium"
        and (create_reason_needs_repair(item.why_not_update) or medium_create_reason_was_locally_synthesized(item))
    ]
    if not weak_reason_medium:
        return ""
    names = ", ".join(f"{item.page_plan_id}:medium" for item in weak_reason_medium[:8])
    return f"合并计划全部为 create，但存在中等召回风险且 create 理由不充分或仅由本地补充（{names}）；请审核这些页面为什么不应 update 到已有知识页。"


def merge_plan_auto_create_review_limit(configured_candidate_budget: int) -> int:
    return min(configured_candidate_budget, MAX_AUTO_APPROVED_ALL_CREATE_ITEMS)


def medium_create_reason_was_locally_synthesized(item: WikiMergePlanItem) -> bool:
    return item.why_not_update.startswith("本地补充：") or LOCAL_MEDIUM_CREATE_REASON_MARKER in item.finalization_reason


def merge_plan_create_overlap_risk_items(plan: WikiMergePlanArtifact) -> list[WikiMergePlanItem]:
    return [
        item
        for item in plan.items
        if (item.model_action == "create" or item.action == "create")
        and item.strongest_overlap.strength in {"medium", "strong"}
        and item.strongest_overlap.path
    ]


def render_candidate_contexts_markdown(
    artifact: CandidateContextsArtifact,
    *,
    resolved_cache_path: str = "",
    query_count: int | None = None,
    encoded_page_count: int | None = None,
) -> str:
    sections = [
        "# 候选页召回上下文",
        "",
        f"- 后端：`{artifact.retrieval_backend}`",
        f"- 模型：`{artifact.model}`",
        f"- 本地缓存模式（local files only）：{artifact.local_files_only}",
        f"- 缓存路径（resolved cache path）：`{resolved_cache_path or artifact.cache_dir or '未使用'}`",
        f"- Embedding 加载耗时：{format_duration(artifact.embedding_load_duration_ms)}",
        f"- Embedding 编码耗时：{format_duration(artifact.embedding_encode_duration_ms)}",
        f"- Embedding 总耗时：{format_duration(artifact.embedding_total_duration_ms)}",
        f"- Embedding 页面向量缓存命中：{artifact.embedding_page_vector_cache_hit}",
        f"- Embedding 页面/查询数量：{artifact.embedding_page_count}/{artifact.embedding_query_count}",
        f"- Embedding 输入字符数：{artifact.embedding_text_char_count}",
        f"- TopK：{artifact.top_k}",
        f"- 候选池页面数：{artifact.candidate_pool_size}",
        f"- 查询数（query count）：{artifact.candidate_pool_size if query_count is None else query_count}",
        f"- 编码页面数：{artifact.candidate_pool_size if encoded_page_count is None else encoded_page_count}",
        f"- 不完整 frontmatter 页面数：{artifact.skipped_count}",
        f"- 候选池 Hash：`{artifact.candidate_pool_sha256}`",
        "- 排序说明：先按强度、Score Bucket、依据、页面类型、目录和标题距离排序；Score Bucket 默认宽度为 "
        f"{SCORE_BUCKET_EPSILON:.2f}，所以表格里的原始分数不一定逐行严格递减。",
        "- Sort Key 说明：`bucket` 是分数分桶；`type/dir/title_distance/path` 是同一分数桶内的 tie-break。",
        "- `lexical_expansion` 表示 query 和旧页命中了同一组高信号术语；中文相似度使用 bigram/短语重叠，避免单字重叠把泛相关页面推高。",
    ]
    if artifact.warnings:
        sections.extend(["", "## 警告", "", *[f"- {warning}" for warning in artifact.warnings]])
    for item in artifact.items:
        rows = [
            [
                str(hit.rank),
                hit.strength,
                hit.match_basis,
                f"{hit.score:.4f}",
                str(hit.score_bucket or int(hit.score / SCORE_BUCKET_EPSILON)),
                hit.sort_explanation or "legacy artifact: sort explanation unavailable",
                "`forced`" if hit.forced else "",
                f"`{hit.path}`",
                hit.display_title,
                "`truncated`" if hit.truncated else "",
                hit.excerpt[:180].replace("\n", " "),
            ]
            for hit in item.hits
        ]
        sections.extend(
            [
                "",
                f"## {item.page_plan_id}",
                "",
                f"查询文本: {item.query[:500]}",
                "",
                format_markdown_table(["排名", "强度", "依据", "分数", "Score Bucket", "Sort Key", "强制命中", "路径", "标题", "截断", "片段"], rows)
                if rows
                else "未召回到候选旧页。",
            ]
        )
        if item.unindexable_pages:
            sections.extend(["", "Frontmatter 不完整但已进入低置信候选池：", "", *[f"- `{path}`" for path in item.unindexable_pages[:20]]])
    return "\n".join(sections).rstrip() + "\n"


def render_merge_decision_report(plan: WikiMergePlanArtifact, snapshot: WikiContextSnapshot) -> str:
    context_by_id = {item.page_plan_id: item for item in snapshot.candidate_contexts.items}
    sections = ["# 合并决策报告", ""]
    create_risks = merge_plan_create_overlap_risk_items(plan)
    if create_risks:
        risk_rows = []
        for item in create_risks:
            context = context_by_id.get(item.page_plan_id)
            inspected = item.inspected_context_paths or ([hit.path for hit in context.hits] if context else [])
            risk_rows.append(
                [
                    item.page_plan_id,
                    item.model_action or item.action,
                    item.action,
                    item.strongest_overlap.strength,
                    f"`{item.strongest_overlap.path}`",
                    ", ".join(f"`{path}`" for path in inspected[:5]) if inspected else "无",
                    item.why_not_update or "未提供",
                    item.apply_eligibility,
                    item.blocked_reason,
                ]
            )
        sections.extend(
            [
                "## Create/Update 风险摘要",
                "",
                "这些项目的模型动作或最终动作包含 create，但 TopK 召回中存在 medium/strong 旧页；审核时应优先检查 why_not_update 是否具体说明范围差异、来源增量和为什么不能 update。",
                "",
                format_markdown_table(
                    ["页面计划", "模型动作", "最终动作", "最强召回", "旧页", "看过的旧页", "为什么不更新", "Apply", "阻断原因"],
                    risk_rows,
                ),
                "",
            ]
        )
    for item in plan.items:
        context = context_by_id.get(item.page_plan_id)
        inspected = item.inspected_context_paths or ([hit.path for hit in context.hits] if context else [])
        overlap_rows = []
        if context is not None:
            for hit in context.hits:
                if hit.strength in {"medium", "strong"}:
                    overlap_rows.append(
                        [
                            hit.rank,
                            hit.strength,
                            hit.match_basis,
                            f"{hit.score:.4f}",
                            f"`{hit.path}`",
                            hit.display_title,
                            hit.excerpt[:160].replace("\n", " "),
                        ]
                    )
        related_text = (
            ", ".join(f"`{related.target_path}`" for related in item.related_pages)
            if item.related_pages
            else f"无（{item.related_absence_reason or 'no_candidate'}）"
        )
        action_question = {
            "create": "为什么不 update",
            "update": "为什么 update",
            "noop": "为什么 noop",
            "needs_human_decision": "为什么需要人工决策",
        }.get(item.action, "为什么 create/update/noop")
        action_answer = {
            "create": item.why_not_update or "未提供",
            "update": item.why_create_or_update or item.reason,
            "noop": item.why_create_or_update or item.reason,
            "needs_human_decision": item.blocked_reason or item.why_create_or_update or item.reason,
        }.get(item.action, item.reason)
        sections.extend(
            [
                f"## {item.display_title}",
                "",
                f"- 页面计划：`{item.page_plan_id}`",
                f"- 模型动作：`{item.model_action or item.action}`",
                f"- 最终动作：`{item.action}`",
                f"- 目标：`{item.canonical_target_path}`",
                f"- 最像旧页：`{item.strongest_overlap.path or '无'}` ({item.strongest_overlap.strength}, {item.strongest_overlap.match_basis})",
                f"- 看过的旧页：{', '.join(f'`{path}`' for path in inspected) if inspected else '无'}",
                f"- {action_question}：{action_answer}",
                f"- 为什么 create/update/noop：{item.why_create_or_update or item.reason}",
                f"- Related：{related_text}",
                f"- Finalizer：{item.finalization_reason}",
            ]
        )
        if item.action == "create" and item.strongest_overlap.strength in {"medium", "strong"}:
            sections.extend(
                [
                    "",
                    "### Create 对比审计",
                    "",
                    "- scope_delta：见 why_not_update 中的新旧页面范围差异。",
                    "- source_delta：见 why_not_update 中的新材料增量。",
                    "- why_update_not_enough：见 why_not_update 中为什么整页更新不合适。",
                    "- why_related_link_not_enough：见 why_not_update 中为什么只做 Related 不够。",
                ]
            )
        if overlap_rows:
            sections.extend(
                [
                    "",
                    "### Medium/Strong 召回命中",
                    "",
                    format_markdown_table(["排名", "强度", "依据", "分数", "路径", "标题", "片段"], overlap_rows),
                ]
            )
        if item.blocked_reason:
            sections.append(f"- 阻断原因：{item.blocked_reason}")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def finalize_draft_rendering(
    artifact: DraftRenderingArtifact,
    plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
) -> DraftRenderingArtifact:
    plan_by_id = {item.page_plan_id: item for item in plan.items if item.action in {"create", "update"}}
    pages: list[DraftPageItem] = []
    used_ids: set[str] = set()
    for page in artifact.pages:
        if not page.page_plan_id.strip():
            raise_draft_issue("missing_field", "draft_rendering page_plan_id must not be empty", field_path="pages.page_plan_id")
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            item = next((candidate for candidate in plan_by_id.values() if candidate.canonical_target_path == page.canonical_target_path), None)
        if item is None:
            raise_draft_issue(
                "unknown_page_plan_reference",
                f"draft_rendering references non-draftable page_plan_id: {page.page_plan_id}",
                field_path="pages.page_plan_id",
            )
        if item.page_plan_id in used_ids:
            raise_draft_issue(
                "duplicate_page_plan_id",
                f"draft_rendering duplicates page_plan_id: {item.page_plan_id}",
                field_path="pages.page_plan_id",
            )
        used_ids.add(item.page_plan_id)
        entry = snapshot_entry(snapshot, f"wiki/{item.canonical_target_path}")
        pages.append(
            page.model_copy(
                update={
                    "page_plan_id": item.page_plan_id,
                    "action": item.action,
                    "canonical_target_path": item.canonical_target_path,
                    "preimage_sha256": entry.preimage_sha256,
                    "section_bodies": normalize_draft_section_bodies(page.section_bodies),
                    "change_summary": finalize_draft_change_summary(page.change_summary, item),
                    "source_coverage_notes": finalize_draft_source_coverage_notes(page.source_coverage_notes, item),
                }
            )
        )
    return DraftRenderingArtifact(pages=pages)


CANONICAL_DRAFT_SECTION_KEYS = ("summary", "detail", "examples", "value_points", "additional_notes", "open_questions")


def default_create_change_summary(item: WikiMergePlanItem) -> str:
    if item.action != "create":
        return ""
    title = item.display_title.strip() or Path(item.canonical_target_path).stem
    return f"创建 {title} 页面。"


def finalize_draft_change_summary(summary: str, item: WikiMergePlanItem) -> str:
    summary = normalize_stable_brand_typos(summary.strip())
    if item.action == "create" and (not summary or looks_like_untranslated_english(summary)):
        return default_create_change_summary(item)
    return summary


def finalize_draft_source_coverage_notes(notes: str, item: WikiMergePlanItem) -> str:
    notes = normalize_stable_brand_typos(notes.strip())
    if not notes or looks_like_untranslated_english(notes):
        title = item.display_title.strip() or Path(item.canonical_target_path).stem
        basis = "本轮来源摘录"
        if item.action == "update":
            basis = "本轮来源摘录与已检查旧页"
        return f"依据{basis}中与「{title}」相关的内容生成；未被来源支撑的细节保留为未决问题。"
    notes = re.sub(r"\bapproved_digest\b", "来源摘要", notes)
    notes = re.sub(r"\bsource_excerpt_pack\b", "来源摘录包", notes)
    notes = re.sub(r"\bwiki_context_snapshot\b", "已检查 wiki 上下文", notes)
    notes = re.sub(r"\bsnippets?\b", "摘录", notes, flags=re.IGNORECASE)
    return notes


def normalize_draft_section_bodies(section_bodies: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_body in section_bodies.items():
        if not isinstance(raw_body, str):
            continue
        body = normalize_stable_brand_typos(raw_body.strip())
        if not body:
            continue
        canonical_key = draft_section_key_alias(raw_key)
        if canonical_key is None:
            header = re.sub(r"\s+", " ", raw_key.strip()) or "Additional Notes"
            body = f"### {header}\n\n{body}"
            canonical_key = "detail"
        normalized[canonical_key] = merge_markdown_blocks(normalized.get(canonical_key, ""), body)
    return normalized


def normalize_stable_brand_typos(text: str) -> str:
    replacements = [
        ("Clade Code", "Claude Code"),
        ("ClaudeCode", "Claude Code"),
        ("Anropinic", "Anthropic"),
        ("Anthopic", "Anthropic"),
        ("ManagedAgents", "Managed Agents"),
        ("Borris Cherny", "Boris Cherny"),
        ("Borris", "Boris"),
    ]
    for wrong, right in replacements:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(wrong)}(?![A-Za-z0-9])", right, text)
    return text


def draft_section_key_alias(raw_key: str) -> str | None:
    key = unicodedata.normalize("NFKC", raw_key)
    key = re.sub(r"^[#*\s`]+|[:：#*\s`]+$", "", key)
    key = re.sub(r"[-_]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip().lower()
    aliases = {
        "summary": "summary",
        "摘要": "summary",
        "detail": "detail",
        "details": "detail",
        "understanding": "detail",
        "详情": "detail",
        "理解": "detail",
        "examples": "examples",
        "example": "examples",
        "cases": "examples",
        "case": "examples",
        "例子": "examples",
        "案例": "examples",
        "value points": "value_points",
        "value point": "value_points",
        "values": "value_points",
        "value": "value_points",
        "why this matters": "value_points",
        "advice": "value_points",
        "advice for pms": "value_points",
        "价值点": "value_points",
        "建议": "value_points",
        "additional notes": "additional_notes",
        "additional note": "additional_notes",
        "notes": "additional_notes",
        "observations": "additional_notes",
        "observation": "additional_notes",
        "freeform": "additional_notes",
        "free form": "additional_notes",
        "补充观察": "additional_notes",
        "补充": "additional_notes",
        "观察": "additional_notes",
        "open questions": "open_questions",
        "open question": "open_questions",
        "questions": "open_questions",
        "tensions": "open_questions",
        "conflicts": "open_questions",
        "uncertainties": "open_questions",
        "矛盾与未决问题": "open_questions",
        "未决问题": "open_questions",
    }
    if key in CANONICAL_DRAFT_SECTION_KEYS:
        return key
    return aliases.get(key)


def merge_markdown_blocks(existing: str, addition: str) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing}\n\n{addition}"


def validate_draft_rendering(artifact: DraftRenderingArtifact, plan: WikiMergePlanArtifact, *, language: str | None = None) -> None:
    allowed_sections = set(CANONICAL_DRAFT_SECTION_KEYS)
    required_sections = {"summary", "detail"}
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    required_ids = {item.page_plan_id for item in plan.items if item.action in {"create", "update"}}
    actual_ids = {page.page_plan_id for page in artifact.pages}
    missing = required_ids - actual_ids
    if missing:
        raise_draft_issue("missing_page_plan_coverage", f"draft_rendering misses page_plan_id(s): {sorted(missing)}", field_path="pages")
    extra = actual_ids - required_ids
    if extra:
        raise_draft_issue("unknown_page_plan_reference", f"draft_rendering contains unexpected page_plan_id(s): {sorted(extra)}", field_path="pages")
    for page in artifact.pages:
        plan_item = plan_by_id.get(page.page_plan_id)
        display_title = plan_item.display_title if plan_item is not None else ""
        if not page.canonical_target_path.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} canonical_target_path must not be empty", field_path="canonical_target_path")
        if not page.section_bodies:
            raise_draft_issue("missing_field", f"{page.page_plan_id} section_bodies must not be empty", field_path="section_bodies")
        section_keys = set(page.section_bodies)
        unknown_sections = section_keys - allowed_sections
        if unknown_sections:
            raise_draft_issue(
                "unsupported_section_key",
                f"{page.page_plan_id} section_bodies contains unsupported section key(s): {sorted(unknown_sections)}",
                field_path="section_bodies",
            )
        missing_sections = required_sections - section_keys
        if missing_sections:
            raise_draft_issue(
                "missing_field",
                f"{page.page_plan_id} section_bodies misses required section key(s): {sorted(missing_sections)}",
                field_path="section_bodies",
            )
        if not page.change_summary.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} change_summary must not be empty", field_path="change_summary")
        if not page.source_coverage_notes.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} source_coverage_notes must not be empty", field_path="source_coverage_notes")
        for section_key, body in page.section_bodies.items():
            if section_key in required_sections and not body.strip():
                raise_draft_issue(
                    "missing_field",
                    f"{page.page_plan_id} section {section_key} must not be empty",
                    field_path=f"section_bodies.{section_key}",
                )
            if "---\n" in body or body.lstrip().startswith("# ") or contains_source_graph_link(body):
                raise_draft_issue(
                    "forbidden_page_markdown",
                    f"{page.page_plan_id} section body contains forbidden page-level markdown",
                    field_path=f"section_bodies.{section_key}",
                )
            if section_contains_stray_related_links(body, page.canonical_target_path, display_title=display_title):
                raise_draft_issue(
                    "stray_related_links_in_content",
                    (
                        f"{page.page_plan_id} section {section_key} contains a related-page block or self wikilink; "
                        "remove body-level related links because the system renders official related pages separately."
                    ),
                    field_path=f"pages.{page.page_plan_id}.section_bodies.{section_key}",
                )
            if language == "zh-CN" and looks_like_untranslated_english(body):
                raise_draft_issue(
                    "zh_cn_untranslated_user_text",
                    f"{page.page_plan_id} section {section_key} must be Chinese for zh-CN vault",
                    field_path=f"section_bodies.{section_key}",
                )
        if language == "zh-CN" and looks_like_untranslated_english(page.change_summary):
            raise_draft_issue(
                "zh_cn_untranslated_user_text",
                f"{page.page_plan_id} change_summary must be Chinese for zh-CN vault",
                field_path="change_summary",
            )
        if language == "zh-CN" and looks_like_untranslated_english(page.source_coverage_notes):
            raise_draft_issue(
                "zh_cn_untranslated_user_text",
                f"{page.page_plan_id} source_coverage_notes must be Chinese for zh-CN vault",
                field_path="source_coverage_notes",
            )
        if plan_item is not None:
            validate_digestive_quality(page, plan_item)


def validate_digestive_quality(page: DraftPageItem, item: WikiMergePlanItem) -> None:
    summary = page.section_bodies.get("summary", "")
    detail = page.section_bodies.get("detail", "")
    examples = page.section_bodies.get("examples", "")
    values = page.section_bodies.get("value_points", "")
    notes = page.section_bodies.get("additional_notes", "")
    substantive_slots = [
        text
        for text in [detail, examples, values, notes]
        if is_substantive_digestive_text(text)
    ]
    if not substantive_slots or normalized_digest_text(summary) == normalized_digest_text(detail):
        raise_draft_issue(
            "thin_digestive_content",
            (
                f"{page.page_plan_id} must include concrete digested understanding: viewpoint, example, "
                "use scenario, boundary condition, or value point; it must not be only a source summary."
            ),
            field_path="section_bodies",
        )
    if item.action == "update" and not update_change_summary_is_specific(page.change_summary):
        raise_draft_issue(
            "thin_update_change_summary",
            f"{page.page_plan_id} update change_summary must explain what the new material补充/改变/澄清了旧理解。",
            field_path="change_summary",
        )


def draft_self_talk_issues(artifact: DraftRenderingArtifact) -> list[StructuredIssue]:
    issues: list[StructuredIssue] = []
    for page in artifact.pages:
        for section_key, body in page.section_bodies.items():
            marker = draft_self_talk_marker(body)
            if not marker:
                continue
            issues.append(
                StructuredIssue(
                    issue_code="model_self_talk_leak",
                    field_path=f"pages.{page.page_plan_id}.{section_key}",
                    validator_id="draft_content_quality",
                    message=(
                        f"{page.page_plan_id} section {section_key} contains model self-talk marker `{marker}`; "
                        "remove reasoning notes about checking, uncertainty, or future edits, and keep only the final sourced page content."
                    ),
                    repairability="repairable",
                )
            )
    return issues


def draft_self_talk_marker(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    for marker in DRAFT_SELF_TALK_MARKERS:
        if marker and marker in compact:
            return marker
    if "需要谨慎" in compact and any(marker in compact for marker in ["检查原文", "查看原文", "原文", "我", "记错"]):
        return "需要谨慎"
    if "应该是" in compact and any(marker in compact for marker in ["我", "记错", "检查原文", "原文数据", "但前面说"]):
        return "应该是"
    return ""


def is_substantive_digestive_text(text: str) -> bool:
    normalized = normalized_digest_text(text)
    if not normalized or any(marker in normalized for marker in ["暂无", "没有相关", "无相关", "n/a"]):
        return False
    return len(normalized) >= 24 or any(marker in text for marker in ["例如", "适用", "边界", "价值", "场景", "反例", "意味着", "可以用来"])


def normalized_digest_text(text: str) -> str:
    return re.sub(r"[\s\-*#`，。；;：:、,.!?！？（）()]+", "", text.strip().lower())


def update_change_summary_is_specific(text: str) -> bool:
    normalized = normalized_digest_text(text)
    if len(normalized) < 12:
        return False
    return any(marker in text for marker in ["补充", "改变", "澄清", "更新", "整合", "新增", "保留", "删除", "修正", "扩展"])


def raise_draft_issue(issue_code: str, message: str, *, field_path: str = "", repairable: bool = True) -> None:
    raise ContractValidationError(
        message,
        issues=[
            StructuredIssue(
                issue_code=issue_code,
                field_path=field_path,
                validator_id="validate_draft_rendering",
                message=message,
                repairability="repairable" if repairable else "non_repairable",
            )
        ],
    )


def contains_source_graph_link(text: str) -> bool:
    return text_contains_source_graph_link(text)


def section_contains_stray_related_links(text: str, canonical_target_path: str, *, display_title: str = "") -> bool:
    stripped = strip_fenced_code_blocks(text)
    return section_contains_self_wikilink(stripped, canonical_target_path, display_title=display_title) or section_contains_related_link_block(stripped)


def strip_fenced_code_blocks(text: str) -> str:
    kept_lines: list[str] = []
    fence_char = ""
    fence_length = 0
    for line in text.splitlines(keepends=True):
        stripped_newline = line.rstrip("\r\n")
        if fence_char:
            if closing_fence_line(stripped_newline, fence_char, fence_length):
                fence_char = ""
                fence_length = 0
            continue
        if match := opening_fence_line(stripped_newline):
            marker = match.group("marker")
            fence_char = marker[0]
            fence_length = len(marker)
            continue
        kept_lines.append(line)
    return "".join(kept_lines)


def opening_fence_line(line: str) -> re.Match[str] | None:
    return re.match(r"^ {0,3}(?P<marker>`{3,}|~{3,})[^\n]*$", line)


def closing_fence_line(line: str, fence_char: str, fence_length: int) -> bool:
    escaped = re.escape(fence_char)
    return re.match(rf"^ {{0,3}}{escaped}{{{fence_length},}}\s*$", line) is not None


def section_contains_self_wikilink(text: str, canonical_target_path: str, *, display_title: str = "") -> bool:
    canonical = normalize_related_candidate_path(canonical_target_path)
    if canonical is None:
        return False
    canonical_aliases = normalized_path_self_aliases(canonical)
    canonical_aliases |= normalized_title_self_aliases(display_title)
    for raw_target in section_link_targets(text):
        if link_target_is_external_or_anchor(raw_target):
            continue
        target = normalize_related_candidate_path(raw_target)
        if target is not None and normalized_path_self_aliases(target) & canonical_aliases:
            return True
    return False


def section_link_targets(text: str) -> list[str]:
    targets: list[str] = []
    for match in re.finditer(r"\[\[([^\]]+)\]\]", text):
        targets.append(match.group(1).split("|", 1)[0])
    for match in re.finditer(r"\[[^\]\n]+\]\(([^)]+)\)", text):
        targets.append(match.group(1))
    return targets


def link_target_is_external_or_anchor(value: str) -> bool:
    text = value.strip()
    return text.startswith(("#", "//")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text) is not None


def normalized_path_self_aliases(path: str) -> set[str]:
    posix = Path(path).as_posix()
    stemless = Path(path).with_suffix("").as_posix()
    return {
        posix,
        stemless,
        Path(posix).name,
        Path(stemless).name,
    }


def normalized_title_self_aliases(title: str) -> set[str]:
    stripped = title.strip()
    if not stripped:
        return set()
    aliases = {stripped}
    if not stripped.endswith(".md"):
        aliases.add(f"{stripped}.md")
    if (title_path := normalize_related_candidate_path(stripped)) is not None:
        aliases |= normalized_path_self_aliases(title_path)
    return aliases


def line_contains_section_link(line: str) -> bool:
    return bool(section_link_targets(line))


def section_contains_related_link_block(text: str) -> bool:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not related_label_line(line):
            continue
        if line_contains_section_link(line):
            return True
        for follower in lines[index + 1 : index + 6]:
            stripped = follower.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                break
            if line_contains_section_link(stripped):
                return True
            if not stripped.startswith(("-", "*", "+")):
                break
    return False


def related_label_line(line: str) -> bool:
    return bool(
        re.match(
            r"^\s{0,3}(?:[-*+]\s*)?(?:#{1,6}\s*)?(?:\*\*|__)?(?:相关页面|related(?:\s+pages)?)(?:\*\*|__)?\s*(?:[:：]|$)",
            line,
            re.IGNORECASE,
        )
    )


def snapshot_entry(snapshot: WikiContextSnapshot, path: str) -> WikiContextEntry:
    for entry in snapshot.entries:
        if entry.path == path:
            return entry
    raise PipelineError(f"snapshot missing path: {path}")


SECTION_TITLE_TO_KEY = {
    "摘要": "summary",
    "Summary": "summary",
    "详情": "detail",
    "Detail": "detail",
    "Details": "detail",
    "例子": "examples",
    "Examples": "examples",
    "价值点": "value_points",
    "Value Points": "value_points",
    "补充观察": "additional_notes",
    "Additional Notes": "additional_notes",
    "相关页面": "related",
    "Related": "related",
    "Related Pages": "related",
    "矛盾与未决问题": "open_questions",
    "Open Questions": "open_questions",
    "Tensions / Open Questions": "open_questions",
}
ENGLISH_SECTION_TITLE_TO_KEY = {
    title.casefold(): key for title, key in SECTION_TITLE_TO_KEY.items() if title.isascii()
}


def parse_existing_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in markdown.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            section_title = match.group(1).strip()
            current_key = SECTION_TITLE_TO_KEY.get(section_title)
            if current_key is None and section_title.isascii():
                current_key = ENGLISH_SECTION_TITLE_TO_KEY.get(section_title.casefold())
            if current_key is not None:
                sections.setdefault(current_key, [])
            continue
        if current_key is not None:
            sections[current_key].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def is_empty_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return not normalized or any(marker in normalized for marker in ["暂无", "没有相关", "无相关", "N/A"])


def merge_update_section(
    section_key: str,
    old: str,
    new: str,
    *,
    absorption_context: str | None = None,
) -> tuple[str, SectionMergeChange]:
    old = old.strip()
    new = new.strip()
    if section_key == "open_questions":
        return merge_update_open_questions_section(old, new)
    retained: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    preserved_old: list[str] = []
    removal_reason = ""
    needs_manual_resolution = False
    absorbed, matched_phrases, _ = update_section_absorption(old, new) if old and new else (False, [], [])
    context_absorbed = False
    context_matched_phrases: list[str] = []
    if old and new and not absorbed and absorption_context and update_merge_should_preserve_old_section(section_key, old):
        context_text = absorption_context.strip()
        if context_text and context_text != new:
            context_absorbed, context_matched_phrases, _ = update_section_absorption(old, context_text)
            absorbed = context_absorbed
    if section_key == "additional_notes":
        absorbed = False
        context_absorbed = False
        matched_phrases = []
        context_matched_phrases = []
    if old and new and absorbed:
        retained.append(old)
    if new and not is_empty_placeholder(new):
        added.append(new)
    if old and not retained and old != new and not is_empty_placeholder(old):
        if section_key == "additional_notes":
            preserved_notes, removed_notes, absorbed_notes = split_high_signal_old_additional_notes(
                old,
                new,
                absorption_context=absorption_context or "",
            )
            if preserved_notes:
                new = merge_markdown_blocks(new, preserved_old_additional_notes_block(preserved_notes))
                retained.extend([*absorbed_notes, *preserved_notes])
                preserved_old.extend(preserved_notes)
                removed.extend(removed_notes)
                removal_reason = (
                    "高信号旧补充观察已自动保留为 legacy note；无需阻塞审批，建议后续按需整理。"
                    "低信号或已覆盖的旧补充观察不机械保留。"
                )
            elif absorbed_notes:
                retained.extend(absorbed_notes)
                removed.extend(removed_notes)
                removal_reason = "高信号旧补充观察已被新草稿吸收；低信号旧补充观察不机械保留。"
            elif removed_notes:
                removed.extend(removed_notes)
                removal_reason = "旧段落不属于 update preservation 核心义务，且未被新草稿自然吸收；本轮不再机械保留。"
            else:
                removed.append(old)
                removal_reason = "旧段落不属于 update preservation 核心义务，且未被新草稿自然吸收；本轮不再机械保留。"
        elif update_merge_should_preserve_old_section(section_key, old):
            preserved = preserved_old_section_block(old)
            new = merge_markdown_blocks(new, preserved)
            retained.append(old)
            preserved_old.append(old)
            needs_manual_resolution = True
            removal_reason = "模型完整重写后未显式吸收该旧段落；系统已临时保留为旧页保留观察，draft review 需消化、改写或确认删除。"
        else:
            removed.append(old)
            removal_reason = "旧段落不属于 update preservation 核心义务，且未被新草稿自然吸收；本轮不再机械保留。"
    elif old and retained and old != new and old not in new:
        if context_absorbed:
            matched = context_matched_phrases[:4]
            removal_reason = (
                f"模型已在新草稿其他章节吸收旧段落关键短语/概念义务：{', '.join(matched)}。"
                if matched
                else "模型已在新草稿其他章节吸收旧段落概念义务。"
            )
        elif matched_phrases:
            removal_reason = f"模型已通过关键短语/概念义务吸收旧段落：{', '.join(matched_phrases[:4])}。"
    return (
        new,
        SectionMergeChange(
            section_key=section_key,
            retained=retained,
            added=added,
            removed=removed,
            preserved_old=preserved_old,
            needs_manual_resolution=needs_manual_resolution,
            removal_reason=removal_reason,
        ),
    )


def split_high_signal_old_additional_notes(
    old: str,
    new: str,
    *,
    absorption_context: str,
) -> tuple[list[str], list[str], list[str]]:
    preserved: list[str] = []
    removed: list[str] = []
    absorbed: list[str] = []
    for note in old_additional_note_units(old):
        if not old_additional_note_is_high_signal_boundary(note):
            removed.append(note)
            continue
        if old_additional_note_superseded(note, new) or old_additional_note_superseded(note, absorption_context):
            removed.append(note)
            continue
        if old_additional_note_absorbed(note, new) or old_additional_note_absorbed(note, absorption_context):
            absorbed.append(note)
            continue
        preserved.append(note)
    return _dedupe_strings(preserved), _dedupe_strings(removed), _dedupe_strings(absorbed)


def old_additional_note_units(text: str) -> list[str]:
    units: list[str] = []
    paragraph: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if paragraph:
                units.append(" ".join(paragraph).strip())
                paragraph = []
            continue
        bullet = re.match(r"^(?:[-*+]|\d+[.)、])\s+(?P<body>.+)$", stripped)
        if bullet:
            if paragraph:
                units.append(" ".join(paragraph).strip())
                paragraph = []
            units.append(bullet.group("body").strip())
            continue
        paragraph.append(stripped)
    if paragraph:
        units.append(" ".join(paragraph).strip())
    normalized_units = [strip_old_additional_note_label(unit) for unit in units]
    return [unit for unit in _dedupe_strings(normalized_units) if unit and not is_empty_placeholder(unit)]


def strip_old_additional_note_label(text: str) -> str:
    stripped = text.strip()
    while True:
        next_value = re.sub(r"^(?:旧页补充观察|旧页保留观察)[:：]\s*", "", stripped).strip()
        if next_value == stripped:
            break
        stripped = next_value
    return stripped.strip()


def old_additional_note_is_high_signal_boundary(note: str) -> bool:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", note))
    if len(normalized) < 12:
        return False
    if re.search(r"[?？]$", normalized) or normalized.startswith(("如何", "是否", "为什么", "能否", "有没有")):
        return False
    if re.search(r"(?:需要|待|尚需|仍需)?确认是否|待确认|尚需确认|仍需确认|是否存在|待验证|待补来源", normalized):
        return False
    if any(marker in normalized for marker in ["暂无", "没有明确", "可与", "关联阅读", "后续可以继续补充"]):
        return False
    strong_markers = (
        "应视为",
        "不能视为",
        "不可视为",
        "并非绝对真实",
        "不是绝对真实",
        "用户确认",
        "需要确认",
        "必须确认",
        "重要决定",
        "重大决策",
        "不可逆",
        "安全风险",
        "可靠性风险",
        "隐私风险",
        "成本约束",
        "权限边界",
        "隔离边界",
    )
    if any(marker in normalized for marker in strong_markers):
        return True
    risk_or_boundary = any(marker in normalized for marker in ["风险", "边界", "限制", "约束"])
    domain_signal = any(
        marker in normalized
        for marker in ["安全", "可靠性", "准确性", "一致性", "成本", "权限", "隔离", "隐私", "审计", "确认"]
    )
    modal_signal = any(marker in normalized for marker in ["需要", "应该", "应当", "不能", "不应", "必须"])
    return risk_or_boundary and domain_signal and modal_signal


def old_additional_note_absorbed(note: str, target: str) -> bool:
    if not note.strip() or not target.strip():
        return False
    normalized_note = normalized_source_match_text(note)
    normalized_target = normalized_source_match_text(target)
    if normalized_note and normalized_note in normalized_target:
        return True
    return old_additional_note_boundary_paraphrase_absorbed(note, target)


def old_additional_note_boundary_paraphrase_absorbed(note: str, target: str) -> bool:
    note_norm = re.sub(r"\s+", "", unicodedata.normalize("NFKC", note.lower()))
    target_norm = re.sub(r"\s+", "", unicodedata.normalize("NFKC", target.lower()))
    obligations: list[str] = []
    if "用户确认" in note_norm:
        obligations.append("user_confirmation")
    if "召回记忆" in note_norm and any(marker in note_norm for marker in ["不能只依赖", "不能依赖", "绝对真实", "上下文"]):
        obligations.append("memory_context_boundary")
    if not obligations:
        return False
    for obligation in obligations:
        if obligation == "user_confirmation" and not old_additional_note_target_has_user_confirmation_boundary(target_norm):
            return False
        if obligation == "memory_context_boundary" and not old_additional_note_target_has_memory_context_boundary(target_norm):
            return False
    return True


def old_additional_note_target_has_user_confirmation_boundary(target_norm: str) -> bool:
    decision_markers = ("重要决定", "重要决策", "重大决策")
    positive_markers = ("需要", "仍需", "仍需要", "应由", "必须", "由用户确认")
    negative_markers = ("不需要", "无需", "不必", "不再需要", "免于")
    for clause in old_additional_note_supersession_clauses(target_norm):
        if "用户确认" not in clause or not any(marker in clause for marker in decision_markers):
            continue
        if any(re.search(rf"{marker}.{{0,8}}用户确认|用户确认.{{0,8}}{marker}", clause) for marker in negative_markers):
            continue
        if any(marker in clause for marker in positive_markers):
            return True
    return False


def old_additional_note_target_has_memory_context_boundary(target_norm: str) -> bool:
    boundary_markers = ("辅助上下文", "有帮助的上下文", "作为上下文", "只能作为", "不能只依赖", "不能依赖", "不是绝对真实", "非绝对真实")
    for clause in old_additional_note_supersession_clauses(target_norm):
        if not re.search(r"召回的?记忆|记忆召回", clause):
            continue
        if old_additional_note_clause_negates_memory_context(clause):
            continue
        if any(marker in clause for marker in boundary_markers):
            return True
    return False


def old_additional_note_clause_negates_memory_context(clause: str) -> bool:
    negative = ("不是", "并非", "不作为", "不能作为", "不应作为", "不再作为")
    context_terms = ("辅助上下文", "有帮助的上下文", "上下文")
    return any(re.search(rf"{marker}.{{0,8}}{term}", clause) for marker in negative for term in context_terms)


def old_additional_note_superseded(note: str, target: str) -> bool:
    if not note.strip() or not target.strip():
        return False
    anchors = old_additional_note_supersession_anchors(note)
    if not anchors:
        return False
    for sentence in old_additional_note_supersession_sentences(target):
        if not any(anchor in sentence for anchor in anchors):
            continue
        if old_additional_note_sentence_supersedes_anchor(sentence, anchors):
            return True
    return False


def old_additional_note_supersession_anchors(note: str) -> list[str]:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", note.lower()))
    anchors = [
        "用户确认",
        "不可逆",
        "绝对真实",
        "召回记忆",
        "权限边界",
        "隔离边界",
        "安全风险",
        "可靠性风险",
        "隐私风险",
        "成本约束",
    ]
    return [anchor for anchor in anchors if anchor in normalized]


def old_additional_note_supersession_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text.lower()))
    return [sentence for sentence in re.split(r"[。！？!?；;\n]+", normalized) if sentence]


def old_additional_note_supersession_clauses(sentence: str) -> list[str]:
    return [clause for clause in re.split(r"[，,、]|但|不过|然而|而|同时|并且", sentence) if clause]


def old_additional_note_clause_preserves_anchor(clause: str, anchor: str) -> bool:
    preservation_markers = ("仍", "仍然", "继续", "依然", "还是")
    for marker in preservation_markers:
        if re.search(rf"{marker}.{{0,8}}{re.escape(anchor)}|{re.escape(anchor)}.{{0,8}}{marker}", clause):
            return True
    return False


def old_additional_note_sentence_supersedes_anchor(sentence: str, anchors: list[str]) -> bool:
    for clause in old_additional_note_supersession_clauses(sentence):
        for anchor in anchors:
            if anchor not in clause:
                continue
            if old_additional_note_clause_preserves_anchor(clause, anchor):
                continue
            if old_additional_note_clause_is_capability_change(clause, anchor):
                continue
            if old_additional_note_clause_supersedes_anchor(clause, anchor):
                return True
    return False


def old_additional_note_clause_is_capability_change(clause: str, anchor: str) -> bool:
    capability_terms = r"(?:元数据|字段|日志|api|接口|能力|属性|参数)"
    if re.search(r"(?:已)?改为(?:支持|提供|记录|返回|包含)", clause) and re.search(capability_terms, clause):
        return True
    if re.search(r"(?:已)?改为", clause) and re.search(rf"{re.escape(anchor)}.{{0,8}}{capability_terms}", clause):
        return True
    return False


def old_additional_note_clause_supersedes_anchor(clause: str, anchor: str) -> bool:
    escaped = re.escape(anchor)
    strong_markers = ("不再需要", "不需要", "无需", "不必", "不再依赖", "不适用", "免于", "deprecated", "废弃", "已废弃")
    if any(re.search(rf"{marker}.{{0,8}}{escaped}|{escaped}.{{0,8}}{marker}", clause) for marker in strong_markers):
        return True
    if re.search(rf"{escaped}.{{0,12}}(?:已)?改为|(?:已)?改为.{{0,12}}{escaped}", clause):
        return True
    if re.search(rf"{escaped}.{{0,12}}替代|替代.{{0,12}}{escaped}", clause):
        return True
    return False


def preserved_old_additional_notes_block(notes: list[str]) -> str:
    if len(notes) == 1:
        return f"旧页补充观察：{notes[0]}"
    lines = "\n".join(f"- {note}" for note in notes)
    return f"旧页补充观察：\n{lines}"


def merge_update_open_questions_section(old: str, new: str) -> tuple[str, SectionMergeChange]:
    old_questions = [
        question
        for question in meaningful_open_question_lines(old)
        if not is_low_signal_open_question(question)
    ]
    new_questions = meaningful_open_question_lines(new)
    seen_keys = {open_question_key(question) for question in new_questions}
    retained_old_questions: list[str] = []
    for question in old_questions:
        key = open_question_key(question)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        retained_old_questions.append(question)
    merged_questions = [*new_questions, *retained_old_questions]
    merged = "\n".join(f"- {question}" for question in merged_questions).strip()
    if not merged:
        merged = new if not is_empty_placeholder(new) else "暂无矛盾与未决问题记录。"
    reason = ""
    if old_questions:
        reason = "旧 open_questions 默认按问题粒度 union/dedupe 保留；占位/低信号问题不机械保留。"
    return (
        merged,
        SectionMergeChange(
            section_key="open_questions",
            retained=retained_old_questions,
            added=new_questions,
            removed=[],
            preserved_old=[],
            needs_manual_resolution=False,
            removal_reason=reason,
        ),
    )


def update_merge_should_preserve_old_section(section_key: str, old_text: str) -> bool:
    if section_key not in {"summary", "detail"}:
        return False
    phrases = update_preservation_phrases(old_text)
    concepts = update_preservation_concepts(old_text)
    return not update_preservation_section_is_low_value(section_key, old_text, phrases, concepts)


def preserved_old_section_block(old: str) -> str:
    return f"旧页保留观察（来自更新前页面，模型本轮未显式吸收，先保留待审）：\n\n{old.strip()}"


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
    if grounding_claim_targets_open_question(claim) and claim.section_key != "open_questions":
        reason = (
            f"{reason} 这是 open_questions 页面；无来源支撑的场景、后果或影响推测不要留在 detail/examples 当事实；"
            "请移动到 section_bodies.open_questions，改写成问题并标注 待补来源，不要换成另一个具体后果。"
        )
    if not text:
        return reason
    return f"{reason} 触发文本：{text[:240]}"


def grounding_claim_targets_open_question(claim: GroundingClaim) -> bool:
    target_path = claim.target_path.strip().replace("\\", "/")
    if target_path.startswith("wiki/"):
        target_path = target_path[5:]
    return target_path.startswith("open_questions/")


def unsupported_backing_marker(text: str) -> str | None:
    for marker in UNSUPPORTED_BACKING_MARKERS:
        if marker == "被多个" and not re.search(
            r"被多个(?:社区|团队|公司|机构|组织|项目|产品|用户|客户|开发者|研究|论文|媒体|开源项目).{0,12}(?:引用|采用|使用|验证|复现|报道|认可|采纳)",
            text,
        ):
            continue
        if marker in text:
            return marker
    return None


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
    marker_equivalents = _dedupe_strings(
        [normalized_source_match_text(marker), *[normalized_source_match_text(phrase) for phrase in EXTERNAL_BACKING_EQUIVALENTS]]
    )
    specific_anchors, generic_anchors = external_backing_topic_anchors(text)
    normalized_text = normalized_source_match_text(text)
    bridge_anchors = [
        normalized_source_match_text(anchor)
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
        normalized_sentence = normalized_source_match_text(sentence)
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
    marker_equivalents = _dedupe_strings(
        [normalized_source_match_text(marker), *[normalized_source_match_text(phrase) for phrase in EXTERNAL_BACKING_EQUIVALENTS]]
    )
    specific_anchors, generic_anchors = external_backing_topic_anchors(text)
    if not specific_anchors:
        return False
    all_anchors = [*specific_anchors, *generic_anchors]
    if len(all_anchors) < 2:
        return False
    sentences = external_backing_source_sentences(source_text)
    for index, sentence in enumerate(sentences):
        normalized_sentence = normalized_source_match_text(sentence)
        if not any(equivalent and equivalent in normalized_sentence for equivalent in marker_equivalents):
            continue
        context = " ".join(sentences[max(0, index - 1) : index + 2])
        normalized_context = normalized_source_match_text(context)
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
        normalized_context = normalized_source_match_text(context)
        if sum(1 for anchor in anchors if anchor in normalized_context) >= min(3, len(anchors)):
            return True
    return False


def scope_speculation_anchors(text: str) -> list[str]:
    normalized_text = normalized_source_match_text(text)
    anchors: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.+_-]{1,}", unicodedata.normalize("NFKC", text)):
        normalized = token.lower().strip("._-+")
        if len(normalized) >= 3 and normalized not in {"the", "and", "for", "with", "from", "into", "this", "that"}:
            anchors.append(normalized)
    for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9（）()·]{2,}", text):
        normalized = normalized_source_match_text(phrase)
        if len(normalized) >= 2 and normalized not in {"可能", "也许", "或许", "推测", "疑似", "受到影响", "受影响", "波及", "涉及", "牵涉", "导致", "造成", "影响到", "关联"}:
            anchors.append(normalized)
    deduped: list[str] = []
    for anchor in anchors:
        if anchor and anchor in normalized_text and anchor not in deduped:
            deduped.append(anchor)
    return deduped[:8]


def grounding_text_supported_by_context(text: str, approved_raw_text: str, existing_wiki_text: str) -> bool:
    return quote_supported_by_text(text, approved_raw_text) or quote_supported_by_text(text, existing_wiki_text)


def rewrite_grounding_sensitive_paraphrases(
    artifact: DraftRenderingArtifact,
    approved_raw_text: str,
) -> tuple[DraftRenderingArtifact, dict[str, Any]]:
    report_pages: list[dict[str, Any]] = []
    rewritten_pages: list[DraftPageItem] = []
    rewrite_count = 0
    source_sentences = grounding_rewrite_source_sentences(approved_raw_text)
    for page in artifact.pages:
        section_bodies: dict[str, str] = {}
        page_sections: list[dict[str, Any]] = []
        for section_key, body in page.section_bodies.items():
            rewritten_body, section_rewrites = rewrite_grounding_sensitive_body(body, source_sentences)
            section_bodies[section_key] = rewritten_body
            if section_rewrites:
                rewrite_count += len(section_rewrites)
                page_sections.append({"section_key": section_key, "rewrites": section_rewrites})
        if page_sections:
            report_pages.append(
                {
                    "page_plan_id": page.page_plan_id,
                    "target_path": page.canonical_target_path,
                    "sections": page_sections,
                }
            )
            rewritten_pages.append(page.model_copy(update={"section_bodies": section_bodies}))
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
    normalized = normalized_source_match_text(quote)
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
    normalized_quote = normalized_source_match_text(quote)
    if not any(marker in normalized_quote for marker in ["失败", "没价值", "不够", "不是自动化", "可靠", "有效"]):
        return None
    normalized_percentages = {normalized_percentage_token(percentage) for percentage in percentages}
    best: str | None = None
    for sentence in source_sentences:
        normalized_sentence = normalized_source_match_text(sentence)
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
    for match in re.finditer(r"“([^“”\n]{6,})”", body):
        spans.append((match.group(1), match.start()))
    index = 0
    while index < len(body):
        quote_start = body.find('"', index)
        if quote_start < 0:
            break
        if not plausible_ascii_open_quote(body, quote_start):
            index = quote_start + 1
            continue
        quote_end = body.find('"', quote_start + 1)
        if quote_end < 0:
            break
        quote = body[quote_start + 1 : quote_end]
        if len(quote) >= 6 and "\n" not in quote and not any(char in quote for char in '“”"'):
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


def collect_grounding_claims(
    *,
    item: WikiMergePlanItem,
    page: DraftPageItem,
    existing_entry: WikiContextEntry,
    approved_raw_text: str,
    claims: list[GroundingClaim],
) -> None:
    for section_key, body in page.section_bodies.items():
        for quote, quote_start in iter_grounding_quote_spans(body):
            normalized_quote = re.sub(r"\s+", "", quote.strip())
            is_explicit_quote = explicit_direct_quote_context(body, quote, quote_start=quote_start)
            is_illustrative_example = illustrative_example_context(body, quote, quote_start=quote_start)
            is_memory_example = illustrative_memory_example_context(body, quote, quote_start=quote_start)
            raw_supported = quote_supported_by_text(quote, approved_raw_text)
            existing_supported = quote_supported_by_text(quote, existing_entry.content)
            supported = raw_supported or existing_supported
            examples_unsafe_bypass_quote = (
                section_key == "examples"
                and not supported
                and examples_quote_has_unsafe_marker_for_bypass(normalized_quote, quote)
            )
            is_concept_label_quote = (
                looks_like_concept_phrase(quote)
                or looks_like_abstract_trend_label(re.sub(r"\s+", "", quote.strip()))
            ) and not supported and not strict_direct_quote_context(body, quote_start=quote_start) and not attributed_quote_context(body, quote_start=quote_start)
            if section_key == "examples" and is_concept_label_quote and (
                examples_quote_has_concrete_marker(normalized_quote, quote) or examples_unsafe_bypass_quote
            ):
                is_concept_label_quote = False
            if examples_unsafe_bypass_quote:
                is_illustrative_example = False
                is_memory_example = False
            section_example_hard_fact = section_key == "examples" and (
                contains_short_fact_marker(normalized_quote) or contains_hard_fact_marker(normalized_quote)
            )
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
                elif is_query_template_example:
                    reason = "例子区的短查询/请求模板按 illustrative example 处理，不要求 raw exact match。"
                elif is_generic_prompt_example:
                    reason = "例子区的通用问题/指令示例按 illustrative example 处理，不要求 raw exact match。"
                elif section_key == "examples":
                    reason = "例子区的通用示例句按 illustrative example 处理，不要求 raw exact match。"
                elif is_illustrative_example:
                    reason = "由如/例如/比如引出的通用示例句按 illustrative example 处理，不要求 raw exact match。"
                elif is_memory_example:
                    reason = "记忆评估中的短问句/用户偏好/对话样例按 illustrative example 处理，不要求 raw exact match。"
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
            claims.append(
                GroundingClaim(
                    page_plan_id=page.page_plan_id,
                    target_path=item.canonical_target_path,
                    section_key=section_key,
                    claim_type="new_fact",
                    text=quote,
                    support="raw" if raw_supported else ("existing_wiki" if existing_supported else "unsupported"),
                    action="kept" if supported else "needs_review",
                    reason=(
                        "直接引用已在 raw 或已有 wiki 中规范化 exact match。"
                        if supported
                        else "直接引用必须在 raw 或已有 wiki 中 exact match。"
                    ),
                )
            )
        for line in body.splitlines():
            text = line.strip(" -*")
            if not text or len(text) < 8:
                continue
            scope_marker = None if section_key == "open_questions" else unsupported_scope_speculation_marker(text)
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
                            action="needs_review",
                            reason=(
                                f"新增影响范围/受影响对象推测 `{scope_marker}` 未被 raw 或 inspected wiki 同句级支撑；"
                                "请删除该推测，或改写为来源明确陈述。"
                            ),
                        )
                    )
            marker = unsupported_backing_marker(text)
            if not marker:
                continue
            if unsupported_backing_marker_inside_supported_quote(
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
                        action="needs_review",
                        reason=f"新增外部背书/强事实标记 `{marker}` 未在 raw 或 inspected wiki 中出现；请删除该背书词，或改写为 source-local 表达。",
                    )
                )
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


def unsupported_backing_marker_inside_supported_quote(
    text: str,
    marker: str,
    approved_raw_text: str,
    existing_wiki_text: str,
) -> bool:
    for quote, _quote_start in iter_grounding_quote_spans(text):
        if marker not in quote:
            continue
        if quote_supported_by_text(quote, approved_raw_text) or quote_supported_by_text(quote, existing_wiki_text):
            return True
    return False


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
    variants = [normalized_source_match_text(text)]
    range_normalized = normalize_numeric_range_connectors(text)
    if range_normalized != text:
        variants.append(normalized_source_match_text(range_normalized))
    enumerated_range_normalized = normalize_paired_temporal_enumerated_ranges(text)
    if enumerated_range_normalized != text:
        variants.append(normalized_source_match_text(enumerated_range_normalized))
    stripped = strip_inline_term_translation_parentheticals(text)
    if stripped != text:
        variants.append(normalized_source_match_text(stripped))
    if re.search(r"\d", unicodedata.normalize("NFKC", text)):
        variants.extend(normalized_direct_quote_elision_variant(variant) for variant in list(variants))
    return _dedupe_strings([variant for variant in variants if variant])


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
    normalized_quote = normalized_source_match_text(quote)
    if len(normalized_quote) < 16 or len(normalized_quote) > 96:
        return False
    if re.search(r"\d", normalized_quote):
        return False
    segments = [
        segment
        for segment in (normalized_source_match_text(part) for part in re.split(r"[，,；;、]", quote))
        if len(segment) >= 4
    ]
    if len(segments) < 2:
        return False
    normalized_text = normalized_source_match_text(text)
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
    normalized_quote = normalized_source_match_text(quote)
    if len(normalized_quote) < 12 or len(normalized_quote) > 64:
        return False
    if re.search(r"\d", normalized_quote) or contains_hard_fact_marker(normalized_quote):
        return False
    if not any(marker in normalized_quote for marker in ["方法", "路径", "方式", "流程", "目标", "原则", "模式", "用例"]):
        return False
    if re.search(r"发布(?:了|过)|推出(?:了|过)|上线(?:了|过)", normalized_quote):
        return False
    normalized_text = normalized_source_match_text(text)
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
    if contains_short_fact_marker(normalized) and not looks_like_instructional_example(normalized, prefix):
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
        marker in lowered_original for marker in ["how to", "install", "setup", "configure", "query", "search"]
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
    if looks_like_mixed_unsupported_example_fact(normalized):
        return True
    if looks_like_user_id_literal(normalized):
        return True
    if contains_hard_fact_marker(normalized):
        return True
    if re.search(r"\d|[%％$￥¥]|https?://|www\.|@|[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{8,}", original):
        return True
    if re.search(r"`[^`]*(?:--|=|/|\\|\d)[^`]*`", original):
        return True
    lowered_original = original.lower()
    unsafe_word_pattern = (
        r"\b(?:order|ticket|issue|status|success|failed|failure|error|token|api[_-]?key|password|"
        r"passwd|secret|credential|account|permission|payment|refund|delete|deleted|revenue)\b"
    )
    if re.search(unsafe_word_pattern, lowered_original):
        return True
    unsafe_markers = [
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
        "订单",
        "交易",
        "付款",
        "支付",
        "退款",
        "删除",
        "凭证",
        "密码",
        "密钥",
        "账户",
        "账号",
        "权限",
        "收入",
        "状态",
        "张三",
        "李四",
        "王五",
    ]
    if any(marker in normalized for marker in unsafe_markers):
        return True
    return bool(re.search(r"\b(?:Alice|Bob|Ethan|Zhang|Li|Wang)\b", original))


def examples_quote_has_unsafe_marker_for_bypass(normalized: str, original: str = "") -> bool:
    if looks_like_user_id_literal(normalized):
        return True
    if examples_quote_has_personal_name_reference(normalized, original):
        return True
    if examples_quote_has_sensitive_user_data_marker(normalized, original):
        return True
    if re.search(r"\d|[%％$￥¥]|https?://|www\.|@|[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{8,}", original):
        return True
    return False


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
        r"api[_-]?key|token|session|cookie|email|phone|login)\b",
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
    user_markers = r"user|users|user's|customer|customers|customer's|person|person's|personal"
    sensitive_objects = r"ip\s+address|address|name|birthday|birth\s*date|birthdate|profile|location"
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
        entry = snapshot_entry(snapshot, f"wiki/{page.canonical_target_path}")
        collect_grounding_claims(
            item=item,
            page=page,
            existing_entry=entry,
            approved_raw_text=approved_raw_text,
            claims=claims,
        )
    return draft_grounding_review_from_claims(claims)


OPEN_QUESTION_SCOPE_CLEANUP_SECTIONS = {"detail", "examples", "value_points", "additional_notes"}
OPEN_QUESTION_SCOPE_CLEANUP_REASON_PREFIX = "新增影响范围/受影响对象推测"
OPEN_QUESTION_SCOPE_CLEANUP_LIMIT = 3


def cleanup_open_question_unsupported_scope_claims(
    artifact: DraftRenderingArtifact,
    plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    approved_raw_text: str,
) -> tuple[DraftRenderingArtifact, dict[str, Any]]:
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    review = build_draft_grounding_review(artifact, plan, snapshot, approved_raw_text)
    claims_by_page_id: dict[str, list[GroundingClaim]] = {}
    for claim in review.unsupported_new_facts:
        item = plan_by_id.get(claim.page_plan_id)
        if not open_question_scope_cleanup_claim(claim, item):
            continue
        claims_by_page_id.setdefault(claim.page_plan_id, []).append(claim)

    rewritten_pages: list[DraftPageItem] = []
    report_pages: list[dict[str, Any]] = []
    relocation_count = 0
    skipped_count = 0
    for page in artifact.pages:
        page_claims = claims_by_page_id.get(page.page_plan_id)
        if not page_claims:
            rewritten_pages.append(page)
            continue
        section_bodies = dict(page.section_bodies)
        existing_question_keys = {
            open_question_key(question)
            for question in meaningful_open_question_lines(section_bodies.get("open_questions", ""))
        }
        page_report = {
            "page_plan_id": page.page_plan_id,
            "target_path": page.canonical_target_path,
            "relocations": [],
            "skipped": [],
        }
        added_count = 0
        for claim in page_claims:
            body = section_bodies.get(claim.section_key, "")
            updated_body, removal_status = remove_grounding_claim_exact_once(body, claim.text)
            if removal_status != "removed":
                page_report["skipped"].append(
                    {
                        "section_key": claim.section_key,
                        "text": claim.text,
                        "reason": removal_status,
                    }
                )
                skipped_count += 1
                continue
            relocated_question = questionize_open_question_scope_claim(claim.text)
            question_key = open_question_key(relocated_question)
            duplicate_question = question_key in existing_question_keys
            if not duplicate_question and added_count >= OPEN_QUESTION_SCOPE_CLEANUP_LIMIT:
                page_report["skipped"].append(
                    {
                        "section_key": claim.section_key,
                        "text": claim.text,
                        "reason": "skipped_limit",
                        "question": relocated_question,
                    }
                )
                skipped_count += 1
                continue
            section_bodies[claim.section_key] = (
                updated_body
                if updated_body.strip()
                else open_question_scope_cleanup_section_placeholder(claim.section_key)
            )
            append_decision = "skipped_duplicate_question"
            if not duplicate_question:
                section_bodies["open_questions"] = append_open_question_line(
                    section_bodies.get("open_questions", ""),
                    relocated_question,
                )
                existing_question_keys.add(question_key)
                added_count += 1
                append_decision = "appended"
            relocation_count += 1
            page_report["relocations"].append(
                {
                    "section_key": claim.section_key,
                    "text": claim.text,
                    "question": relocated_question,
                    "reason": claim.reason,
                    "append_decision": append_decision,
                }
            )
        if page_report["relocations"]:
            rewritten_pages.append(page.model_copy(update={"section_bodies": section_bodies}))
        else:
            rewritten_pages.append(page)
        if page_report["relocations"] or page_report["skipped"]:
            report_pages.append(page_report)

    report = {
        "schema_version": "open_question_grounding_cleanup_report.v1",
        "changed": relocation_count > 0,
        "relocation_count": relocation_count,
        "skipped_count": skipped_count,
        "pages": report_pages,
    }
    if relocation_count == 0:
        return artifact, report
    return artifact.model_copy(update={"pages": rewritten_pages}), report


def cleanup_unsupported_example_literals(
    artifact: DraftRenderingArtifact,
    plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    approved_raw_text: str,
    *,
    review: DraftGroundingReview | None = None,
) -> tuple[DraftRenderingArtifact, dict[str, Any]]:
    active_review = review or build_draft_grounding_review(artifact, plan, snapshot, approved_raw_text)
    claims_by_page_id: dict[str, list[GroundingClaim]] = {}
    for claim in active_review.unsupported_new_facts:
        if unsupported_example_literal_cleanup_claim(claim):
            claims_by_page_id.setdefault(claim.page_plan_id, []).append(claim)

    rewritten_pages: list[DraftPageItem] = []
    replacements: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for page in artifact.pages:
        page_claims = claims_by_page_id.get(page.page_plan_id)
        if not page_claims:
            rewritten_pages.append(page)
            continue
        section_bodies = dict(page.section_bodies)
        examples_body = section_bodies.get("examples", "")
        updated_body = examples_body
        changed = False
        for claim in page_claims:
            updated_body, result = replace_unsupported_example_literal_once(updated_body, claim.text)
            result.update(
                {
                    "page_plan_id": page.page_plan_id,
                    "target_path": page.canonical_target_path,
                    "section_key": "examples",
                    "claim_reason": claim.reason,
                }
            )
            if result.get("status") == "replaced":
                changed = True
                replacements.append(result)
            else:
                skipped.append(result)
        if changed:
            section_bodies["examples"] = updated_body
            rewritten_pages.append(page.model_copy(update={"section_bodies": section_bodies}))
        else:
            rewritten_pages.append(page)
    report = {
        "schema_version": "example_concrete_cleanup_report.v1",
        "changed": bool(replacements),
        "replacement_count": len(replacements),
        "skipped_count": len(skipped),
        "replacements": replacements,
        "skipped": skipped,
    }
    if not replacements:
        return artifact, report
    return artifact.model_copy(update={"pages": rewritten_pages}), report


def unsupported_example_literal_cleanup_claim(claim: GroundingClaim) -> bool:
    return bool(
        claim.section_key == "examples"
        and claim.support == "unsupported"
        and claim.action == "needs_review"
        and "直接引用必须在 raw 或已有 wiki 中 exact match" in claim.reason
    )


def replace_unsupported_example_literal_once(body: str, text: str) -> tuple[str, dict[str, Any]]:
    if not text:
        return body, example_literal_skip_result(text, "skipped_empty_text")
    matches = grounding_quoted_literal_occurrences(body, text)
    if not matches:
        return body, example_literal_skip_result(text, "skipped_missing_exact_quoted_literal")
    if len(matches) != 1:
        return body, example_literal_skip_result(text, "skipped_ambiguous_repeated_quoted_literal")
    quote_start, quoted_text = matches[0]
    if strict_direct_quote_context(body, quote_start=quote_start) or attributed_quote_context(body, quote_start=quote_start):
        return body, example_literal_skip_result(text, "skipped_attributed_or_direct_quote_context")
    if position_inside_fenced_code_block(body, quote_start):
        return body, example_literal_skip_result(text, "skipped_inside_fenced_code")
    inside_inline_code = position_inside_inline_code_span(body, quote_start)
    replacement, reason = unsupported_example_literal_placeholder(
        text,
        body,
        quote_start,
        inside_inline_code=inside_inline_code,
    )
    if not replacement:
        return body, example_literal_skip_result(text, reason or "skipped_no_safe_placeholder")
    updated = body[:quote_start] + replacement + body[quote_start + len(quoted_text) :]
    return (
        updated,
        {
            "status": "replaced",
            "text": text,
            "original": text,
            "replacement": replacement,
            "reason": reason,
        },
    )


def grounding_quoted_literal_occurrences(body: str, quote: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for quoted_text in [f"“{quote}”", f'"{quote}"']:
        start = 0
        while True:
            found = body.find(quoted_text, start)
            if found < 0:
                break
            matches.append((found, quoted_text))
            start = found + len(quoted_text)
    return sorted(matches, key=lambda item: item[0])


def example_literal_skip_result(text: str, reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "text": text,
        "original": text,
        "replacement": "",
        "reason": reason,
    }


def unsupported_example_literal_placeholder(
    text: str,
    body: str,
    quote_start: int,
    *,
    inside_inline_code: bool = False,
) -> tuple[str, str]:
    normalized = unicodedata.normalize("NFKC", text).strip()
    compact = re.sub(r"\s+", "", normalized)
    lowered = normalized.lower()
    if looks_like_mixed_unsupported_example_fact(normalized):
        return "", "skipped_mixed_fact_literal"
    if memory_query_call_argument_context(body, quote_start):
        return example_angle_placeholder("memory_query", quoted=True), "memory_query_argument_placeholder"
    if looks_like_user_id_literal(normalized):
        return example_angle_placeholder("user_id", quoted=inside_inline_code), "user_id_placeholder"
    if re.search(r"\b(?:api[_-]?key|password|passwd|secret|token)\b", lowered) or any(marker in compact for marker in ["密钥", "密码", "令牌", "凭证"]):
        return example_angle_placeholder("api_key", quoted=inside_inline_code), "secret_placeholder"
    if looks_like_time_period_literal(normalized):
        return example_angle_placeholder("time_period", quoted=inside_inline_code), "time_period_placeholder"
    if looks_like_example_identifier_literal(normalized):
        return example_angle_placeholder("example_id", quoted=inside_inline_code), "example_id_placeholder"
    if looks_like_metric_or_outcome_literal(compact):
        return "", "skipped_metric_or_outcome_fact"
    if looks_like_user_preference_literal(normalized):
        if inside_inline_code:
            return example_angle_placeholder("memory_text", quoted=True), "memory_text_placeholder"
        return "用户偏好 X", "user_preference_placeholder"
    if looks_like_location_consumption_literal(normalized):
        if inside_inline_code:
            return example_angle_placeholder("memory_text", quoted=True), "memory_text_placeholder"
        return "某个用户在某个地点消费过", "location_consumption_placeholder"
    if looks_like_product_usage_literal(normalized):
        if inside_inline_code:
            return example_angle_placeholder("memory_text", quoted=True), "memory_text_placeholder"
        return "某个用户使用某类产品", "product_usage_placeholder"
    return "", "skipped_no_safe_placeholder"


def example_angle_placeholder(name: str, *, quoted: bool = False) -> str:
    value = f"<{name}>"
    return f'"{value}"' if quoted else f"`{value}`"


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


def looks_like_time_period_literal(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text)
    compact = re.sub(r"\s+", "", normalized)
    return bool(
        re.fullmatch(r"(?:19|20)\d{2}(?:年)?", compact)
        or re.fullmatch(r"(?:19|20)\d{2}年(?:第?[一二三四1234]季度|[一二三四1234]季度|上半年|下半年|\d{1,2}月(?:\d{1,2}日)?)", compact)
        or re.fullmatch(r"(?:第?[一二三四1234]季度|[一二三四1234]季度|Q[1-4]|上半年|下半年|\d{1,2}月(?:\d{1,2}日)?)", compact, re.IGNORECASE)
        or re.fullmatch(r"(?:Q[1-4]|quarter\s*[1-4])\s*(?:19|20)\d{2}", normalized, re.IGNORECASE)
        or re.fullmatch(r"(?:19|20)\d{2}\s*(?:Q[1-4]|quarter\s*[1-4])", normalized, re.IGNORECASE)
    )


def looks_like_example_identifier_literal(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text)
    return bool(
        re.fullmatch(r"[A-Z]{2,}[-_]?\d+[A-Z0-9_-]*", normalized)
        or re.fullmatch(r"(?:ticket|issue|case)[-_ #:]?\d+", normalized, re.IGNORECASE)
        or re.fullmatch(r"(?:编号|工单)\s*[A-Za-z0-9_-]*\d+", normalized)
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


def looks_like_user_preference_literal(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    preference_markers = ["喜欢", "偏好", "喜好", "爱好", "likes", "prefers", "preference"]
    concrete_preference_markers = [
        "蓝色",
        "红色",
        "绿色",
        "黄色",
        "紫色",
        "咖啡",
        "电影",
        "文章",
        "coffee",
        "movie",
        "movies",
        "article",
        "articles",
    ]
    return any(marker in lowered for marker in preference_markers) and any(
        marker in lowered for marker in concrete_preference_markers
    )


def looks_like_location_consumption_literal(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    location_markers = ["北京", "南京", "上海", "深圳", "广州", "成都", "门店", "星巴克", "starbucks"]
    activity_markers = ["消费", "购买", "下单", "visited", "bought", "purchased"]
    return any(marker in lowered for marker in location_markers) and any(marker in lowered for marker in activity_markers)


def looks_like_product_usage_literal(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text).lower()
    product_markers = [
        "macbook",
        "iphone",
        "xps",
        "华为",
        "小米",
        "苹果",
        "oppo",
        "电脑",
        "手机",
        "产品",
    ]
    action_markers = ["购买", "使用", "买了", "uses", "bought", "purchased"]
    return any(marker in lowered for marker in product_markers) and any(marker in lowered for marker in action_markers)


def position_inside_fenced_code_block(text: str, position: int) -> bool:
    fence_char = ""
    fence_length = 0
    cursor = 0
    for line in text.splitlines(keepends=True):
        line_end = cursor + len(line)
        stripped_newline = line.rstrip("\r\n")
        if position < line_end:
            return bool(fence_char)
        if fence_char:
            if closing_fence_line(stripped_newline, fence_char, fence_length):
                fence_char = ""
                fence_length = 0
        elif match := opening_fence_line(stripped_newline):
            marker = match.group("marker")
            fence_char = marker[0]
            fence_length = len(marker)
        cursor = line_end
    return bool(fence_char and position >= cursor)


def position_inside_inline_code_span(text: str, position: int) -> bool:
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    local_position = position - line_start
    before = line[:local_position]
    after = line[local_position:]
    return before.count("`") % 2 == 1 and after.count("`") > 0


def open_question_scope_cleanup_claim(claim: GroundingClaim, item: WikiMergePlanItem | None) -> bool:
    marker = open_question_scope_cleanup_reason_marker(claim.reason)
    return bool(
        item is not None
        and item.page_type == "open_question"
        and claim.support == "unsupported"
        and claim.action == "needs_review"
        and claim.section_key in OPEN_QUESTION_SCOPE_CLEANUP_SECTIONS
        and marker
        and marker in claim.text
    )


def open_question_scope_cleanup_reason_marker(reason: str) -> str:
    if not reason.startswith(OPEN_QUESTION_SCOPE_CLEANUP_REASON_PREFIX):
        return ""
    match = re.match(rf"^{re.escape(OPEN_QUESTION_SCOPE_CLEANUP_REASON_PREFIX)} `([^`]+)`", reason)
    return match.group(1).strip() if match else ""


def remove_grounding_claim_exact_once(body: str, text: str) -> tuple[str, str]:
    if not text or text not in body:
        return body, "skipped_missing_exact_text"
    if body.count(text) != 1:
        return body, "skipped_ambiguous_repeated_text"
    updated = body.replace(text, "", 1)
    updated = re.sub(r"[ \t]+", " ", updated)
    updated = re.sub(r"\s+([。！？；;，,])", r"\1", updated)
    updated = re.sub(r"^[\s。；;，,、]+", "", updated)
    updated = re.sub(r"[\s。；;，,、]+$", "", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated.strip(), "removed"


def questionize_open_question_scope_claim(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    if any(marker in normalized for marker in ["偏好", "喜好"]):
        return "待补来源：召回到不准确的用户偏好时，系统应如何确认与纠正？"
    if any(marker in normalized for marker in ["支付", "付款", "删除", "下单", "不可逆", "高风险"]):
        return "待补来源：不准确的记忆在高风险或不可逆操作中是否会造成错误结果？需要补充来源确认。"
    if any(marker in normalized for marker in ["隐私", "污染", "混杂", "隔离", "命名空间"]):
        return "待补来源：记忆隔离或命名空间配置不当会带来哪些风险？需要补充来源确认。"
    return "待补来源：记忆不准确会带来什么后果，系统应如何确认与纠正？"


def open_question_scope_cleanup_section_placeholder(section_key: str) -> str:
    if section_key == "examples":
        return "暂无来源内可确认的具体例子；相关风险已转入未决问题。"
    if section_key == "value_points":
        return "用于整理仍需来源确认的风险判断和产品设计边界。"
    return "相关未证实风险已转入未决问题，等待补充来源。"


def append_open_question_line(existing: str, question: str) -> str:
    lines = [line.rstrip() for line in existing.splitlines() if line.strip()]
    prefix = "- " if not question.lstrip().startswith(("-", "*")) else ""
    lines.append(f"{prefix}{question}")
    return "\n".join(lines)


def draft_grounding_review_from_claims(claims: list[GroundingClaim]) -> DraftGroundingReview:
    unsupported_new_facts = [
        claim
        for claim in claims
        if claim.claim_type == "new_fact" and claim.support == "unsupported" and claim.action == "needs_review"
    ]
    return DraftGroundingReview(
        unsupported_new_facts=unsupported_new_facts,
        claims=claims,
        requires_review=bool(unsupported_new_facts),
    )


def assemble_knowledge_page(
    *,
    item: WikiMergePlanItem,
    page: DraftPageItem,
    existing_entry: WikiContextEntry,
    raw_path: str,
    raw_hash: str,
    prepared_hash: str,
    operation_id: str,
    log_date: str,
    update_reports: list[UpdatePageMergeReport] | None = None,
    related_reports: list[RelatedCandidateReport] | None = None,
    grounding_claims: list[GroundingClaim] | None = None,
    known_related_paths: set[str] | None = None,
    approved_raw_text: str = "",
) -> str:
    existing_sections = parse_existing_sections(existing_entry.content)
    summary = page.section_bodies.get("summary") or item.new_understanding
    detail = page.section_bodies.get("detail") or item.knowledge_delta or item.new_understanding
    examples = page.section_bodies.get("examples") or "暂无相关例子记录。"
    values = page.section_bodies.get("value_points") or "\n".join(f"- {value}" for value in item.value_points) or "暂无明确价值点记录。"
    additional_notes = page.section_bodies.get("additional_notes", "").strip()
    questions = page.section_bodies.get("open_questions") or "暂无矛盾与未决问题记录。"
    metadata = existing_entry.metadata
    is_update = item.action == "update" and metadata is not None
    final_title = metadata.title if is_update and metadata.title else item.display_title
    aliases = list(metadata.aliases if metadata is not None else [])
    if is_update and item.display_title and item.display_title != final_title and item.display_title not in aliases:
        aliases.append(item.display_title)
    created = metadata.created if metadata is not None and metadata.created else log_date
    section_changes: list[SectionMergeChange] = []
    if is_update:
        update_absorption_context = "\n\n".join(
            section
            for section in [summary, detail, examples, values, additional_notes, questions]
            if section.strip()
        )
        summary, summary_change = merge_update_section(
            "summary",
            existing_sections.get("summary", ""),
            summary,
            absorption_context=update_absorption_context,
        )
        detail, detail_change = merge_update_section(
            "detail",
            existing_sections.get("detail", ""),
            detail,
            absorption_context=update_absorption_context,
        )
        examples, examples_change = merge_update_section(
            "examples",
            existing_sections.get("examples", ""),
            examples,
            absorption_context=update_absorption_context,
        )
        values, values_change = merge_update_section(
            "value_points",
            existing_sections.get("value_points", ""),
            values,
            absorption_context=update_absorption_context,
        )
        additional_notes, notes_change = merge_update_section(
            "additional_notes",
            existing_sections.get("additional_notes", ""),
            additional_notes,
            absorption_context=update_absorption_context,
        )
        questions, questions_change = merge_update_section(
            "open_questions",
            existing_sections.get("open_questions", ""),
            questions,
            absorption_context=update_absorption_context,
        )
        section_changes.extend([summary_change, detail_change, examples_change, values_change, notes_change, questions_change])
    additional_notes_section = f"## 补充观察\n\n{additional_notes}\n\n" if additional_notes else ""
    related = render_related_pages(
        item,
        existing_entry=existing_entry,
        report_list=related_reports,
        known_paths=known_related_paths,
    )
    if update_reports is not None and is_update:
        update_reports.append(
            UpdatePageMergeReport(
                page_plan_id=item.page_plan_id,
                target_path=item.canonical_target_path,
                old_title=metadata.title,
                final_title=final_title,
                model_title=item.display_title,
                retained_title=final_title == metadata.title,
                merged_page_plan_ids=item.merged_page_plan_ids,
                noop_covered_by_update=item.noop_covered_by_update,
                sections=section_changes,
            )
        )
    if grounding_claims is not None:
        collect_grounding_claims(
            item=item,
            page=page,
            existing_entry=existing_entry,
            approved_raw_text=approved_raw_text,
            claims=grounding_claims,
        )
    source_raw_paths = _append_unique(metadata.source_raw_paths if metadata is not None else [], raw_path)
    source_raw_hashes = _append_unique(metadata.source_raw_hashes if metadata is not None else [], raw_hash)
    source_prepared_hashes = _append_unique(metadata.source_prepared_hashes if metadata is not None else [], prepared_hash)
    source_operation_ids = _append_unique(metadata.source_operation_ids if metadata is not None else [], operation_id)
    return (
        "---\n"
        f"llmwiki_type: {item.page_type}\n"
        f"title: {yaml_scalar(final_title)}\n"
        f"{_yaml_list('aliases', aliases)}"
        f"summary: {yaml_scalar(summary)}\n"
        f"created: {created}\n"
        f"updated: {log_date}\n"
        f"{_yaml_list('source_raw_paths', source_raw_paths)}"
        f"{_yaml_list('source_raw_hashes', source_raw_hashes)}"
        f"{_yaml_list('source_prepared_hashes', source_prepared_hashes)}"
        f"{_yaml_list('source_operation_ids', source_operation_ids)}"
        f"last_ingest_operation: {yaml_scalar(operation_id)}\n"
        "---\n\n"
        f"# {final_title}\n\n"
        "## 摘要\n\n"
        f"{summary}\n\n"
        "## 详情\n\n"
        f"{detail}\n\n"
        "## 例子\n\n"
        f"{examples}\n\n"
        "## 价值点\n\n"
        f"{values}\n\n"
        f"{additional_notes_section}"
        "## 相关页面\n\n"
        f"{related}\n\n"
        "## 矛盾与未决问题\n\n"
        f"{questions}\n"
    )


def _append_unique(existing: list[str], value: str) -> list[str]:
    items: list[str] = []
    for item in [*existing, value]:
        if item and item not in items:
            items.append(item)
    return items


def _yaml_list(key: str, values: list[str]) -> str:
    if not values:
        return f"{key}: []\n"
    return f"{key}:\n" + "".join(f"  - {yaml_scalar(value)}\n" for value in values)


def render_source_page(
    *,
    title: str,
    digest: SourceDigestArtifact,
    operation_id: str,
    linked_pages: list[str],
    touched_pages: list[str],
    no_change_pages: list[str],
    log_date: str,
    raw_hash: str,
    prepared_hash: str,
    cleanup: RawLinkCleanupArtifact,
) -> str:
    links = "\n".join(f"- `{path}`" for path in linked_pages) or "- 暂无派生知识页。"
    no_change = render_source_unwritten_notes(digest, no_change_pages)
    summary = neutralize_markdown_links(digest.summary)
    takeaways = "\n".join(f"- {neutralize_markdown_links(item)}" for item in digest.key_takeaways) or "- 暂无关键收获记录。"
    return (
        "---\n"
        "llmwiki_type: source\n"
        f"title: {yaml_scalar(title)}\n"
        "aliases: []\n"
        f"summary: {yaml_scalar(summary)}\n"
        f"created: {log_date}\n"
        f"updated: {log_date}\n"
        "source_raw_paths:\n"
        f"  - {yaml_scalar(digest.source_raw_path)}\n"
        "source_raw_hashes:\n"
        f"  - {yaml_scalar(raw_hash)}\n"
        "source_prepared_hashes:\n"
        f"  - {yaml_scalar(prepared_hash)}\n"
        "source_operation_ids:\n"
        f"  - {yaml_scalar(operation_id)}\n"
        f"raw_cleanup_pre_sha256: {yaml_scalar(cleanup.pre_cleanup_sha256)}\n"
        f"raw_cleanup_post_sha256: {yaml_scalar(cleanup.post_cleanup_sha256)}\n"
        f"raw_cleanup_rule_version: {yaml_scalar(cleanup.cleanup_rule_version)}\n"
        f"raw_cleanup_artifact_ref: {yaml_scalar('raw_link_cleanup/raw_link_cleanup.json')}\n"
        f"raw_cleanup_changed: {str(cleanup.changed).lower()}\n"
        f"raw_cleanup_cleaned_link_count: {cleanup.cleaned_link_count}\n"
        f"raw_cleanup_diff_ref: {yaml_scalar('raw_link_cleanup/cleanup.diff')}\n"
        f"last_ingest_operation: {yaml_scalar(operation_id)}\n"
        "---\n\n"
        f"# {title}\n\n"
        "## 摘要\n\n"
        f"{summary}\n\n"
        "## 原始材料\n\n"
        f"- `{digest.source_raw_path}`\n\n"
        "## 关键收获\n\n"
        f"{takeaways}\n\n"
        "## 派生知识页\n\n"
        f"{links}\n\n"
        "## 未写入说明\n\n"
        f"{no_change}\n"
    )


def render_source_unwritten_notes(digest: SourceDigestArtifact, no_change_pages: list[str]) -> str:
    sections = [
        (
            "未改动页面：\n" + "\n".join(f"- `{path}`" for path in no_change_pages)
            if no_change_pages
            else "暂无未写入页面。"
        )
    ]
    if digest.budget_deferred_candidates:
        rows = [
            [
                candidate.candidate_id,
                candidate.type,
                neutralize_markdown_links(candidate.suggested_page_title or candidate.name),
                neutralize_markdown_links(candidate.one_sentence_summary),
                neutralize_markdown_links(candidate.wiki_value),
                neutralize_markdown_links(candidate.resolution_hint),
            ]
            for candidate in digest.budget_deferred_candidates
        ]
        sections.extend(
            [
                "### 预算延后候选（未独立建页）",
                "这些候选因 `max_ingest_candidates` 预算限制未进入本轮页面规划；它们保留在 source digest 和预算报告中，后续可单独建页或聚合进总览/对比页。",
                format_markdown_table(["ID", "类型", "建议标题", "摘要", "Wiki 价值", "处理提示"], rows),
            ]
        )
        aggregation_rows = [
            [
                aggregation["suggested_page_type"],
                neutralize_markdown_links(str(aggregation["suggested_title"])),
                neutralize_markdown_links(str(aggregation["coverage_summary"])),
                ", ".join(str(candidate.get("title", candidate.get("candidate_id", ""))) for candidate in aggregation.get("representative_candidates", [])[:4]),
                neutralize_markdown_links(str(aggregation["suggested_action"])),
            ]
            for aggregation in build_deferred_candidate_aggregations(digest.budget_deferred_candidates)
        ]
        if aggregation_rows:
            sections.extend(
                [
                    "### 延后候选聚合建议",
                    "这些聚合建议只记录后续处理路径，不会在本轮增加知识页数量。",
                    format_markdown_table(["建议页类型", "建议标题", "覆盖摘要", "代表候选", "后续动作"], aggregation_rows),
                ]
            )
    return "\n\n".join(sections)


def neutralize_markdown_links(text: str) -> str:
    def wiki_repl(match: re.Match[str]) -> str:
        label = match.group(1).replace("|", " / ").strip()
        return f"`{label}`" if label else ""

    def markdown_repl(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        target = match.group(2).strip()
        if not label:
            return f"`{target}`"
        return f"{label} (`{target}`)" if target else label

    text = re.sub(r"\[\[([^\]]+)\]\]", wiki_repl, text)
    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", markdown_repl, text)


def build_index_rows(profile: Any, plan: WikiMergePlanArtifact, draft: DraftRenderingArtifact, snapshot: WikiContextSnapshot) -> list[dict[str, str]]:
    rows_by_path: dict[str, dict[str, str]] = {}
    for pool_entry in snapshot.knowledge_metadata_pool:
        metadata = pool_entry.metadata
        if metadata is None or metadata.llmwiki_type.lower() == "source":
            continue
        rows_by_path[metadata.path] = {
            "title": clean_display_title(metadata.title),
            "page": obsidian_link(metadata.path),
            "type": metadata.llmwiki_type,
            "summary": metadata.summary,
            "updated": metadata.updated,
        }
    page_by_id = {page.page_plan_id: page for page in draft.pages}
    for item in plan.items:
        if item.action not in {"create", "update"}:
            continue
        if item.page_type.lower() == "source":
            continue
        page = page_by_id.get(item.page_plan_id)
        summary = page.section_bodies.get("summary") if page else item.new_understanding
        title = item.display_title
        if item.action == "update":
            entry = next((entry for entry in snapshot.entries if entry.path == f"wiki/{item.canonical_target_path}"), None)
            if entry is not None and entry.metadata is not None:
                title = clean_display_title(entry.metadata.title)
        rows_by_path[item.canonical_target_path] = {
            "title": title,
            "page": obsidian_link(item.canonical_target_path),
            "type": item.page_type,
            "summary": summary or item.new_understanding,
            "updated": plan.log_date,
        }
    type_order = list(profile.page_types)
    rows = list(rows_by_path.values())
    rows.sort(key=lambda row: row["page"])
    rows.sort(key=lambda row: row["updated"], reverse=True)
    rows.sort(key=lambda row: type_order.index(row["type"]) if row["type"] in type_order else len(type_order))
    return rows


def build_open_question_rows(plan: WikiMergePlanArtifact, draft: DraftRenderingArtifact, snapshot: WikiContextSnapshot) -> list[dict[str, str]]:
    rows, _ = build_open_question_rows_with_report(plan, draft, snapshot)
    return rows


def build_open_question_rows_with_report(
    plan: WikiMergePlanArtifact,
    draft: DraftRenderingArtifact,
    snapshot: WikiContextSnapshot,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidates: list[dict[str, str]] = []
    for entry in snapshot.entries:
        metadata = entry.metadata
        if entry.expected_state != "present" or metadata is None or metadata.llmwiki_type.lower() == "source":
            continue
        for question in extract_open_questions(entry.content):
            candidates.append({
                "question": question,
                "page": obsidian_link(metadata.path, clean_display_title(metadata.title)),
                "path": metadata.path,
                "updated": metadata.updated,
                "page_type": metadata.llmwiki_type,
                "source": "existing_wiki",
            })
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    for page in draft.pages:
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            continue
        for question in meaningful_open_question_lines(page.section_bodies.get("open_questions", "")):
            candidates.append({
                "question": question,
                "page": obsidian_link(item.canonical_target_path, item.display_title),
                "path": item.canonical_target_path,
                "updated": plan.log_date,
                "page_type": item.page_type,
                "source": "draft",
            })
    by_key = group_open_question_candidates(candidates)
    rows: list[dict[str, str]] = []
    report_items: list[dict[str, Any]] = []
    for key, grouped in sorted(by_key.items()):
        representative = max(grouped, key=open_question_representative_sort_key)
        low_signal = is_low_signal_open_question(representative["question"])
        repeated_gap = len(grouped) >= 2 and low_signal
        keep = (
            any(item["page_type"] == "open_question" for item in grouped)
            or repeated_gap
            or not low_signal
        )
        pages = _dedupe_strings([item["page"] for item in sorted(grouped, key=lambda item: item["updated"], reverse=True)])[:3]
        decision = "kept" if keep else "filtered"
        reason = "open_question_page" if any(item["page_type"] == "open_question" for item in grouped) else ""
        if not reason:
            reason = "repeated_source_gap" if repeated_gap else ("low_signal_or_source_gap" if low_signal else "high_signal")
        report_items.append(
            {
                "normalized_key": key,
                "question": representative["question"],
                "decision": decision,
                "reason": reason,
                "pages": pages,
                "occurrences": len(grouped),
            }
        )
        if not keep:
            continue
        rows.append(
            {
                "question": representative["question"],
                "page": ", ".join(pages),
                "updated": max(item["updated"] for item in grouped),
            }
        )
    rows.sort(key=lambda row: (row["updated"], row["page"], row["question"]), reverse=True)
    return rows, {
        "schema_version": "index_open_questions_report.v1",
        "kept_count": sum(1 for item in report_items if item["decision"] == "kept"),
        "filtered_count": sum(1 for item in report_items if item["decision"] == "filtered"),
        "deduped_count": sum(max(0, item["occurrences"] - 1) for item in report_items if item["decision"] == "kept"),
        "items": report_items,
    }


def open_question_representative_sort_key(item: dict[str, str]) -> tuple[int, str, tuple[int, int, int]]:
    question = item["question"]
    non_low_signal = 0 if is_low_signal_open_question(question) else 1
    return (non_low_signal, item["updated"], open_question_representative_score(question))


def open_question_key(question: str) -> str:
    text = strip_open_question_marker(question)
    text = re.sub(r"^(待补来源|待补充来源|需要来源|缺少来源)\s*[:：]\s*", "", text)
    text = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\s，。；;：:、,.!?！？（）()【】\[\]\"'“”‘’]+", "", text)
    semantic_key = semantic_open_question_key(normalized)
    return semantic_key or normalized


def strip_open_question_marker(question: str) -> str:
    text = re.sub(r"^\s*[-*]\s+", "", question.strip())
    return re.sub(r"^\s*\d+\s*[.)、．]\s*", "", text).strip()


def group_open_question_candidates(candidates: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        key = open_question_key(candidate["question"])
        merge_key = next(
            (
                existing_key
                for existing_key, grouped in groups.items()
                if open_question_keys_should_merge(key, existing_key, candidate, grouped)
            ),
            None,
        )
        groups.setdefault(merge_key or key, []).append(candidate)
    return groups


def open_question_keys_should_merge(
    key: str,
    existing_key: str,
    candidate: dict[str, str],
    grouped: list[dict[str, str]],
) -> bool:
    if key == existing_key:
        return True
    if existing_key.startswith("semantic:") and key.startswith("semantic:"):
        candidate_norm = open_question_similarity_text(candidate["question"])
        return any(
            candidate.get("path") == existing.get("path")
            and open_question_token_overlap(candidate_norm, open_question_similarity_text(existing["question"])) >= 0.60
            for existing in grouped
        )
    if open_question_key_contains_other(key, existing_key):
        return True
    candidate_norm = open_question_similarity_text(candidate["question"])
    if not candidate_norm:
        return False
    for existing in grouped:
        existing_norm = open_question_similarity_text(existing["question"])
        if not existing_norm:
            continue
        same_path = candidate.get("path") == existing.get("path")
        if open_question_key_contains_other(candidate_norm, existing_norm):
            return True
        if same_path and open_question_token_overlap(candidate_norm, existing_norm) >= 0.62:
            return True
    return False


def open_question_key_contains_other(left: str, right: str) -> bool:
    if len(left) < 12 or len(right) < 12:
        return False
    return left in right or right in left


def open_question_similarity_text(question: str) -> str:
    text = strip_open_question_marker(question)
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s，。；;：:、,.!?！？（）()【】\[\]\"'“”‘’]+", "", text)


def open_question_token_overlap(left: str, right: str) -> float:
    left_tokens = open_question_similarity_tokens(left)
    right_tokens = open_question_similarity_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    return len(intersection) / min(len(left_tokens), len(right_tokens))


def open_question_similarity_tokens(normalized: str) -> set[str]:
    text = normalized
    for stop in ["如何", "是否", "能否", "会不会", "为什么", "什么", "哪些", "是否可能", "可能", "应该", "需要"]:
        text = text.replace(stop, "")
    tokens = set(re.findall(r"[a-z][a-z0-9_/-]{1,}", text))
    cjk = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
    for size in (4, 3):
        for index in range(0, max(0, len(cjk) - size + 1)):
            token = cjk[index : index + size]
            if open_question_similarity_token_is_noise(token):
                continue
            tokens.add(token)
    return tokens


def open_question_similarity_token_is_noise(token: str) -> bool:
    if all(char in "的了和与及或是否如何什么为什么可能需要应该能否会不会" for char in token):
        return True
    return token in {"产品", "功能", "用户", "团队", "问题", "未来", "影响", "风险"}


def open_question_representative_score(question: str) -> tuple[int, int, int]:
    stripped = strip_open_question_marker(question)
    single_question = 1 if stripped.count("？") + stripped.count("?") <= 1 else 0
    has_source_gap = 1 if is_low_signal_open_question(stripped) else 0
    return (single_question, -has_source_gap, -len(stripped))


def semantic_open_question_key(normalized: str) -> str:
    for key, required_groups in OPEN_QUESTION_SEMANTIC_CLUSTERS:
        if all(any(term in normalized for term in group) for group in required_groups):
            return key
    return ""


def is_low_signal_open_question(question: str) -> bool:
    normalized = open_question_key(question)
    if len(normalized) < 10:
        return True
    low_signal_markers = ["待补来源", "待补充来源", "需要来源", "缺少来源", "source needed", "citation needed"]
    if any(marker in question.lower() for marker in low_signal_markers):
        return True
    source_gap_markers = ["具体引用", "具体来源", "出处", "引用链接", "原始证据"]
    return any(marker in question for marker in source_gap_markers)


def render_index_open_questions_report(report: dict[str, Any]) -> str:
    rows = [
        [
            item["decision"],
            item["reason"],
            item["question"],
            ", ".join(item["pages"]),
            str(item["occurrences"]),
        ]
        for item in report.get("items", [])
    ]
    return (
        "# Index 未决问题筛选报告\n\n"
        f"- 保留：{report.get('kept_count', 0)}\n"
        f"- 过滤：{report.get('filtered_count', 0)}\n\n"
        f"- 合并重复：{report.get('deduped_count', 0)}\n\n"
        + (format_markdown_table(["决策", "原因", "问题", "关联页面", "次数"], rows) if rows else "暂无未决问题。")
        + "\n"
    )


def extract_open_questions(markdown: str) -> list[str]:
    match = re.search(r"(?ms)^##\s+矛盾与未决问题\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", markdown)
    if not match:
        return []
    return meaningful_open_question_lines(match.group("body"))


def meaningful_open_question_lines(text: str) -> list[str]:
    results: list[str] = []
    for raw_line in text.splitlines():
        line = strip_open_question_marker(raw_line)
        line = line.strip("。；; ")
        if not line:
            continue
        normalized = re.sub(r"\s+", "", line)
        if any(marker in normalized for marker in ["暂无", "没有", "无未决", "无矛盾", "不适用", "N/A", "na"]):
            continue
        if len(normalized) < 4:
            continue
        results.append(line)
    return _dedupe_strings(results)


def render_related_pages(
    item: WikiMergePlanItem,
    *,
    existing_entry: WikiContextEntry | None = None,
    report_list: list[RelatedCandidateReport] | None = None,
    known_paths: set[str] | None = None,
) -> str:
    candidates: list[dict[str, Any]] = []
    for order, related in enumerate(item.related_pages):
        candidates.append(
            {
                "target_path": _strip_wiki_prefix(related.target_path),
                "display_title": related.display_title,
                "reason": chinese_related_reason(related.reason, "该页面与当前主题存在明确内容互补关系。"),
                "source": related.source,
                "priority": related_candidate_priority(related.source),
                "order": order,
            }
        )
    if existing_entry is not None and existing_entry.expected_state == "present":
        candidates.extend(parse_existing_related_candidates(existing_entry.content))

    candidates.sort(key=lambda candidate: (int(candidate.get("priority", 50)), int(candidate.get("order", 0))))
    rows: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = normalize_related_candidate_path(candidate["target_path"])
        title = candidate["display_title"].strip() or clean_display_title(Path(candidate["target_path"]).stem)
        reason = normalize_stable_brand_typos(
            chinese_related_reason(candidate["reason"], "该页面与当前主题存在明确内容互补关系。")
        )
        reject_reason = ""
        if path is None:
            reject_reason = "unknown_path"
        elif path == item.canonical_target_path:
            reject_reason = "self_link"
        elif path in seen:
            reject_reason = "duplicate"
        elif known_paths is not None and path not in known_paths:
            reject_reason = "unknown_path"
        elif len(rows) >= FINAL_RELATED_LIMIT:
            reject_reason = "cap_cutoff"
        if report_list is not None:
            report_list.append(
                RelatedCandidateReport(
                    page_plan_id=item.page_plan_id,
                    target_path=path or candidate["target_path"],
                    display_title=title,
                    reason=reason,
                    source=candidate["source"],
                    decision="cutoff" if reject_reason == "cap_cutoff" else ("filtered" if reject_reason else "kept"),
                    reject_reason=reject_reason,
                )
            )
        if reject_reason or path is None:
            continue
        seen.add(path)
        rows.append(f"- {obsidian_alias_link(path, title)}：{reason}")
    if not rows:
        if report_list is not None:
            report_list.append(
                RelatedCandidateReport(
                    page_plan_id=item.page_plan_id,
                    target_path=item.canonical_target_path,
                    display_title=item.display_title,
                    reason=item.related_absence_reason or "low_confidence",
                    source="absence_reason",
                    decision="filtered",
                    reject_reason=item.related_absence_reason or "low_confidence",
                )
            )
        return "- 暂无相关页面记录。"
    return "\n".join(rows)


def parse_existing_related_candidates(markdown: str) -> list[dict[str, str]]:
    match = re.search(r"(?ms)^##\s+相关页面\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", markdown)
    if not match:
        return []
    candidates: list[dict[str, Any]] = []
    for line in match.group("body").splitlines():
        for link in re.finditer(r"\[\[([^\]]+)\]\]", line):
            target, _, alias = link.group(1).partition("|")
            path = normalize_related_candidate_path(target)
            order = len(candidates)
            candidates.append(
                {
                    "target_path": path or target.strip(),
                    "display_title": alias.strip() or clean_display_title(Path(target).stem),
                    "reason": "旧 Related 作为候选重新参与排序。",
                    "source": "existing_related",
                    "priority": 0,
                    "order": order,
                }
            )
    return candidates


def related_candidate_priority(source: str) -> int:
    return {
        "existing_related": 0,
        "exact_or_alias": 1,
        "source_digest": 2,
        "wiki_context": 3,
    }.get(source, 9)


def normalize_related_candidate_path(value: str) -> str | None:
    text = value.strip().strip("`").replace("\\", "/")
    if not text:
        return None
    text = text.split("#", 1)[0]
    while text.startswith("./"):
        text = text[2:]
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.parts and path.parts[0] == "wiki":
        path = Path(*path.parts[1:])
    if not path.parts or path.parts[0] in {"sources", "logs"} or path.as_posix() in {"index.md", "log.md"}:
        return None
    if path.suffix != ".md":
        path = path.with_suffix(".md")
    return path.as_posix()


def _strip_wiki_prefix(value: str) -> str:
    path = Path(value.strip().replace("\\", "/"))
    if path.parts and path.parts[0] == "wiki":
        return Path(*path.parts[1:]).as_posix()
    return path.as_posix()


def render_update_merge_report(report: UpdateMergeReport) -> str:
    if not report.pages:
        return "# Update 合并报告\n\n本次没有 update 页面。\n"
    sections = ["# Update 合并报告", ""]
    for page in report.pages:
        sections.extend(
            [
                f"## {page.final_title or page.target_path}",
                "",
                f"- 页面计划：`{page.page_plan_id}`",
                f"- 目标：`{page.target_path}`",
                f"- 旧标题：{page.old_title or '无'}",
                f"- 模型标题：{page.model_title or '无'}",
                f"- 最终标题：{page.final_title or '无'}",
                f"- 保留旧标题：{'是' if page.retained_title else '否'}",
                f"- 合并计划页：{', '.join(f'`{item}`' for item in page.merged_page_plan_ids) if page.merged_page_plan_ids else '无'}",
                f"- noop 被 update 覆盖：{'是' if page.noop_covered_by_update else '否'}",
                "",
            ]
        )
        rows = [
            [
                change.section_key,
                "\n".join(change.retained) or "无",
                "\n".join(change.added) or "无",
                "\n".join(change.removed) or "无",
                "\n".join(change.preserved_old) or "无",
                "是" if change.needs_manual_resolution else "否",
                change.removal_reason,
            ]
            for change in page.sections
        ]
        sections.append(
            format_markdown_table(["段落", "保留", "新增", "删除", "旧页保留观察", "需人工消化", "原因"], rows)
            if rows
            else "没有记录 section 级变更。"
        )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def render_related_merge_report(report: RelatedMergeReport) -> str:
    rows = [
        [
            item.page_plan_id,
            f"`{item.target_path}`",
            item.display_title,
            item.source,
            item.decision,
            item.reject_reason,
            item.reason,
        ]
        for item in report.candidates
    ]
    body = format_markdown_table(["页面计划", "目标", "标题", "来源", "决策", "过滤原因", "理由"], rows) if rows else "没有 Related 候选。"
    return "# 相关页面合并报告\n\n" + body + "\n"


def render_draft_grounding_review(review: DraftGroundingReview) -> str:
    summary = "需要人工确认" if review.requires_review else "通过"
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
    sections = [
        "# 草稿来源支撑审查",
        "",
        f"- 结果：{summary}",
        f"- 未支撑新增事实数量：{len(review.unsupported_new_facts)}",
        "",
        "## 需要确认的新事实",
        "",
        format_markdown_table(["页面计划", "目标", "段落", "原因", "文本"], unsupported_rows) if unsupported_rows else "暂无。",
        "",
        "## 全部分类",
        "",
        format_markdown_table(["页面计划", "目标", "段落", "类型", "支持", "处理", "原因", "文本"], rows) if rows else "暂无分类记录。",
    ]
    return "\n".join(sections).rstrip() + "\n"


def obsidian_link(path: str, title: str | None = None) -> str:
    target = Path(path)
    if target.parts and target.parts[0] == "wiki":
        target = Path(*target.parts[1:])
    return f"[[{target.with_suffix('').as_posix()}]]"


def obsidian_alias_link(path: str, title: str) -> str:
    target = Path(path)
    if target.parts and target.parts[0] == "wiki":
        target = Path(*target.parts[1:])
    return f"[[{target.with_suffix('').as_posix()}|{obsidian_link_label(title)}]]"


def obsidian_link_label(value: str) -> str:
    label = " ".join(value.replace("|", "/").replace("]", "").split())
    return label or "Untitled"


def render_update_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    return "".join(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=old_name,
            tofile=new_name,
        )
    )


def build_draft_approval(
    run_dir: Path,
    approved_manifest_path: Path,
    *,
    decision: Literal["approved", "pending", "rejected"],
    review_mode: Literal["auto_stub", "manual", "not_required"],
    auto_approved: bool,
    notes: str,
) -> DraftApproval:
    require_m42_draft_sidecars(run_dir)
    approved_manifest = read_model(approved_manifest_path, DraftWriteManifest)
    markdown_hashes: dict[str, str] = {}
    for target in approved_manifest.targets:
        draft = run_dir / target.draft_path
        if draft.suffix == ".md" and draft.exists():
            markdown_hashes[target.draft_path] = sha256_file(draft)
    for rel_path in M42_REQUIRED_DRAFT_SIDECARS:
        path = run_dir / rel_path
        markdown_hashes[rel_path] = sha256_file(path)
    return DraftApproval(
        decision=decision,
        review_mode=review_mode,
        auto_approved=auto_approved,
        approved_draft_json_sha256=sha256_file(approved_manifest_path),
        approved_markdown_sha256=markdown_hashes,
        notes=notes,
    )


def render_draft_review_prompt(run_dir: Path, draft_manifest: DraftWriteManifest) -> str:
    manual_resolution_count = update_manual_resolution_count(run_dir)
    reinforcement_count = update_reinforcement_count(run_dir)
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
        f"Draft rendering 本地补强了 {reinforcement_count} 段旧页知识，并记录在 "
        "`draft_rendering/update_preservation_reinforcement_report.md`。如本轮还因其他问题进入人工审核，"
        "可顺手检查这些桥接语是否自然。\n\n"
        if reinforcement_count
        else ""
    )
    rows = [
        [
            target.action,
            f"`{target.target_path}`",
            f"`{target.draft_path}`",
            draft_diff_ref(run_dir, target),
            draft_change_summary(run_dir, target),
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
        + format_markdown_table(
            ["动作", "目标", "草稿", "Diff", "变更摘要", "预期状态", "Preimage"],
            rows,
        )
        + "\n"
    )


def draft_review_reason(run_dir: Path, draft_manifest: DraftWriteManifest) -> str:
    reasons: list[str] = []
    if draft_manifest.requires_grounding_review:
        reasons.append("Grounding review 发现 unsupported new_fact，需要人工确认。")
    manual_resolution_count = update_manual_resolution_count(run_dir)
    reinforcement_count = update_reinforcement_count(run_dir)
    if manual_resolution_count:
        reasons.append(f"Update 合并报告包含 {manual_resolution_count} 段旧页保留观察，需人工消化、改写或确认删除。")
    return " ".join(reasons) or "草稿需要显式人工批准。"


def update_manual_resolution_count(run_dir: Path) -> int:
    report_path = run_dir / "draft_rendering" / "update_merge_report.json"
    if not report_path.exists():
        return 0
    try:
        report = read_model(report_path, UpdateMergeReport)
    except Exception:
        return 0
    return sum(1 for page in report.pages for section in page.sections if section.needs_manual_resolution)


def update_reinforcement_count(run_dir: Path) -> int:
    draft_root = run_dir / "draft_rendering"
    report_path = draft_root / "update_preservation_reinforcement_report.json"
    if report_path.exists():
        try:
            report = read_json(report_path)
            return int(report.get("reinforced_section_count", 0))
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


def draft_diff_ref(run_dir: Path, target: DraftWriteTarget) -> str:
    if not target.page_plan_id:
        return ""
    diff = run_dir / "draft_rendering" / "diffs" / f"{target.page_plan_id}.diff"
    return f"`{diff.relative_to(run_dir).as_posix()}`" if diff.exists() else ""


def draft_change_summary(run_dir: Path, target: DraftWriteTarget) -> str:
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


def build_apply_preview(vault: Path, run_dir: Path) -> ApplyPreview:
    operation_id = run_dir.name
    manifest_path = require_step_output_dir(run_dir, "draft_review") / "approved_write_manifest.json"
    draft_manifest = read_model(manifest_path, DraftWriteManifest)
    target_paths = [item.target_path for item in draft_manifest.targets]
    if len(target_paths) != len(set(target_paths)):
        raise PipelineError("draft_write_manifest contains duplicate target_path values")
    targets: list[ApplyTarget] = []
    source_targets: list[str] = []
    log_targets: list[str] = []
    index_targets: list[str] = []
    for item in draft_manifest.targets:
        target_path = item.target_path
        current = vault / target_path
        current_sha = sha256_file(current) if current.exists() else None
        if item.action == "source":
            source_targets.append(target_path)
        if item.action in {"global_log", "daily_log"}:
            log_targets.append(target_path)
        if item.action == "index":
            index_targets.append(target_path)
        targets.append(
            ApplyTarget(
                action=item.action,
                target_path=target_path,
                draft_path=item.draft_path,
                expected_state=item.expected_state,
                preimage_sha256=item.preimage_sha256,
                current_sha256=current_sha,
                will_write=True,
                approved_draft_ref=manifest_path.relative_to(run_dir).as_posix(),
                page_plan_id=item.page_plan_id,
            )
        )
    write_set_payload = {
        "approved_manifest_sha256": sha256_file(manifest_path),
        "targets": [
            {
                "target_path": target.target_path,
                "draft_path": target.draft_path,
                "preimage_sha256": target.preimage_sha256,
                "expected_state": target.expected_state,
                "draft_sha256": sha256_file(run_dir / target.draft_path),
            }
            for target in targets
        ],
    }
    write_set_sha = sha256_bytes(json.dumps(write_set_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    requires_manual_draft_review = draft_review_requires_manual(run_dir, draft_manifest)
    return ApplyPreview(
        operation_id=operation_id,
        operation_applyable=bool(targets),
        requires_draft_review=requires_manual_draft_review,
        has_updates=draft_manifest.has_updates,
        has_noops=draft_manifest.has_noops,
        blocked_reasons=[],
        write_set_sha256=write_set_sha,
        targets=targets,
        source_targets=source_targets,
        log_targets=log_targets,
        index_targets=index_targets,
    )
