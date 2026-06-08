import pytest

from helpers import draft_body
import llmwiki_engine.draft_grounding as draft_grounding
from llmwiki_engine.models import (
    DraftGroundingReview,
    DraftPageItem,
    DraftRenderingArtifact,
    GroundingClaim,
    SourceBasis,
    WikiContextEntry,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
)


def test_grounding_examples_do_not_require_raw_exact_match_for_generic_prompts() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-X",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_X.md",
        display_title="示例提示",
        page_type="concept",
        new_understanding="示例提示帮助说明 Agent 使用边界。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-X",
                action="create",
                canonical_target_path="concepts/Concept_X.md",
                summary="示例提示。",
                body_markdown=draft_body(detail="这个页面说明如何处理通用问题。", examples="- “公司报销政策是什么？”\n- “你是一位客服代表，用礼貌的语气回答。”"),
                change_summary="创建示例提示页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_X.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(draft, WikiMergePlanArtifact(log_date="2026-06-06", items=[item]), snapshot, "")

    assert review.requires_review is False
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def build_examples_grounding_case(
    examples: str,
    *,
    detail: str = "这个页面说明例子 grounding。",
) -> tuple[DraftRenderingArtifact, WikiMergePlanArtifact, WikiContextSnapshot]:
    item = WikiMergePlanItem(
        page_plan_id="PP-EXAMPLES",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Examples.md",
        display_title="例子页",
        page_type="concept",
        new_understanding="例子页用于测试 grounding。",
        section_plans={"examples": "例子"},
        reason="测试 grounding examples。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="例子页。",
                body_markdown=draft_body(detail=detail, examples=examples),
                change_summary="创建例子页。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Examples.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[item])
    return draft, plan, snapshot


def build_examples_grounding_review(examples: str) -> DraftGroundingReview:
    draft, plan, snapshot = build_examples_grounding_case(examples)
    return draft_grounding.build_draft_grounding_review(
        draft,
        plan,
        snapshot,
        "",
    )


def test_grounding_examples_allow_abstract_placeholder_quotes() -> None:
    review = build_examples_grounding_review("- “某个用户曾在某家店消费过”\n- “该用户表示喜欢某类产品”")

    assert review.requires_review is False
    assert [claim.action for claim in review.warnings] == ["warn", "warn"]


def test_grounding_examples_allow_user_preference_placeholder() -> None:
    review = build_examples_grounding_review("- “用户偏好 X”\n- “<example_id>”\n- “<time_period>”")

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["用户偏好 X", "<example_id>", "<time_period>"]


def test_grounding_examples_allow_abstract_memory_query_literals() -> None:
    review = build_examples_grounding_review(
        '- `recall("用户最近的工单信息")`\n'
        '- `search_context("用户之前提到的项目截止日期")`'
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [
        "用户最近的工单信息",
        "用户之前提到的项目截止日期",
    ]
    assert review.requires_review is False


def test_grounding_examples_allow_short_query_template_quotes() -> None:
    review = build_examples_grounding_review(
        "一个问答代理经常被问及“Redis 的安装方法”。\n"
        "用户如果用“怎么安装Redis”询问，语义缓存可命中。\n"
        "类似“查询某个用户的记忆片段”的请求可以作为模板。\n"
        "用户如果用“Redis 连接地址配置方法”询问，也是在描述技术主题。"
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [
        "Redis 的安装方法",
        "怎么安装Redis",
        "查询某个用户的记忆片段",
        "Redis 连接地址配置方法",
    ]
    reasons = {claim.text: claim.reason for claim in review.claims}
    assert reasons["Redis 的安装方法"] == "例子区的短查询/请求模板按 illustrative example 处理，不要求 raw exact match。"
    assert reasons["怎么安装Redis"] == "例子区的短查询/请求模板按 illustrative example 处理，不要求 raw exact match。"


@pytest.mark.parametrize(
    "examples",
    [
        "例如“某个用户的账户余额是多少？”",
        "类似“查看某个用户的账户余额”的请求",
        "例如“如何重置密码”",
        "比如“忘记密码怎么办”",
        "类似“query account balance”的请求",
        "例如“reset password”",
    ],
)
def test_grounding_examples_sensitive_dynamic_queries_do_not_block_ingest(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert all(claim.action != "needs_review" for claim in review.claims)


def test_grounding_detail_sensitive_dynamic_query_does_not_block_ingest() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail="语义缓存可以处理常见问题，例如“忘记密码怎么办”。",
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "detail",
    [
        "记忆评估问题“某个用户的账户余额是多少？”用于测试召回。",
        "记忆评估问题“我的账户余额是多少？”用于测试召回。",
        "记忆评估问句“查询某个用户的手机号”用于测试召回。",
        "语义缓存可以处理常见问题，例如“密码忘了怎么办”。",
        "语义缓存可以处理常见问题，例如“密码找不回怎么办”。",
        "语义缓存可以处理常见问题，例如“forgot my password”。",
        "语义缓存可以处理常见问题，例如“how to reset my password”。",
        "语义缓存可以处理常见问题，例如“change my password”。",
        "语义缓存可以处理常见问题，例如“cannot login”。",
        "语义缓存可以处理常见问题，例如“登录失败怎么处理”。",
        "语义缓存可以处理常见问题，例如“登录报错怎么处理”。",
        "语义缓存可以处理常见问题，例如“登录问题”。",
        "语义缓存可以处理常见问题，例如“账号问题”。",
        "语义缓存可以处理常见问题，例如“账户问题”。",
        "语义缓存可以处理常见问题，例如“账号登录问题”。",
        "语义缓存可以处理常见问题，例如“login failed”。",
        "语义缓存可以处理常见问题，例如“login error”。",
        "语义缓存可以处理常见问题，例如“login problem”。",
        "语义缓存可以处理常见问题，例如“sign in failed”。",
        "语义缓存可以处理常见问题，例如“sign-in failed”。",
        "语义缓存可以处理常见问题，例如“sign-in problem”。",
        "语义缓存可以处理常见问题，例如“sign-in error”。",
        "语义缓存可以处理常见问题，例如“failed login”。",
        "语义缓存可以处理常见问题，例如“failed sign-in”。",
        "语义缓存可以处理常见问题，例如“error log-in”。",
        "语义缓存可以处理常见问题，例如“log-in failed”。",
        "语义缓存可以处理常见问题，例如“log-in problem”。",
        "语义缓存可以处理常见问题，例如“account login error”。",
        "语义缓存可以处理常见问题，例如“account login problem”。",
        "语义缓存可以处理常见问题，例如“user login problem”。",
        "语义缓存可以处理常见问题，例如“account problems”。",
        "语义缓存可以处理常见问题，例如“user account problems”。",
        "技术排障示例可以写成“登不上账号”。",
        "技术排障示例可以写成“登不上账户”。",
        "技术排障示例可以写成“登不上后台”。",
        '技术排障示例可以写成"登不上账号"。',
        '技术排障示例可以写成"login failed"。',
        '技术排障示例可以写成"login problem"。',
        '技术排障示例可以写成"account problem"。',
        "语义缓存可以处理常见问题，例如“query user emails”。",
        "语义缓存可以处理常见问题，例如“query users emails”。",
        "语义缓存可以处理常见问题，例如“query user addresses”。",
        "语义缓存可以处理常见问题，例如“query user profiles”。",
    ],
)
def test_grounding_detail_memory_examples_sensitive_dynamic_queries_do_not_block_ingest(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert all(claim.action != "needs_review" for claim in review.claims)


def test_grounding_body_markdown_sensitive_dynamic_query_does_not_block_ingest() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="语义缓存需要区分稳定知识和动态查询。",
                body_markdown="### 风险边界\n\n用户查询订单状态时，系统不能把这个动态场景当成可缓存知识。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown, expected_marker",
    [
        ("### 医疗建议\n\n患者每天服用阿司匹林可以预防心梗。", "阿司匹林"),
        ("### 法律判断\n\n员工签署竞业协议后一定不能加入竞争公司。", "竞业"),
        ("### 金融建议\n\n普通用户应该把大部分存款投入高收益债券。", "存款"),
    ],
)
def test_grounding_body_markdown_high_risk_domain_advice_does_not_block_ingest(
    body_markdown: str,
    expected_marker: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert expected_marker


def test_grounding_body_markdown_high_risk_domain_advice_allows_source_supported_claim() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    sentence = "患者每天服用阿司匹林可以预防心梗。"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown=f"### 医疗建议\n\n{sentence}",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, sentence)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_open_questions_high_risk_domain_gap_is_not_blocked_as_fact() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown="### 边界\n\n这里不把高风险建议写成事实。",
                open_questions="- 待补来源：患者每天服用阿司匹林是否可以预防心梗？",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_low_risk_open_question_quote_warns_without_review() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="PM 角色演变仍待讨论。",
                body_markdown="### 背景\n\n这里把问题保留为待研究方向。",
                open_questions="- 待补来源：“AGI后PM是否必要？”",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["AGI后PM是否必要？"]
    assert review.warnings[0].action == "warn"


def test_grounding_low_risk_body_quote_warns_without_review() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="产品直觉需要长期训练。",
                body_markdown="### 表达方式\n\n这里把“好的产品判断往往来自长期实践中形成的经验直觉”当作一个低风险表述来记录。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["好的产品判断往往来自长期实践中形成的经验直觉"]


@pytest.mark.parametrize(
    "body_markdown, expected",
    [
        ("### 公司关系\n\n“OpenAI 收购了 Anthropic”是一个需要来源支撑的公司关系。", "OpenAI 收购了 Anthropic"),
        ("### 身份关系\n\n“Sam Altman 担任 Anthropic CEO”是一个需要来源支撑的身份关系。", "Sam Altman 担任 Anthropic CEO"),
        ("### 产品关系\n\n“Claude 由 Google 发布”是一个需要来源支撑的产品关系。", "Claude 由 Google 发布"),
    ],
)
def test_grounding_severe_factual_relationship_quote_warns_without_raw_contradiction(body_markdown: str, expected: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [expected]


@pytest.mark.parametrize(
    "body_markdown, expected",
    [
        ("### 公司关系\n\nOpenAI 收购了 Anthropic。", "OpenAI 收购了 Anthropic。"),
        ("### 身份关系\n\nSam Altman 担任 Anthropic CEO。", "Sam Altman 担任 Anthropic CEO。"),
        ("### 产品关系\n\nClaude 由 Google 发布。", "Claude 由 Google 发布。"),
    ],
)
def test_grounding_severe_factual_relationship_unquoted_warns_without_raw_contradiction(body_markdown: str, expected: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [expected]


def test_grounding_severe_factual_relationship_quote_warning_is_not_duplicated_by_unquoted_scan() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系需要来源支撑。",
                body_markdown="### 公司关系\n\n“OpenAI 收购了 Anthropic”是一个需要来源支撑的公司关系。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["OpenAI 收购了 Anthropic"]


def test_grounding_severe_factual_relationship_blocks_when_raw_contradicts() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系与 raw 不符才阻断。",
                body_markdown="### 公司关系\n\nOpenAI 收购了 Anthropic。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Anthropic 收购了 OpenAI。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 收购了 Anthropic。"]
    assert "明显不符" in review.unsupported_new_facts[0].reason


def test_grounding_scans_body_markdown_heading_text_for_contradictions() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="标题里的事实关系也需要来源一致。",
                body_markdown="### OpenAI 收购了 Anthropic\n\n正文只补充说明这个标题。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Anthropic 收购了 OpenAI。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 收购了 Anthropic"]
    assert "明显不符" in review.unsupported_new_facts[0].reason


def test_grounding_role_relationship_blocks_when_raw_names_different_org() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="身份关系与 raw 不符才阻断。",
                body_markdown="### 身份关系\n\nSam Altman 担任 Anthropic CEO。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Sam Altman 担任 OpenAI CEO。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["Sam Altman 担任 Anthropic CEO。"]


def test_grounding_severe_factual_relationship_blocks_explicit_raw_negation() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="显式否定与肯定冲突才阻断。",
                body_markdown="### 公司关系\n\nOpenAI 收购了 Anthropic。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "OpenAI 没有收购 Anthropic。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 收购了 Anthropic。"]


def test_grounding_creator_relationship_blocks_same_object_different_creator() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="同一对象不同创建方与 raw 不符才阻断。",
                body_markdown="### 创建关系\n\nOpenAI 创建了 Claude。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Anthropic 创建了 Claude。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 创建了 Claude。"]


def test_grounding_release_relationship_same_subject_different_object_only_warns() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="同一主体发布另一个对象只是缺支撑，不算 raw 矛盾。",
                body_markdown="### 发布关系\n\nOpenAI 发布了 Sora。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "OpenAI 发布了 ChatGPT。")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert [claim.text for claim in review.warnings] == ["OpenAI 发布了 Sora。"]


@pytest.mark.parametrize(
    ("body_markdown", "approved_raw", "expected"),
    [
        ("### 创建关系\n\nClaude 由 OpenAI 创建。", "Claude 由 Anthropic 创建。", "Claude 由 OpenAI 创建。"),
        ("### 发布关系\n\nSora 由 OpenAI 发布。", "Sora 由 Anthropic 发布。", "Sora 由 OpenAI 发布。"),
        ("### Creation\n\nClaude was developed by OpenAI.", "Claude was developed by Anthropic.", "Claude was developed by OpenAI."),
    ],
)
def test_grounding_by_actor_relationship_blocks_same_object_different_actor(
    body_markdown: str,
    approved_raw: str,
    expected: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="由某方创建/发布的同一对象不同主体与 raw 不符才阻断。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, approved_raw)

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == [expected]


@pytest.mark.parametrize(
    ("body_markdown", "approved_raw"),
    [
        ("### 收购关系\n\nAnthropic 被 OpenAI 收购。", "OpenAI 收购了 Anthropic。"),
        ("### 创建关系\n\nClaude 由 Anthropic 创建。", "Anthropic 创建了 Claude。"),
    ],
)
def test_grounding_active_passive_paraphrase_does_not_count_as_raw_contradiction(
    body_markdown: str,
    approved_raw: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="主动/被动同义转述不应被当成 raw 矛盾。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, approved_raw)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_severe_factual_relationship_unquoted_allows_source_supported_claim() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    sentence = "OpenAI 收购了 Anthropic。"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系如果来自来源则保留。",
                body_markdown=f"### 公司关系\n\n{sentence}",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, sentence)

    assert review.requires_review is False
    assert [claim.action for claim in review.claims if claim.text == sentence] == ["kept"]


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 技术解释\n\nAI Agent 由模型、工具和记忆组成。",
        "### 技术解释\n\nPython 支持异步编程。",
        "### 技术解释\n\nRedis 支持语义缓存。",
    ],
)
def test_grounding_weak_technical_relationships_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="普通技术解释不应该被严重事实关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 导入流程\n\nLLM 读取源文档，提取关键信息，并更新或创建相关 wiki 页面。",
        "### Wiki 层\n\nLLM 完全拥有这一层：创建、更新、删除页面，维护交叉引用，保持一致性。",
        "### 摘要撰写\n\nLLM 在 wiki 中创建该源的摘要页面，记录来源信息及主要贡献。",
        "### 查询流程\n\n当用户向 wiki 提出问题时，LLM 会搜索相关页面并合成答案。",
        "### 自定义工具\n\n开发者可以注册自定义工具（使用 `@register_tool` 装饰器），例如创建一个图像生成工具，然后实例化 `Assistant` 并配置 LLM 服务。",
        "### 智能体创建\n\n3. **创建智能体**：通过 `Assistant` 类实例化，集成工具使用与文件读取能力。",
    ],
)
def test_grounding_wiki_operation_create_and_question_flow_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="Wiki 操作流程不应该被严重事实关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### LLM 层\n\n通过 `BaseChatModel` 基类封装大语言模型接口，提供统一的 `chat` 方法，支持流式输出和函数调用。",
        "### 工具调用\n\n默认模板支持并行工具调用。",
        "### API 兼容\n\nQwen-Agent 支持 OpenAI-compatible API。",
    ],
)
def test_grounding_technical_support_capabilities_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="技术能力说明不应该被支持关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 可选依赖\n\n支持可选依赖，如 GUI（基于 Gradio）、RAG（检索增强生成）、代码解释器、MCP（模型上下文协议）等。",
        "### MCP 集成\n\nMCP 集成：支持模型上下文协议，使用 MCP 工具需要安装 Node.js、uv、Git 等依赖（详见 README）。",
        "### 版本更新\n\n框架持续更新，近期版本包括 Qwen3.5 支持、DeepPlanning 评测基准发布等。",
        "### 智能体创建\n\n`Assistant` 是一个能够使用工具并读取文件的智能体，其创建示例如下（来自源代码步骤 3）：",
        "### 工具循环\n\n工具调用后，结果返回给 Agent，再由 Agent 决定下一步动作。",
        "### 初始化示例\n\n示例中通过 `Assistant(llm, function_list, files)` 创建（见 README 步骤3代码）。",
        "### 模板配置\n\n工具调用支持多种模板，通过 `fncall_prompt_type` 参数配置，默认为 `nous`（Qwen3 推荐）。",
        "### 代码解释器\n\n当智能体决定使用代码解释器时，框架会在本地 Docker 环境中创建一个隔离容器。",
        "### 框架定位\n\nQwen-Agent 是一个基于 Qwen 模型的 Agent 开发框架，提供 LLM、Tool、Agent 等组件，支持自定义工具、代码解释器、MCP 集成，并作为 Qwen Chat 的后端运行。",
        "### 模型服务\n\nQwen-Agent 支持接入阿里云 DashScope 服务提供的 Qwen 模型服务，也支持通过 OpenAI API 方式接入开源的 Qwen 模型服务。",
        "### 工具解析\n\n部署时注意：对于 QwQ 和 Qwen3 模型，建议不开启 vLLM 的 `--enable-auto-tool-choice` 和 `--tool-call-parser hermes`，由 Qwen-Agent 自行解析工具输出。",
        "### DeepPlanning\n\nDeepPlanning 是用于评估 Agent 规划能力的开源基准测试，由 Qwen 团队发布。",
        "### DeepPlanning\n\nDeepPlanning 是一个用于评估大语言模型智能体规划能力的开源基准测试，由 Qwen 团队在 2026 年 1 月发布。",
    ],
)
def test_grounding_qwen_agent_technical_documentation_does_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="Qwen-Agent README 的技术说明不应该被严重事实关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_allows_placeholder_api_key_in_configuration_example() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail=(
            "2. 配置 LLM，例如使用 DashScope："
            "`{'model': 'qwen3-32b', 'model_type': 'qwen_dashscope', 'api_key': '<DASHSCOPE_API_KEY>'}`。"
        ),
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_blocks_real_api_key_literal_in_configuration_example() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail="示例配置里写了 api_key: sk-live-secret-value。",
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "detail",
    [
        "配置：api_key: <DASHSCOPE_API_KEY>; token: sk-live-secret-value。",
        "配置：token: sk-live-secret-value; api_key: <DASHSCOPE_API_KEY>。",
    ],
)
def test_grounding_placeholder_secret_does_not_hide_real_secret(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown, expected_marker",
    [
        ("### 方案归属\n\nKarpathy 提出 llm-wiki 方案。", "提出"),
        ("### 产品归属\n\nOpenAI 创建了一个 Anthropic 竞品。", "创建"),
        ("### 服务关系\n\nOpenAI 支持 Anthropic 服务。", "支持"),
        ("### 产品关系\n\nOpenAI 创建了 Assistant 产品。", "创建"),
    ],
)
def test_grounding_real_create_and_propose_relationships_warn_without_raw_contradiction(
    body_markdown: str,
    expected_marker: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="真实创建/提出关系缺支撑时只提醒。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert review.warnings[0].action == "warn"
    assert expected_marker in review.warnings[0].reason


def test_grounding_security_sandbox_advice_does_not_require_review() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="防御性安全工程描述不应自动 review。",
                body_markdown=(
                    "### 沙箱边界\n\n"
                    "Docker 沙箱提供了一定程度的隔离，但生产环境中可能需要更严格的沙箱方案（如 gVisor、Firecracker）来增强安全性。"
                ),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_quote_uses_sentence_context_without_blocking_high_risk_domain() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown="### 医疗建议\n\n医疗上通常推荐“方案A”。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 安全协作\n\nAI 安全团队可以把评估结果作为产品发布节奏的参考。",
        "### 金融产品表达\n\n金融产品可以用更清晰的解释降低用户理解成本。",
        "### 法务协作\n\n法务团队可以提前参与合同评审流程。",
        "### 医疗表述\n\n这段材料适合讨论阿司匹林风险。",
        "### 金融表述\n\n这段材料适合讨论高收益债券。",
        "### 法律表述\n\n这段材料适用于合同评审案例。",
        "### 安全表述\n\n这个案例适合讨论明文密码风险。",
        "### Security Framing\n\nThis setup is suitable for discussing passwords in plaintext.",
    ],
)
def test_grounding_high_risk_domain_meta_statements_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="普通领域元叙述不应该被高风险建议误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 医疗建议\n\n患者可以服用阿司匹林。",
        "### 医疗建议\n\n患者可以吃阿司匹林。",
        "### 医疗建议\n\n患者可以口服阿司匹林。",
        "### Medical Advice\n\nPatients can use aspirin for chest pain.",
        "### 金融建议\n\n普通用户可以投资高收益债券。",
        "### 金融建议\n\n普通用户可以配置高收益理财产品。",
        "### 法律建议\n\n法律上可以解除这份合同。",
        "### 法律建议\n\n员工可以起诉雇主要求赔偿。",
        "### 法律建议\n\n员工可以签竞业协议。",
        "### Legal Advice\n\nYou can sue your employer without an attorney.",
        "### 安全建议\n\n管理员可以绕过认证后删除日志。",
        "### 安全建议\n\n管理员可以删日志。",
        "### 安全建议\n\n管理员推荐用于删除日志。",
        "### Security Advice\n\nUsers can store passwords in plaintext for convenience.",
        "### Security Advice\n\nThis setup is recommended for storing passwords in plaintext.",
    ],
)
def test_grounding_high_risk_domain_actionable_can_statements_do_not_block_ingest(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="可执行高风险建议需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_render_draft_grounding_review_shows_warning_section() -> None:
    claim = GroundingClaim(
        page_plan_id="PP-WARN",
        target_path="concepts/Concept_Warn.md",
        section_key="detail",
        claim_type="new_fact",
        text="好的产品判断往往来自长期实践中形成的经验直觉",
        support="unsupported",
        action="warn",
        reason="低风险未支撑引号内容仅记录为 warning，不阻塞自动 ingest；如需严谨可人工回看来源。",
    )
    review = DraftGroundingReview(warnings=[claim], claims=[claim], requires_review=False)

    markdown = draft_grounding.render_draft_grounding_review(review)

    assert "- 结果：通过，有非阻塞提醒" in markdown
    assert "- 非阻塞提醒数量：1" in markdown
    assert "## 非阻塞提醒" in markdown
    assert "好的产品判断往往来自长期实践中形成的经验直觉" in markdown


@pytest.mark.parametrize(
    "detail",
    [
        "例如，在 AI 代理的客服场景中，用户反复询问与某个订单状态相关的相似问题时，语义缓存可识别语义相似性。",
        "例如，在 Redis 语义缓存场景中，用户反复询问与某个订单状态相关的相似问题时，语义缓存可复用回答。",
        "例如，Redis 语义缓存可以帮助客服处理用户想修改那个订单的地址的请求。",
        "例如，客服排查用户登录问题时会查询登录状态。",
        "例如，API 场景中用户反复询问某个订单状态。",
        "例如，客服接口场景中用户想修改那个订单的地址。",
        "例如，参数配置场景中客户查询订单信息。",
        "例如，API 场景中用户查看某个订单状态。",
        "例如，接口场景中客户获取订单信息。",
        "例如，参数配置场景中用户搜索订单状态。",
        "例如，客服 API 中客户申请订单退款。",
        "In a Redis semantic cache scenario, a user asks about order status repeatedly.",
        "In an API scenario, a user asks about order status repeatedly.",
        "In an API scenario, a user checks order status.",
        "In a support API scenario, a customer looks up order details.",
        "场景：用户想修改那个订单的地址，代理需要关联长期记忆。",
        "在连续对话场景中，假设用户先询问某个过去的订单信息，随后用户说修改那个订单的地址。",
    ],
)
def test_grounding_detail_unquoted_dynamic_user_scenarios_do_not_block_ingest(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_examples_unquoted_dynamic_user_scenario_does_not_block_ingest() -> None:
    review = build_examples_grounding_review("例如用户反复询问与某个订单状态相关的相似问题。")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_sensitive_dynamic_query_passes_when_source_supported() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail="客服文档原文示例是“忘记密码怎么办”。",
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "忘记密码怎么办")

    assert review.requires_review is False
    assert any(claim.text == "忘记密码怎么办" and claim.support == "raw" for claim in review.claims)


@pytest.mark.parametrize(
    ("detail", "raw"),
    [
        ('客服文档原文示例是"登不上账号"。', "登不上账号"),
        ('客服文档原文示例是"login problem"。', "login problem"),
        ('客服文档原文示例是"sign-in problem"。', "sign-in problem"),
        ("客服文档原文示例是“账号问题”。", "账号问题"),
    ],
)
def test_grounding_sensitive_dynamic_query_short_quote_passes_when_source_supported(detail: str, raw: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail=detail,
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, raw)

    assert review.requires_review is False
    assert any(claim.text == raw and claim.support == "raw" for claim in review.claims)


def test_grounding_unquoted_dynamic_user_scenario_passes_when_source_supported() -> None:
    detail = "例如，在 AI 代理的客服场景中，用户反复询问与某个订单状态相关的相似问题时，语义缓存可识别语义相似性。"
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, detail)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_examples_still_allow_safe_technical_query_template_after_sensitive_guard() -> None:
    review = build_examples_grounding_review("类似“Redis 地址配置方法”的请求")

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["Redis 地址配置方法"]


@pytest.mark.parametrize(
    "detail",
    [
        "技术排障示例可以写成“无法连接 Redis”。",
        "技术排障示例可以写成“不能安装 Redis”。",
        "技术排障示例可以写成“打不开配置文件”。",
        "技术排障示例可以写成“Redis 无法启动”。",
        '技术排障示例可以写成"cache"。',
        '技术排障示例可以写成"Redis config"。',
        '技术排障示例可以写成"Redis problem"。',
        '技术排障示例可以写成"config problem"。',
        '技术排障示例可以写成"cache problem"。',
        '技术排障示例可以写成"service account config"。',
        '技术排障示例可以写成"service account issue"。',
        '技术排障示例可以写成"service account error"。',
        '技术排障示例可以写成"service account problems"。',
        '技术排障示例可以写成"login configuration"。',
        '技术排障示例可以写成"sign-in configuration"。',
        '技术排障示例可以写成"log-in configuration"。',
        "例如，Redis 配置问题可以通过文档排查。",
        "订单状态字段用于排序。",
        "订单状态 schema 示例用于说明字段。",
        "服务会缓存订单状态字段。",
        "订单状态 API 示例用于说明接口。",
        "OAuth 回调接口说明包含订单状态参数。",
        "Redis 数据库字段 order_status 用于缓存订单状态。",
        "用户字段 API 参数说明包含 user_id。",
        "API docs show order status lookup parameters.",
        "payment API parameter describes refund status.",
    ],
)
def test_grounding_detail_allows_safe_technical_troubleshooting_queries(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_unquoted_dynamic_scenario_scanner_ignores_quoted_claims() -> None:
    review = build_examples_grounding_review("例如“查看某个用户的账户余额”这类问题。")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_unquoted_dynamic_scenario_scanner_ignores_open_questions_section() -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    page = draft.pages[0]
    draft = draft.model_copy(
        update={"pages": [page.model_copy(update={"open_questions": "- 待补来源：用户订单状态场景是否适合语义缓存？"})]}
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_examples_query_template_without_local_context_warns() -> None:
    review = build_examples_grounding_review("- “Redis 的安装方法”")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["Redis 的安装方法"]


def test_grounding_examples_query_template_direct_quote_warns_without_support() -> None:
    review = build_examples_grounding_review("原文称：“Redis 的安装方法”。")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["Redis 的安装方法"]


@pytest.mark.parametrize(
    "examples",
    [
        "类似“查询某个用户的记忆片段”的请求",
        "类似“查询某个用户的记忆片段”的请求可以作为模板。",
        "类似“Redis setup guide”的请求",
        "类似“query Redis memory”的请求",
        "类似“query redis memory”的请求",
        "类似“查询Redis记忆”的请求",
        "类似“Redis IP address config”的请求",
        "类似“Redis address configuration”的请求",
        "类似“Redis configuration guide”的请求",
    ],
)
def test_grounding_examples_allow_isolated_query_template_contexts(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False


@pytest.mark.parametrize(
    "examples",
    [
        "类似“Redis 支持集群模式”的问题",
        "类似“用户 1234 删除了凭证”的请求",
        "类似“Alice uses MacBook in 2026”的请求",
        "类似“Alice uses MacBook”的请求",
        "类似“用户使用华为手机”的请求",
        "类似“查询王小明的记忆片段”的请求",
        "类似“查询某个用户的手机号”的请求",
        "类似“查询某个用户的邮箱地址”的请求",
        "类似“查询用户登录记录”的请求",
        "类似“query a user's email address”的请求",
        "类似“查询Alice的记忆片段”的请求",
        "类似“查询Charlie的记忆片段”的请求",
        "类似“查询alice的记忆片段”的请求",
        "类似“查询alice的对话摘要”的请求",
        "类似“查找Alice记忆片段”的请求",
        "类似“查找王小明记忆片段”的请求",
        "类似“query a user's address”的请求",
        "类似“query user's address”的请求",
        "类似“query customer address”的请求",
        "类似“query user IP address”的请求",
        "类似“query a user's birthday”的请求",
        "类似“query a user's name”的请求",
        "类似“query person profile”的请求",
        "类似“query user IP”的请求",
        "类似“query customer IP”的请求",
        "类似“query users profiles”的请求",
        "类似“query users addresses”的请求",
        "类似“search customer profiles”的请求",
        "类似“find user addresses”的请求",
        "类似“query user IP config”的请求",
        "类似“query users IPs”的请求",
        "类似“query users emails”的请求",
        "类似“query customer cookies”的请求",
        "类似“query user IDs”的请求",
        "类似“query customer card”的请求",
        "类似“query user credit card”的请求",
        "类似“query users tokens”的请求",
        "类似“query user sessions”的请求",
        "类似“query user passwords”的请求",
        "类似“query people's addresses”的请求",
        "类似“find people profiles”的请求",
        "类似“search persons addresses”的请求",
        "类似“query people locations”的请求",
        "类似“query users' emails”的请求",
        "类似“search persons' addresses”的请求",
        "类似“Redis config supports cluster”的请求",
        "类似“Redis config improves latency”的请求",
        "类似“Redis configuration is best”的请求",
        "例如“Redis config supports cluster”",
        "比如“Redis config improves latency”",
        "示例“Redis configuration is best”",
        "例如“Redis supports cluster”",
        "例如“Redis 支持集群模式”",
        "比如“Redis 最佳实践”",
        "示例“Redis 配置是最佳方案”",
        "例如“Redis 已经发布新功能”",
        "例如“Redis 推出企业版”",
        "例如“Redis 配置推荐用于生产环境”",
        "比如“Redis 配置导致错误”",
        "类似“query user SSN”的请求",
        "类似“query users SSNs”的请求",
        "类似“query user social security number”的请求",
        "类似“query customer passport number”的请求",
        "类似“query customer license number”的请求",
        "类似“query user api key”的请求",
        "类似“query user API key”的请求",
        "类似“query user api keys”的请求",
        "类似“query users API keys”的请求",
        "类似“query user secrets”的请求",
        "类似“query customer passwds”的请求",
        "例如“query users API keys”",
    ],
)
def test_grounding_examples_query_template_does_not_block_without_raw_contradiction(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "examples",
    [
        "- “权限配置方法”",
        "例如“权限配置方法”",
        "例如“payment setup guide”",
        "类似“payment setup guide”的请求",
    ],
)
def test_grounding_examples_sensitive_quotes_do_not_block_ingest(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "examples",
    [
        "问句“查询某个用户的手机号”用于评估。",
        "记忆评估问题“用户登录记录是什么？”",
        "记忆评估问题“王小明的手机号是多少？”",
    ],
)
def test_grounding_examples_sensitive_memory_eval_quotes_do_not_block_ingest(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "examples",
    [
        "- “该用户喜欢蓝色”",
        "- “Alice 在 2026 年 3 月购买了 MacBook。”",
        "- “Build number 1234 completed with status success”",
        "- 原文称：“某个用户曾在某家店消费过”",
        "- “Alice uses MacBook”",
        "- “Alice likes coffee”",
        "- “MacBook syncs memory”",
        "- “某个用户在星巴克消费过”",
        "- “某个用户购买了华为手机”",
        "- “某个用户在北京门店消费过”",
        "- “某个用户购买了小米手机”",
        "- “某个用户在南京门店消费过”",
        "- “某个用户喜欢黄色”",
        "- “某个用户购买了OPPO手机”",
        "- “某个用户在成都门店消费过”",
        "- “某个用户喜欢紫色”",
        '- `recall("张三的工单 1234")`',
        '- `search_context("Alice order 1234")`',
        '- `recall("用户喜欢蓝色")`',
        '- `recall("某个用户在北京门店消费过")`',
        '- `recall("用户最近的订单状态")`',
        '- `recall("查询某个用户的手机号")`',
        '- `mem0 search "用户最近的工单信息" --user-id user123`',
    ],
)
def test_grounding_examples_placeholder_bypass_warns_for_concrete_or_attributed_quotes(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert review.warnings or review.claims


def test_grounding_examples_hard_facts_warn_without_support() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FACT-EXAMPLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fact_Example.md",
        display_title="事实例子",
        page_type="concept",
        new_understanding="事实例子需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FACT-EXAMPLE",
                action="create",
                canonical_target_path="concepts/Concept_Fact_Example.md",
                summary="事实例子。",
                body_markdown=draft_body(detail="这个页面说明事实型例子需要来源。", examples="- “销量增长三倍”"),
                change_summary="创建事实例子页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Fact_Example.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["销量增长三倍"]


def test_grounding_examples_cli_argument_literals_warn_without_repair() -> None:
    review = build_examples_grounding_review('- `mem0 add --user-id user123 --text "用户喜欢科技类文章"`')

    assert review.requires_review is False
    assert review.warnings


def test_grounding_detail_illustrative_examples_do_not_require_raw_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-STYLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Style.md",
        display_title="写作风格",
        page_type="concept",
        new_understanding="写作风格描述 AI 上下文文件中的表达方式。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-STYLE",
                action="create",
                canonical_target_path="concepts/Concept_Style.md",
                summary="写作风格示例。",
                body_markdown=draft_body(detail="解释性风格提供理由，如“因为性能原因，使用列表推导”；条件性风格指定条件，如“如果代码量超过 100 行，请拆分”。", examples="暂无相关例子记录。"),
                change_summary="创建写作风格页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Style.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["因为性能原因，使用列表推导", "如果代码量超过 100 行，请拆分"]


def test_grounding_memory_example_questions_do_not_require_raw_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-MEMORY",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_记忆评估示例.md",
        display_title="记忆评估示例",
        page_type="concept",
        new_understanding="记忆评估常用短问句和偏好样例解释不同记忆层级。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding memory examples。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-MEMORY",
                action="create",
                canonical_target_path="concepts/Concept_记忆评估示例.md",
                summary="MemBench 用短样例解释不同记忆任务。",
                body_markdown=draft_body(detail="事实记忆的问题示例包括“用户哥哥的名字是什么？”，反思记忆示例包括“用户喜欢重口味”。", examples="暂无相关例子记录。"),
                change_summary="创建记忆评估示例页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_记忆评估示例.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["用户哥哥的名字是什么？", "用户喜欢重口味"]
    assert {claim.reason for claim in review.claims} == {"记忆评估中的短问句/用户偏好/对话样例按 illustrative example 处理，不要求 raw exact match。"}


def test_grounding_detail_memory_examples_do_not_require_raw_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-MEM-DETAIL",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Memory_Detail.md",
        display_title="事实记忆",
        page_type="concept",
        new_understanding="事实记忆包含不同评估子任务。",
        section_plans={"detail": "详情"},
        reason="测试 detail memory examples。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-MEM-DETAIL",
                action="create",
                canonical_target_path="concepts/Concept_Memory_Detail.md",
                summary="摘要。",
                body_markdown=draft_body(detail="事实记忆的子任务包括单跳（如“用户表哥的名字？”）和知识更新"
                        "（如“用户修改了年龄后，现在多大？”）。在参与场景中，例如，用户说"
                        "“我的表哥Ethan身高162cm”，智能体回应“明白了，Ethan身高162厘米”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Memory_Detail.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [
        "用户表哥的名字？",
        "用户修改了年龄后，现在多大？",
        "我的表哥Ethan身高162cm",
        "明白了，Ethan身高162厘米",
    ]
    assert {claim.reason for claim in review.claims} == {"记忆评估中的短问句/用户偏好/对话样例按 illustrative example 处理，不要求 raw exact match。"}


def test_grounding_short_concept_phrases_do_not_require_raw_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-SCALING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Scaling.md",
        display_title="Scaling Managed Agents",
        page_type="concept",
        new_understanding="Scaling 讨论管理型 Agent 的协作边界。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding 短语。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-SCALING",
                action="create",
                canonical_target_path="concepts/Concept_Scaling.md",
                summary="页面围绕“宠物 vs 牛”和“解耦大脑与双手”两个概念展开。",
                body_markdown=draft_body(detail="还保留“会话作为持久上下文对象”这个标题式表达。", examples="暂无相关例子记录。"),
                change_summary="创建 Scaling 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Scaling.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["宠物 vs 牛", "解耦大脑与双手", "会话作为持久上下文对象"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_short_concept_phrases_in_body_markdown_do_not_require_raw_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-SCALING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Scaling.md",
        display_title="Scaling Managed Agents",
        page_type="concept",
        new_understanding="Scaling 讨论管理型 Agent 的协作边界。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding 短语。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-SCALING",
                action="create",
                canonical_target_path="concepts/Concept_Scaling.md",
                summary="页面围绕“宠物 vs 牛”和“解耦大脑与双手”两个概念展开。",
                body_markdown="### 概念框架\n\n还保留“会话作为持久上下文对象”这个标题式表达。",
                change_summary="创建 Scaling 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Scaling.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["宠物 vs 牛", "解耦大脑与双手", "会话作为持久上下文对象"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_external_backing_claim_uses_trigger_sentence() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-EVAL",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Eval.md",
        display_title="评估（Eval）",
        page_type="concept",
        new_understanding="Eval 在产品开发中用于判断功能风险。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 外部背书。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EVAL",
                action="create",
                canonical_target_path="concepts/Concept_Eval.md",
                summary="摘要。",
                body_markdown=draft_body(detail="在Anthropic，评估被广泛使用于产品开发。Cat Wu指出，评估的重要性因功能而异。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Eval.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "Cat Wu指出，评估的重要性因功能而异。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["在Anthropic，评估被广泛使用于产品开发。"]
    assert review.warnings[0].action == "warn"
    assert "被广泛使用" in review.warnings[0].reason
    assert "非阻塞提醒" in review.warnings[0].reason


def test_grounding_external_backing_issue_message_warns_with_source_local_guidance() -> None:
    claim = GroundingClaim(
        page_plan_id="PP-REDIS",
        target_path="concepts/Concept_Redis.md",
        section_key="detail",
        claim_type="new_fact",
        text="Redis最初作为高性能缓存、分析和消息代理广泛使用。",
        support="unsupported",
        action="warn",
        reason="新增外部背书/强事实标记 `广泛使用` 未在 raw 或 inspected wiki 中出现；作为非阻塞提醒保留，必要时可改写为 source-local 表达。",
    )

    message = draft_grounding.grounding_issue_message(claim)

    assert "非阻塞提醒" in message
    assert "adoption/authority 表达最好有来源意识" in message
    assert "source-local 表达" in message
    assert "触发文本：Redis最初作为高性能缓存、分析和消息代理广泛使用。" in message


def test_grounding_external_backing_detects_adoption_and_best_practice_real_path() -> None:
    concept_item = WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 可以用作缓存。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试广泛采用 marker。",
    )
    question_item = WikiMergePlanItem(
        page_plan_id="PP-SEARCH",
        source_basis=SourceBasis(source_candidate_ids=["CAND002"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_混合搜索策略.md",
        display_title="混合搜索策略",
        page_type="open_question",
        new_understanding="混合搜索策略仍需确认。",
        section_plans={"open_questions": "记录待补来源问题。"},
        reason="测试最佳实践 marker。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail="Redis 被广泛采用作为缓存和消息代理。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            ),
            DraftPageItem(
                page_plan_id="PP-SEARCH",
                action="create",
                canonical_target_path="open_questions/Open_Question_混合搜索策略.md",
                summary="摘要。",
                body_markdown=draft_body(detail="整理仍需确认的策略问题。"),
                open_questions="- 目前是否存在最佳实践？",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            ),
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            ),
            WikiContextEntry(
                path="wiki/open_questions/Open_Question_混合搜索策略.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            ),
        ],
    )
    raw = "Redis 可以作为缓存和消息代理使用。材料讨论了混合搜索与向量搜索的融合策略需要继续验证。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[concept_item, question_item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert {claim.action for claim in review.warnings} == {"warn"}
    reasons_by_text = {claim.text: claim.reason for claim in review.warnings}
    assert "Redis 被广泛采用作为缓存和消息代理。" in reasons_by_text
    assert "目前是否存在最佳实践？" in reasons_by_text
    assert "被广泛采用" in reasons_by_text["Redis 被广泛采用作为缓存和消息代理。"]
    assert "最佳实践" in reasons_by_text["目前是否存在最佳实践？"]


@pytest.mark.parametrize(
    "examples",
    [
        "例子写成“Redis 被广泛采用”。",
        "类似“是否存在广泛采用的方案？”的问题",
        "类似“是否存在公认方案？”的问题",
    ],
)
def test_grounding_examples_external_backing_quotes_do_not_bypass_as_inference(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert [claim.action for claim in review.warnings] == ["warn"]


def test_grounding_external_backing_quote_only_detail_does_not_bypass_as_concept_label() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 作为缓存能力被讨论。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试引号内外部背书 marker。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail="主题写成“Redis 被广泛采用”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["Redis 被广泛采用"]
    assert review.warnings[0].action == "warn"


def test_grounding_external_backing_supported_quote_does_not_hide_later_unsupported_marker() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 作为缓存能力被讨论。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试 supported quote 后的额外 marker。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail="材料写到“Redis 被广泛采用作为缓存”，因此 MongoDB 被广泛采用。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "Redis 被广泛采用作为缓存。",
    )

    assert review.requires_review is False
    assert any("MongoDB 被广泛采用" in claim.text for claim in review.warnings)
    assert any("被广泛采用" in claim.reason for claim in review.warnings)


@pytest.mark.parametrize(
    ("outside_claim", "expected_marker"),
    [
        ("因此 MongoDB 广泛采用。", "广泛采用"),
        ("因此 MongoDB 公认可靠。", "公认"),
        ("这说明 Redis 是行业最佳。", "行业最佳"),
        ("这说明 MongoDB 有最佳实践明确支持。", "最佳实践"),
    ],
)
def test_grounding_external_backing_supported_quote_does_not_hide_different_later_marker(
    outside_claim: str,
    expected_marker: str,
) -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 作为缓存能力被讨论。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试 supported quote 后的不同 marker。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"材料写到“Redis 被广泛采用作为缓存”，{outside_claim}", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "Redis 被广泛采用作为缓存。",
    )

    assert review.requires_review is False
    assert any(outside_claim.rstrip("。") in claim.text for claim in review.warnings)
    assert any(expected_marker in claim.reason for claim in review.warnings)


def test_grounding_external_backing_does_not_flag_internal_multiple_components() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-MANY-HANDS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Many_Hands.md",
        display_title="多脑多手扩展",
        page_type="concept",
        new_understanding="多脑多手扩展描述大脑和沙箱的组合方式。",
        section_plans={"detail": "详情"},
        reason="测试 `被多个` 不误伤内部组件关系。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-MANY-HANDS",
                action="create",
                canonical_target_path="concepts/Concept_Many_Hands.md",
                summary="摘要。",
                body_markdown=draft_body(detail="一个沙箱可以被多个适配框架共享以保持状态一致性。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Many_Hands.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_still_flags_multiple_community_claim() -> None:
    assert draft_grounding.unsupported_backing_marker("该方案被多个社区引用。") == "被多个"


def test_grounding_flags_unsupported_scope_speculation() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-COWORK",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_Cowork.md",
        display_title="Cowork",
        page_type="entity",
        new_understanding="Cowork 是知识工作产品。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 范围推测。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-COWORK",
                action="create",
                canonical_target_path="entities/Entity_Cowork.md",
                summary="Cowork 是知识工作产品。",
                body_markdown=draft_body(detail="Cowork 用于综合信息和创建文档。", additional_notes="源代码泄露事件中，Cowork 的组件可能也受到影响，但访谈中未详细说明。"),
                change_summary="创建 Cowork 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Cowork.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "Claude Code 的源代码泄露被归因于人为错误。Cowork 是另一款知识工作产品。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [
        "源代码泄露事件中，Cowork 的组件可能也受到影响，但访谈中未详细说明。"
    ]
    assert review.warnings[0].action == "warn"
    assert "受影响对象推测" in review.warnings[0].reason


def test_grounding_scope_speculation_allows_open_question() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-QUESTION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_发布一致性.md",
        display_title="发布一致性",
        page_type="open_question",
        new_understanding="快速发布有一致性问题。",
        section_plans={"open_questions": "未决问题"},
        reason="测试 grounding 未决问题。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-QUESTION",
                action="create",
                canonical_target_path="open_questions/Open_Question_发布一致性.md",
                summary="快速发布和产品一致性之间存在张力。",
                body_markdown=draft_body(detail="访谈提到团队追求快速发布。"),
                open_questions="快速发布是否可能影响长期产品一致性？",
                change_summary="创建未决问题页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/open_questions/Open_Question_发布一致性.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "团队追求快速发布。",
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_accepts_english_widely_used_anchor() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-NYU-CTF",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_NYU CTF Bench.md",
        display_title="NYU CTF Bench",
        page_type="entity",
        new_understanding="NYU CTF Bench 是静态 CTF benchmark。",
        section_plans={"detail": "详情"},
        reason="测试英文论文 backing marker。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-NYU-CTF",
                action="create",
                canonical_target_path="entities/Entity_NYU CTF Bench.md",
                summary="摘要。",
                body_markdown=draft_body(detail="NYU CTF Bench 被广泛用于评估 LLM 智能体在网络安全任务中的表现。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_NYU CTF Bench.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "To evaluate these agents, CTF benchmarks have become the de-facto standard. "
        "These benchmarks have also been widely used in evaluating recent LLM models."
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_accepts_widely_across_tasks_anchor() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        matched_page="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 Managed Agents 生态中的 harness。",
        section_plans={"detail": "详情"},
        reason="测试英文 widely across tasks 支撑。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 是 Anthropic 开发的 harness，在团队内部被广泛使用。",
                body_markdown=draft_body(detail="Claude Code 作为 Managed Agents 的一个 harness 示例，被广泛用于多种任务。"),
                change_summary="更新页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content="",
            )
        ],
    )
    raw = "For example, Claude Code is an excellent harness that we use widely across tasks."

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_accepts_retained_existing_fact_with_bridge_prefix() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        matched_page="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Cat Wu 访谈补充 Claude Code 产品管理细节。",
        section_plans={"detail": "详情"},
        reason="测试 update preservation 旧事实桥接前缀。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 是 Anthropic 开发的一款编程助手产品。",
                body_markdown=draft_body(detail="从 Managed Agents / 托管智能体 等旧页视角看，本材料将 Claude Code 描述为“出色的 harness”，在各种任务中广泛使用。"),
                change_summary="更新页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content="# Claude Code\n\n## 详细说明\n\n本材料将 Claude Code 描述为“出色的 harness”，在各种任务中广泛使用。\n",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "Cat Wu 访谈讨论 Claude Code 产品团队。",
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert any(claim.claim_type == "retained_fact" and claim.support == "existing_wiki" for claim in review.claims)


def test_grounding_external_backing_uses_same_line_pronoun_context() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-NYU-PRONOUN",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_NYU CTF Bench.md",
        display_title="NYU CTF Bench",
        page_type="entity",
        new_understanding="NYU CTF Bench 是静态 CTF benchmark。",
        section_plans={"summary": "摘要"},
        reason="测试英文论文 backing marker 的代词上下文。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-NYU-PRONOUN",
                action="create",
                canonical_target_path="entities/Entity_NYU CTF Bench.md",
                summary="NYU CTF Bench 是用于评估 LLM 智能体的 CTF 基准。它被广泛使用，但存在数据污染风险。",
                body_markdown=draft_body(detail="静态 CTF 基准可能高估模型表现，实时 CTF 可以降低公开题解带来的污染。", examples="例如，公开 write-up 会影响静态题库。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_NYU CTF Bench.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "NYU CTF Bench was the first benchmark to use CTF problems for evaluating cybersecurity agents. "
        "These benchmarks have also been widely used in evaluating recent LLM models."
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_requires_specific_anchor_not_only_generic_widely_used() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FAKE-BENCH",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_FooBench.md",
        display_title="FooBench",
        page_type="entity",
        new_understanding="FooBench 是一个评估基准。",
        section_plans={"summary": "摘要"},
        reason="测试英文 backing 不能只凭泛词放行。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FAKE-BENCH",
                action="create",
                canonical_target_path="entities/Entity_FooBench.md",
                summary="FooBench 是用于评估 LLM 智能体的基准。它被广泛使用，但存在数据污染风险。",
                body_markdown=draft_body(detail="静态基准可能高估模型表现。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_FooBench.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "These benchmarks have also been widely used in evaluating recent LLM models."
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["它被广泛使用，但存在数据污染风险。"]
    assert review.warnings[0].action == "warn"


def test_grounding_quoted_conceptual_release_process_is_not_direct_quote() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-RESEARCH-PREVIEW",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="designs/Design_Research_Preview.md",
        display_title="研究预览发布模式",
        page_type="design",
        new_understanding="研究预览是一种发布模式。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 发布流程概念短语。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-RESEARCH-PREVIEW",
                action="create",
                canonical_target_path="designs/Design_Research_Preview.md",
                summary="摘要。",
                body_markdown=draft_body(detail="该模式与“可重复发布流程”和“设定清晰目标”形成配套。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/designs/Design_Research_Preview.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["可重复发布流程", "设定清晰目标"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_quoted_product_choice_label_context_is_not_direct_quote() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-PRODUCT-CHOICE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Claude_Code_Cowork_Choice.md",
        display_title="Claude Code 与 Cowork 的产品选择",
        page_type="concept",
        new_understanding="产品选择标签不应被当成直接引用。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 产品选择标签。",
    )
    quote = "何时使用Claude Code与Cowork"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-PRODUCT-CHOICE",
                action="create",
                canonical_target_path="concepts/Concept_Claude_Code_Cowork_Choice.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本概念源自访谈中关于“{quote}”的讨论。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Claude_Code_Cowork_Choice.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_concept_label_after_broad_mention_is_not_direct_quote() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-HARNESS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 harness 示例。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 宽泛提到短概念。",
    )
    quote = "优秀的适配框架"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-HARNESS",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"文中提到它是“{quote}”，展示了元适配框架可以容纳不同类型的 harness。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_attributed_concept_label_is_not_dequoted_or_bypassed() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-ATTRIBUTED-LABEL",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Context_Object.md",
        display_title="会话上下文对象",
        page_type="concept",
        new_understanding="会话可被理解成上下文对象。",
        section_plans={"detail": "详情"},
        reason="测试 attributed concept label。",
    )
    quote = "会话作为持久上下文对象"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-ATTRIBUTED-LABEL",
                action="create",
                canonical_target_path="concepts/Concept_Context_Object.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"文中称“{quote}”，因此该页面保留这个概念。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Context_Object.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is False
    assert f"“{quote}”" in rewritten.pages[0].body_markdown
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_attributed_concept_label_with_punctuation_is_not_bypassed() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-ATTRIBUTED-PUNCTUATION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Context_Object.md",
        display_title="会话上下文对象",
        page_type="concept",
        new_understanding="会话可被理解成上下文对象。",
        section_plans={"detail": "详情"},
        reason="测试 attributed punctuation。",
    )
    quote = "会话作为持久上下文对象"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-ATTRIBUTED-PUNCTUATION",
                action="create",
                canonical_target_path="concepts/Concept_Context_Object.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"文中称：“{quote}”，因此该页面保留这个概念。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Context_Object.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is False
    assert f"“{quote}”" in rewritten.pages[0].body_markdown
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_quoted_abstract_trend_label_context_is_not_direct_quote() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-PM-SKILLS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_AI_PM_Skills.md",
        display_title="AI PM 技能变化",
        page_type="open_question",
        new_understanding="抽象趋势标签不应被当成直接引用。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 抽象趋势标签。",
    )
    quote = "技术壁垒正在降低"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-PM-SKILLS",
                action="create",
                canonical_target_path="open_questions/Open_Question_AI_PM_Skills.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本问题源自她提到的“{quote}”的趋势。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/open_questions/Open_Question_AI_PM_Skills.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_explicit_direct_quote_mismatch_warns_without_review() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-QUOTE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Quote.md",
        display_title="直接引用",
        page_type="concept",
        new_understanding="直接引用需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-QUOTE",
                action="create",
                canonical_target_path="concepts/Concept_Quote.md",
                summary="原文说“解耦大脑与双手”。",
                body_markdown=draft_body(detail="暂无更多细节。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Quote.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert review.warnings[0].text == "解耦大脑与双手"
    assert "直接引用/作者归因" in review.warnings[0].reason


def test_grounding_direct_quote_accepts_normalized_source_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-PAPER",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="designs/Design_Paper.md",
        display_title="论文方法",
        page_type="design",
        new_understanding="论文方法句需要来源支撑。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 规范化 exact match。",
    )
    quote = "Based on MemEngine (Zhang et al., 2025), we implement seven memory mechanisms, using Qwen2.5-7B as the base model"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-PAPER",
                action="create",
                canonical_target_path="designs/Design_Paper.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"源摘录中提到“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/designs/Design_Paper.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "To eliminate other designs on results, we make no modifications to components."
        "Based on MemEngine (Zhang et al., 2025 ), we implement seven memory mechanisms, "
        "using Qwen2.5-7B as the base model for the agent applications on our benchmark."
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].support == "raw"
    assert review.claims[0].reason == "直接引用已在 raw 或已有 wiki 中规范化 exact match。"


def test_grounding_direct_quote_accepts_time_range_transcript_variant() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-PM-ROLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_PM角色演变.md",
        display_title="PM角色演变",
        page_type="concept",
        new_understanding="PM负责从当前状态到长期愿景之间的路径。",
        section_plans={"detail": "详情"},
        reason="测试 transcript 数字范围近似直引。",
    )
    quote = "弄清楚从今天到3-6个月后愿景之间的路径"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-PM-ROLE",
                action="create",
                canonical_target_path="concepts/Concept_PM角色演变.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat Wu提到，PM的工作是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_PM角色演变.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "Boris 非常擅长设定方向，比如这就是产品在3个月、6个月后需要成为的样子。"
        "而我的很多职责是弄清楚从今天到那个3到6个月后的愿景之间的路径是什么。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].support == "raw"
    assert review.claims[0].reason == "直接引用已在 raw 或已有 wiki 中规范化 exact match。"


def test_grounding_direct_quote_accepts_paired_month_enumeration_as_range() -> None:
    quote = "产品在3-6个月后需要成为的样子"
    raw = "Boris 非常擅长设定方向，比如这就是产品在3个月、6个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is True


def test_grounding_direct_quote_paired_month_enumeration_requires_same_numbers() -> None:
    quote = "产品在3-9个月后需要成为的样子"
    raw = "Boris 非常擅长设定方向，比如这就是产品在3个月、6个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is False


def test_grounding_direct_quote_does_not_collapse_three_item_timeline() -> None:
    quote = "产品在3-6个月后需要成为的样子"
    raw = "路线图分别记录产品在3个月、6个月、9个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is False


def test_grounding_direct_quote_range_does_not_match_partial_numeric_token() -> None:
    quote = "产品在3-6个月后需要成为的样子"
    raw = "Boris 讨论的是产品在13个月、6个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is False


def test_grounding_direct_quote_time_range_variant_requires_same_numbers() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-PM-ROLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_PM角色演变.md",
        display_title="PM角色演变",
        page_type="concept",
        new_understanding="PM负责从当前状态到长期愿景之间的路径。",
        section_plans={"detail": "详情"},
        reason="测试 transcript 数字范围不能误配。",
    )
    quote = "弄清楚从今天到3-9个月后愿景之间的路径"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-PM-ROLE",
                action="create",
                canonical_target_path="concepts/Concept_PM角色演变.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat Wu提到，PM的工作是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_PM角色演变.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "我的职责是弄清楚从今天到那个3到6个月后的愿景之间的路径是什么。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_short_domain_quote_accepts_normalized_source_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE",
        source_basis=SourceBasis(source_candidate_ids=["auto-ent-claudecode"]),
        action="create",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 Managed Agents 生态中的 harness。",
        section_plans={"detail": "详情"},
        reason="测试短 domain quote 的规范化 exact match。",
    )
    quote = "出色的 harness"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"源材料在正文中提到，Claude Code 已经作为“{quote}”被集成到 Managed Agents 架构中。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "例如，**Claude Code** 是一个出色的 **harness（适配框架）**，我们在各种任务中广泛使用它。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].text == quote
    assert review.claims[0].support == "raw"


def test_grounding_domain_quote_accepts_source_match_with_parenthetical_translation() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE-LONG",
        source_basis=SourceBasis(source_candidate_ids=["auto-ent-claudecode"]),
        action="create",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 Managed Agents 生态中的 harness。",
        section_plans={"detail": "详情"},
        reason="测试 domain quote 可省略英文术语后的中文括注。",
    )
    quote = "一个出色的 harness，我们在各种任务中广泛使用它"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE-LONG",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"原文提到“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "例如，**Claude Code** 是一个出色的 **harness（适配框架）**，我们在各种任务中广泛使用它。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].text == quote
    assert review.claims[0].support == "raw"


def test_grounding_short_numeric_quote_accepts_exact_numeric_source_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-BORIS",
        source_basis=SourceBasis(source_candidate_ids=["E002"]),
        action="create",
        canonical_target_path="entities/Entity_Boris Cherny.md",
        display_title="Boris Cherny",
        page_type="entity",
        new_understanding="Boris 与 Cat Wu 的协作模式。",
        section_plans={"detail": "详情"},
        reason="测试短数字 quote 的规范化 exact match。",
    )
    quote = "80% 是心灵融合"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-BORIS",
                action="create",
                canonical_target_path="entities/Entity_Boris Cherny.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat 形容他们的合作“{quote}”，剩余 20% 由各自在意的事情驱动。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Boris Cherny.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = '我觉得我们大概80%是心灵融合，然后有20%的事情我更在意，我就多推动那些。'

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].text == quote
    assert review.claims[0].support == "raw"


def test_grounding_short_numeric_quote_does_not_match_decimal_collapse() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-BORIS",
        source_basis=SourceBasis(source_candidate_ids=["E002"]),
        action="create",
        canonical_target_path="entities/Entity_Boris Cherny.md",
        display_title="Boris Cherny",
        page_type="entity",
        new_understanding="Boris 与 Cat Wu 的协作模式。",
        section_plans={"detail": "详情"},
        reason="测试短数字 quote 不把小数错配成整数百分比。",
    )
    quote = "9.5% 是心灵融合"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-BORIS",
                action="create",
                canonical_target_path="entities/Entity_Boris Cherny.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat 形容他们的合作“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/entities/Entity_Boris Cherny.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "我觉得我们大概95%是心灵融合。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_ascii_closing_quote_is_not_treated_as_new_quote_start() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-PETS-CATTLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Pets_Cattle.md",
        display_title="Pets vs Cattle",
        page_type="concept",
        new_understanding="容器失败应像 cattle 一样被自动替换。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 半角引号边界。",
    )
    quote = (
        "If the container died, the harness caught the failure as a tool-call error "
        "and passed it back to Claude."
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-PETS-CATTLE",
                action="create",
                canonical_target_path="concepts/Concept_Pets_Cattle.md",
                summary="摘要。",
                body_markdown=draft_body(detail='解耦后，container 变成"牲畜"——如果它死了，harness 将失败捕获为工具调用错误，'
                        f'传回 Claude。原文描述："{quote}"', examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Pets_Cattle.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        quote,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].support == "raw"


def test_grounding_quoted_evaluation_question_template_is_not_direct_quote() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-EVAL-QUESTION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Eval.md",
        display_title="概率性 AI 产品评估",
        page_type="open_question",
        new_understanding="评估问题模板不是直接事实引用。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 评估问句模板。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-EVAL-QUESTION",
                action="create",
                canonical_target_path="open_questions/Open_Question_Eval.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes="不是“是否回答正确”，而是“在多少比例下用户满意”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/open_questions/Open_Question_Eval.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["是否回答正确", "在多少比例下用户满意"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_quoted_compact_paraphrase_uses_nearby_source_support() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FAST-ITERATION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Iteration.md",
        display_title="快速迭代流程",
        page_type="concept",
        new_understanding="清晰目标帮助团队快速迭代。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 压缩概括。",
    )
    quote = "核心用户是专业开发者，主要问题是权限提示疲劳"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FAST-ITERATION",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Iteration.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"设定清晰目标（如“{quote}”）可以减少 LLM 通用性带来的模糊。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Iteration.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "所以我认为一个优秀的PM能够说：好的，我们的核心用户是专业开发者。"
        "我们这个功能要解决的主要问题可能是权限提示太多了，人们感到疲劳。"
        "我们的用例是：我们希望企业里的专业开发者能够安全地实现零权限提示。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"
    assert review.claims[0].support == "raw"
    assert "压缩概括" in review.claims[0].reason


def test_grounding_quoted_method_goal_paraphrase_uses_nearby_source_support() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FAST-SHIPPING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Shipping.md",
        display_title="快速交付方法",
        page_type="concept",
        new_understanding="团队用目标短语总结快速交付方法。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 方法目标短语。",
    )
    quote = "找到最快将功能交到用户手中的方法"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FAST-SHIPPING",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Shipping.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"PM 关注的是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Shipping.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "我们怎样才能找到最快把东西推出去的方法？"
        "我们怎样才能创建一个产品套件的概念角落，让工程师或 PM 有一个想法，"
        "到周末就能把功能交到用户手中。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"
    assert review.claims[0].support == "raw"
    assert "压缩概括" in review.claims[0].reason


def test_grounding_quoted_method_goal_paraphrase_warns_without_nearby_support() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FAST-SHIPPING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Shipping.md",
        display_title="快速交付方法",
        page_type="concept",
        new_understanding="团队用目标短语总结快速交付方法。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 方法目标短语。",
    )
    quote = "找到最快将功能交到用户手中的方法"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FAST-SHIPPING",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Shipping.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"PM 关注的是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Shipping.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "我们怎样才能找到最快把东西推出去的方法？这里没有说明最终交付给谁。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_numeric_reliability_paraphrase_rewrites_to_source_sentence() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-AUTOMATION-RELIABILITY",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI自动化可靠性.md",
        display_title="AI自动化可靠性",
        page_type="concept",
        new_understanding="100%可靠性原则来自访谈。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 百分比 paraphrase 改写。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-AUTOMATION-RELIABILITY",
                action="create",
                canonical_target_path="concepts/Concept_AI自动化可靠性.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes="100%可靠性原则也适用于AI产品自身的质量，正如Cat Wu所说“95%对AI来说就是失败”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    raw = "如果自动化不是100%有效，它真的不是自动化。95%的自动化真的没什么价值。"
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_AI自动化可靠性.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, raw)
    body = rewritten.pages[0].body_markdown
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert report["changed"] is True
    assert report["rewrite_count"] == 1
    assert "95%对AI来说就是失败" not in body
    assert "正如Cat Wu所说，95%的自动化真的没什么价值。" in body
    assert review.requires_review is False


def test_grounding_numeric_reliability_paraphrase_warns_without_source_sentence() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-AUTOMATION-RELIABILITY",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI自动化可靠性.md",
        display_title="AI自动化可靠性",
        page_type="concept",
        new_understanding="100%可靠性原则来自访谈。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 百分比 paraphrase 改写。",
    )
    quote = "95%对AI来说就是失败"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-AUTOMATION-RELIABILITY",
                action="create",
                canonical_target_path="concepts/Concept_AI自动化可靠性.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"正如Cat Wu所说“{quote}”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_AI自动化可靠性.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "AI工具需要继续提升可靠性。")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "AI工具需要继续提升可靠性。",
    )

    assert report["changed"] is False
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_rewrite_translates_known_english_harness_quote() -> None:
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail="Claude Code 被描述为“an excellent harness that provides a focused coding experience”，可接入 Managed Agents。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    body = rewritten.pages[0].body_markdown

    assert report["changed"] is True
    assert "an excellent harness" not in body
    assert "被描述为一种优秀的 harness，提供聚焦的编码体验" in body


def test_grounding_rewrite_dequotes_internal_digest_paraphrase() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-SECURITY",
        source_basis=SourceBasis(source_candidate_ids=["D001"]),
        action="create",
        canonical_target_path="designs/Design_Security.md",
        display_title="安全令牌隔离",
        page_type="design",
        new_understanding="安全令牌隔离减少凭证暴露。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试内部 artifact paraphrase 去引号。",
    )
    quote = "在耦合架构中，sandbox 与凭证共存，攻击者可通过提示注入窃取令牌；此设计从结构上消除了该风险。"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-SECURITY",
                action="create",
                canonical_target_path="designs/Design_Security.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"该设计对应 approved_digest 中 D001 的 why_matters 描述：“{quote}”"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/designs/Design_Security.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    body = rewritten.pages[0].body_markdown
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert "approved_digest" not in body
    assert f"“{quote}”" not in body
    assert "对应的来源要点是：" in body
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_non_explicit_scope_paraphrase() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-CONTEXT",
        source_basis=SourceBasis(source_candidate_ids=["Q001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Context.md",
        display_title="长时上下文管理",
        page_type="open_question",
        new_understanding="长时上下文管理仍有开放问题。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 scope paraphrase 去引号。",
    )
    quote = "文章仅提出会话作为持久化存储，但未深入讨论智能压缩、索引或预取"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-CONTEXT",
                action="create",
                canonical_target_path="open_questions/Open_Question_Context.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本材料中提到“{quote}”，这正是该问题的来源。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/open_questions/Open_Question_Context.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_long_non_explicit_paraphrase_with_fact_markers() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-MANY-HANDS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Many_Hands.md",
        display_title="多大脑多手",
        page_type="concept",
        new_understanding="多大脑多手来自大脑与双手解耦。",
        section_plans={"detail": "详情"},
        reason="测试长 paraphrase 去引号。",
    )
    quote = (
        "将大脑与双手解耦解决了我们最早的客户投诉之一。当团队希望 Claude 使用他们自己 VPC 中的资源时，"
        "唯一的路径是将他们的网络与我们的做对等互连，因为持有 harness 的容器假定每个资源都在它旁边。"
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-MANY-HANDS",
                action="create",
                canonical_target_path="concepts/Concept_Many_Hands.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"材料指出：“{quote}”解耦后，资源可以位于任何位置。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Many_Hands.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_open_question_quote_with_growth_marker() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-LOG-GROWTH",
        source_basis=SourceBasis(source_candidate_ids=["Q001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Log_Growth.md",
        display_title="会话日志增长管理",
        page_type="open_question",
        new_understanding="会话日志增长管理仍待设计。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试开放问题问句去引号。",
    )
    quote = "日志大小增长如何管理？是否需要引入日志压缩或归档策略？"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-LOG-GROWTH",
                action="create",
                canonical_target_path="open_questions/Open_Question_Log_Growth.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"该开放问题来自来源材料中明确提到的“{quote}”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/open_questions/Open_Question_Log_Growth.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_source_local_context_window_paraphrase() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-CONTEXT-WINDOW",
        source_basis=SourceBasis(source_candidate_ids=["Q001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Context_Window.md",
        display_title="未来上下文工程不可预测性",
        page_type="open_question",
        new_understanding="会话和上下文窗口的边界可能变化。",
        section_plans={"examples": "例子"},
        reason="测试 source-local concept paraphrase 去引号。",
    )
    quote = "会话不是 Claude 的上下文窗口"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-CONTEXT-WINDOW",
                action="create",
                canonical_target_path="open_questions/Open_Question_Context_Window.md",
                summary="摘要。",
                body_markdown=draft_body(detail="暂无更多细节。", examples=f"原文提到“{quote}”，但未说明未来原生长上下文是否会改变当前架构。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/open_questions/Open_Question_Context_Window.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_short_slogan_label() -> None:
    quote = "快速行动，打破常规"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-META-CULTURE",
                action="create",
                canonical_target_path="comparisons/Comparison_Meta_Culture.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Meta 速度实验驱动：推崇“{quote}”，产品决策依赖 A/B 测试和快速迭代。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(
        draft,
        "Meta 的文化是速度驱动的。Move fast and break things。你不需要完美的文档。",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert "推崇快速行动，打破常规" in rewritten.pages[0].body_markdown


def test_grounding_rewrite_does_not_dequote_short_hard_fact_label() -> None:
    quote = "用户增长，收入下降"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-HARD-FACT",
                action="create",
                canonical_target_path="concepts/Concept_Hard_Fact.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"报告称“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "源材料没有这句话。")

    assert report["changed"] is False
    assert f"“{quote}”" in rewritten.pages[0].body_markdown


def test_grounding_numeric_reliability_rewrite_does_not_match_decimal_percent() -> None:
    quote = "9.5%对AI来说就是失败"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-AUTOMATION-RELIABILITY",
                action="create",
                canonical_target_path="concepts/Concept_AI自动化可靠性.md",
                summary="摘要。",
                body_markdown=draft_body(additional_notes=f"正如Cat Wu所说“{quote}”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(
        draft,
        "95%的自动化真的没什么价值。",
    )

    assert report["changed"] is False
    assert rewritten.pages[0].body_markdown == draft.pages[0].body_markdown


def test_grounding_attributed_paraphrase_warns_without_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FAST-SHIPPING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Shipping.md",
        display_title="快速交付方法",
        page_type="concept",
        new_understanding="团队用目标短语总结快速交付方法。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 人物归因短语。",
    )
    quote = "找到最快将功能交到用户手中的方法"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FAST-SHIPPING",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Shipping.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat Wu指出“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Shipping.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "我们怎样才能找到最快把东西推出去的方法？"
        "让工程师或 PM 有一个想法，到周末就能把功能交到用户手中。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )
    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, raw)

    assert report["changed"] is False
    assert rewritten.pages[0].body_markdown == draft.pages[0].body_markdown
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_named_tool_concept_label_with_digits_is_not_direct_quote() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-AGENT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI_Agent.md",
        display_title="AI Agent 架构",
        page_type="concept",
        new_understanding="Agent 和工作流适用场景不同。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 工具标题标签。",
    )
    quote = "N8N工作流与Agent构建对比"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-AGENT",
                action="create",
                canonical_target_path="concepts/Concept_AI_Agent.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本页可与设计模式“{quote}”联动阅读。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_AI_Agent.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_named_tool_label_with_numeric_fact_warns_without_support() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-AGENT-FACT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI_Agent.md",
        display_title="AI Agent 架构",
        page_type="concept",
        new_understanding="Agent 系统包含多个步骤。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试带数字的短事实不能伪装成概念标题。",
    )
    quote = "Agent系统有3个步骤"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-AGENT-FACT",
                action="create",
                canonical_target_path="concepts/Concept_AI_Agent.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本页暂以“{quote}”作为结构提示。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_AI_Agent.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_quoted_compact_paraphrase_warns_without_support_for_each_part() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FAST-ITERATION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Iteration.md",
        display_title="快速迭代流程",
        page_type="concept",
        new_understanding="清晰目标帮助团队快速迭代。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 压缩概括。",
    )
    quote = "核心用户是专业开发者，主要问题是权限提示疲劳"
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FAST-ITERATION",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Iteration.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"设定清晰目标（如“{quote}”）可以减少 LLM 通用性带来的模糊。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Iteration.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "团队原则中写到，核心用户是专业开发者。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_short_fact_phrases_warn_without_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-FACT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fact.md",
        display_title="短事实",
        page_type="concept",
        new_understanding="短事实需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-FACT",
                action="create",
                canonical_target_path="concepts/Concept_Fact.md",
                summary="结果包括例如“销量增长三倍”、“裁撤一半团队”和“预算超过百万”。",
                body_markdown=draft_body(detail="暂无更多细节。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Fact.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["销量增长三倍", "裁撤一半团队", "预算超过百万"]


def test_grounding_quoted_release_event_warns_without_exact_match() -> None:
    item = WikiMergePlanItem(
        page_plan_id="PP-RELEASE-FACT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Release_Fact.md",
        display_title="发布事实",
        page_type="concept",
        new_understanding="发布事件需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding 发布事实。",
    )
    draft = DraftRenderingArtifact(
        pages=[
            DraftPageItem(
                page_plan_id="PP-RELEASE-FACT",
                action="create",
                canonical_target_path="concepts/Concept_Release_Fact.md",
                summary="团队“发布了重大功能”。",
                body_markdown=draft_body(detail="暂无更多细节。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            WikiContextEntry(
                path="wiki/concepts/Concept_Release_Fact.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["发布了重大功能"]
