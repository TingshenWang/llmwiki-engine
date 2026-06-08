from llmwiki_engine import pipeline as pipeline_module
from llmwiki_engine import source_excerpt


def test_pipeline_does_not_reexport_source_excerpt_helpers() -> None:
    assert not hasattr(pipeline_module, "source_snippets_for_cues")
    assert not hasattr(pipeline_module, "find_source_cue")
    assert not hasattr(pipeline_module, "source_semantic_match_terms")


def test_source_snippets_match_nfkc_parenthetical_title_variants() -> None:
    text = (
        "# Cat Wu 访谈\n\n"
        + ("开头填充段落。\n" * 80)
        + "归根结底还是产品品味（Product Taste）。当代码越来越廉价时，决定写什么更有价值。\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(
        text,
        ["产品品味 (Product Taste)"],
        max_chars=240,
    )

    assert snippets
    assert snippets[0]["cue"] != "fallback_start"
    assert "产品品味" in snippets[0]["text"]
    assert source_excerpt.find_source_cue(text, "产品品味 (Product Taste)") >= 0


def test_source_snippets_use_locator_heading_instead_of_start_fallback() -> None:
    text = (
        "# Cat Wu 访谈\n\n"
        + ("开头填充段落。\n" * 80)
        + "### 为什么95%自动化不够\n\n"
        "如果自动化不是100%有效，它真的不是自动化。95%的自动化真的没什么价值。\n\n"
        "### 构建你每天使用的应用，而不是原型\n\n"
        "后续小节内容。\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(
        text,
        ["自动化100%价值 (Value of 100% Automation)", "访谈‘为什么95%自动化不够’部分"],
        max_chars=260,
    )

    assert snippets
    assert snippets[0]["cue"] != "fallback_start"
    assert "为什么95%自动化不够" in snippets[0]["text"]
    assert "95%的自动化真的没什么价值" in snippets[0]["text"]
    assert "开头填充段落" not in snippets[0]["text"]


def test_source_snippets_rank_specific_window_over_generic_frontmatter() -> None:
    text = (
        "---\n"
        "title: Cat Wu at Anthropic\n"
        "description: Anthropic 的 Claude Code 和 Cowork 访谈。\n"
        "---\n\n"
        "## 访谈全文\n\n"
        + ("开头背景段落。\n" * 60)
        + "### 为什么构建Eval被低估了\n\n"
        "所以我认为Eval是被低估的东西，更多的PM和工程师应该做这个。"
        "仅仅构建10个出色的Eval，对于帮助团队量化目标是什么、他们离目标进展如何、以及缺少什么，就很重要。"
        "团队会更精确地理解Claude Code行为，以及最大的改进领域是什么。\n\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(
        text,
        [
            "评估（Eval）",
            "评估在AI产品开发中被低估，它帮助量化成功、指导模型改进。",
            "阐述评估在Anthropic中的应用：如何编写评估、以及评估如何辅助产品决策。",
        ],
        max_chars=360,
    )

    assert snippets
    snippet_text = "\n".join(snippet["text"] for snippet in snippets)
    assert "为什么构建Eval被低估了" in snippet_text
    assert "量化目标" in snippet_text
    assert "title: Cat Wu" not in snippet_text


def test_source_semantic_match_terms_preserve_api_terms_with_translated_cues() -> None:
    cues = [
        "对比通过 ingest() 自动从对话提取记忆与通过 remember() 让模型直接存储已知记忆的优劣",
        "讨论提示中要求模型将 recall 召回记忆视为有用上下文而非绝对真相",
        "S010-S011",
        "帮助开发者根据场景选择合适的记忆写入方式",
        "团队内部技术方案评审",
        "用户体验敏感的场景设计",
    ]

    terms = source_excerpt.source_semantic_match_terms(cues)

    assert {"ingest", "remember", "recall"} <= set(terms)
    assert "s010s011" not in terms
    assert any("记忆写入" in term for term in terms)


def test_source_snippets_resolve_multiple_section_locators_without_start_fallback() -> None:
    text = (
        "# Cloudflare Agent Memory: Get started\n\n"
        "Documentation chrome that should not be selected.\n\n"
        "## How agent memory works\n\n"
        "Use recall when the model needs relevant memory.\n\n"
        "## Extract memories from conversation\n\n"
        "Use ingest when you have conversation messages and want Agent Memory to extract durable memories automatically.\n\n"
        "## Store explicit memories when needed\n\n"
        "Use remember when your agent already knows the exact memory to store.\n\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(text, ["S003, S004"], max_chars=360)

    snippet_text = "\n".join(snippet["text"] for snippet in snippets)
    assert snippets
    assert snippets[0]["cue"].startswith("source_locator:S003")
    assert "Extract memories from conversation" in snippet_text
    assert "Store explicit memories when needed" in snippet_text
    assert "Documentation chrome" not in snippet_text


def test_source_snippets_resolve_section_locator_ranges_and_labels() -> None:
    text = (
        "# Memory docs\n\n"
        "Intro chrome.\n\n"
        "## Add memory recall as a tool\n\n"
        "The MEMORY_CONTEXT prompt tells the model to treat recalled memories as useful context, not absolute truth.\n\n"
        "## Extract memories from conversation\n\n"
        "Use ingest after idle windows rather than after every model turn.\n\n"
    )

    label_snippets = source_excerpt.source_snippets_for_cues(text, ["S002 MEMORY_CONTEXT 提示"], max_chars=260)
    range_snippets = source_excerpt.source_snippets_for_cues(text, ["S002-S003"], max_chars=360)

    assert "MEMORY_CONTEXT prompt" in "\n".join(snippet["text"] for snippet in label_snippets)
    range_text = "\n".join(snippet["text"] for snippet in range_snippets)
    assert "Add memory recall as a tool" in range_text
    assert "Extract memories from conversation" in range_text


def test_source_snippets_distribute_tight_budget_across_three_section_locators() -> None:
    text = (
        "# Memory docs\n\n"
        "Intro chrome.\n\n"
        "## Alpha locator section\n\n"
        "Alpha key evidence appears immediately. " + ("Alpha filler. " * 25) + "\n\n"
        "## Beta locator section\n\n"
        "Beta key evidence appears immediately. " + ("Beta filler. " * 25) + "\n\n"
        "## Gamma locator section\n\n"
        "Gamma key evidence appears immediately. " + ("Gamma filler. " * 25) + "\n\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(text, ["S002, S003, S004"], max_chars=480)

    snippet_text = "\n".join(snippet["text"] for snippet in snippets)
    assert "Alpha key evidence" in snippet_text
    assert "Beta key evidence" in snippet_text
    assert "Gamma key evidence" in snippet_text
    assert sum(len(snippet["text"]) for snippet in snippets) <= 480


def test_source_snippets_stale_section_locator_does_not_suppress_semantic_fallback() -> None:
    text = (
        "# Memory docs\n\n"
        "Documentation chrome that mentions setup but not the target API terms.\n\n"
        "## Extract memories from conversation\n\n"
        "Use `ingest()` when you have conversation messages and want Agent Memory to extract durable memories automatically. "
        "Use `remember()` only when the agent already knows the exact memory to store.\n\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(
        text,
        ["S001", "自动摄取通过 ingest() 提取对话记忆，并用 remember() 保存明确记忆"],
        max_chars=300,
    )

    snippet_text = "\n".join(snippet["text"] for snippet in snippets)
    assert snippets[0]["cue"].startswith("fallback_semantic:")
    assert "Use `ingest()`" in snippet_text
    assert "Use `ingest()`" in snippets[0]["text"]
    assert "Documentation chrome" not in snippets[0]["text"]
    assert sum(len(snippet["text"]) for snippet in snippets) <= 300


def test_source_snippets_correct_section_locator_dedupes_semantic_fallback() -> None:
    text = (
        "# Memory docs\n\n"
        "Intro chrome.\n\n"
        "## Extract memories from conversation\n\n"
        "Use `ingest()` when you have conversation messages and want Agent Memory to extract durable memories automatically. "
        "Use `remember()` only when the agent already knows the exact memory to store.\n\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(
        text,
        ["S002", "自动摄取通过 ingest() 提取对话记忆，并用 remember() 保存明确记忆"],
        max_chars=300,
    )

    assert snippets[0]["cue"].startswith("source_locator:S002")
    assert len(snippets) == 1
    assert "Extract memories from conversation" in snippets[0]["text"]
    assert sum(len(snippet["text"]) for snippet in snippets) <= 300


def test_source_snippets_use_semantic_fallback_when_exact_cue_is_not_contiguous() -> None:
    text = (
        "# 偏好学习笔记\n\n"
        + ("开头背景段落，不包含目标知识。\n\n" * 45)
        + "训练偏好系统时，人类反馈会先被整理成比较数据。随后团队训练奖励模型，"
        "让模型学习哪些回答更符合人类偏好，并把这种偏好信号用于后续对齐。\n\n"
        + ("结尾填充段落。\n" * 20)
    )

    snippets = source_excerpt.source_snippets_for_cues(
        text,
        ["人类反馈训练奖励模型可以改善偏好对齐"],
        max_chars=280,
    )

    assert snippets
    assert snippets[0]["cue"].startswith("fallback_semantic:")
    assert "人类反馈" in snippets[0]["text"]
    assert "奖励模型" in snippets[0]["text"]
    assert "开头背景段落" not in snippets[0]["text"]


def test_source_snippets_semantic_fallback_ignores_generic_cues() -> None:
    text = (
        "# 普通材料\n\n"
        + ("开头背景段落，用来模拟很长的输入。\n" * 40)
        + "\n\n这里讨论一个具体实现，但没有足够的候选主题词。\n"
    )

    snippets = source_excerpt.source_snippets_for_cues(
        text,
        ["为什么这个问题重要", "来源定位：讨论部分"],
        max_chars=220,
    )

    assert snippets
    assert snippets[0]["cue"] == "fallback_start"
