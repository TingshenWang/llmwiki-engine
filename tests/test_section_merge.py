import llmwiki_engine.pipeline as pipeline_module


def _assert_pipeline_does_not_reexport(names: set[str]) -> None:
    leaked = sorted(name for name in names if hasattr(pipeline_module, name))
    assert leaked == []


def test_pipeline_does_not_reexport_section_merge_helpers() -> None:
    _assert_pipeline_does_not_reexport(
        {"merge_update_section", "merge_update_open_questions_section", "split_high_signal_old_additional_notes", "old_additional_note_units", "update_merge_should_preserve_old_section", "preserved_old_section_block"}
    )


def test_pipeline_does_not_reexport_open_question_helpers() -> None:
    _assert_pipeline_does_not_reexport(
        {"open_question_key", "meaningful_open_question_lines", "is_low_signal_open_question", "extract_open_questions", "OPEN_QUESTION_SEMANTIC_CLUSTERS"}
    )


def test_pipeline_does_not_reexport_open_question_index_or_wiki_markup_helpers() -> None:
    _assert_pipeline_does_not_reexport(
        {"build_open_question_rows_with_report", "render_index_open_questions_report", "clean_display_title", "obsidian_link", "obsidian_alias_link", "obsidian_link_label", "draft_page_summary", "draft_page_core_markdown", "draft_page_open_questions"}
    )
