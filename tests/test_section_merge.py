import llmwiki_engine.pipeline as pipeline_module


def test_pipeline_does_not_reexport_section_merge_helpers() -> None:
    assert not hasattr(pipeline_module, "merge_update_section")
    assert not hasattr(pipeline_module, "merge_update_open_questions_section")
    assert not hasattr(pipeline_module, "split_high_signal_old_additional_notes")
    assert not hasattr(pipeline_module, "old_additional_note_units")
    assert not hasattr(pipeline_module, "update_merge_should_preserve_old_section")
    assert not hasattr(pipeline_module, "preserved_old_section_block")


def test_pipeline_does_not_reexport_open_question_helpers() -> None:
    assert not hasattr(pipeline_module, "open_question_key")
    assert not hasattr(pipeline_module, "meaningful_open_question_lines")
    assert not hasattr(pipeline_module, "is_low_signal_open_question")
    assert not hasattr(pipeline_module, "extract_open_questions")
    assert not hasattr(pipeline_module, "OPEN_QUESTION_SEMANTIC_CLUSTERS")


def test_pipeline_does_not_reexport_open_question_index_or_wiki_markup_helpers() -> None:
    assert not hasattr(pipeline_module, "build_open_question_rows_with_report")
    assert not hasattr(pipeline_module, "render_index_open_questions_report")
    assert not hasattr(pipeline_module, "clean_display_title")
    assert not hasattr(pipeline_module, "obsidian_link")
    assert not hasattr(pipeline_module, "obsidian_alias_link")
    assert not hasattr(pipeline_module, "obsidian_link_label")
    assert not hasattr(pipeline_module, "draft_page_summary")
    assert not hasattr(pipeline_module, "draft_page_core_markdown")
    assert not hasattr(pipeline_module, "draft_page_open_questions")
