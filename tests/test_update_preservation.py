import llmwiki_engine.pipeline as pipeline_module


def test_pipeline_does_not_reexport_update_preservation_helpers() -> None:
    assert not hasattr(pipeline_module, "build_update_preservation_pack")
    assert not hasattr(pipeline_module, "update_preservation_issues")
    assert not hasattr(pipeline_module, "update_preservation_concepts")
    assert not hasattr(pipeline_module, "update_preservation_section_absorption")
    assert not hasattr(pipeline_module, "draft_page_text_for_preservation_section")
    assert not hasattr(pipeline_module, "draft_field_for_preservation_section")
    assert not hasattr(pipeline_module, "parse_existing_sections")


def test_pipeline_does_not_reexport_draft_rendering_runner_helpers() -> None:
    for helper in [
        "run_draft_rendering_model",
        "run_single_draft_rendering_model_call",
        "build_draft_rendering_missing_page_repair_payload",
        "build_draft_rendering_page_repair_payload",
        "draft_repair_accepted_page_refs",
        "merge_repaired_draft_with_accepted_pages",
        "preserve_active_repair_page_issues",
        "draft_repair_page_plan_ids_from_issues",
        "accepted_partial_page_copy_issues",
        "extract_valid_partial_draft_rendering",
        "write_draft_aux_report_if_active",
        "write_draft_rendering_batch_reports",
        "render_draft_rendering_batch_report",
        "finalize_draft_rendering",
    ]:
        assert not hasattr(pipeline_module, helper)
