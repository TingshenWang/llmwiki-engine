import llmwiki_engine.pipeline as pipeline_module


def test_pipeline_does_not_reexport_update_preservation_helpers() -> None:
    assert not hasattr(pipeline_module, "build_update_preservation_pack")
    assert not hasattr(pipeline_module, "update_preservation_issues")
    assert not hasattr(pipeline_module, "update_preservation_concepts")
    assert not hasattr(pipeline_module, "update_preservation_section_absorption")
    assert not hasattr(pipeline_module, "draft_page_text_for_preservation_section")
    assert not hasattr(pipeline_module, "draft_field_for_preservation_section")
    assert not hasattr(pipeline_module, "parse_existing_sections")
