from pathlib import Path

import pytest

from llmwiki_engine.models import EvidencePolicy
from llmwiki_engine.profiles import builtin_profile_names, load_profile


def test_builtin_profiles_load() -> None:
    names = builtin_profile_names()
    assert {"project_basic", "research_basic", "memory_basic"}.issubset(names)
    profile = load_profile("project_basic")
    assert profile.page_types["source"].evidence_policy == EvidencePolicy.strict
    assert "concept" in profile.page_types


def test_unknown_profile_fails() -> None:
    with pytest.raises(Exception):
        load_profile("missing_profile")


def test_custom_profile_validates(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
name: custom
version: "1"
default_page_type: note
source_page_type: source
page_types:
  note:
    directory: notes
    title_prefix: "Note_"
    template: note.md
    required_sections: ["Summary"]
    evidence_policy: none
  source:
    directory: sources
    title_prefix: "Source_"
    template: source.md
    required_sections: ["Summary"]
    evidence_policy: light
""",
        encoding="utf-8",
    )
    assert load_profile(profile_path).name == "custom"

