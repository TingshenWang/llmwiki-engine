from pathlib import Path

import pytest
from pydantic import ValidationError

from llmwiki_engine.models import EvidencePolicy
from llmwiki_engine.profiles import builtin_profile_names, load_profile


def test_builtin_profiles_load() -> None:
    names = builtin_profile_names()
    assert {"project_basic", "research_basic", "memory_basic"}.issubset(names)
    profile = load_profile("project_basic")
    assert profile.version == "2"
    assert profile.page_types["source"].evidence_policy == EvidencePolicy.light
    assert profile.page_types["source"].required_sections == ["Summary", "Raw", "Key Takeaways", "Derived Wiki Pages"]
    assert "concept" in profile.page_types
    assert not hasattr(profile.page_types["concept"], "template")


def test_unknown_profile_fails() -> None:
    with pytest.raises(Exception):
        load_profile("missing_profile")


def test_custom_profile_validates(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
name: custom
version: "2"
default_page_type: note
source_page_type: source
page_types:
  note:
    directory: notes
    title_prefix: "Note_"
    required_sections: ["Summary"]
    evidence_policy: none
  source:
    directory: sources
    title_prefix: "Source_"
    required_sections: ["Summary"]
    evidence_policy: light
""",
        encoding="utf-8",
    )
    assert load_profile(profile_path).name == "custom"


def test_custom_profile_rejects_template_fields(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
name: custom
version: "2"
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
    required_sections: ["Summary"]
    evidence_policy: light
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="template"):
        load_profile(profile_path)


def test_custom_profile_requires_current_version(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
name: custom
version: unsupported
default_page_type: note
source_page_type: source
page_types:
  note:
    directory: notes
    title_prefix: "Note_"
    required_sections: ["Summary"]
    evidence_policy: none
  source:
    directory: sources
    title_prefix: "Source_"
    required_sections: ["Summary"]
    evidence_policy: light
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="version"):
        load_profile(profile_path)


def test_custom_profile_rejects_nested_page_type_name(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
name: custom
version: "2"
default_page_type: note
source_page_type: source
page_types:
  note:
    name: note
    directory: notes
    title_prefix: "Note_"
    required_sections: ["Summary"]
    evidence_policy: none
  source:
    directory: sources
    title_prefix: "Source_"
    required_sections: ["Summary"]
    evidence_policy: light
""",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="nested name"):
        load_profile(profile_path)
