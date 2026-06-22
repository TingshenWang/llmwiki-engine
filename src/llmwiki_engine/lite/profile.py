from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .io import write_text


class PageTypeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str
    title_prefix: str
    required_sections: list[str] = Field(default_factory=list)
    evidence_policy: str = "light"


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str
    default_page_type: str
    source_page_type: str
    page_types: dict[str, PageTypeSpec]

    def page_type(self, page_type: str) -> PageTypeSpec:
        return self.page_types.get(page_type, self.page_types[self.default_page_type])

PROJECT_BASIC = {
    "name": "project_basic",
    "version": "2-lite",
    "description": "Lite ingest 使用的中文项目知识库 profile。",
    "default_page_type": "concept",
    "source_page_type": "source",
    "page_types": {
        "source": {
            "directory": "sources",
            "title_prefix": "Source_",
            "required_sections": ["摘要", "原始材料", "关键收获", "派生知识页"],
            "evidence_policy": "light",
        },
        "concept": {
            "directory": "concepts",
            "title_prefix": "Concept_",
            "required_sections": ["摘要", "核心内容", "矛盾与未决问题"],
            "evidence_policy": "light",
        },
        "entity": {
            "directory": "entities",
            "title_prefix": "Entity_",
            "required_sections": ["摘要", "核心内容", "矛盾与未决问题"],
            "evidence_policy": "light",
        },
        "comparison": {
            "directory": "comparisons",
            "title_prefix": "Comparison_",
            "required_sections": ["摘要", "核心内容", "矛盾与未决问题"],
            "evidence_policy": "light",
        },
        "overview": {
            "directory": "overviews",
            "title_prefix": "Overview_",
            "required_sections": ["摘要", "核心内容", "矛盾与未决问题"],
            "evidence_policy": "light",
        },
        "design": {
            "directory": "designs",
            "title_prefix": "Design_",
            "required_sections": ["摘要", "核心内容", "矛盾与未决问题"],
            "evidence_policy": "light",
        },
        "event": {
            "directory": "events",
            "title_prefix": "Event_",
            "required_sections": ["摘要", "核心内容", "矛盾与未决问题"],
            "evidence_policy": "light",
        },
        "open_question": {
            "directory": "open_questions",
            "title_prefix": "Open_Question_",
            "required_sections": ["摘要", "核心内容", "矛盾与未决问题"],
            "evidence_policy": "light",
        },
    },
}

BUILTIN_PROFILES = {"project_basic": PROJECT_BASIC}


def load_profile(vault: Path, profile_name: str = "project_basic") -> Profile:
    profile_path = vault / ".llmwiki" / "profiles" / profile_name / "profile.yaml"
    if profile_path.exists():
        return Profile.model_validate(yaml.safe_load(profile_path.read_text(encoding="utf-8")))
    if profile_name not in BUILTIN_PROFILES:
        raise ValueError(f"未知 profile：{profile_name}")
    return Profile.model_validate(BUILTIN_PROFILES[profile_name])


def write_profile(vault: Path, profile: Profile) -> None:
    path = vault / ".llmwiki" / "profiles" / profile.name / "profile.yaml"
    payload = yaml.safe_dump(profile.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
    write_text(path, payload)
