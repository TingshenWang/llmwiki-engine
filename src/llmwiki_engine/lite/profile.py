from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .io import write_text


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str

PROJECT_BASIC = {
    "name": "project_basic",
    "version": "2-lite",
    "description": "Lite ingest 使用的中文项目知识库 profile。",
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
