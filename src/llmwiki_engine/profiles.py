from __future__ import annotations

from importlib import resources
from pathlib import Path

from .io import read_yaml
from .models import PageTypeSpec, ProfileSpec


BUILTIN_PROFILE_ROOT = resources.files("llmwiki_engine") / "builtin_profiles"


class ProfileError(RuntimeError):
    pass


def builtin_profile_names() -> list[str]:
    return sorted(
        item.name
        for item in BUILTIN_PROFILE_ROOT.iterdir()
        if item.is_dir() and (item / "profile.yaml").is_file()
    )


def load_profile(name_or_path: str | Path) -> ProfileSpec:
    path = Path(name_or_path)
    if path.exists():
        profile_path = path / "profile.yaml" if path.is_dir() else path
        return _load_profile_file(profile_path)
    profile_path = Path(str(BUILTIN_PROFILE_ROOT / str(name_or_path) / "profile.yaml"))
    if not profile_path.exists():
        raise ProfileError(f"Unknown profile: {name_or_path}")
    return _load_profile_file(profile_path)


def _load_profile_file(path: Path) -> ProfileSpec:
    raw = read_yaml(path)
    page_types = {}
    for key, value in raw.get("page_types", {}).items():
        data = dict(value)
        if "name" in data:
            raise ProfileError(f"page_types.{key} must not contain a nested name field")
        page_types[key] = PageTypeSpec(name=key, **data)
    raw["page_types"] = page_types
    return ProfileSpec.model_validate(raw)


def profile_to_yaml_data(profile: ProfileSpec) -> dict[str, object]:
    data = profile.model_dump(mode="json")
    data["page_types"] = {
        key: {field: value for field, value in spec.model_dump(mode="json").items() if field != "name"}
        for key, spec in profile.page_types.items()
    }
    return data


def page_output_path(root: Path, profile: ProfileSpec, page_type: str, title: str) -> Path:
    spec = profile.page_types.get(page_type)
    if spec is None:
        raise ProfileError(f"Unknown page type {page_type!r} for profile {profile.name}")
    filename = safe_filename(title if title.startswith(spec.title_prefix) else f"{spec.title_prefix}{title}")
    return root / spec.directory / f"{filename}.md"


def safe_filename(value: str) -> str:
    bad = '\\/:*?"<>|#^[]'
    cleaned = "".join("_" if char in bad else char for char in value).strip(" .")
    return cleaned or "untitled"
