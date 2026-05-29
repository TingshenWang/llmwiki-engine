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
    raw["page_types"] = {
        key: PageTypeSpec(name=key, **value) for key, value in raw.get("page_types", {}).items()
    }
    raw["template_root"] = path.parent / "templates"
    return ProfileSpec.model_validate(raw)


def profile_template_text(profile: ProfileSpec, page_type: str) -> str:
    spec = profile.page_types.get(page_type)
    if spec is None:
        raise ProfileError(f"Profile {profile.name} does not define page type {page_type!r}")
    template_root = profile.template_root or Path(str(BUILTIN_PROFILE_ROOT / profile.name / "templates"))
    template_path = Path(str(template_root / spec.template))
    if not template_path.exists():
        raise ProfileError(f"Template not found for {profile.name}:{page_type}: {spec.template}")
    return template_path.read_text(encoding="utf-8")


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
