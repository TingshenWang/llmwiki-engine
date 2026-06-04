from pathlib import Path

import pytest

from llmwiki_engine.steps import STEP_NAMES


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "marker"),
    [
        (ROOT / "docs" / "cli-reference.en.md", "Valid steps:"),
        (ROOT / "docs" / "cli-reference.zh-CN.md", "合法 step："),
    ],
)
def test_cli_reference_resume_from_steps_match_step_metadata(path: Path, marker: str) -> None:
    assert _fenced_text_after_marker(path, marker) == STEP_NAMES


def test_user_facing_docs_do_not_reintroduce_removed_pipeline_or_commit_promise() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "docs" / "cli-reference.en.md",
        ROOT / "docs" / "cli-reference.zh-CN.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for removed_step in ["source_page_rendering", "raw_index", "extraction_windows", "claim_extraction", "page_planning"]:
        assert removed_step not in combined
    assert "Write `vault/wiki/` and create a Git commit" not in combined
    assert "写入 `vault/wiki/` 后创建 Git commit" not in combined
    for removed_page_contract in ["Understanding", "{{understanding}}", "Final Wiki Pages", "Source Pages", "[[sources/"]:
        assert removed_page_contract not in combined


def _fenced_text_after_marker(path: Path, marker: str) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    assert marker in text
    after_marker = text.split(marker, 1)[1]
    assert "```" in after_marker
    fence_content = after_marker.split("```", 1)[1]
    assert "```" in fence_content
    block = fence_content.split("```", 1)[0]
    lines = block.splitlines()
    if lines and lines[0].strip() and lines[0].strip() not in STEP_NAMES:
        lines = lines[1:]
    return tuple(line.strip() for line in lines if line.strip())
