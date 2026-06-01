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
