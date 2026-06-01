from pathlib import Path

import pytest

from llmwiki_engine.eval import run_eval
from llmwiki_engine.steps import EVAL_MODULES


def test_eval_run_writes_report(tmp_path: Path) -> None:
    dataset = Path(__file__).parent / "fixtures" / "evals" / "page_planning"
    result = run_eval("page_planning", dataset, tmp_path)
    assert result.parse_success_rate == 1.0
    assert result.results[0].page_type_accuracy == 1.0
    assert (tmp_path / f"{result.run_id}.json").exists()


def test_unsupported_eval_module_lists_supported_modules(tmp_path: Path) -> None:
    dataset = Path(__file__).parent / "fixtures" / "evals" / "page_planning"
    with pytest.raises(ValueError) as exc:
        run_eval("missing_module", dataset, tmp_path)
    message = str(exc.value)
    assert "Unsupported eval module: missing_module" in message
    for module in EVAL_MODULES:
        assert module in message
