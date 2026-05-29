from pathlib import Path

from llmwiki_engine.eval import run_eval


def test_eval_run_writes_report(tmp_path: Path) -> None:
    dataset = Path(__file__).parent / "fixtures" / "evals" / "page_planning"
    result = run_eval("page_planning", dataset, tmp_path)
    assert result.parse_success_rate == 1.0
    assert result.results[0].page_type_accuracy == 1.0
    assert (tmp_path / f"{result.run_id}.json").exists()

