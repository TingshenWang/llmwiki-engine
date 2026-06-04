from pathlib import Path

import pytest

from llmwiki_engine.eval import run_eval
from llmwiki_engine.io import read_json, write_json
from llmwiki_engine.steps import EVAL_MODULES


def test_eval_run_writes_report(tmp_path: Path) -> None:
    dataset = Path(__file__).parent / "fixtures" / "evals" / "source_digest"
    result = run_eval("source_digest", dataset, tmp_path)
    assert result.parse_success_rate == 1.0
    assert result.results[0].page_type_accuracy == 1.0
    assert (tmp_path / f"{result.run_id}.json").exists()


def test_source_digest_eval_schema_validation_rejects_v1(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    case = dataset / "case1"
    case.mkdir(parents=True)
    source = Path(__file__).parent / "fixtures" / "evals" / "source_digest" / "case1" / "actual.json"
    data = read_json(source)
    data["schema_version"] = "source_digest.v1"
    write_json(case / "actual.json", data)

    result = run_eval("source_digest", dataset, tmp_path / "out")

    assert result.results[0].parse_success is True
    assert result.results[0].schema_valid is False
    assert "source_digest.v1 is incompatible" in result.results[0].errors[0]


def test_source_digest_eval_schema_validation_rejects_invalid_expected(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    case = dataset / "case1"
    case.mkdir(parents=True)
    source = Path(__file__).parent / "fixtures" / "evals" / "source_digest" / "case1" / "actual.json"
    actual = read_json(source)
    expected = dict(actual)
    expected["schema_version"] = "source_digest.v1"
    write_json(case / "actual.json", actual)
    write_json(case / "expected.json", expected)

    result = run_eval("source_digest", dataset, tmp_path / "out")

    assert result.results[0].parse_success is True
    assert result.results[0].schema_valid is False
    assert "source_digest.v1 is incompatible" in result.results[0].errors[0]


def test_source_digest_eval_schema_validation_rejects_extra_field(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    case = dataset / "case1"
    case.mkdir(parents=True)
    source = Path(__file__).parent / "fixtures" / "evals" / "source_digest" / "case1" / "actual.json"
    data = read_json(source)
    data["unexpected"] = "blocked"
    write_json(case / "actual.json", data)

    result = run_eval("source_digest", dataset, tmp_path / "out")

    assert result.results[0].parse_success is True
    assert result.results[0].schema_valid is False
    assert "Extra inputs are not permitted" in result.results[0].errors[0]


def test_unsupported_eval_module_lists_supported_modules(tmp_path: Path) -> None:
    dataset = Path(__file__).parent / "fixtures" / "evals" / "source_digest"
    with pytest.raises(ValueError) as exc:
        run_eval("missing_module", dataset, tmp_path)
    message = str(exc.value)
    assert "Unsupported eval module: missing_module" in message
    for module in EVAL_MODULES:
        assert module in message
