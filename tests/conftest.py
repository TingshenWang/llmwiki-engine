from __future__ import annotations

from pathlib import Path

import pytest

from llmwiki_engine.lite import embeddings


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())


@pytest.fixture(autouse=True)
def fake_sentence_transformer_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed_texts(texts: list[str], config: embeddings.EmbeddingConfig, *, is_query: bool) -> list[list[float]]:
        return [embeddings.hashing_vector(text, config.dimensions) for text in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
