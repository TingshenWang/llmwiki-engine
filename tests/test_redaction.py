from pydantic import BaseModel

from llmwiki_engine.redaction import Redactor, redact_model


class SecretModel(BaseModel):
    title: str
    nested: dict[str, object]
    items: list[str]


def test_redact_model_redacts_nested_model_dump_values() -> None:
    model = SecretModel(
        title="token-123 appears here",
        nested={"value": "keep token-123 hidden", "count": 2},
        items=["safe", "token-123"],
    )

    redacted = redact_model(Redactor(("token-123",)), model, SecretModel)

    assert isinstance(redacted, SecretModel)
    assert redacted.title == "[REDACTED] appears here"
    assert redacted.nested == {"value": "keep [REDACTED] hidden", "count": 2}
    assert redacted.items == ["safe", "[REDACTED]"]
