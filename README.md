# llmwiki-engine

`llmwiki-engine` is a Python 3.11+ CLI and engine for modular, profile-driven
knowledge compilation.

The first runnable path is a simplified Ingest pipeline. It prepares noisy raw
material into a canonical `prepared_raw/prepared.md` artifact before indexing,
extraction windows, and wiki draft rendering:

```bash
llmwiki init /path/to/vault --profile project_basic
llmwiki ingest run /path/to/vault raw/project_note.md --fixture-dir tests/fixtures/simple_project/mock
llmwiki ingest status /path/to/vault <operation_id>
llmwiki ingest apply /path/to/vault <operation_id>
```

The default runtime is deterministic: model-backed modules use fixture-backed
`MockProvider` outputs so the pipeline, artifacts, validators, logs, profiles,
and drafts can be tested before real models are introduced.

## Design

- Canonical artifacts are JSON/JSONL.
- `raw_prepare` turns original raw material into the canonical prepared raw used
  by downstream knowledge compilation.
- `extraction_windows` are engineering context windows for structured
  extraction; they are not semantic knowledge units.
- Human-editable profiles are YAML.
- Human review and drafts are Markdown.
- Providers are pluggable per module.
- Validators are hard gates; LLM critics are optional semantic reviewers.
