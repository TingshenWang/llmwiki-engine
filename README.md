# llmwiki-engine

English | [中文](README.zh-CN.md)

`llmwiki-engine` is a Python 3.11+ CLI and engine for modular, profile-driven
knowledge compilation. Its goal is to compile noisy source material into a local
wiki while keeping each step independently testable, evaluable, and optimizable.

The first runnable path is a simplified Ingest pipeline. It prepares noisy raw
material into a canonical `raw_prepare/prepared.md` artifact before indexing,
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

## Design Principles

- Canonical artifacts are JSON/JSONL.
- `raw_prepare` turns original raw material into the canonical prepared raw used
  by downstream knowledge compilation.
- `extraction_windows` are engineering context windows for structured
  extraction; they are not semantic knowledge units.
- Human-editable profiles are YAML.
- Human review and drafts are Markdown.
- Providers are pluggable per module.
- Validators are hard gates; future LLM critics are reserved semantic reviewers.

Provider selection is configured per task in `.llmwiki/config.yaml`. Each
operation records the provider context used by model-backed steps. Provider
records keep only runtime fields such as `spec`, `endpoint`, `api_key_env`, and
`fixture_dir`; plaintext API keys are rejected.

```yaml
providers:
  raw_prepare: mock:fixture
  page_planning:
    spec: mock:fixture
    fixture_dir: tests/fixtures/simple_project/mock
  claim_extraction:
    spec: openai:gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  default:
    spec: ollama:llama3
    endpoint: http://localhost:11434/api/generate
```

Plain `llmwiki ingest resume` reuses the provider context already recorded in
the operation manifest and does not reread the current config.

To intentionally rerun from a step with the current provider config:

```bash
llmwiki ingest resume /path/to/vault <operation_id> --from claim_extraction --refresh-providers
```
