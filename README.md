# llmwiki-engine

English | [中文](README.zh-CN.md)

`llmwiki-engine` is a Python 3.11+ CLI and engine for modular, profile-driven
knowledge compilation. Its goal is to compile noisy source material into a local
wiki while keeping each step independently testable, evaluable, and optimizable.

For complete command usage, parameters, config rules, Git boundaries, and common
errors, see [docs/cli-reference.en.md](docs/cli-reference.en.md). Chinese users
can read [docs/cli-reference.zh-CN.md](docs/cli-reference.zh-CN.md).

The current runnable path is the M4 Ingest MVP. It first normalizes Obsidian
text wikilinks in the target `raw/` file in place, then prepares noisy raw
material into a canonical `raw_prepare/prepared.md` artifact, creates a
`source_digest`, checks source duplicates, plans candidate pages from the
approved prepared text, freezes a wiki context snapshot with local embedding
retrieval evidence, model-plans create/update/noop decisions, renders reviewed
drafts, and produces an apply preview:

```bash
llmwiki init /path/to/vault --profile project_basic
llmwiki ingest run /path/to/vault raw/project_note.md --prepare auto --mock-fixture-dir tests/fixtures/simple_project/mock
llmwiki providers check /path/to/vault
llmwiki ingest status /path/to/vault <operation_id>
llmwiki ingest apply /path/to/vault <operation_id>
```

Mock/fixture runs are deterministic when provider YAML includes `fixture_dir`,
or when a run is forced with `--mock-fixture-dir`. This lets the pipeline,
artifacts, validators, logs, profiles, and drafts be tested before real models
are introduced.

## Design Principles

- Canonical artifacts are JSON/JSONL.
- `raw/` is the normalized material layer. The MVP only unwraps Obsidian text
  wikilinks such as `[[Page]]` and `[[Page|Alias]]`; web links, Markdown links,
  media embeds, and code blocks are preserved.
- `raw_prepare` turns original raw material into the canonical prepared raw used
  by downstream knowledge compilation. Use `--prepare auto` for normal model
  cleanup, `--prepare skip` when the user explicitly wants Markdown passthrough,
  and `--prepare force` when the user wants to force model cleanup.
- `source_digest` is the complete single-source digestion artifact used for
  human review of candidate knowledge.
- `candidate_resolution` plans wiki topics from the approved prepared text and
  digest; `wiki_context_snapshot` retrieves the most relevant existing wiki
  pages per planned topic; `wiki_merge_planning` uses that frozen evidence to
  decide create/update/noop/needs-human-decision.
- Update writes are supported as whole-page draft replacement, but must pass
  explicit draft review before validation/apply.
- Knowledge pages use sparse deterministic `Related` wikilinks; source pages
  stay out of the Obsidian knowledge graph.
- Human-editable profiles are YAML.
- Human review and drafts are Markdown.
- Providers are pluggable per module.
- Validators are hard gates; future LLM critics are reserved semantic reviewers.

Provider selection is configured in YAML. A vault can use local
`.llmwiki/config.yaml`, while shared provider defaults can live in
`~/.llmwiki/config.yaml`. Each operation records a sanitized provider context
for the current execution. Provider records keep only non-secret fields such as
`spec`, `endpoint`, `fixture_dir`, `max_retries`, and
`retry_backoff_seconds`; plaintext API keys are allowed only in config files and
are never written to run artifacts, status JSON, receipts, or CLI output.

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-chat
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: sk-...
    max_retries: 2
    retry_backoff_seconds: 1.0
  source_digest:
    spec: mock:fixture
    fixture_dir: tests/fixtures/simple_project/mock
```

Check provider configuration before a run:

```bash
llmwiki providers check /path/to/vault
llmwiki providers check /path/to/vault --live
```

`--live` sends a small JSON-mode real-model probe, then falls back once to a
prompt-only JSON probe with a warning when JSON mode is clearly unsupported. It
does not use transient retry. Normal ingest calls retry transient
OpenAI-compatible provider failures such as timeouts, connection resets, 429,
408/409/425, and 5xx responses; `max_retries` is one shared transient retry
budget per logical model call. JSON-mode compatibility fallback may add a
prompt-only request and uses only the remaining retry budget.

Embedding retrieval is configured in `.llmwiki/config.json`, not provider YAML.
New vaults default to local CPU `sentence_transformers` with
`Qwen/Qwen3-Embedding-0.6B`; install it with `uv sync --extra embedding`.
Model files are cached globally under `~/.llmwiki/cache/embeddings` so multiple
vaults can share the same download. By default retrieval loads embeddings from
the local cache only (`local_files_only: true`), which keeps hand-tests from
making background HF Hub requests; set it to `false` only for an explicit
download-enabled run. Mock/fixture runs use an exact lexical retriever and do not
download embedding models.

Plain `llmwiki ingest resume` reads the current merged provider config for
model-backed steps that still need to execute. Completed steps are not rerun just
because config changed. To rerun from a step with the current config:

```bash
llmwiki ingest resume /path/to/vault <operation_id> --from source_digest
```
