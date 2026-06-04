# llmwiki-engine

English | [中文](README.zh-CN.md)

`llmwiki-engine` is a Python 3.11+ CLI and engine for modular, profile-driven
knowledge compilation. Its goal is to compile noisy source material into a local
wiki while keeping each step independently testable, evaluable, and optimizable.

For complete command usage, parameters, config rules, Git boundaries, and common
errors, see [docs/cli-reference.en.md](docs/cli-reference.en.md). Chinese users
can read [docs/cli-reference.zh-CN.md](docs/cli-reference.zh-CN.md).

The current runnable path is the M3 Ingest MVP. It first normalizes Obsidian
text wikilinks in the target `raw/` file in place, then prepares noisy raw
material into a canonical `raw_prepare/prepared.md` artifact, creates a
`source_digest`, checks source duplicates, plans candidate pages from the
approved prepared text, freezes a wiki context snapshot, model-plans
create/update/noop decisions, renders reviewed drafts, and produces an apply
preview:

```bash
llmwiki init /path/to/vault --profile project_basic
llmwiki ingest run /path/to/vault raw/project_note.md --fixture-dir tests/fixtures/simple_project/mock
llmwiki providers check /path/to/vault
llmwiki ingest status /path/to/vault <operation_id>
llmwiki ingest apply /path/to/vault <operation_id>
```

The default runtime is deterministic: model-backed modules use fixture-backed
`MockProvider` outputs so the pipeline, artifacts, validators, logs, profiles,
and drafts can be tested before real models are introduced.

## Design Principles

- Canonical artifacts are JSON/JSONL.
- `raw/` is the normalized material layer. The MVP only unwraps Obsidian text
  wikilinks such as `[[Page]]` and `[[Page|Alias]]`; web links, Markdown links,
  media embeds, and code blocks are preserved.
- `raw_prepare` turns original raw material into the canonical prepared raw used
  by downstream knowledge compilation.
- `source_digest` is the complete single-source digestion artifact used for
  human review of candidate knowledge.
- `candidate_resolution` plans wiki topics from the approved prepared text and
  digest; `wiki_merge_planning` uses a frozen snapshot to decide
  create/update/noop/needs-human-decision.
- Update writes are supported in dev mode as whole-page draft replacement, but
  must pass explicit draft review before validation/apply.
- Knowledge pages use deterministic `Related` wikilinks; source pages stay out
  of the Obsidian knowledge graph.
- Human-editable profiles are YAML.
- Human review and drafts are Markdown.
- Providers are pluggable per module.
- Validators are hard gates; future LLM critics are reserved semantic reviewers.

Provider selection is configured in YAML. A vault can use local
`.llmwiki/config.yaml`, while shared provider defaults can live in
`~/.llmwiki/config.yaml`. Each operation records a sanitized provider context
for the current execution. Provider records keep only `spec`, `endpoint`, and
`fixture_dir`; plaintext API keys are allowed only in config files and are never
written to run artifacts, status JSON, receipts, or CLI output.

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-chat
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: sk-...
  source_digest:
    spec: mock:fixture
    fixture_dir: tests/fixtures/simple_project/mock
```

Check provider configuration before a run:

```bash
llmwiki providers check /path/to/vault
llmwiki providers check /path/to/vault --live
```

`--live` sends a small real-model probe. For thinking models it uses a
`max_tokens=512` completion cap and prefers JSON mode, then falls back to a
prompt-only JSON probe with a warning when JSON mode is clearly unsupported.

Plain `llmwiki ingest resume` reads the current merged provider config for
model-backed steps that still need to execute. Completed steps are not rerun just
because config changed. To rerun from a step with the current config:

```bash
llmwiki ingest resume /path/to/vault <operation_id> --from source_digest
```
