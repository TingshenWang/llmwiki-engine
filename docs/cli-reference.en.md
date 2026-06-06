# llmwiki CLI Reference

This document is the command compass for `llmwiki-engine`. It explains common commands, parameters, config rules, Git boundaries, and frequent errors.

## Core Concepts

`vault`

A local knowledge base directory. `llmwiki init` creates `raw/`, `wiki/`, and `.llmwiki/` inside it.

`raw`

Source material to ingest. The `RAW` argument for `ingest run` must point to a file under `vault/raw/`. During `raw_link_cleanup`, the engine rewrites the target raw file in place as the normalized material layer. This MVP only unwraps Obsidian text wikilinks such as `[[Page]]` and `[[Page|Alias]]`; URL links, bare URLs, HTML links, reference links, relative Markdown links, media embeds, fenced code blocks, and inline code are preserved.

`.llmwiki/`

Local runtime state. It stores config, profiles, runs, manifests, events, and applied receipts. It is written to `.gitignore` and should not be committed.

`operation_id`

The ID for one ingest operation, for example:

```text
ING-2026-06-01T085014Z-manual
```

It maps to:

```text
<vault>/.llmwiki/runs/ingest/<operation_id>/
```

`fixture_dir`

The mock provider answer directory. It usually contains:

```text
raw_prepare.json
source_digest.json
```

`provider`

The execution source for model-backed steps. Public provider specs are:

- `mock:fixture`: read deterministic fixture files for tests.
- `human`: manual handoff placeholder.
- `openai_compatible:<model>`: call a Chat Completions-compatible API.

`apply`

Write draft pages from a run into `vault/wiki/`.

`apply --commit`

Disabled in this MVP. The flag is retained for CLI compatibility and fails before any verify or wiki write.

`staged`

Git's index. After `git add file`, the file is staged and a normal `git commit` would include it.

## Config Files

llmwiki uses YAML config:

```text
~/.llmwiki/config.yaml
<vault>/.llmwiki/config.yaml
```

Global config only supports `providers`. Vault config owns the current vault `profile` and may override providers.

Merge rule:

```text
global providers -> vault providers
same provider key is replaced as a whole
no field-level merge
```

Allowed provider keys:

```text
default
raw_prepare
source_digest
candidate_resolution
wiki_merge_planning
draft_rendering
```

`default` is the fallback provider. In normal real-model runs, configure only this key. A concrete step key is only needed when one step should use a different model.

Mock config example:

```yaml
profile: project_basic
providers:
  default:
    spec: mock:fixture
    fixture_dir: /path/to/mock
```

For real-model runs, prefer the global config at `~/.llmwiki/config.yaml` so vaults do not need repeated provider setup:

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-chat
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: sk-...
    max_retries: 2
    retry_backoff_seconds: 1.0
```

OpenAI-compatible config example:

```yaml
profile: project_basic
providers:
  default:
    spec: openai_compatible:deepseek-chat
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: sk-...
    max_retries: 2
    retry_backoff_seconds: 1.0
  source_digest:
    spec: openai_compatible:stronger-digest
    endpoint: https://example.test/v1/chat/completions
    api_key: sk-...
```

`max_retries` and `retry_backoff_seconds` are optional and only apply to
`openai_compatible` ingest calls. `max_retries` is one shared transient retry
budget per logical model call. JSON-mode compatibility fallback may add a
prompt-only request and uses only the remaining retry budget. Retry covers
transient transport failures, 408/409/425/429, and 5xx-like responses, but does
not retry ordinary bad requests.

Plaintext API keys are allowed in local config files, but are not written to manifests, events, provider results, status JSON, applied receipts, or CLI output.

## Quick Mock Flow

```bash
cd /Users/wangtingshen/Documents/llmwiki-engine

VAULT="$(mktemp -d -t llmwiki-vault)"
FIXTURE="$PWD/tests/fixtures/simple_project/mock"

uv run llmwiki init "$VAULT"

cp tests/fixtures/simple_project/raw_project_note.md "$VAULT/raw/"
RAW="$VAULT/raw/raw_project_note.md"

uv run llmwiki providers check "$VAULT"
uv run llmwiki ingest run "$VAULT" "$RAW" --fixture-dir "$FIXTURE" --slug manual

OP="$(ls -1 "$VAULT/.llmwiki/runs/ingest" | tail -n 1)"

uv run llmwiki ingest status "$VAULT" "$OP"
uv run llmwiki ingest status "$VAULT" "$OP" --verify
uv run llmwiki ingest apply "$VAULT" "$OP"
```

## Command Overview

```bash
llmwiki init <vault> [--profile project_basic]
llmwiki providers list
llmwiki providers check <vault> [--live]
llmwiki ingest raw-prepare-check <vault> <raw> [--skip-prepare|--force-prepare] [--json]
llmwiki ingest run <vault> <raw> [--fixture-dir PATH|--mock-fixture-dir PATH] [--profile NAME] [--slug TEXT] [--mode dev|standard] [--skip-prepare|--force-prepare] [--json]
llmwiki ingest run-next <vault> [--include-changed] [--dry-run] [--fixture-dir PATH|--mock-fixture-dir PATH] [--profile NAME] [--slug TEXT] [--mode dev|standard] [--skip-prepare|--force-prepare] [--json]
llmwiki ingest status <vault> [operation_id] [--verify] [--json]
llmwiki ingest inspect <vault> [operation_id] [--json]
llmwiki ingest raw-candidates <vault> [--all] [--limit N] [--json]
llmwiki ingest raw-import-url <vault> <url> [--title TEXT] [--output PATH] [--overwrite] [--dedupe-url|--no-dedupe-url] [--arxiv-html|--no-arxiv-html] [--timeout SECONDS] [--max-bytes BYTES] [--json]
llmwiki ingest raw-import-arxiv <vault> <query> [--limit N] [--dry-run] [--overwrite] [--dedupe-url|--no-dedupe-url] [--sort-by VALUE] [--sort-order VALUE] [--min-relevance-score N] [--timeout SECONDS] [--max-bytes BYTES] [--json]
llmwiki ingest resume <vault> <operation_id> [--from STEP] [--mock-fixture-dir PATH] [--skip-prepare|--force-prepare] [--mode dev|standard]
llmwiki ingest apply <vault> <operation_id>
llmwiki profile list
llmwiki profile validate <path_or_name>
llmwiki eval run <module> <dataset> [--output-root PATH]
llmwiki eval report <run>
```

## `llmwiki init`

Initialize a vault.

```bash
uv run llmwiki init "$VAULT"
uv run llmwiki init "$VAULT" --profile project_basic
```

Arguments and options:

- `VAULT`: vault path.
- `--profile`: profile to initialize with. Default: `project_basic`.

It creates:

```text
raw/
wiki/
.gitignore
.llmwiki/config.yaml
.llmwiki/profiles/
.llmwiki/runs/
.llmwiki/applied/
```

It does not run `git init`.

## `llmwiki providers list`

List public provider types.

```bash
uv run llmwiki providers list
```

Expected provider names:

```text
human
mock
openai_compatible
```

## `llmwiki providers check`

Check provider config without creating a run or writing a manifest.

```bash
uv run llmwiki providers check "$VAULT"
```

It checks:

- whether global and vault config can be read and merged;
- whether provider keys are limited to `default` and model-backed steps;
- whether `spec`, `endpoint`, `api_key`, `fixture_dir`, `max_retries`, and `retry_backoff_seconds` match the provider type;
- whether a mock provider lacks `fixture_dir`;
- whether `.llmwiki/` is tracked or staged by Git.

Common warning:

```text
mock provider ... has no fixture_dir; ingest will require --fixture-dir.
```

This is not a failure. It means config does not include the mock answer directory, so a real `ingest run` must pass:

```bash
--fixture-dir "$FIXTURE"
```

With `--live`, openai-compatible providers receive a minimal connectivity probe:

```bash
uv run llmwiki providers check "$VAULT" --live
```

`--live` does not write the vault, does not create a run, and is not a full ingest. Mock providers only check fixtures. Human providers do not make network requests. Real providers receive a small Chat Completions probe:

- `temperature=0`
- `max_tokens=512`
- preferred `response_format={"type": "json_object"}`

`max_tokens=512` is a completion cap, not a fixed cost. Thinking models usually stop earlier, but may consume up to that limit. If JSON mode is unsupported, the check falls back once with a prompt-only JSON probe and reports a warning if that succeeds. The live probe itself does not use transient retry, so provider checks stay quick. `temperature=0` does not promise deterministic behavior for every thinking model.

## `llmwiki ingest raw-prepare-check`

Preview whether `raw_prepare` will use deterministic passthrough or model cleanup before starting a real ingest. This command is read-only: it does not create an operation and does not modify the raw file.

```bash
uv run llmwiki ingest raw-prepare-check "$VAULT" "$RAW"
uv run llmwiki ingest raw-prepare-check "$VAULT" "$RAW" --json
```

It simulates the text after `raw_link_cleanup`, reads the current raw_prepare provider, then reuses the real `raw_prepare` fast-path rules to report:

- whether the current provider allows the deterministic fast path;
- whether auto mode will use the fast path;
- why auto mode would call the model;
- whether `--skip-prepare` is available, and which auto blockers it would suppress;
- whether the selected policy and auto policy are expected to call the raw_prepare provider;
- whether podcast/video transcript, translated transcript, timestamp/speaker-turn, or media embed risk was detected.

If the raw file has already been human-audited and is structurally clean, pass `--skip-prepare` to `ingest run` or to `resume` before `raw_prepare` to save model time. If the material is likely low-quality ASR or translated transcript text, keep auto mode or pass `--force-prepare` to explicitly request model cleanup.

## `llmwiki ingest run`

Start an ingest operation.

```bash
uv run llmwiki ingest run "$VAULT" "$RAW" --fixture-dir "$FIXTURE" --slug manual
```

Arguments and options:

- `VAULT`: vault path.
- `RAW`: raw file path. It must be under `VAULT/raw/`.
- `--fixture-dir PATH`: mock provider fixture directory. Real providers do not need it.
- `--mock-fixture-dir PATH`: force all model-backed steps to use `mock:fixture` with this fixture directory.
- `--profile NAME`: temporarily override the vault config profile.
- `--slug TEXT`: readable suffix for the operation ID.
- `--mode dev|standard`: run mode. Default: `dev`.
- `--skip-prepare`: use deterministic passthrough for eligible Markdown raw; hard blockers such as empty or non-Markdown raw still fall back to the configured `raw_prepare` provider.
- `--force-prepare`: force model `raw_prepare` cleanup and disable the deterministic fast path.

`--slug manual` only affects the operation ID, for example:

```text
ING-2026-06-01T085014Z-manual
```

It does not affect providers, models, or outputs.

## `llmwiki ingest status`

Show operation status.

```bash
uv run llmwiki ingest status "$VAULT" "$OP"
```

If `OP` is omitted, the latest operation is used:

```bash
uv run llmwiki ingest status "$VAULT"
```

Arguments and options:

- `VAULT`: vault path.
- `OPERATION_ID`: operation ID. Optional.
- `--verify`: recompute raw and artifact hashes without writing files.
- `--json`: print full JSON for scripts.

Examples:

```bash
uv run llmwiki ingest status "$VAULT" "$OP" --verify
uv run llmwiki ingest status "$VAULT" "$OP" --json
```

The table shows each step's review state, attempt count, last duration, total
duration, and provider. It also prints useful run artifact paths such as raw
cleanup audit files, review prompts, draft pages, diffs, and apply preview when
they exist.

## `llmwiki ingest resume`

Continue a failed or pending operation.

```bash
uv run llmwiki ingest resume "$VAULT" "$OP"
```

Default behavior:

- resume from the first failed or pending step;
- read current merged provider config for model-backed steps that will execute;
- do not rerun completed steps just because config changed.

Rerun from a step and downstream:

```bash
uv run llmwiki ingest resume "$VAULT" "$OP" --from source_digest
```

Resume can also override providers or raw preparation policy for steps that will rerun:

```bash
uv run llmwiki ingest resume "$VAULT" "$OP" --from raw_prepare --skip-prepare
uv run llmwiki ingest resume "$VAULT" "$OP" --from raw_prepare --force-prepare
uv run llmwiki ingest resume "$VAULT" "$OP" --mock-fixture-dir "$FIXTURE"
```

`--from STEP` will:

- verify the operation is not applied;
- run verification;
- parse and validate current provider config;
- delete the chosen step module directory and downstream module directories;
- clear current outputs for those steps;
- rerun from `STEP`.

Valid steps:

```text
raw_link_cleanup
raw_prepare
prepared_raw_review
source_digest
source_digest_review
source_duplicate_guard
candidate_resolution
wiki_context_snapshot
wiki_merge_planning
merge_plan_review
draft_rendering
draft_review
validation
apply_preview
```

## `llmwiki ingest review / approve / revise`

Inspect and resolve real review gates. The current gates are:

- `merge_plan_review`: review which pages will be written and why. If the plan
  contains `needs_human_decision`, or embedding retrieval finds medium/strong
  overlap while the plan still creates everything, the pipeline stops here.
  Inspect `wiki_context_snapshot/candidate_contexts.md` and
  `wiki_merge_planning/merge_decision_report.md`.
- `draft_review`: review the concrete page content. Update drafts and revised
  drafts require explicit approval.

Show review artifacts:

```bash
uv run llmwiki ingest review "$VAULT" "$OP" merge_plan_review
uv run llmwiki ingest review "$VAULT" "$OP" draft_review
```

If `merge_plan_review` is waiting, edit
`merge_plan_review/pending_merge_plan.json` in the operation directory and
change `needs_human_decision` to `create`, `update`, or `noop`, then approve:

```bash
uv run llmwiki ingest approve "$VAULT" "$OP" merge_plan_review
uv run llmwiki ingest resume "$VAULT" "$OP"
```

If `draft_review` is waiting, inspect `draft_review/review_prompt.md`,
`draft_rendering/diffs/`, and `draft_rendering/draft_pages/`, then approve:

```bash
uv run llmwiki ingest approve "$VAULT" "$OP" draft_review
uv run llmwiki ingest resume "$VAULT" "$OP"
```

Ask the model to regenerate the upstream content for a review gate:

```bash
uv run llmwiki ingest revise "$VAULT" "$OP" merge_plan_review
uv run llmwiki ingest revise "$VAULT" "$OP" draft_review
uv run llmwiki ingest resume "$VAULT" "$OP"
```

## `llmwiki ingest apply`

Write draft pages into `vault/wiki/`.

```bash
uv run llmwiki ingest apply "$VAULT" "$OP"
```

Plain `apply` is currently available only for `dev` operations. It does not touch Git and does not require the vault to be a Git repository.

`--commit` is disabled in this MVP:

```bash
uv run llmwiki ingest apply "$VAULT" "$OP" --commit
```

It fails before verify, preimage checks, manifest writes, receipt writes, or wiki writes. Target-scoped Git transaction support is a later follow-up.

## `profile` Commands

List built-in profiles:

```bash
uv run llmwiki profile list
```

Validate a profile:

```bash
uv run llmwiki profile validate project_basic
uv run llmwiki profile validate /path/to/profile
```

## `eval` Commands

Run a module eval:

```bash
uv run llmwiki eval run source_digest tests/fixtures/evals/source_digest
```

Arguments and options:

- `MODULE`: eval module name.
- `DATASET`: fixture dataset directory.
- `--output-root PATH`: eval output directory. Default: `eval_runs`.

Read an eval report:

```bash
uv run llmwiki eval report eval_runs/<run_id>.json
```

## Common Errors

`mock provider ... has no fixture_dir`

The mock provider has no fixture directory. Pass:

```bash
--fixture-dir "$FIXTURE"
```

or put it in config:

```yaml
providers:
  default:
    spec: mock:fixture
    fixture_dir: /path/to/mock
```

`Raw input must be inside the vault raw/ directory.`

The `RAW` argument is outside `VAULT/raw/`. Copy the file into `raw/` first.

`Raw input file does not exist: raw/...`

The `RAW` file does not exist.

`raw file hash changed`

The raw file changed after the operation was created. Resume/apply blocks for auditability.

`artifact hash changed`

A run artifact changed or was corrupted. Resume/apply blocks.

`Applied operations are immutable. Start a new operation instead.`

An applied operation cannot be resumed. Start a new `ingest run`.

`.llmwiki/ must not be tracked or staged by Git`

`.llmwiki/` is local runtime/config state and should not be committed. Remove it from tracked/staged Git state first.

`operation is incompatible with current MVP pipeline; rerun ingest`

The current MVP pipeline changed. Old development runs are not migrated; start a new ingest operation.

## Git Boundary

`.gitignore` prevents `.llmwiki/` from being added by default.

Plain `apply` does not check Git and does not commit.

`apply --commit` is disabled in this MVP and fails before writing. Future auto-apply/commit support will use a target-scoped Git transaction.
