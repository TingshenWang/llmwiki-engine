# llmwiki CLI Reference

This document is the command compass for `llmwiki-engine`. It explains common commands, parameters, config rules, Git boundaries, and frequent errors.

## Core Concepts

`vault`

A local knowledge base directory. `llmwiki init` creates `raw/`, `wiki/`, and `.llmwiki/` inside it.

`raw`

Source material to ingest. The `RAW` argument for `ingest run` must point to a file under `vault/raw/`.

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
claim_extraction.json
page_planning.json
```

`provider`

The execution source for model-backed steps. Public provider specs are:

- `mock:fixture`: read deterministic fixture files for tests.
- `human`: manual handoff placeholder.
- `openai_compatible:<model>`: call a Chat Completions-compatible API.

`apply`

Write draft pages from a run into `vault/wiki/`.

`apply --commit`

Write `vault/wiki/` and create a Git commit. The commit only includes `wiki/`, not unrelated files the user already staged.

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
claim_extraction
page_planning
```

`default` is the fallback provider. A concrete step overrides `default`.

Mock config example:

```yaml
profile: project_basic
providers:
  default:
    spec: mock:fixture
    fixture_dir: /path/to/mock
```

OpenAI-compatible config example:

```yaml
profile: project_basic
providers:
  default:
    spec: openai_compatible:deepseek-chat
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: sk-...
  page_planning:
    spec: openai_compatible:stronger-planner
    endpoint: https://example.test/v1/chat/completions
    api_key: sk-...
```

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
llmwiki ingest run <vault> <raw> [--fixture-dir PATH] [--profile NAME] [--slug TEXT] [--mode dev|standard]
llmwiki ingest status <vault> [operation_id] [--verify] [--json]
llmwiki ingest resume <vault> <operation_id> [--from STEP] [--mode dev|standard]
llmwiki ingest apply <vault> <operation_id> [--commit]
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
- whether `spec`, `endpoint`, `api_key`, and `fixture_dir` match the provider type;
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

`--live` does not write the vault and does not create a run. Mock providers only check fixtures. Human providers do not make network requests.

## `llmwiki ingest run`

Start an ingest operation.

```bash
uv run llmwiki ingest run "$VAULT" "$RAW" --fixture-dir "$FIXTURE" --slug manual
```

Arguments and options:

- `VAULT`: vault path.
- `RAW`: raw file path. It must be under `VAULT/raw/`.
- `--fixture-dir PATH`: mock provider fixture directory. Real providers do not need it.
- `--profile NAME`: temporarily override the vault config profile.
- `--slug TEXT`: readable suffix for the operation ID.
- `--mode dev|standard`: run mode. Default: `dev`.

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
uv run llmwiki ingest resume "$VAULT" "$OP" --from claim_extraction
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
raw_prepare
raw_index
extraction_windows
claim_extraction
page_planning
draft_rendering
validation
apply_preview
```

## `llmwiki ingest apply`

Write draft pages into `vault/wiki/`.

```bash
uv run llmwiki ingest apply "$VAULT" "$OP"
```

Plain `apply` does not touch Git and does not require the vault to be a Git repository.

With `--commit`:

```bash
uv run llmwiki ingest apply "$VAULT" "$OP" --commit
```

`apply --commit` will:

- require the vault to be a Git repository;
- check before writing that `.llmwiki/` is not tracked or staged;
- write `wiki/`;
- create a commit containing only the `wiki/` path.

If unrelated files were already staged, `apply --commit` does not commit them and does not unstage them.

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
uv run llmwiki eval run page_planning tests/fixtures/evals/page_planning
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

`Unsupported manifest schema_version`

The current code does not support that old run manifest. This stage does not migrate old in-flight operations. Start a new run.

## Git Boundary

`.gitignore` prevents `.llmwiki/` from being added by default.

`apply --commit` limits this commit to `wiki/`.

They are separate protections:

```text
.gitignore       prevents default git add
apply --commit   limits this commit path
```

Plain `apply` does not check Git and does not commit.

`apply --commit` requires a Git repository and checks before writing that `.llmwiki/` is not tracked or staged.
