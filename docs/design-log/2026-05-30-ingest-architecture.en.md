# Ingest Architecture Design Log

English | [中文](2026-05-30-ingest-architecture.zh-CN.md)

Date: 2026-05-30
Status: Accepted for current MVP direction

This note records the design decisions behind the current `llmwiki-engine`
Ingest architecture. It is not a transcript. It is the durable project memory
for why the engine moved from an agent-controlled skill flow toward a modular,
auditable, and resumable CLI workflow.

## Context

`llmwiki-engine` is meant to support incremental knowledge compilation for a
local wiki. Inputs may be noisy and inconsistent: Markdown notes, video
transcripts, PDF or DOCX exports, HTML clips, tables, diaries, and other
user-authored or user-studied material.

The earlier skill-driven approach made the LLM repeatedly read procedural
instructions and control the whole flow. That was expensive, hard to test, easy
to derail, and difficult to optimize module by module. The new direction is to
make the engine itself own the workflow while using LLM calls only for bounded
model tasks with explicit inputs, schemas, validation, artifacts, and review.

## Goals

- Keep the core Ingest flow runnable from a CLI.
- Give every step fixed inputs, outputs, logs, and artifacts.
- Make each module independently testable, evaluable, and optimizable.
- Keep the formal wiki small and clean.
- Keep local run cache separate from committed wiki output.
- Make `resume` and `apply` safe against raw, artifact, and wiki target drift;
  provider config changes require an explicit refresh.
- Allow providers to differ by module so cost and quality can be tuned locally.
- Avoid making an agent the default runtime dependency.

## Storage Boundary

The vault layout separates source material, formal wiki pages, and local run
state:

```text
<vault>/
  raw/
  wiki/
  .llmwiki/
    config.yaml
    profiles/
    applied/operations.jsonl
    runs/ingest/<operation_id>/
```

`wiki/` is only for final pages. Run artifacts, drafts, previews, model calls,
and manifests stay under `.llmwiki/runs/`.

`.llmwiki/runs/` is local cache. `.llmwiki/config.yaml`, profiles, and applied
receipts can be committed because they are small, intentional, and useful for
auditing workflow decisions.

## Original Raw And Prepared Raw

Original raw material is not automatically treated as clean truth. It may
contain transcript artifacts, repeated bilingual lines, timestamp noise,
recognition errors, formatting damage, or irrelevant wrapper text.

The current design introduces `raw_prepare` as the first model-backed bounded
step. It converts original raw material into canonical prepared raw:

```text
original raw -> raw_prepare -> raw_prepare/prepared.md
```

Downstream extraction and wiki compilation treat prepared raw as the factual
input for the run. This is a deliberate trust boundary: the prepared raw must be
auditable, but once accepted it is usually more valuable than forcing every
downstream step to reason over the original noisy input.

The preparation step records:

- the prepared Markdown text;
- operations applied during cleaning;
- uncertain or risky items;
- a human-readable preparation review;
- structured artifacts for validation and replay.

## Image Handling

Images are intentionally out of scope for the core MVP reasoning path.

Markdown, HTML, and DOCX may preserve explicit image links or extracted image
assets later, but image contents should not silently enter wiki knowledge unless
a human or an explicit future OCR/vision module turns them into text.

For PDFs, embedded images can be dropped in the initial version. Users can add a
separate raw note when an image carries important knowledge.

This avoids turning the Ingest engine into an OCR and vision system before the
text compilation path is stable.

## Extraction Windows Instead Of Semantic Aggregation

The earlier `semantic_aggregation` idea was removed from the main path.

The concern was that an aggregation layer could pretend to define knowledge
units while actually making lossy or mistaken grouping decisions. For example,
a conversation might group one question and answer too narrowly, or a transcript
might combine unrelated fragments because they are adjacent.

The replacement is `extraction_windows`.

Extraction windows are engineering context windows, not semantic knowledge
units. They exist to give claim extraction enough local context while preserving
traceability back to prepared raw spans.

```text
prepared raw -> raw_index -> extraction_windows -> claim_extraction
```

Claims must point to a `source_window_id`, and their evidence must bind back to
prepared raw spans. This makes extraction testable without pretending that
window boundaries are conceptual boundaries.

## Current Linear Pipeline

The current MVP pipeline is:

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

`validation` is an outputless gate step. It validates upstream artifacts and
does not create a `validation/` module directory.

`apply` remains an explicit command after preview. It verifies the run again,
checks target preimage hashes, writes final wiki pages, and appends an applied
receipt.

## Manifest, Status, Resume, Apply

The manifest is the local workflow contract. It records raw bindings, step
attempts, current artifact references, provider context records, and run
status.

`status` is for both humans and machines:

- default output should be clean and action-oriented;
- `--json` should expose complete structured state;
- `--verify` should recompute integrity checks without writing files.

`resume` defaults to the first failed or pending step and reuses the provider
context already recorded in the manifest. It does not reread the current
`.llmwiki/config.yaml`. `resume --from STEP` deletes the chosen step's module
directory and downstream module directories, marks those steps pending, and
reruns from there. `resume --from STEP --refresh-providers` first resolves the
current provider config and records a new provider context for the rerun range.

Applied operations cannot be resumed. Raw drift, required artifact drift, or
apply preimage drift must block execution.

## Provider Runtime Direction

Providers should be selected per module, not globally. This enables cheap local
models for preparation, reserved future critique steps, and stronger API models
for harder extraction or planning tasks.

The intended provider direction is:

- `MockProvider` for deterministic fixtures and tests;
- `OpenAIProvider` for hosted model calls;
- `OllamaProvider` for local models;
- `LocalHTTPProvider` for custom local services;
- `HumanProvider` for explicit manual handoff points.

Every model-backed step should use structured calls with schema validation,
bounded repair, cost and latency capture, and failed-output diagnostics.
Provider context records are the only provider execution snapshots. They store
standard runtime fields such as `spec`, `endpoint`, `api_key_env`, and
`fixture_dir`; plaintext API keys must not be recorded.

## Test And Evaluation Direction

The engine should continue to grow through module-level tests and evals rather
than only end-to-end demos.

Important checks include:

- schema valid rate;
- parse success rate;
- repair success rate;
- evidence quote validity;
- page type accuracy;
- provider cost and latency;
- resume safety;
- apply preimage safety.

The smoke path remains useful as a fast confidence check:

```text
init -> ingest run -> status -> status --verify -> apply
```

It does not replace unit tests, regression tests, or module evals.

## Open Questions

- How strict should `raw_prepare` be when the model detects uncertain cleanup?
- Should prepared raw require an optional human approval gate before extraction?
- How should non-Markdown formats normalize into the same prepared raw contract?
- What is the right claim schema before deduplication and page planning become
  more sophisticated?
- Which modules deserve local-model defaults, and which should default to
  stronger hosted providers?
