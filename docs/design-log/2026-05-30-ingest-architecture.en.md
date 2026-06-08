# Ingest Architecture Design Log

English | [中文](2026-05-30-ingest-architecture.zh-CN.md)

Date: 2026-05-30
Status: Updated by the 2026-06-02 M1 ingest redesign; the modular CLI, artifact, resume, and apply boundaries still hold

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
  model-backed resume steps read current provider config while completed steps
  remain unchanged unless explicitly rerun with `--from`.
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

`.llmwiki/` is local runtime, configuration, and audit state. It may contain
plaintext provider credentials in config, so `init` ignores the whole directory
with `.gitignore`. Git-backed review and rollback should focus on `wiki/` and
source material, not run cache.

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

## Source Digest Instead Of Semantic Aggregation

The earlier `semantic_aggregation` idea and strict evidence-window flow were
removed from the main path.

The concern was that an aggregation layer could pretend to define knowledge
units while actually making lossy or mistaken grouping decisions. For example,
a conversation might group one question and answer too narrowly, or a transcript
might combine unrelated fragments because they are adjacent.

The current replacement is `source_digest`.

Source digest is a single-raw digestion artifact for human review, not a final
wiki page. It asks the model to list entity, concept, design, comparison, and
open-question candidates as completely as possible before later
resolution/merge steps decide which candidates update existing pages and which
create new pages.

```text
prepared raw -> source_digest -> candidate_resolution -> wiki_merge_planning
```

Strict evidence chains are no longer a hard main-path contract. As long as the
source page links back to raw, reviewers can return to the source when needed.
The MVP review surface is whether the digest is complete, candidate resolution
is reasonable, and the final drafts are useful.

## Current Linear Pipeline

The current MVP pipeline is:

```text
raw_prepare
prepared_raw_review
source_digest
source_digest_review
candidate_resolution
wiki_merge_planning
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

`resume` defaults to the first failed or pending step. For the current
execution, it reads the merged provider config and records a new sanitized
provider context. Step attempts remain the authority for which provider each
step actually used. Completed steps are not rerun just
because config changed. `resume --from STEP` first validates the current provider
execution context, then deletes the chosen step's module directory and
downstream module directories, marks those steps pending, and reruns from there.

Applied operations cannot be resumed. Raw drift, required artifact drift, or
apply preimage drift must block execution.

## Provider Runtime Direction

Providers should be selected per module, not globally. This enables cheap local
models for preparation, reserved future critique steps, and stronger API models
for harder extraction or planning tasks.

The MVP provider set is:

- `MockProvider` for deterministic fixtures and tests;
- `OpenAICompatibleProvider` for hosted or routed Chat Completions-compatible APIs.

Every model-backed step should use structured calls with schema validation,
bounded repair, cost and latency capture, and failed-output diagnostics.
Provider context records are the only provider execution records. They store
only non-secret runtime fields such as `spec`, `endpoint`, and `fixture_dir`.
Plaintext API keys are allowed in local config files, but they must not be
recorded in manifests, events, provider results, receipts, status JSON, or CLI
output. Real provider execution uses an in-memory `ProviderExecutionContext`
instead of reconstructing credentials from the manifest.

If providers beyond `openai_compatible` later need live checks, the live-check
interface should become explicit, for example with a shared protocol and result
object. JSON mode fallback should remain specific to OpenAI-compatible probes so
dynamic `check_live(...)` calls do not become brittle as provider types grow.

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

## Closed Follow-ups From Review

- An automated end-to-end API key regression test now verifies that keys do not
  spread into manifests, events, provider results, status JSON, applied receipts,
  or CLI output.
- Module directory reads and writes now derive from `StepSpec.output_dir`; production
  code no longer relies on `run_dir / step_name` matching the current directory names.
- Step and eval support lists now derive from `StepSpec`, including `resume --from`
  help text and eval module validation.
- Provider config errors now include global/vault source information and the
  provider key that produced the error.
- CLI tests now verify that old or future manifest schemas such as `operation_manifest.v3`
  and `operation_manifest.v5` are rejected clearly.

## Known Follow-ups From Review

- `resume --help` tests can be tightened to fail if stale or unsupported step
  names are accidentally mixed into the generated help text.
- Provider config safety can later add stronger detection or redaction for
  secrets mistakenly pasted into unknown provider keys, unsupported field names,
  or endpoint fields beyond the current MVP endpoint guards.

## Open Questions

- How strict should `raw_prepare` be when the model detects uncertain cleanup?
- Should prepared raw require an optional human approval gate before extraction?
- How should non-Markdown formats normalize into the same prepared raw contract?
- How should the source digest candidate schema evolve before deduplication,
  candidate resolution, and merge planning become more sophisticated?
- Which modules deserve local-model defaults, and which should default to
  stronger hosted providers?
