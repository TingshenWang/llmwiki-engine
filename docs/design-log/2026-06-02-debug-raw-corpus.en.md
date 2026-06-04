# Development Debug Raw Corpus Design Log

English | [中文](2026-06-02-debug-raw-corpus.zh-CN.md)

Date: 2026-06-02
Status: Accepted as the current development debug corpus

This note records the fixed raw corpus currently used to develop and debug
`llmwiki-engine`. It is not a committed fixture, and it does not copy source
content into this repository. It records the three local raw files used for
real ingest debugging and explains why they cover the next round of page
specification, profile, validator, and eval design.

## Decision

The current development debug corpus uses three AIAgent-PM raw files:

- `/Users/wangtingshen/Documents/AIAgent-PM/raw/How Anthropic产品团队如何以超快速度开发产品 - Cat Wu访谈.md`
- `/Users/wangtingshen/Documents/AIAgent-PM/raw/Scaling Managed Agents - 将大脑与双手解耦（中文翻译）.md`
- `/Users/wangtingshen/Documents/AIAgent-PM/raw/Stop Applying to AI PM Jobs Until You Watch This（中文翻译）.md`

Development debugging should prioritize these materials for realistic behavior
checks instead of relying only on the simplified project note under
`tests/fixtures/simple_project/`.

## Why These Three Matter

### Cat Wu Interview

This file is a long video transcript with two speakers. The model needs to
infer which speaker is the host and which speaker is responding. It tests:

- whether `raw_prepare` can handle long transcripts, speech noise, and speaker attribution;
- whether `source_digest` can extract stable knowledge candidates from scattered dialogue;
- whether `candidate_resolution` can map one interview into several page types;
- whether wiki pages can organize entities, concepts, designs, comparisons, and events instead of producing loose summaries.

This file is the default single-source debug raw.

### Scaling Managed Agents

This file discusses a future Agent concept and is useful for more abstract
design and concept consolidation. It covers:

- Design pages for future Agent architecture, plans, components, and boundaries;
- Concept pages for managed agents, brain/hands decoupling, and related ideas;
- Comparison pages for different Agent working modes or product forms;
- cross-source updates where later raw files enrich an existing concept.

### Stop Applying To AI PM Jobs

This file is a podcast/interview-style source focused on AI PM role judgment.
It tests:

- cleanup and structuring for video or podcast transcripts;
- splitting opinions, role requirements, skill models, and action advice;
- linking people, organizations, roles, concepts, and comparisons;
- long-term knowledge accumulation around the AIAgent-PM domain.

## Debugging Mode

Development should observe two run modes:

- Single-source ingest: ingest one raw file at a time, then inspect the source page, page split, candidate resolution, and draft usability.
- Sequential multi-source ingest: ingest the three raw files one by one into the same vault, then inspect how existing pages are updated, enriched, or kept from being duplicated.

Multi-source debugging still keeps the operation boundary at one raw per ingest.
This preserves operation-level manifests, artifacts, resume behavior, and apply
auditing.

## Implications For Page Specification

This corpus exposes the main reason the current generated pages are not yet
usable: page shape has not been productized. Future profiles should not only
describe page types, directories, and templates. They also need to describe:

- operation intent: whether an ingest absorbs an interview, organizes opinions, records a design, or updates existing pages;
- page archetype: what information job each page type performs;
- section contract: which sections must appear in each page type and what knowledge job each section performs;
- link/update policy: when to create a new page, update an existing page, or only add links;
- validation/eval: whether a page is useful, not only whether claim ids exist.

These three raw files are the primary development references for future page
product specifications and eval cases.
