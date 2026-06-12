# llmwiki-engine Lite

**Language:** English | [中文](README.zh-CN.md)

Fresh Lite branch for a fully automatic LLM-Wiki ingest core.

The current Lite implementation is a fully automatic ingest pipeline:

```text
raw_file
-> source_digest
-> wiki_snapshot
-> candidate_pages
-> candidate_contexts
-> merge_plan
-> composition_plan
-> final_pages
-> validation
-> knowledge_write
-> source_record_write
-> index_log_write
-> embedding_cache_refresh
-> receipt
```

There is no manual review, no manual apply, and no raw prepare step in this
branch. The `raw/` file remains the durable evidence layer; generated wiki
pages are a compiled navigation and understanding layer.

`wiki_snapshot` freezes the current knowledge pool and synchronizes the
current-state page embedding cache. `candidate_contexts` then retrieves the
top related old pages for each generated candidate page. The cache is
path-current: the same wiki path has one latest vector record, and stale
records are pruned during cache sync.

## Install From GitHub Release

Download the macOS/Linux release bundle from GitHub:

```bash
curl -L -O https://github.com/TingshenWang/llmwiki-engine/releases/download/v0.2.0/llmwiki-engine-0.2.0-macos-linux.tar.gz
curl -L -O https://github.com/TingshenWang/llmwiki-engine/releases/download/v0.2.0/llmwiki-engine-0.2.0-macos-linux.tar.gz.sha256
```

Verify the archive:

```bash
shasum -a 256 -c llmwiki-engine-0.2.0-macos-linux.tar.gz.sha256
```

Unpack and run the installer:

```bash
tar -xzf llmwiki-engine-0.2.0-macos-linux.tar.gz
cd llmwiki-engine-0.2.0-macos-linux
./install.sh
```

The installer creates a virtual environment under
`~/.local/share/llmwiki-engine` and symlinks the CLI to
`~/.local/bin/llmwiki`. If your shell cannot find `llmwiki`, add this to
`~/.zshrc` or `~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Custom install directory:

```bash
./install.sh /path/to/install/llmwiki-engine
```

## Quick Start

Configure a real provider. For example, DeepSeek:

```bash
mkdir -p ~/.llmwiki
cat > ~/.llmwiki/config.yaml <<'YAML'
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: your-deepseek-api-key
    max_tokens: 262144
    timeout_seconds: 300
YAML
```

Create a vault, put Markdown files under `raw/`, and run ingest:

```bash
llmwiki init ~/my-llmwiki-vault
mkdir -p ~/my-llmwiki-vault/raw
cp /path/to/example.md ~/my-llmwiki-vault/raw/

llmwiki ingest run ~/my-llmwiki-vault raw/example.md
llmwiki ingest status ~/my-llmwiki-vault --verify
llmwiki ingest raw-candidates ~/my-llmwiki-vault --all
```

Generated knowledge pages are written under `wiki/`. The original file under
`raw/` remains the durable evidence layer.

## Release Build

Build a macOS/Linux release bundle from the committed source tree:

```bash
scripts/build_release.sh
```

The script writes:

- `dist/releases/llmwiki-engine-<version>-macos-linux.tar.gz`
- `dist/releases/llmwiki-engine-<version>-macos-linux.tar.gz.sha256`

The bundle contains the universal wheel, sdist, install script, and English
and Chinese READMEs. The install script creates a venv and installs
`llmwiki-engine[embeddings]` so candidate recall uses local Qwen embeddings.

## Provider Config

Lite Ingest requires a real OpenAI-compatible chat completions provider for
model-backed steps. If no real provider is configured, `llmwiki ingest run`
fails before writing any wiki pages. Configure the default provider or
step-specific providers in `~/.llmwiki/config.yaml` or in the vault-level
`.llmwiki/config.yaml`. A vault-level config overrides the global config.

```yaml
providers:
  default:
    spec: openai_compatible:gpt-4.1-mini
    endpoint: https://api.openai.com/v1/chat/completions
    api_key: your-openai-api-key
    json_mode: json_schema
    json_schema_strict: false
    temperature: 0
    max_tokens: 262144
    max_retries: 2
```

For DeepSeek, Lite automatically uses `json_object` mode because DeepSeek's
current JSON Output guide documents `response_format: {"type": "json_object"}`
rather than JSON Schema response format:

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: your-deepseek-api-key
    max_tokens: 262144
```

Supported model-backed step keys are `source_digest`, `candidate_pages`,
`merge_plan`, `composition_plan`, and `final_pages`. Provider prompts and
response schemas are written under each step's `model_calls/` directory; API
keys are not written to artifacts or receipts.

If an older vault already has `.llmwiki/config.yaml` with `local:heuristic`,
replace that file with the real provider config or delete it so the global
config can take effect. `local:heuristic` is rejected by `llmwiki providers
check --live` and by normal ingest runs.

## Local Qwen Embeddings

New vaults default to local Qwen embeddings through Sentence Transformers.
`candidate_contexts` requires this real embedding backend for top-k retrieval;
it does not fall back to lexical, title, exact, or hashing recall.

```json
{
  "embedding": {
    "enabled": true,
    "backend": "sentence_transformers",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "dimensions": 1024,
    "top_k_pages": 5,
    "max_page_chars": 6000,
    "max_query_chars": 4000,
    "batch_size": 4,
    "query_prompt_name": "query"
  }
}
```

Lite no longer ships a hashing embedding backend. Non-`sentence_transformers`
embedding configs are rejected before candidate recall.
