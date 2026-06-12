# llmwiki-engine Lite

Fresh Lite branch for a fully automatic LLM-Wiki ingest core.

The first implementation target is a deterministic local pipeline:

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
    api_key_env: DEEPSEEK_API_KEY
    max_tokens: 262144
    timeout_seconds: 300
YAML

export DEEPSEEK_API_KEY="your-api-key"
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

## 中文安装与使用

从 GitHub Release 下载 macOS/Linux 通用包：

```bash
curl -L -O https://github.com/TingshenWang/llmwiki-engine/releases/download/v0.2.0/llmwiki-engine-0.2.0-macos-linux.tar.gz
curl -L -O https://github.com/TingshenWang/llmwiki-engine/releases/download/v0.2.0/llmwiki-engine-0.2.0-macos-linux.tar.gz.sha256
```

校验下载文件：

```bash
shasum -a 256 -c llmwiki-engine-0.2.0-macos-linux.tar.gz.sha256
```

解压并运行安装脚本：

```bash
tar -xzf llmwiki-engine-0.2.0-macos-linux.tar.gz
cd llmwiki-engine-0.2.0-macos-linux
./install.sh
```

默认安装位置是 `~/.local/share/llmwiki-engine`，命令入口会链接到
`~/.local/bin/llmwiki`。如果终端提示找不到 `llmwiki`，把下面这一行加到
`~/.zshrc` 或 `~/.bashrc`：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

也可以指定安装目录：

```bash
./install.sh /path/to/install/llmwiki-engine
```

配置真实模型 provider。DeepSeek 示例：

```bash
mkdir -p ~/.llmwiki
cat > ~/.llmwiki/config.yaml <<'YAML'
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key_env: DEEPSEEK_API_KEY
    max_tokens: 262144
    timeout_seconds: 300
YAML

export DEEPSEEK_API_KEY="你的 API Key"
```

创建 vault，把 Markdown 原始材料放进 `raw/`，然后运行 Ingest：

```bash
llmwiki init ~/my-llmwiki-vault
mkdir -p ~/my-llmwiki-vault/raw
cp /path/to/example.md ~/my-llmwiki-vault/raw/

llmwiki ingest run ~/my-llmwiki-vault raw/example.md
llmwiki ingest status ~/my-llmwiki-vault --verify
llmwiki ingest raw-candidates ~/my-llmwiki-vault --all
```

生成的知识页会写入 `wiki/`；`raw/` 下的原文会保留为可追溯证据层。第一次使用
Qwen embedding 时，`sentence-transformers` 可能需要从 Hugging Face 下载模型；
如果本机已有缓存，会直接复用。

## Release Build

Build a macOS/Linux release bundle from the committed source tree:

```bash
scripts/build_release.sh
```

The script writes:

- `dist/releases/llmwiki-engine-<version>-macos-linux.tar.gz`
- `dist/releases/llmwiki-engine-<version>-macos-linux.tar.gz.sha256`

The bundle contains the universal wheel, sdist, install script, README, and
Lite requirements document. The install script creates a venv and installs
`llmwiki-engine[embeddings]` so candidate recall uses local Qwen embeddings.

## Provider Config

New vaults default to deterministic local heuristics in `.llmwiki/config.yaml`:

```yaml
providers:
  default:
    spec: local:heuristic
```

To use a real OpenAI-compatible chat completions endpoint for model-backed
steps, configure the default provider or a step-specific provider:

```yaml
providers:
  default:
    spec: openai_compatible:gpt-4.1-mini
    endpoint: https://api.openai.com/v1/chat/completions
    api_key_env: OPENAI_API_KEY
    json_mode: json_schema
    json_schema_strict: false
    temperature: 0
    max_tokens: 262144
    max_retries: 2
  merge_plan:
    spec: local:heuristic
```

For DeepSeek, Lite automatically uses `json_object` mode because DeepSeek's
current JSON Output guide documents `response_format: {"type": "json_object"}`
rather than JSON Schema response format:

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key_env: DEEPSEEK_API_KEY
    max_tokens: 262144
```

Supported model-backed step keys are `source_digest`, `candidate_pages`,
`merge_plan`, `composition_plan`, and `final_pages`. Provider prompts and
response schemas are written under each step's `model_calls/` directory; API
keys are not written to artifacts or receipts.

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

The hashing backend remains available only for low-level deterministic tests.
Ingest candidate recall rejects hashing so the fifth step always uses
embedding vectors.
