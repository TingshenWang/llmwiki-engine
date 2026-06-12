# llmwiki-engine Lite

**语言：** [English](README.md) | 中文

这是一个从零重写的 Lite 分支，核心目标是实现全自动 LLM-Wiki Ingest。

当前 Lite 实现是一条全自动 Ingest 编译流水线：

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

这个分支没有人工 review、没有人工 apply，也没有 raw prepare 步骤。`raw/`
文件始终保留为可追溯证据层；生成的 wiki 页面是编译后的导航和理解层。

`wiki_snapshot` 会冻结当前知识池，并同步当前状态的页面 embedding 缓存。
`candidate_contexts` 会为每个候选页面召回 top 相关旧页面。缓存是 path-current
语义：同一个 wiki 路径只保留最新向量记录，缓存同步时会清理陈旧记录。

## 从 GitHub Release 安装

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

## 快速开始

先配置真实模型 provider。DeepSeek 示例：

```bash
mkdir -p ~/.llmwiki
cat > ~/.llmwiki/config.yaml <<'YAML'
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: 你的 DeepSeek API Key
    max_tokens: 262144
    timeout_seconds: 300
YAML
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

生成的知识页会写入 `wiki/`；`raw/` 下的原文会保留为可追溯证据层。

## Release 打包

从已提交源码构建 macOS/Linux release 包：

```bash
scripts/build_release.sh
```

脚本会生成：

- `dist/releases/llmwiki-engine-<version>-macos-linux.tar.gz`
- `dist/releases/llmwiki-engine-<version>-macos-linux.tar.gz.sha256`

包内包含 universal wheel、sdist、安装脚本、英文 README 和中文 README。安装脚本会
创建 venv 并安装 `llmwiki-engine[embeddings]`，确保候选页召回使用本地 Qwen
embedding。

## Provider 配置

Lite Ingest 的模型步骤必须使用真实 OpenAI-compatible chat completions
provider。如果没有配置真实 provider，`llmwiki ingest run` 会在写入任何 wiki 页面前
直接失败。可以在 `~/.llmwiki/config.yaml` 配置默认 provider，也可以在 vault 内的
`.llmwiki/config.yaml` 配置默认或分步骤 provider。vault 内配置会覆盖全局配置。

```yaml
providers:
  default:
    spec: openai_compatible:gpt-4.1-mini
    endpoint: https://api.openai.com/v1/chat/completions
    api_key: 你的 OpenAI API Key
    json_mode: json_schema
    json_schema_strict: false
    temperature: 0
    max_tokens: 262144
    max_retries: 2
```

DeepSeek 示例：

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: 你的 DeepSeek API Key
    max_tokens: 262144
```

Lite 会对 DeepSeek 自动使用 `json_object` 模式，因为 DeepSeek 当前 JSON Output
文档使用的是 `response_format: {"type": "json_object"}`，不是 JSON Schema
response format。

支持模型调用的步骤 key 包括 `source_digest`、`candidate_pages`、`merge_plan`、
`composition_plan` 和 `final_pages`。每个步骤的 prompt 和 response schema 会写入
对应的 `model_calls/` 目录；API key 不会写入 artifact 或 receipt。

如果旧 vault 里已经有 `.llmwiki/config.yaml`，并且内容还是 `local:heuristic`，
请直接把那份文件替换成真实 provider 配置，或删除它，让全局配置生效。
`llmwiki providers check --live` 和正常 Ingest 都会拒绝 `local:heuristic`。

## 本地 Qwen Embeddings

新 vault 默认通过 Sentence Transformers 使用本地 Qwen embedding。`candidate_contexts`
要求使用真实 embedding backend 做 top-k 召回；不会回退到词面、标题、精确匹配或
hashing 召回。

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

Lite 不再内置 hashing embedding backend。非 `sentence_transformers` 的
embedding 配置会在候选页召回前被拒绝。
