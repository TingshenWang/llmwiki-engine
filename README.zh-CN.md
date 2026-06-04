# llmwiki-engine

[English](README.md) | 中文

`llmwiki-engine` 是一个面向模块化知识编译的 Python 3.11+ CLI 和引擎。
它的目标是把噪声较多的原始材料编译进本地 wiki，同时让每个步骤都可以
独立测试、评估和优化。

完整命令、参数、配置、Git 边界和常见错误说明见
[docs/cli-reference.zh-CN.md](docs/cli-reference.zh-CN.md)。英文版见
[docs/cli-reference.en.md](docs/cli-reference.en.md)。

当前可运行路径是 M3 Ingest MVP。它会先把目标 `raw/` 文件里的 Obsidian 文本
wikilink 原地规范化，再把 noisy raw material 整理成规范的
`raw_prepare/prepared.md`，然后生成 `source_digest`、做来源重复检查、基于全文和
digest 做候选页面规划、冻结 wiki context snapshot、用模型判断
create/update/noop/needs-human-decision、生成可审核草稿和 apply preview：

```bash
llmwiki init /path/to/vault --profile project_basic
llmwiki ingest run /path/to/vault raw/project_note.md --fixture-dir tests/fixtures/simple_project/mock
llmwiki providers check /path/to/vault
llmwiki ingest status /path/to/vault <operation_id>
llmwiki ingest apply /path/to/vault <operation_id>
```

默认运行时是确定性的：模型相关模块使用 fixture 驱动的 `MockProvider` 输出，
这样可以在接入真实模型前测试 pipeline、artifacts、validators、logs、profiles
和 drafts。

## 设计原则

- 规范 artifacts 使用 JSON/JSONL。
- `raw/` 是规范化材料层。MVP 只展开 `[[Page]]`、`[[Page|Alias]]` 这类
  Obsidian 文本 wikilink；网页链接、Markdown 链接、媒体 embed 和代码块保持不变。
- `raw_prepare` 将规范化后的 raw 转换为下游知识编译使用的 canonical prepared raw。
- `source_digest` 是单篇 raw 的完整消化文件，用于人工审核候选知识。
- `candidate_resolution` 基于 approved prepared 全文和 digest 规划 wiki 选题；
  `wiki_merge_planning` 基于冻结 snapshot 判断
  create/update/noop/needs-human-decision。
- dev 模式支持 update 整页草稿替换，但必须经过显式 draft review 才能
  validation/apply。
- 知识页使用确定性的 `Related` wikilink；source 页不进入 Obsidian 知识图谱。
- 人类可编辑的 profile 使用 YAML。
- 人类 review 和 drafts 使用 Markdown。
- Provider 可以按模块配置。
- Validator 是硬门禁；未来的 LLM critic 保留为语义审查者。

Provider 使用 YAML 配置。单个 vault 可以使用 `.llmwiki/config.yaml`，共享的
provider 默认配置可以放在 `~/.llmwiki/config.yaml`。每个 operation 会记录本次执行
解析出的 sanitized provider context；每一步实际用了什么 provider 由 step attempt
记录。Provider 记录只保留 `spec`、`endpoint`、
`fixture_dir`；明文 API key 只允许存在于 config 文件，不会写入运行 artifacts、
status JSON、applied receipt 或 CLI 输出。

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

正式运行前可以检查 provider 配置：

```bash
llmwiki providers check /path/to/vault
llmwiki providers check /path/to/vault --live
```

`--live` 会发起一次小型真实模型探针。面向 thinking 模型时，它使用
`max_tokens=512` 的 completion 上限，并优先使用 JSON mode；如果 API 明确不支持
JSON mode，则 fallback 到 prompt-only JSON probe，并给出 warning。

普通 `llmwiki ingest resume` 会为仍需执行的模型步骤读取当前合并后的 provider
配置。已经完成的 step 不会因为 config 改变自动重跑。

如果需要从某一步开始，用当前配置重跑该 step 及下游：

```bash
llmwiki ingest resume /path/to/vault <operation_id> --from source_digest
```
