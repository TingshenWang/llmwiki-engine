# llmwiki-engine

[English](README.md) | 中文

`llmwiki-engine` 是一个面向模块化知识编译的 Python 3.11+ CLI 和引擎。
它的目标是把噪声较多的原始材料编译进本地 wiki，同时让每个步骤都可以
独立测试、评估和优化。

完整命令、参数、配置、Git 边界和常见错误说明见
[docs/cli-reference.zh-CN.md](docs/cli-reference.zh-CN.md)。英文版见
[docs/cli-reference.en.md](docs/cli-reference.en.md)。

当前第一个可运行路径是简化版 Ingest 流程。它会先把 noisy raw material
整理成规范的 `raw_prepare/prepared.md`，再进入索引、抽取窗口和 wiki 草稿渲染：

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
- `raw_prepare` 将 original raw 转换为下游知识编译使用的 canonical prepared raw。
- `extraction_windows` 是结构化抽取的工程上下文窗口，不是语义知识单元。
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
  page_planning:
    spec: mock:fixture
    fixture_dir: tests/fixtures/simple_project/mock
```

正式运行前可以检查 provider 配置：

```bash
llmwiki providers check /path/to/vault
llmwiki providers check /path/to/vault --live
```

普通 `llmwiki ingest resume` 会为仍需执行的模型步骤读取当前合并后的 provider
配置。已经完成的 step 不会因为 config 改变自动重跑。

如果需要从某一步开始，用当前配置重跑该 step 及下游：

```bash
llmwiki ingest resume /path/to/vault <operation_id> --from claim_extraction
```
