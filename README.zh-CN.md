# llmwiki-engine

[English](README.md) | 中文

`llmwiki-engine` 是一个面向模块化知识编译的 Python 3.11+ CLI 和引擎。
它的目标是把噪声较多的原始材料编译进本地 wiki，同时让每个步骤都可以
独立测试、评估和优化。

当前第一个可运行路径是简化版 Ingest 流程。它会先把 noisy raw material
整理成规范的 `prepared_raw/prepared.md`，再进入索引、抽取窗口和 wiki 草稿渲染：

```bash
llmwiki init /path/to/vault --profile project_basic
llmwiki ingest run /path/to/vault raw/project_note.md --fixture-dir tests/fixtures/simple_project/mock
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
- Validator 是硬门禁；LLM critic 是可选的语义审查者。

Provider 在 `.llmwiki/config.yaml` 里按任务配置，并会在每次运行前写入
run snapshot：

```yaml
providers:
  raw_prepare: mock:fixture
  claim_extraction: openai:gpt-4.1-mini
  page_planning: ollama:llama3
```
