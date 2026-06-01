# Ingest 架构设计日志

[English](2026-05-30-ingest-architecture.en.md) | 中文

日期：2026-05-30
状态：已采纳为当前 MVP 方向

这篇文档记录当前 `llmwiki-engine` Ingest 架构背后的设计决策。它不是聊天记录，
而是项目长期维护时需要保留的设计记忆：为什么这个引擎从 Agent 控制的 skill
流程，转向模块化、可审计、可恢复的 CLI workflow。

## 背景

`llmwiki-engine` 的目标是支持本地 wiki 的增量式知识编译。输入材料可能非常不
稳定：Markdown 笔记、视频转录稿、PDF 或 DOCX 导出、HTML clip、表格、日记，
以及其他用户学习或整理过的材料。

早期 skill 驱动的方案，会让 LLM 不断读取流程说明并控制整个过程。这种方式成本
高、难测试、容易跑偏，也很难对单个模块做独立优化。新的方向是：由 engine 掌控
workflow，只在边界清楚的模型任务中调用 LLM，并且每次调用都必须有明确输入、
schema、校验、artifact 和 review。

## 目标

- 核心 Ingest 流程必须可以通过 CLI 运行。
- 每个步骤都有固定输入、输出、日志和 artifact。
- 每个模块都可以独立测试、评估和优化。
- 正式 wiki 保持小而干净。
- 本地运行缓存与可提交的 wiki 输出分离。
- `resume` 和 `apply` 必须能防止 raw、artifact、wiki target 漂移；需要执行的
  模型步骤会读取当前 provider config，已完成步骤不会自动重跑，除非显式 `--from`。
- Provider 可以按模块配置，从而按步骤优化成本和质量。
- Agent 不作为默认运行时依赖。

## 存储边界

Vault 布局会区分原始材料、正式 wiki 页面和本地运行态：

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

`wiki/` 只放正式页面。运行 artifacts、drafts、previews、model calls 和 manifests
都留在 `.llmwiki/runs/`。

`.llmwiki/` 是本地运行状态、配置和审计状态。它可能在 config 中包含明文
provider credentials，所以 `init` 会在 `.gitignore` 中忽略整个目录。Git 层面的
review 和回退应该聚焦 `wiki/` 与 source material，而不是 run cache。

## Original Raw 与 Prepared Raw

Original raw 不被自动视为干净事实。它可能包含视频转录噪声、重复的双语句子、嵌入
时间戳、识别错误、格式错乱，或者无关的包装文本。

当前设计引入 `raw_prepare` 作为第一个模型驱动但边界清楚的步骤。它把 original raw
整理成 canonical prepared raw：

```text
original raw -> raw_prepare -> raw_prepare/prepared.md
```

下游抽取和 wiki 编译会把 prepared raw 当作该次运行的事实输入。这是一个刻意设计的
信任边界：prepared raw 必须可审计，但一旦被接受，它通常比让每个下游步骤都直接面对
高噪声 original raw 更有价值。

`raw_prepare` 会记录：

- 整理后的 Markdown 正文；
- 执行过的清洗操作；
- 不确定或有风险的项目；
- 面向人类的 preparation review；
- 用于校验和回放的结构化 artifact。

## 图片处理

图片暂时不进入 MVP 核心推理路径。

Markdown、HTML 和 DOCX 后续可以保留明确的图片链接或抽取出的图片资产，但图片内容
不应该静默进入 wiki knowledge，除非人类或未来明确的 OCR/vision 模块把它转换成文本。

PDF 中的内嵌图片在初始版本中可以直接丢弃。如果某张图片确实承载重要知识，用户可以
额外写一个 raw note 来补充。

这样可以避免在文本编译路径稳定之前，把 Ingest engine 过早扩展成 OCR 和视觉系统。

## 用 Extraction Windows 替代 Semantic Aggregation

早期的 `semantic_aggregation` 已经从主路径移除。

原因是 aggregation 层可能会假装自己在定义知识单元，但实际上做出有损或错误的分组。
例如，对话材料可能把一问一答切得过窄；视频转录稿也可能因为相邻而把无关片段合并。

替代方案是 `extraction_windows`。

Extraction windows 是工程上下文窗口，不是语义知识单元。它们的作用是给 claim
extraction 足够的局部上下文，同时保留对 prepared raw spans 的可追溯性。

```text
prepared raw -> raw_index -> extraction_windows -> claim_extraction
```

Claims 必须指向 `source_window_id`，并且 evidence 必须绑定回 prepared raw spans。
这样抽取过程可以被测试，而不会假装窗口边界就是概念边界。

## 当前线性 Pipeline

当前 MVP pipeline 是：

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

`validation` 是不产出文件的 gate step。它只校验上游 artifacts，不创建
`validation/` 模块目录。

`apply` 仍然是 preview 之后的显式命令。它会再次 verify run，检查 target preimage
hash，写入正式 wiki 页面，并追加 applied receipt。

## Manifest、Status、Resume、Apply

Manifest 是本地 workflow contract。它记录 raw bindings、step attempts、当前
artifact references、provider context records 和运行状态。

`status` 同时面向人类和机器：

- 默认输出应该清爽，并给出下一步建议；
- `--json` 暴露完整结构化状态；
- `--verify` 重新计算完整性检查，但不写文件。

`resume` 默认从第一个 failed 或 pending step 继续。对于本次执行，它会读取当前合并后
的 provider config，并记录新的 sanitized provider context；每个 step 实际用了什么
provider，以 step attempt 为准。已经完成的 step 不会仅因为 config 改变而自动重跑。
`resume --from STEP` 会先验证
当前 provider execution context，再删除目标 step 及其下游模块目录，把这些 step
标记为 pending，然后从那里重跑。

已经 applied 的 operation 不能 resume。raw drift、required artifact drift、
或 apply preimage drift 都必须阻断执行。

## Provider Runtime 方向

Provider 应该按模块选择，而不是全局只选一个模型。这可以让准备、未来保留的审查等
模块使用便宜的本地模型，同时让更难的抽取或规划任务使用更强的 API 模型。

MVP provider 集合包括：

- `MockProvider`：用于确定性 fixtures 和测试；
- `OpenAICompatibleProvider`：用于 hosted 或 API 中转站的 Chat Completions-compatible 接口；
- `HumanProvider`：用于显式人工交接点。

每个模型驱动步骤都应该使用 structured calls，并记录 schema validation、有限
repair、cost、latency 和失败输出诊断。
Provider context records 是唯一的 provider 执行快照，只记录非 secret 运行字段：
`spec`、`endpoint`、`fixture_dir`。明文 API key 允许保存在本地 config，但不能进入
manifest、events、provider result、receipt、status JSON 或 CLI 输出。真实 provider
执行使用内存中的 `ProviderExecutionContext`，不能从 manifest 反推出 credentials。

未来如果 `openai_compatible` 之外的 provider 也需要 live check，应把 live-check
接口显式化，例如定义统一的协议和返回对象；JSON mode fallback 仍应保持为
OpenAI-compatible 探针的专属行为，避免动态 `check_live(...)` 调用在新增 provider
类型后变脆。

## 测试与评估方向

引擎后续应该通过模块级测试和 eval 增长，而不是只依靠端到端 demo。

重要指标包括：

- schema valid rate；
- parse success rate；
- repair success rate；
- evidence quote validity；
- page type accuracy；
- provider cost and latency；
- resume safety；
- apply preimage safety。

smoke path 仍然适合作为快速信心检查：

```text
init -> ingest run -> status -> status --verify -> apply
```

但它不能替代 unit tests、regression tests 或 module evals。

## Review 已收口项

- API key 不扩散已有端到端自动化回归测试覆盖，确认 key 不会进入 manifest、events、
  provider result、status JSON、applied receipt 或 CLI 输出。
- 模块目录读写已经从 `StepSpec.output_dir` 派生，生产代码不再依赖
  `run_dir / step_name` 恰好等于当前目录名。

## Review 已知后续项

当前 MVP 有意把下面这些清理项留到后续单独收口：

- 移除剩余手写 step/module 列表，例如 `resume --from` help 文案和 eval module 声明；
- provider config 错误中补充 global/vault 配置来源信息，方便定位是哪份 config 出错；
- 把剩余 provider “snapshot/快照” 表述改成 provider execution record / provider 执行记录，
  避免和已经删除的 `snapshots/` 布局混淆；
- CLI 层也同时覆盖旧 `operation_manifest.v3` 和 `operation_manifest.v4`，不只在底层
  manifest 测试里覆盖。

## 开放问题

- 当模型发现清洗存在不确定性时，`raw_prepare` 应该多严格？
- prepared raw 是否需要可选的人类 approval gate？
- 非 Markdown 格式应该如何归一化到同一个 prepared raw contract？
- 在 deduplication 和 page planning 更复杂之前，claim schema 应该如何演进？
- 哪些模块适合默认本地模型，哪些模块应该默认使用更强的线上模型？
