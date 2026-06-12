# Lite Ingest 需求文档

版本：v0.1 draft  
日期：2026-06-11  
范围：`llmwiki-engine` Lite 分支的 Ingest 核心重写需求  
状态：需求规格，不是实现方案

## 1. 背景

当前 `llmwiki-engine` 已经具备一条较完整的 M4 Ingest MVP 路径：初始化 vault、读取 `raw/`、模型生成 `source_digest`、候选规划、带 embedding 召回的 wiki context snapshot、merge planning、draft rendering、review、validation、apply preview 和手动 apply。

这套系统证明了几个重要能力：

- JSON/JSONL artifacts 可以让每个步骤可审计、可恢复、可测试。
- provider 可以按步骤配置，并记录非密文执行上下文。
- embedding retrieval 可以在非空 vault 中帮助定位相关旧页面。
- validator 可以把路径、schema、source refs、merge/action 合法性变成硬门禁。
- apply preview / preimage / receipt 可以保证写入前后可追踪。

但当前代码已经混入较多历史路径、人工 review 节点、apply 命令、raw cleanup / raw prepare 兼容逻辑、旧 grounding / Harness 策略和多轮修补痕迹。Lite 分支的目标不是在现有流程上继续加功能，而是围绕 Ingest 核心重新建立更清晰、更自动化、更硬的结构契约。

## 2. 核心产品判断

Lite 的目标不是把模型语义判断做到绝对正确，而是把结构做稳。

核心原则：

```text
结构要稳准狠，语义接受不确定。
```

含义：

- `/raw` 是永久证据层。raw 必须保留、可定位、可 hash 校验。
- `wiki/` 是压缩后的理解层、导航层和长期记忆索引，不是唯一真相层。
- Ingest 可以全自动运行，不因普通语义不确定而停下来等人工审核。
- 如果 wiki 没有回答某个问题，后续 Query 应能回到 raw，而不是假装 wiki 已包含全部信息。
- 系统硬度来自路径、hash、source refs、schema、transaction、receipt、snapshot drift 检查，而不是来自模型每次理解都完美。

### 2.1 中文输出契约

Lite 的默认用户可见语言是中文。

- 中间 artifacts 中给人阅读的摘要、理由、warnings、Markdown 报告和页面草稿必须使用中文。
- 最终知识库页面、source page、index、log、Related 章节和校验报告必须使用中文。
- CLI 的普通人类展示必须使用中文，包括步骤名、状态、计数字段、校验信息和 raw 候选列表。
- schema 字段名、枚举值、路径、hash、operation id、provider spec、必要产品名或专有名词可以保留英文机器值。
- provider 输出若在用户可读字段中返回英文段落，应被视为结构契约失败，而不是静默写入 wiki。

## 3. Lite 总目标

Lite Ingest 应支持一条全自动编译链：

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

关键变化：

- 去掉所有人工 review 节点。
- 去掉手动 apply 命令；写入变成 Ingest 自动流程的最终事务步骤。
- 去掉 `raw_prepare` 模型清洗步骤。
- 输入 raw 默认已经是适合编译的 Markdown/文本；文档清洗由上游负责。
- `candidate_pages` 在 `wiki_snapshot` 之后、`merge_plan` 之前生成。
- 先生成忠于 raw 的候选知识页面，再与已有 wiki 对比决定 create/update/noop/split/merge。
- 最后通过 `composition_plan` 调整每篇最终页面的写作顺序，再重新生成 `final_pages`。
- embedding cache 的旧页面同步归属 `wiki_snapshot`；每个候选页的 top5 召回归属 `candidate_contexts`。
- `knowledge_write` 只写知识页；`source_record_write` / `index_log_write` 位于尾部；`embedding_cache_refresh` 在写入后刷新最终 cache；`receipt` 最后记录最终 cache 状态。

## 4. 非目标

Lite 第一阶段不做：

- 不做 UI。
- 不做通用 PDF/OCR/网页清洗。
- 不做人工 review / approve / revise 交互。
- 不保留手动 apply 命令作为主流程。
- 不追求每次 candidate / merge 语义判断完美。
- 不把 grounding warning 变成普通技术材料的阻塞门禁。
- 不为了兼容旧 run manifest 保留复杂迁移逻辑。
- 不先做 Query 产品，但 Ingest 输出必须为未来 Query 回 raw 留足索引和 receipt。

## 5. 命名纪律

保留：

- `raw_file`：原始材料文件，位于 vault 的 `raw/` 下。
- `source_digest`：对某一个固定 raw/source 的消化结果。
- `source page`：wiki 中记录某个 raw/source 的来源页。

避免：

- 不把第一次生成的知识页面叫 `source_pages`。
- 不把第一次生成的知识页面叫 `source_drafts`。
- 不把候选知识页面和 wiki source page 混用。

推荐术语：

- `candidate_pages`：基于本次 raw 生成的候选知识页面，尚未决定最终 create/update/noop。
- `merge_plan`：把 candidate pages 对齐到现有 wiki 的结构决策。
- `composition_plan`：每个最终页面的写作顺序、保留/新增/重排规则。
- `final_pages`：最终准备写入 wiki 的页面。
- `knowledge_write`：自动写入经过验证的知识页，包含 preimage 和 atomic writes。
- `source_record_write`：写入对应 raw/source 的 source page。
- `index_log_write`：最后维护 index/log 等系统页。
- `embedding_cache_refresh`：在知识页、source page、index/log 写入后刷新当前 wiki 知识页 embedding cache，只保留最新状态。
- `receipt`：最后记录 operation、raw bindings、written targets、warnings、artifact hashes 和最终 embedding cache 状态。

## 6. 继承当前项目能力

Lite 应继承或等价保留这些能力：

- CLI-first：仍然是 Python 3.11+ CLI/engine，可本地运行和测试。
- profile：保留 profile 定义 page types、目录、语言和基础写作规则。
- provider config：保留全局/vault provider 配置、mock fixture、OpenAI-compatible provider。
- structured calls：模型输出必须进入 Pydantic schema，不接受自由文本作为核心 artifact。
- model repair：保留，但优先本地 normalization 修复小结构问题，避免无意义二次模型调用。
- manifest：每个 operation 有 manifest、step attempts、provider context、raw binding、artifact refs。
- events / metrics：保留 step duration、model call count、repair count、payload chars、warnings。
- verification：保留 raw hash 和 required artifact hash 的 verify。
- wiki snapshot：保留 snapshot preimage hash、candidate pool hash、drift 检查。
- embedding retrieval：必须使用真实 embedding 后端，默认 `sentence_transformers` + `Qwen/Qwen3-Embedding-0.6B`，并记录 model、revision、cache、query count、page count；`candidate_contexts` 不允许 exact/hash/lexical/title fallback。
- validators：保留路径安全、schema、candidate coverage、merge action、source refs、write target 等硬门禁。
- receipts：写入后记录 operation、raw bindings、written targets、warnings、artifact hashes。
- tests：保留 fixture / mock deterministic path 和真实 raw E2E 压测样本。

## 7. 从线上 Issues 吸收的需求

### 7.1 Issue #1：本地修复小结构问题

问题：`source_digest` 中 `suggested_page_title == ""` 会触发完整模型 repair，导致耗时接近翻倍。

Lite 要求：

- `source_digest` 的小字段缺失应优先本地 normalization。
- 可本地修复的字段不应触发完整模型 repair。
- repair report 必须标明 repair 原因、是否本地修复、是否调用模型。
- 验收：空 `suggested_page_title` 可以用 `name` 或安全标题补齐，最终 artifact 合法，模型调用次数不增加。

### 7.2 Issue #2：清理历史代码

问题：当前代码有历史逻辑、旧边界、兼容绕路和不再使用的 helper。

Lite 要求：

- Lite 分支默认重写核心，不逐段搬旧代码。
- 旧系统作为行为 contract、回归样本和反面教材使用。
- 所有不服务 Lite 主链的 review/apply/raw_prepare 旧路径默认不迁移。
- 开发步骤必须包含专门的 legacy pruning / parity review。

### 7.3 Issue #5：模型步骤动态反馈

问题：长模型步骤等待时 CLI 像卡住。

Lite 要求：

- 所有 model-backed steps 有可见动态进度。
- 非 TTY 和 `--json` 模式禁止动画。
- repair / retry 必须明确展示。
- 每个步骤结束输出耗时、模型调用次数、repair 次数、关键数量。

### 7.4 Issue #6：CLI 显示关键数量

问题：用户不知道 digest / candidate 阶段提取了多少内容。

Lite 要求：

- `source_digest` 完成后显示候选数量、弱/噪声数量、deferred 数量。
- `candidate_pages` 完成后显示生成页面数量。
- `candidate_contexts` 完成后显示 query count、top_k 和 retrieval backend。
- `merge_plan` 完成后显示 create/update/noop/split/merge 数量。
- `final_pages` 完成后显示最终写入页面数量。
- `knowledge_write` 完成后显示知识页写入 target 数。
- `embedding_cache_refresh` 完成后显示 cache refreshed/pruned 数。
- `receipt` 完成后显示 receipt path，并能展示最终 cache stats。

### 7.5 Issue #7：写入后 embedding cache refresh

问题：新建/更新页面后，后续 retrieval 可能重复编码。

Lite 要求：

- 当前知识页 cache 同步职责在 `wiki_snapshot`。
- 每个 candidate page 的召回职责在 `candidate_contexts`。
- `index_log_write` 成功后执行最终 `embedding_cache_refresh`，随后由 `receipt` 记录最终 cache metrics。
- cache key 至少包含 page path、content hash、embedding model、model revision、embedding input version、truncation contract。
- 下一次 `wiki_snapshot` 对未变化页面复用 cache。
- cache 必须是 current-state only：同一路径只保留最新向量，当前 wiki 中不存在的路径必须被 prune。
- 最终 cache report 必须区分并列出 `cache_hit_paths`、`cache_created_paths`、`cache_updated_paths`、`cache_refreshed_paths`、`cache_pruned_paths`。
- 不创建与现有 vector cache 冲突的平行缓存系统；优先统一 cache contract。

## 8. 运行时执行步骤

### Step 0：Operation 初始化

目标：创建一次可恢复、可审计的 ingest operation。

输入：

- `vault`
- `raw_file`
- profile
- provider config
- Lite config

输出：

- `.llmwiki/runs/ingest/<operation_id>/manifest.json`
- `events.jsonl`
- `operation_config_snapshot.json`

是否调用模型：否。

硬门禁：

- raw_file 必须在 `vault/raw/` 下。
- operation_id 必须是安全单段 ID。
- `.llmwiki/` 必须未被 Git tracked/staged。
- provider config 不能把 secret 写入 manifest/events/artifacts。

CLI 输出：

- operation id
- raw path
- profile
- provider summary
- Lite mode：`full_auto`

### Step 1：Raw Binding

目标：冻结本次 raw 输入身份。

输入：

- raw file bytes

输出：

- `raw_binding.json`
- raw sha256
- raw size
- raw mtime

是否调用模型：否。

硬门禁：

- raw 不存在：失败。
- raw 为空：失败或明确 warning，取决于配置。
- raw 路径逃逸：失败。

软门禁：

- raw 是否已经清洗不做模型判断。
- raw 内容质量差只 warning，不清洗。

说明：

- Lite 不再执行 `raw_link_cleanup` 和 `raw_prepare`。
- 如果未来仍需 wikilink normalization，应作为显式上游命令，不属于默认 ingest。

### Step 2：Source Digest

目标：对单个 raw/source 生成可结构化消费的 digest。

输入：

- raw text
- profile
- language config
- source identity

输出：

- `source_digest/source_digest.json`
- `source_digest/source_digest.md`
- `source_digest/source_map.json`
- `source_digest/source_map.md`
- `structured_repair_report.json`

是否调用模型：是。

核心字段：

- source_raw_path
- summary
- key_takeaways
- entities
- concepts
- designs
- comparisons
- open_questions
- budget_deferred_candidates
- weak_or_noise_items

硬门禁：

- JSON parse/schema 失败且无法 repair：失败。
- source_raw_path 不匹配本次 raw：失败。
- candidate_id 重复：失败。
- 正式候选缺少 name/type/summary/source basis：失败。

本地 normalization：

- 空 `suggested_page_title` 用 `name` 或安全标题补齐。
- 可安全推导的 page title/path stem 不触发模型 repair。
- 弱相关或明确不建页内容进入 `weak_or_noise_items` 或 deferred。

软门禁：

- digest 不完美不阻塞。
- 某些候选遗漏不阻塞。
- source locator 粗糙不阻塞，但写入 warning。

CLI 输出：

- 耗时
- 候选总数
- weak/noise 数量
- deferred 数量
- model call / repair count

### Step 3：Wiki Snapshot

目标：冻结当前 wiki 状态，并同步当前知识页的 embedding cache。

输入：

- source_digest
- current `wiki/`
- profile page types
- embedding config

输出：

- `wiki_snapshot/wiki_snapshot.json`
- `wiki_snapshot/knowledge_pool.json`
- `wiki_snapshot/embedding_cache_report.json`

是否调用模型：否。

职责：

- 扫描 `wiki/` 中可索引知识页。
- 排除 source pages、logs、system pages。
- 读取 frontmatter、title、summary、aliases、source refs。
- 为每个当前知识页生成 page-level embedding input。
- 读取或刷新 page vector cache。
- 同一路径只保留最新 content hash 对应的向量记录。
- 删除当前 wiki 中已经不存在的旧 cache 记录。
- 记录 candidate pool hash。

embedding 归属：

- 旧页面 embedding cache 同步发生在本步骤。
- 不做 candidate query；候选页还没有生成。
- 本步骤不写最终 wiki 页面。

硬门禁：

- snapshot 超过 max context：失败。
- embedding backend 配置错误：失败。
- snapshot 条目路径不安全：失败。

软门禁：

- embedding cache 缺失、禁用或使用非真实后端会阻塞 `candidate_contexts`。
- 低分命中只影响 warning/strength，不阻塞。

CLI 输出：

- knowledge pool size
- retrieval backend
- encoded page count
- cache hit/miss
- stale/pruned count

### Step 4：Candidate Pages

目标：先基于本次 raw/source 生成候选知识页面，尚不决定 create/update。

输入：

- raw text 或 source excerpt pack
- source_digest
- wiki_snapshot 的轻量上下文
- profile

输出：

- `candidate_pages/candidate_pages.json`
- `candidate_pages/candidate_pages.md`
- `candidate_pages/page_<id>.md`

是否调用模型：是。

设计重点：

- candidate pages 是“页面形态的候选表达”，不是最终 wiki 输出。
- candidate pages 应并发生成：以 source_digest candidates 为并发边界，同时发起多个页面生成请求。
- 并发结果必须合并为一个 `candidate_pages.json`，再进入统一校验。
- candidate pages 忠于本次 raw/source。
- candidate pages 可以参考 wiki_snapshot 避免明显重复表达，但不在本步骤决定目标旧页。
- 每个 candidate page 必须有 stable id。
- 每个 candidate page 必须带 raw/source ref。

推荐字段：

- candidate_page_id
- source_candidate_ids
- title
- proposed_page_type
- proposed_path_hint
- summary
- body_markdown
- open_questions
- source_refs
- evidence_notes
- confidence

硬门禁：

- candidate page id 重复：失败。
- candidate page 无 source ref：失败。
- body 为空：失败。
- page type 不合法且无法映射 profile：失败。
- 输出 page-level markdown 包含 frontmatter/target path 假写入：失败。

软门禁：

- 标题不完美不阻塞。
- 页面拆分粒度不完美不阻塞。
- related links 不全不阻塞。

CLI 输出：

- candidate page count
- covered digest candidate count
- skipped/deferred count
- model call / repair count

### Step 5：Candidate Contexts

目标：为每一个 candidate page 找到 top5 相关旧页面。

输入：

- candidate_pages
- wiki_snapshot
- page embedding cache
- embedding config

输出：

- `candidate_contexts/candidate_contexts.json`
- `candidate_contexts/candidate_contexts.md`

是否调用模型：否。

职责：

- 每个 candidate_page_id 独立生成 query。
- 使用 page-level embedding cache 对当前知识池召回 top5。
- 记录命中 path、title、score、match basis、page hash、excerpt。
- 不在本步骤判断 create/update/noop。
- 不做 chunk 级召回；Lite MVP 先保持 page-level。

软门禁：

- 召回分数低不阻塞。
- top5 不完整不阻塞。

CLI 输出：

- candidate page count
- query count
- top_k
- retrieval backend

### Step 6：Merge Plan

目标：比较 candidate pages 与 wiki_snapshot，决定每个候选页面如何进入 wiki。

输入：

- candidate_pages
- wiki_snapshot
- candidate_contexts
- source_digest
- profile

输出：

- `merge_plan/merge_plan.json`
- `merge_plan/merge_plan.md`
- `merge_plan/merge_decision_report.md`

是否调用模型：是。空 vault 或确定性场景允许 local shortcut。

动作：

- `create`
- `update`
- `noop`
- `split`
- `merge`

不再使用：

- `needs_human_decision` 作为阻塞动作。

如果模型不确定：

- 仍必须自动选择 create/update/noop/split/merge。
- 不确定性写入 `uncertainties` / `warnings` / receipt。
- 结构无法安全决定时，整次 operation fail，而不是等待人工 review。

merge_plan 必须说明：

- candidate_page_id
- action
- target_path
- matched_existing_paths
- inspected_context_paths
- strongest_overlap
- why_create / why_update / why_noop
- split/merge mapping
- source_refs
- warnings

硬门禁：

- 每个 candidate page 必须被消费。
- create/update target_path 不能重复冲突。
- update 必须指向 snapshot 中存在的旧页面。
- noop 必须指向已检查旧页面或说明由某个 update 覆盖。
- split/merge 必须保持 source refs 不丢失。
- target path 必须位于 `wiki/` 下且符合 profile routing。
- source page / logs / index 不能作为知识页 update 目标。

软门禁：

- create/update 判断语义不完美不阻塞。
- medium overlap create 允许，但必须写 why_not_update。
- grounding 弱只 warning。

CLI 输出：

- create/update/noop/split/merge count
- warning count
- inspected old page count

### Step 6：Composition Plan

目标：为最终写入页面确定阅读顺序、结构和旧内容保留策略。

输入：

- merge_plan
- candidate_pages
- wiki_snapshot full entries
- profile

输出：

- `composition_plan/composition_plan.json`
- `composition_plan/composition_plan.md`

是否调用模型：可以是模型步骤，也可以先本地规则 + 模型补充。

职责：

- 为每个 final target 聚合来自 candidate pages 和旧页面的内容。
- 决定段落顺序。
- 指明旧内容保留、移动、删除、补充的位置。
- 对 update 明确 old/new 的融合策略。
- 对 split/merge 明确 candidate fragment 去向。

推荐字段：

- final_page_id
- target_path
- action
- candidate_page_ids
- existing_page_refs
- section_order
- preserve_rules
- insert_rules
- delete_rules
- source_ref_rules
- readability_goal
- warnings

硬门禁：

- merge_plan 中 create/update/split/merge 产生的写入目标必须有 composition item。
- update 必须引用 preimage hash。
- composition 不允许丢 source refs。
- 不允许把 source page 写作规则混进知识页。

软门禁：

- 写作顺序不完美不阻塞。
- section 命名不完美可由 final_pages validator 修正。

CLI 输出：

- final target count
- update target count
- pages requiring old-content preservation count

### Step 7：Final Pages

目标：根据 composition plan 重新生成最终可写入页面。

输入：

- composition_plan
- candidate_pages
- wiki_snapshot entries
- raw/source excerpts
- source_digest
- profile

输出：

- `final_pages/final_pages.json`
- `final_pages/pages/<target_path>.md`
- `final_pages/final_page_manifest.json`
- `final_pages/diffs/*.diff`
- grounding/warning report

是否调用模型：是。

要求：

- final pages 必须是最终写入形态。
- final pages 应并发生成：以 composition_plan items 为并发边界，同时发起多个页面生成请求。
- 并发结果必须合并为一个 `final_pages.json`，再由系统统一校验覆盖关系。
- 每页 frontmatter 由系统组装或严格验证。
- 每页必须包含 source refs。
- update 页面必须吸收旧页面仍然有价值的内容。
- 页面正文以可读性为优先，避免机械拼贴。

硬门禁：

- final target 缺页：失败。
- final page target path 与 composition 不一致：失败。
- frontmatter 缺失必要字段：失败。
- source refs 丢失：失败。
- update preimage 不匹配 snapshot：失败。
- page-level markdown 结构非法：失败。
- 模型 self-talk 泄漏：失败或本地 repair。

软门禁：

- unsupported factual expansion warning only，除非与 raw 明确矛盾。
- 普通 quote mismatch warning only。
- related links 不充分 warning only。

CLI 输出：

- final page count
- diff count
- warning count
- model call / repair count

### Step 8：Validation

目标：在写入前做最终硬门禁。

输入：

- all prior artifacts
- final_page_manifest
- write_set
- current filesystem

输出：

- `validation/validation_report.json`
- `validation/validation_report.md`

是否调用模型：否。

硬门禁：

- raw hash drift：失败。
- wiki snapshot drift：失败。
- final pages 缺失：失败。
- target path escape：失败。
- duplicate target：失败。
- unsafe overwrite：失败。
- source refs 缺失：失败。
- page type / directory routing 不合法：失败。
- write_set hash 不一致：失败。
- secret literal / real credential 泄漏：失败。
- actionable medical/legal/financial/security advice 由模型扩写且 raw 不支持：失败。

warning only：

- grounding support weak。
- create/update 判断低置信度。
- related links 不全。
- open questions 可能遗漏。
- source digest coverage 不完美。

### Step 9：Knowledge Write

目标：全自动、安全写入最终知识页。

输入：

- final_page_manifest
- validated final_pages
- current wiki filesystem

输出：

- 写入 `wiki/<knowledge directories>/`
- `knowledge_write/write_set.json`
- `knowledge_write/preimages.json`
- `knowledge_write/write_result.json`

是否调用模型：否。

职责：

- 写入前再次校验 knowledge page preimage。
- 使用 atomic write。
- 只写知识页，不写 source/log/index。
- 记录 knowledge written_targets。

硬门禁：

- preimage drift：失败，不写。
- target 出现在 preview 后：失败。
- write_set target 重复：失败。

### Step 10：Source Record Write

目标：写入本次 raw/source 对应的来源记录页。

输出：

- `wiki/sources/Source_<raw>.md`
- `source_record_write/write_result.json`
- `source_record_write/write_item.json`

职责：

- 记录 raw path、raw hash、operation id、source digest summary。
- 记录本次派生的 knowledge pages。
- 不进入 embedding index。

### Step 11：Index / Log Write

目标：在所有知识页和 source page 写入后，维护系统页。

输出：

- `wiki/index.md`
- `wiki/log.md`
- `index_log_write/write_result.json`
- `index_log_write/write_items.json`

职责：

- `index.md` 反映当前知识页列表。
- `log.md` 追加本次 operation。
- 不进入 embedding index。

### Step 12：Embedding Cache Refresh

目标：最后刷新当前 wiki 知识页 embedding cache。

输出：

- `embedding_cache_refresh/embedding_cache_refresh.json`

职责：

- 重新扫描当前知识页。
- 刷新新建/更新后的知识页向量。
- 删除当前 wiki 已不存在路径的旧 cache 记录。
- 确保 cache 只保留最新状态。

CLI 输出：

- embedding cache refreshed count
- cache hit/pruned count

### Step 13：Receipt

目标：让一次自动 ingest 事后可理解。

输出：

- `receipt/receipt.json`
- `.llmwiki/applied/operations.jsonl`

职责：

- 记录 operation、raw bindings、artifact hashes、written targets、action counts。
- 记录 provider contexts、model call count、repair count、最终 embedding cache stats。
- receipt 写出后，本次 raw 才被视为 processed。

硬门禁：

- receipt 已存在：失败。

### Operation Summary Fields

目标：让一次自动 ingest 事后可理解。

输出：

- operation_id
- raw path/hash
- source_digest hash
- wiki_snapshot hash
- candidate_pages hash
- merge_plan hash
- composition_plan hash
- final_pages hash
- write_set hash
- written targets
- create/update/noop/split/merge count
- warnings
- provider contexts
- model call count
- repair count
- embedding backend/cache stats
- engine version/profile version

## 9. 状态机

Lite 推荐状态：

- `created`
- `running`
- `failed`
- `validated`
- `written`
- `source_recorded`

删除：

- `awaiting_review`
- `drafted`
- `applied` 作为用户手动 apply-ready 状态
- `apply_failed` 作为独立主状态

说明：

- 如果没有知识页变化但 source page/log 被记录，可用 `source_recorded`。
- 如果写入中失败，可用 `failed` 并在 failure artifact 中记录 partial writes。
- 是否需要单独保留 `written` vs `completed` 可在实现时确认。

## 10. CLI 需求

核心命令：

```bash
llmwiki init <vault> [--profile NAME]
llmwiki ingest run <vault> <raw> [--profile NAME] [--slug TEXT] [--json]
llmwiki ingest status <vault> [operation_id] [--verify] [--json]
llmwiki ingest inspect <vault> [operation_id] [--json]
llmwiki ingest raw-candidates <vault> [--all] [--limit N] [--json]
llmwiki providers check <vault> [--live]
```

删除或降级：

- `llmwiki ingest apply` 不作为 Lite 主流程命令。
- `llmwiki ingest approve/revise` 不进入 Lite。
- `--prepare auto|skip|force` 不进入 Lite 默认 ingest。

进度显示：

- TTY：Rich spinner / progress live。
- 非 TTY：普通事件行。
- `--json`：仅机器可读 JSON，不输出动画。

每个 step 完成行应包含：

- step name
- duration
- artifact count
- model call count
- repair count
- key domain counts

## 11. Artifact 目录建议

```text
.llmwiki/runs/ingest/<operation_id>/
  manifest.json
  events.jsonl
  run_metrics.json
  raw_binding/
  source_digest/
  wiki_snapshot/
  candidate_pages/
  candidate_contexts/
  merge_plan/
  composition_plan/
  final_pages/
  validation/
  knowledge_write/
  source_record_write/
  index_log_write/
  embedding_cache_refresh/
  receipt/
```

## 12. Hard Gates vs Warnings

Hard fail：

- raw 不存在、路径逃逸、hash drift。
- schema/JSON 无法解析。
- required artifact 丢失或 hash drift。
- wiki snapshot drift。
- target path 不安全。
- duplicate writable target。
- source refs 丢失。
- final page 空内容。
- write transaction preimage mismatch。
- receipt 冲突。
- secret literal 泄漏。

Warning：

- source_digest coverage 不完美。
- candidate page 粒度不完美。
- create/update/noop 低置信度。
- grounding weak。
- related links 不充分。
- open questions 可能遗漏。
- embedding retrieval 未使用真实 embedding 后端。

## 13. 开发步骤

### Dev Step 1：冻结 Lite 规格和测试样本

目标：

- 将本文档作为 Lite 初始需求。
- 固定 3-5 个真实 raw 作为回归样本。

建议样本：

- Cat Wu 访谈 raw。
- Scaling Managed Agents。
- Stop Applying to AI PM Jobs。
- Qwen-Agent README CN。
- Karpathy llm-wiki gist。

产出：

- `docs/lite-ingest-requirements.zh-CN.md`
- `docs/lite-ingest-design-log.zh-CN.md`，可选。
- fixture/e2e 样本清单。

验收：

- 文档被确认。
- 样本路径可读。

### Dev Step 2：建立 Lite schema skeleton

目标：

- 定义 Lite artifacts 的 Pydantic models。
- 不接模型，不写业务逻辑。

范围：

- RawBinding
- SourceDigest
- WikiSnapshot
- CandidatePages
- MergePlan
- CompositionPlan
- FinalPages
- ValidationReport
- WriteTransaction
- Receipt

验收：

- schema 单测覆盖 required/forbid extra/path validation。
- 所有 schema 可 JSON roundtrip。

### Dev Step 3：建立 Lite manifest 和 step runtime

目标：

- 用更少状态重建 operation runner。

范围：

- step registry
- manifest write/read
- attempt tracking
- artifact refs
- verify
- event logging
- metrics

验收：

- mock no-op operation 可从 created 跑到 failed/written。
- artifact hash drift 能被 verify 捕获。

### Dev Step 4：实现 raw_binding + source_digest

目标：

- 接入第一个模型步骤。

重点：

- source_digest schema。
- 本地 normalization 修复空 title 等小问题。
- structured repair report。
- CLI count summary。
- spinner/progress。

验收：

- Issue #1 的场景不触发二次模型 repair。
- mock fixture deterministic。
- 真实 Qwen raw source_digest 可跑通。

### Dev Step 5：实现 wiki_snapshot + embedding cache contract

目标：

- 冻结 wiki 状态并召回相关旧页。

范围：

- knowledge pool。
- exact backend。
- sentence_transformers backend。
- page vector cache。
- candidate context projection。
- drift check。

验收：

- 空 vault 可返回空 pool。
- 非空 vault 能召回相关旧页。
- cache key 包含 path/content hash/model/revision/input version。
- 重新运行能命中 cache。

### Dev Step 6：实现 candidate_pages

目标：

- 基于 raw/source 先生成候选知识页面。

范围：

- CandidatePages schema。
- prompt/payload。
- validator。
- markdown sidecars。

验收：

- 每个 candidate page 有 stable id/source refs。
- 输出不能伪造最终 target write。
- 能覆盖 source_digest 主要候选。

### Dev Step 7：实现 merge_plan

目标：

- 比较 candidate pages 和 wiki snapshot，决定 create/update/noop/split/merge。

范围：

- MergePlan schema。
- matching payload。
- validators。
- local shortcut for empty vault。
- decision report。

验收：

- 每个 candidate page 被消费。
- update 必须绑定旧页。
- duplicate target 被拦截。
- all create 不是默认风险阻塞，但必须有 inspected evidence。

### Dev Step 8：实现 composition_plan

目标：

- 从 merge decision 过渡到最终页面结构。

范围：

- CompositionPlan schema。
- update preserve rules。
- split/merge mapping。
- section order。

验收：

- 每个 final target 有 composition item。
- source refs 不丢。
- update 保留旧页有价值内容。

### Dev Step 9：实现 final_pages

目标：

- 重新生成最终写入页面。

范围：

- FinalPages schema。
- markdown assembly。
- frontmatter assembly。
- diff generation。
- grounding warnings。

验收：

- create/update 页面均可生成。
- update diff 可读。
- self-talk / illegal markdown 被捕获。
- warning 不阻塞普通技术 prose。

### Dev Step 10：实现 validation + split writes

目标：

- 全自动写入，替代手动 apply。

范围：

- final validation。
- write_set hash。
- preimage verification。
- atomic write。
- knowledge_write。
- source_record_write。
- index_log_write。
- embedding cache refresh。
- receipt。

验收：

- preimage drift 不写。
- duplicate target 不写。
- 成功后写 wiki、source page、index/log。
- 成功后 cache refresh report 记录新建/更新页面。
- receipt 最后记录最终 cache metrics。

### Dev Step 11：CLI polish and issue closure

目标：

- 处理线上 issue 对应的可见体验。

对应：

- #5 spinner/progress。
- #6 counts。
- #7 cache refresh。
- #1 local normalization。
- #2 legacy prune。

验收：

- TTY 有动态反馈。
- `--json` 稳定。
- status/inspect 能展示 counts、warnings、receipt、cache stats。

### Dev Step 12：E2E 压测与旧系统对照

目标：

- 证明 Lite 主链可用，并对照旧系统关键能力未丢。

验收：

- mock fixture 全绿。
- 单 raw 空 vault 跑通。
- 多 raw 非空 vault 跑通。
- Git diff 可读。
- receipt 能解释写入原因。
- Query 缺答案时可根据 source refs 回 raw。

## 14. 验收总标准

Lite MVP 完成条件：

- 无 manual review。
- 无 manual apply。
- 无 raw_prepare。
- 一条命令完成 raw 到 wiki 自动写入。
- 所有写入页面可追溯到 raw。
- 每次写入有 receipt。
- 写入前检查 raw/wiki drift。
- embedding 在 wiki_snapshot 中可用，并在写入后可刷新 cache。
- create/update/noop/split/merge 有结构化 merge_plan。
- final pages 经过 composition_plan 二次生成。
- 结构错误 hard fail，语义不确定 warning。
- 至少 3 个真实 raw E2E 跑通。

## 15. 待讨论问题

1. Lite 是否保留 source page？

建议保留。source page 不进入知识图谱，但作为 raw 的 human-readable receipt。

2. `noop` 是否仍写 source page/log？

建议写。即使没有知识页变化，也记录 raw 已被看过。

3. `split/merge` 第一版是否实现？

建议 schema 先支持，MVP 可先限制为 create/update/noop，split/merge 进入第二阶段。

4. final pages 的 frontmatter 由模型写还是系统组装？

建议系统组装，模型只写 summary/body/open_questions/change_summary。

5. 是否保留 `source_digest` eval？

建议保留并扩展到 candidate_pages / merge_plan。

6. 是否自动 git commit？

建议不自动 commit。engine 写 wiki 和 receipt，用户用 Git 做文件级回溯。
