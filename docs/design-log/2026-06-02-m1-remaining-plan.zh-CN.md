# M1 剩余计划

日期：2026-06-02
状态：M1 骨架已实现；当前 review 为显式 auto stub；以下为剩余工作清单

这份文档保存 2026-06-02 讨论后尚未完成的计划。当前已经实现的是新的 M1
主链路骨架：`raw_prepare -> prepared_raw_review -> source_digest ->
source_digest_review -> candidate_resolution -> wiki_merge_planning ->
draft_rendering -> validation -> apply_preview`。

下面只列还需要继续做的部分。

当前短期约定：保留 `prepared_raw_review` / `source_digest_review` 等 step
名，不改成 `_stub`。M1 仍然会自动通过这两个审核节点，但
`review_decision.json` 必须写明 `review_mode: auto_stub` 与
`auto_approved: true`，`ingest status` 也必须能看见该状态，避免把自动通过
误读成人工审核。

## P0：交互审核与可修改循环

### Awaiting Review 状态机

- 下一优先级是把 review step 做成真实暂停点，而不是继续自动穿过。
- 保持现有 step 名：`prepared_raw_review`、`source_digest_review`，不改成
  `_stub`。
- 进入 review step 时写出 `review_prompt.md`、待审核 artifact 与当前建议的
  approved artifact。
- 将当前 step 标记为 `awaiting_review`；operation status 后续需要增加
  `awaiting_review`，表示 pipeline 正在等待用户审核。
- pipeline 在该 step 停止，下游 step 保持 `pending`。
- `resume` 不能自动越过 `awaiting_review`；必须先完成审核决策。
- 计划增加 CLI：
  - `llmwiki ingest review "$VAULT" "$OP" source_digest_review`
  - `llmwiki ingest approve "$VAULT" "$OP" source_digest_review`
  - `llmwiki ingest revise "$VAULT" "$OP" source_digest_review`
- MVP 行为可以先简单：`review` 展示或打开待审核 artifact；用户可手动编辑
  approved 文件；`approve` 做基本校验、写入 `review_decision.json`，再把 step
  标记为 `completed`。
- approve 后再 `resume`，从下一个 step 继续执行。

### Prepared Raw Review

- 在 `raw_prepare` 后展示清洗后的 Markdown。
- MVP 使用 terminal/Rich 展示；后续再调研 terminal markdown 工具和 Obsidian 展示方式。
- 人类可以和模型讨论重点、侧重点、key takeaways、清洗错误和需要保留的上下文。
- 允许模型根据局部反馈修改 prepared raw，而不是只能全量重跑。
- 保存 `review_feedback.jsonl`、`review_decision.json` 和最终 `approved_prepared.md`。

### Source Digest Review

- `source_digest` 是单篇 raw 的完整消化文件，不与已有 wiki 发生合并。
- 审核界面需要支持按候选类型查看：entity、concept、design、comparison、open question。
- 展示候选表：candidate id、类型、名称、一句话总结、wiki value、duplicate risk、suggested action、source locator。
- 人类可以要求模型局部新增、删除、改写、合并候选。
- 审核通过后写入 `approved_digest.json` 和 `approved_digest.md`。

### Candidate Resolution Review

- 在 apply 前必须展示候选解析表。
- 表格至少包括：candidate、target page、action、matched page、duplicate risk、affected pages、reason。
- 允许人类要求模型局部修改解析结果，例如改成更新已有页、换目标页、拆分候选、合并候选。
- 不做 deferred/not_applied 状态；通过审核的候选全部进入 create/update/cross-reference/tension 等明确动作。

### Final Apply Preview Review

- apply 前展示最终 preview 表。
- 表格至少包括：target path、create/update、preimage status、source page、knowledge page、system page。
- `index.md`、`log.md`、`logs/YYYY-MM-DD.md` 必须出现在 preview 中，不能在 apply 后偷偷写。
- 人类确认后才允许 apply；`standard` 在这阶段仍只自动到 draft/apply preview，不自动写入。

## P0：增量合并与系统页

### Wiki Merge

- 当前 M1 骨架可以生成 draft；后续需要真正读取已有 wiki 页面并做增量更新。
- 更新已有页面时不能简单覆盖整页，需要按 section 合并。
- 新旧说法冲突时，不直接静默改写；应写入 tension/open question 或给出冲突提示。
- 需要处理 cross-references：新页面链接旧页面，旧页面也能补回相关链接。

### Index

- `wiki/index.md` 是全局索引页，初始化时创建空文件。
- 更新时应保留已有页面，并追加或更新本次涉及页面。
- 固定 MVP 字段：页面、类型、一句话总结、更新时间。
- 后续字段应进入全局字段表管理，并支持按页面类型配置。

### Log

- `wiki/log.md` 是日志索引页。
- 每日日志放在 `wiki/logs/YYYY-MM-DD.md`。
- 每次 ingest 追加当日日志，不应覆盖同一天已有记录。
- 日志应记录 operation id、raw、created、updated、duplicate/noop、tension、apply 状态。

## P0：页面结构与字段管理

- 定义全局 frontmatter 字段表，MVP 先固定，未来支持 profile/page type 配置。
- 每类页面必须有完整 section 结构。
- 没有内容的 section 不删除，写空态提示，例如 `No related knowledge recorded yet.`。
- Source page 只链接 raw 和最终 wiki 页面，不链接 prepared/digest/resolution 等中间 artifact。
- Knowledge page 的基础结构继续使用：Summary、Understanding、Related、Sources、Tensions / Open Questions。

## P0：质量评分

- 不要求用户手工评分，由 Agent 根据 raw/prepared/digest/final draft 自动评分。
- 评分维度建议：
  - coverage：是否漏掉重要候选知识；
  - correctness：是否引入原文不支持的内容；
  - usefulness：页面是否像可复用 wiki，而不是普通摘要；
  - merge quality：新增/更新/合并判断是否合理；
  - link quality：raw、source page、最终 wiki 页面链接是否正确；
  - section quality：各 section 是否承担了清楚的信息功能。
- 评分结果先作为 debug artifact，不进入正式 wiki。

## P1：Obsidian 与用户体验

- MVP 继续 terminal/Rich；后续调研 Obsidian CLI。
- 目标 UX 是弹框或 Obsidian 内友好的审核展示，而不是让用户手工翻 artifact 目录。
- 需要支持从 terminal 打开相关 draft/source/wiki 页面。
- Obsidian 集成只作为展示和审核辅助，不改变 engine 的 artifact/source-of-truth 边界。

## P1：Debug Corpus 与 Eval

- 固定三篇 AIAgent-PM raw 作为真实开发调试材料：
  - `/Users/wangtingshen/Documents/AIAgent-PM/raw/How Anthropic产品团队如何以超快速度开发产品 - Cat Wu访谈.md`
  - `/Users/wangtingshen/Documents/AIAgent-PM/raw/Scaling Managed Agents - 将大脑与双手解耦（中文翻译）.md`
  - `/Users/wangtingshen/Documents/AIAgent-PM/raw/Stop Applying to AI PM Jobs Until You Watch This（中文翻译）.md`
- 单篇调试：一次 ingest 一篇 raw，检查 digest、候选解析和最终草稿。
- 多篇顺序调试：同一个 vault 中一篇一篇 ingest，检查已有页面如何被扩充和链接。
- Eval 需要从 source_digest 扩展到最终页面质量、增量合并质量和 cross-reference 质量。
- `source_digest` eval 后续需要更真实的指标和 fixture：schema validation、
  candidate coverage/precision、source_locator、duplicate risk、action 判断，以及
  weak/noise 是否被错误转成页面。
- 这部分先不改当前 M1 骨架，后续单独设计 eval 合同。

## P1：未来配置化

- 页面类型、section、frontmatter、空态提示、source page 结构都应逐步可配置。
- MVP 可以继续使用全局固定结构。
- 配置化不能破坏 run artifact 的可恢复、可验证和可审计边界。
- 当前 `templates/` 标记为“暂未生效 / 未来配置化”。短期不恢复模板驱动；
  页面结构还在讨论，M1 继续稳定硬编码渲染。
- 模板系统未来必须彻底解决：当页面合同稳定后，再把 source/knowledge page
  渲染抽成 template-driven，并补“自定义模板确实影响输出”的测试。
- source page 是否完全脱离 profile、变成不可配置系统页，后续再定；当前先保留
  profile 中的 source 类型，但不得把模板误认为已生效。

## 暂不做

- 不恢复强证据链作为主路径硬合同。
- 不做 deferred/not_applied 候选池。
- 不让 source page 链接中间 artifact。
- 不做自动写入的 `standard` 模式；在审核系统稳定前，`standard` 只到 draft/apply preview。
