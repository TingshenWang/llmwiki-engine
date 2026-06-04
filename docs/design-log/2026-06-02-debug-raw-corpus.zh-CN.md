# 开发调试 Raw Corpus 设计日志

[English](2026-06-02-debug-raw-corpus.en.md) | 中文

日期：2026-06-02
状态：已采纳为当前开发调试语料

这篇文档记录 `llmwiki-engine` 当前用于开发和调试的固定 raw corpus。它不是正式
fixture，也不复制原始内容进仓库；它记录本机上用于真实 ingest 调试的三篇 raw
材料，以及为什么它们足够覆盖下一阶段的页面规格、profile、validator 和 eval 设计。

## 决策

当前开发调试 corpus 使用三篇 AIAgent-PM raw：

- `/Users/wangtingshen/Documents/AIAgent-PM/raw/How Anthropic产品团队如何以超快速度开发产品 - Cat Wu访谈.md`
- `/Users/wangtingshen/Documents/AIAgent-PM/raw/Scaling Managed Agents - 将大脑与双手解耦（中文翻译）.md`
- `/Users/wangtingshen/Documents/AIAgent-PM/raw/Stop Applying to AI PM Jobs Until You Watch This（中文翻译）.md`

开发调试时优先用这些材料验证真实行为，而不是只依赖 `tests/fixtures/simple_project/`
里的简化项目笔记。

## 为什么这三篇足够重要

### Cat Wu 访谈

这篇材料是长视频转写稿，两个人对话，且主持人与回答者需要由模型判断。它能测试：

- `raw_prepare` 能否处理长转录稿、口语噪声和说话人归属；
- `source_digest` 能否从分散对话中提取稳定候选知识；
- `candidate_resolution` 能否把同一篇访谈落到多个页面类型；
- wiki 页面是否能组织实体、概念、设计方案、对比和事件，而不是只生成松散摘要。

这篇材料作为默认单篇 debug raw。

### Scaling Managed Agents

这篇材料讨论未来 Agent 构想，适合测试更抽象的设计和概念沉淀。它能覆盖：

- Design 页面：未来 Agent 架构、方案、组件和边界；
- Concept 页面：managed agents、brain/hands decoupling 等概念；
- Comparison 页面：不同 Agent 工作方式或产品形态的比较；
- 多篇 raw 之间对同一概念的补充和更新。

### Stop Applying to AI PM Jobs

这篇材料是播客/访谈型内容，聚焦 AI PM 岗位判断。它能测试：

- 视频或播客转写材料的清洗和结构化；
- 观点、岗位要求、能力模型和行动建议的拆分；
- 人物、组织、角色、概念和对比页面之间的连接；
- 与 AIAgent-PM 主题相关的长期知识积累。

## 调试方式

开发时需要观察两类运行：

- 单篇 ingest：每次只 ingest 一篇 raw，检查该 raw 的 source page、页面拆分、候选解析和草稿可用性。
- 多篇顺序 ingest：在同一个 vault 中一篇一篇 ingest 三篇 raw，检查已有页面如何被更新、补充或避免重复新建。

多篇调试仍然保持“一次 ingest 一篇 raw”的操作边界。这样可以保留 operation 级别的
manifest、artifact、resume 和 apply 审计能力。

## 对页面规格的影响

这组 corpus 暴露出当前页面不可用的核心原因：页面形态没有被产品化定义。后续
profile 不应该只描述 page type、目录和模板，还需要描述：

- operation intent：本次 ingest 是吸收访谈、整理观点、沉淀设计，还是更新已有页面；
- page archetype：不同页面类型应该承担什么信息功能；
- section contract：每类页面必须出现哪些段落，以及每段应该承担什么知识功能；
- link/update policy：什么时候新建页面，什么时候更新已有页面，什么时候只建立连接；
- validation/eval：页面是否可用需要被校验，而不只是检查 claim id 是否存在。

这三篇 raw 将作为后续页面产品规格和 eval case 的主要开发参照。
