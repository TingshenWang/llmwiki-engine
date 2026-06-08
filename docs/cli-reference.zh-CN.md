# llmwiki CLI 使用手册

这份文档是 `llmwiki-engine` 的命令罗盘。它解释每个常用命令、参数含义、配置关系、Git 边界和常见错误。

## 核心概念

`vault`

一个知识库目录。`llmwiki init` 会在里面创建 `raw/`、`wiki/` 和 `.llmwiki/`。

`raw`

待入库的原始材料。`ingest run` 的 `RAW` 参数必须指向 `vault/raw/` 里面的文件。`raw_link_cleanup` 会把目标 raw 文件原地改写为规范化材料层。本轮 MVP 只展开 `[[Page]]`、`[[Page|Alias]]` 这类 Obsidian 文本 wikilink；网页链接、裸 URL、HTML 链接、reference-style Markdown 链接、普通相对 Markdown 链接、媒体 embed、fenced code block 和 inline code 都保持不变。

`.llmwiki/`

本地运行目录，保存 config、profiles、runs、manifest、events 和 applied receipts。它默认写入 `.gitignore`，不应该进入 Git。

`operation_id`

一次 ingest operation 的 ID，例如：

```text
ING-2026-06-01T085014Z-manual
```

它对应目录：

```text
<vault>/.llmwiki/runs/ingest/<operation_id>/
```

`fixture_dir`

mock provider 的测试答案目录。目录里通常有：

```text
raw_prepare.json
source_digest.json
```

`provider`

模型步骤的执行来源。目前公开支持：

- `mock:fixture`：从 fixture 文件读取固定输出，适合测试。
- `human`：人工交接占位，会提示需要手动提供 artifact。
- `openai_compatible:<model>`：调用 Chat Completions-compatible API。

`apply`

把 run 里的 draft pages 写入 `vault/wiki/`。

`staged`

Git 的暂存区。`git add file` 后，文件就是 staged，默认会被下一次普通 `git commit` 提交。

## 配置文件

llmwiki 使用 YAML 配置：

```text
~/.llmwiki/config.yaml
<vault>/.llmwiki/config.yaml
```

global config 只允许配置 `providers`。vault config 配置当前 vault 的 `profile`，也可以覆盖 provider。

合并规则：

```text
global providers -> vault providers
同名 provider key 整体覆盖
不做字段级 merge
```

合法 provider key：

```text
default
raw_prepare
source_digest
candidate_resolution
wiki_merge_planning
draft_rendering
```

`default` 是默认 provider；通常只需要配置这一项。具体 step key 只在你想让某一步使用不同模型时覆盖 `default`。

mock 配置示例：

```yaml
profile: project_basic
providers:
  default:
    spec: mock:fixture
    fixture_dir: /path/to/mock
```

真实模型建议放在全局配置 `~/.llmwiki/config.yaml`，这样不同 vault 不用重复配置：

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-chat
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: sk-...
    max_retries: 2
    retry_backoff_seconds: 1.0
```

openai-compatible 配置示例：

```yaml
profile: project_basic
providers:
  default:
    spec: openai_compatible:deepseek-chat
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key: sk-...
    max_retries: 2
    retry_backoff_seconds: 1.0
  source_digest:
    spec: openai_compatible:stronger-digest
    endpoint: https://example.test/v1/chat/completions
    api_key: sk-...
```

`max_retries` 和 `retry_backoff_seconds` 是可选项，只对 `openai_compatible`
的正式 ingest 调用生效。`max_retries` 是每个逻辑模型调用共享的一组 transient retry
budget；JSON-mode 兼容 fallback 可能额外增加一次 prompt-only 请求，但只使用剩余
retry budget。retry 只覆盖 transient transport failure、408/409/425/429 和 5xx 类响应，不会
重试普通 bad request。

API key 允许明文保存在本地 config 中，但不会写入 manifest、events、provider_result、status JSON、applied receipt 或 CLI 输出。

## 快速测试流程

```bash
cd /Users/wangtingshen/Documents/llmwiki-engine

VAULT="$(mktemp -d -t llmwiki-vault)"
FIXTURE="$PWD/tests/fixtures/simple_project/mock"

uv run llmwiki init "$VAULT"

cp tests/fixtures/simple_project/raw_project_note.md "$VAULT/raw/"
RAW="$VAULT/raw/raw_project_note.md"

uv run llmwiki providers check "$VAULT"
uv run llmwiki ingest run "$VAULT" "$RAW" --fixture-dir "$FIXTURE" --slug manual

OP="$(ls -1 "$VAULT/.llmwiki/runs/ingest" | tail -n 1)"

uv run llmwiki ingest status "$VAULT" "$OP"
uv run llmwiki ingest status "$VAULT" "$OP" --verify
uv run llmwiki ingest apply "$VAULT" "$OP"
```

## 命令总览

```bash
llmwiki init <vault> [--profile project_basic]
llmwiki providers list
llmwiki providers check <vault> [--live]
llmwiki ingest run <vault> <raw> [--fixture-dir PATH|--mock-fixture-dir PATH] [--profile NAME] [--slug TEXT] [--mode dev|standard] [--prepare auto|skip|force] [--json]
llmwiki ingest run-next <vault> [--include-changed] [--dry-run] [--fixture-dir PATH|--mock-fixture-dir PATH] [--profile NAME] [--slug TEXT] [--mode dev|standard] [--prepare auto|skip|force] [--json]
llmwiki ingest status <vault> [operation_id] [--verify] [--json]
llmwiki ingest inspect <vault> [operation_id] [--json]
llmwiki ingest raw-candidates <vault> [--all] [--limit N] [--json]
llmwiki ingest raw-import-url <vault> <url> [--title TEXT] [--output PATH] [--overwrite] [--dedupe-url|--no-dedupe-url] [--arxiv-html|--no-arxiv-html] [--timeout SECONDS] [--max-bytes BYTES] [--json]
llmwiki ingest raw-import-arxiv <vault> <query> [--limit N] [--dry-run] [--overwrite] [--dedupe-url|--no-dedupe-url] [--sort-by VALUE] [--sort-order VALUE] [--min-relevance-score N] [--timeout SECONDS] [--max-bytes BYTES] [--json]
llmwiki ingest resume <vault> <operation_id> [--from STEP] [--mock-fixture-dir PATH] [--prepare auto|skip|force] [--mode dev|standard]
llmwiki ingest apply <vault> <operation_id>
llmwiki profile list
llmwiki profile validate <path_or_name>
llmwiki eval run <module> <dataset> [--output-root PATH]
llmwiki eval report <run>
```

## `llmwiki init`

初始化一个 vault。

```bash
uv run llmwiki init "$VAULT"
uv run llmwiki init "$VAULT" --profile project_basic
```

参数：

- `VAULT`：vault 路径。
- `--profile`：初始化使用的 profile，默认是 `project_basic`。

它会创建：

```text
raw/
wiki/
.gitignore
.llmwiki/config.yaml
.llmwiki/profiles/
.llmwiki/runs/
.llmwiki/applied/
```

它不会自动 `git init`。

## `llmwiki providers list`

列出当前公开 provider 类型。

```bash
uv run llmwiki providers list
```

当前输出应包含：

```text
human
mock
openai_compatible
```

## `llmwiki providers check`

检查 provider 配置，不创建 run，不写 manifest。

```bash
uv run llmwiki providers check "$VAULT"
```

它会检查：

- global/vault config 是否能读取并合并；
- provider key 是否只包含 `default` 和模型步骤；
- `spec`、`endpoint`、`api_key`、`fixture_dir`、`max_retries`、`retry_backoff_seconds` 是否符合 provider 类型；
- mock provider 是否缺少 `fixture_dir`；
- `.llmwiki/` 是否被 Git tracked 或 staged。

常见 warning：

```text
mock provider ... has no fixture_dir; ingest will require --fixture-dir.
```

这不是失败。意思是 config 里没有写 mock 答案目录，真实 `ingest run` 时需要传：

```bash
--fixture-dir "$FIXTURE"
```

加 `--live` 会对 `openai_compatible` 发一个最小连通性请求：

```bash
uv run llmwiki providers check "$VAULT" --live
```

`--live` 不会写 vault，不会创建 run，也不等同于完整 ingest。mock 只检查 fixture，
human 不发请求。真实 provider 会收到一次小型 Chat Completions 探针：

- `temperature=0`
- `max_tokens=512`
- 优先使用 `response_format={"type": "json_object"}`

`max_tokens=512` 是 completion 上限，不代表固定消耗；thinking 模型通常会提前停止，
但最多可能用到这个上限。JSON mode 不支持时，会 fallback 一次到 prompt-only JSON probe，
通过后仍会给 warning。live probe 本身不使用 transient retry，所以 provider 检查仍然保持轻量。
`temperature=0` 也不承诺所有 thinking 模型都完全确定性。

## `llmwiki ingest run`

启动一次 ingest operation。

```bash
uv run llmwiki ingest run "$VAULT" "$RAW" --fixture-dir "$FIXTURE" --slug manual
```

参数：

- `VAULT`：vault 路径。
- `RAW`：raw 文件路径，必须在 `VAULT/raw/` 下。
- `--fixture-dir PATH`：mock provider 的 fixture 目录。真实 provider 不需要。
- `--mock-fixture-dir PATH`：强制所有模型步骤使用指定目录的 `mock:fixture`。
- `--profile NAME`：临时覆盖 vault config 里的 profile。
- `--slug TEXT`：operation ID 的可读后缀，方便手动测试辨认。
- `--mode dev|standard`：运行模式，默认 `dev`。
- `--prepare auto|skip|force`：选择 raw_prepare 策略。`auto` 使用模型清洗，`skip` 明确对非空 Markdown 做本地透传，`force` 明确要求模型清洗。

`--slug manual` 只影响 operation ID，例如：

```text
ING-2026-06-01T085014Z-manual
```

不影响模型、provider 或输出内容。

## `llmwiki ingest status`

查看 operation 状态。

```bash
uv run llmwiki ingest status "$VAULT" "$OP"
```

如果不传 `OP`，默认使用最新 operation：

```bash
uv run llmwiki ingest status "$VAULT"
```

参数：

- `VAULT`：vault 路径。
- `OPERATION_ID`：operation ID，可省略。
- `--verify`：重新计算 raw 和 artifact hash，不写文件。
- `--json`：输出完整 JSON，适合脚本使用。

例子：

```bash
uv run llmwiki ingest status "$VAULT" "$OP" --verify
uv run llmwiki ingest status "$VAULT" "$OP" --json
```

状态表会展示每一步的 review 状态、attempt 次数、最近耗时、总耗时和 provider。
如果对应 artifact 已存在，也会输出 raw cleanup 审计文件、review prompt、draft
pages、diff 和 apply preview 等路径。

## `llmwiki ingest resume`

继续一个失败或 pending 的 operation。

```bash
uv run llmwiki ingest resume "$VAULT" "$OP"
```

默认行为：

- 从第一个 failed/pending step 继续；
- 对本次会执行的模型步骤读取当前 merged provider config；
- 已完成 step 不会因为 config 改变自动重跑。

从指定 step 及下游重跑：

```bash
uv run llmwiki ingest resume "$VAULT" "$OP" --from source_digest
```

resume 也可以覆盖本次会重跑步骤的 provider 或 raw_prepare 策略：

```bash
uv run llmwiki ingest resume "$VAULT" "$OP" --from raw_prepare --prepare skip
uv run llmwiki ingest resume "$VAULT" "$OP" --from raw_prepare --prepare force
uv run llmwiki ingest resume "$VAULT" "$OP" --mock-fixture-dir "$FIXTURE"
```

`--from STEP` 会：

- 先验证当前 operation 未 applied；
- 先执行 verify；
- 先解析并验证当前 provider config；
- 然后删除 `STEP` 及下游模块目录；
- 清空这些 step 的 current outputs；
- 从 `STEP` 重新执行。

合法 step：

```text
raw_link_cleanup
raw_prepare
prepared_raw_review
source_digest
source_digest_review
source_duplicate_guard
candidate_resolution
wiki_context_snapshot
wiki_merge_planning
merge_plan_review
draft_rendering
draft_review
validation
apply_preview
```

## `llmwiki ingest review / approve / revise`

查看并处理真实 review gate。当前 gate 有两个：

- `merge_plan_review`：审核“写哪些页面、为什么写”。如果计划里有
  `needs_human_decision`，或 embedding 召回发现中/强相关旧页但计划仍全部 create，
  pipeline 会停在这里。相关证据见 `wiki_context_snapshot/candidate_contexts.md`
  和 `wiki_merge_planning/merge_decision_report.md`。
- `draft_review`：审核“具体写什么”。update 或 revise 后的草稿必须显式 approve。

查看 review artifacts：

```bash
uv run llmwiki ingest review "$VAULT" "$OP" merge_plan_review
uv run llmwiki ingest review "$VAULT" "$OP" draft_review
```

如果 `merge_plan_review` 停住，先编辑 operation 目录下的
`merge_plan_review/pending_merge_plan.json`，把 `needs_human_decision` 改成
`create`、`update` 或 `noop`，再 approve：

```bash
uv run llmwiki ingest approve "$VAULT" "$OP" merge_plan_review
uv run llmwiki ingest resume "$VAULT" "$OP"
```

如果 `draft_review` 停住，先检查 `draft_review/review_prompt.md`、
`draft_rendering/diffs/` 和 `draft_rendering/draft_pages/`，再 approve：

```bash
uv run llmwiki ingest approve "$VAULT" "$OP" draft_review
uv run llmwiki ingest resume "$VAULT" "$OP"
```

要求模型重新生成当前 review 对应的上游内容：

```bash
uv run llmwiki ingest revise "$VAULT" "$OP" merge_plan_review
uv run llmwiki ingest revise "$VAULT" "$OP" draft_review
uv run llmwiki ingest resume "$VAULT" "$OP"
```

## `llmwiki ingest apply`

把 draft pages 写入 `vault/wiki/`。

```bash
uv run llmwiki ingest apply "$VAULT" "$OP"
```

普通 `apply` 目前只对 `dev` operation 可用。不碰 Git，不要求 vault 是 Git repo。

## `profile` 命令

列出内置 profile：

```bash
uv run llmwiki profile list
```

校验 profile：

```bash
uv run llmwiki profile validate project_basic
uv run llmwiki profile validate /path/to/profile
```

## `eval` 命令

运行模块 eval：

```bash
uv run llmwiki eval run source_digest tests/fixtures/evals/source_digest
```

参数：

- `MODULE`：eval 模块名。
- `DATASET`：fixture dataset 目录。
- `--output-root PATH`：eval 输出目录，默认 `eval_runs`。

查看 eval report：

```bash
uv run llmwiki eval report eval_runs/<run_id>.json
```

## 常见错误和解释

`mock provider ... has no fixture_dir`

mock provider 没有配置 fixture 目录。可以在 `ingest run` 里加：

```bash
--fixture-dir "$FIXTURE"
```

或写进 config：

```yaml
providers:
  default:
    spec: mock:fixture
    fixture_dir: /path/to/mock
```

`Raw input must be inside the vault raw/ directory.`

`RAW` 参数不在 `VAULT/raw/` 下。把文件复制到 `raw/` 后再运行。

`Raw input file does not exist: raw/...`

`RAW` 指向的文件不存在。

`raw file hash changed`

operation 创建后 raw 文件被改过。为了可审计，resume/apply 会阻止继续。

`artifact hash changed`

run artifact 被修改过或损坏，resume/apply 会阻止继续。

`Applied operations are immutable. Start a new operation instead.`

已经 applied 的 operation 不允许 resume。需要重新 `ingest run`。

`.llmwiki/ must not be tracked or staged by Git`

`.llmwiki/` 是本地运行和配置目录，不应该进入 Git。需要先从 Git tracked/staged 状态移除。

`operation is incompatible with current MVP pipeline; rerun ingest`

当前 MVP pipeline 已变化。开发期旧 run 不做迁移，直接重新 ingest。

## Git 边界

`.gitignore` 负责让 `.llmwiki/` 默认不进入 Git。

普通 `apply` 不检查 Git repo，也不提交。
