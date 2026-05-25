# llmsdk-anthropic · examples

Anthropic provider 端到端可用性 smoke 测试。一条命令把 `llmsdk-anthropic`
当前实现的所有公开能力（Messages / Files / Skills / typed server tools）
都跑一遍，用来确认本地配置 / API key / 模型权限 / SDK 行为是否符合预期。

> **不是单元测试。** 这里调用真实 Anthropic 端点，会**消耗 API 配额**
> （chat / stream 几乎可忽略，thinking / web-search 稍贵；files / skills
> 走 beta 端点需账户已开通对应 beta）。

## 快速开始

```bash
# 1. 在仓库根准备 .env（.gitignore 已忽略）
cp .env.example .env
$EDITOR .env     # 至少填 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN

# 2. 跑全部 9 个 demo
cargo run -p llmsdk-anthropic --example anthropic_smoke

# 3. 或只跑单个
cargo run -p llmsdk-anthropic --example anthropic_smoke -- stream
```

启动时会打印实际加载到的 `.env` 路径，便于排查"为什么没读到"。

## 示例清单

`anthropic_smoke` 第一个 CLI 参数选择 demo（默认 `all`）：

| 子命令         | 接口                                             | 验证点                                                                |
| -------------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| `chat`         | `Anthropic::messages → do_generate`              | 基础文本生成；system + temperature + max_output_tokens                |
| `stream`       | `Anthropic::messages → do_stream`                | **SSE 流式**；逐 token 打印 + 终态 `Finish` 帧聚合 usage              |
| `tools`        | `do_generate × 2`                                | 函数工具多轮：模型 ToolCall → 本地"执行" → 喂回 ToolResult → 总结    |
| `json`         | `ResponseFormat::Json + structuredOutputMode`    | 带 JSON Schema 的结构化输出（走 wire `output_config.format`）         |
| `vision`       | `UserPart::File + FileData::Url`                 | 多模态：图片 URL 输入                                                 |
| `thinking`     | `provider_options.anthropic.thinking.enabled`    | Extended thinking；识别 `Content::Reasoning` 块（默认 budget 2048）   |
| `web-search`   | `tools::web_search_20260209` typed factory       | 服务端 provider tool；自动 beta header 路由 + `Content::Source` 引用  |
| `files`        | `Anthropic::files → upload_file`                 | Files API（`POST /v1/files`，beta `files-api-2025-04-14`）            |
| `skills`       | `Anthropic::skills → upload_skill`               | Skills API（`POST /v1/skills`，beta `skills-2025-10-02`）             |

任意单个 demo 失败不会终止其它 demo；结束时打印 `N 通过 / M 失败`，
进程退出码反映整体结果（有失败即非零）。

## 命令一览

```bash
# 全跑
cargo run -p llmsdk-anthropic --example anthropic_smoke

# Messages API
cargo run -p llmsdk-anthropic --example anthropic_smoke -- chat
cargo run -p llmsdk-anthropic --example anthropic_smoke -- stream
cargo run -p llmsdk-anthropic --example anthropic_smoke -- tools
cargo run -p llmsdk-anthropic --example anthropic_smoke -- json
cargo run -p llmsdk-anthropic --example anthropic_smoke -- vision
cargo run -p llmsdk-anthropic --example anthropic_smoke -- thinking
cargo run -p llmsdk-anthropic --example anthropic_smoke -- web-search

# Files / Skills 上传端点
cargo run -p llmsdk-anthropic --example anthropic_smoke -- files
cargo run -p llmsdk-anthropic --example anthropic_smoke -- skills

# 临时覆盖模型 / endpoint（不改 .env）
ANTHROPIC_CHAT_MODEL=claude-3-5-haiku-latest \
  cargo run -p llmsdk-anthropic --example anthropic_smoke -- chat

ANTHROPIC_BASE_URL=https://my-proxy.example.com/v1 \
  cargo run -p llmsdk-anthropic --example anthropic_smoke -- chat
```

## 环境变量

最少只需 `ANTHROPIC_API_KEY`（或 `ANTHROPIC_AUTH_TOKEN`）。其余按需覆盖：

| 变量                          | 默认                            | 说明                                                                |
| ----------------------------- | ------------------------------- | ------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`           | (二选一)                        | 设置后走 `x-api-key` header                                         |
| `ANTHROPIC_AUTH_TOKEN`        | (二选一)                        | 设置后走 `Authorization: Bearer`（与 `ANTHROPIC_API_KEY` 互斥）     |
| `ANTHROPIC_BASE_URL`          | `https://api.anthropic.com/v1`  | 自建网关 / 代理 / Bedrock 转发                                      |
| `ANTHROPIC_VERSION`           | `2023-06-01`                    | `anthropic-version` header                                          |
| `ANTHROPIC_CHAT_MODEL`        | `claude-3-5-sonnet-latest`      | `chat` / `stream` / `tools` / `json`                                |
| `ANTHROPIC_VISION_MODEL`      | `claude-3-5-sonnet-latest`      | `vision`                                                            |
| `ANTHROPIC_THINKING_MODEL`    | `claude-3-7-sonnet-latest`      | `thinking`（必须是支持 extended thinking 的模型）                   |
| `ANTHROPIC_WEB_SEARCH_MODEL`  | `claude-3-5-sonnet-latest`      | `web-search`                                                        |
| `ANTHROPIC_VISION_IMAGE_URL`  | wiki 一张 320px JPEG            | `vision` 输入图片                                                   |

`ANTHROPIC_API_KEY` 与 `ANTHROPIC_AUTH_TOKEN` 同时设置时 builder 会直接报错；
示例代码也按"先 api_key 再 auth_token"的顺序选择，避免冲突。

## `.env` 加载顺序

实际进程环境变量**永远最高优先**；如果未设置，按以下顺序合并 `.env`
（先到先得，已存在的 key 不会被后续覆盖）：

1. `$CWD/.env`                                — 在仓库根运行时
2. `crates/llmsdk-anthropic/.env`             — crate-local 覆盖
3. workspace 根 `.env`                        — 通过 `CARGO_MANIFEST_DIR/../..` 编译期路径
4. `crates/llmsdk-anthropic/examples/.env`    — example 同目录

加载器是零依赖手写的（约 25 行）。支持：
- `KEY=VALUE` / `export KEY=VALUE`
- `#` 行注释
- 首尾的单引号 / 双引号会被剥掉
- 不展开变量、不做转义（够用即可）

## 常见问题

**`401 Unauthorized` / "API key not loaded"**
检查启动时打印的"加载到 N 个 .env 文件"那一行；如果是 0，说明
没找到任何 `.env`。最稳的位置是仓库根 `.env`。

**`thinking` demo 没打印任何 `(thinking, ...)` 行**
请确认 `ANTHROPIC_THINKING_MODEL` 指向支持 extended thinking 的模型
（如 `claude-3-7-sonnet-latest` / `claude-opus-4-*` 等）。
3.5 系列不支持，会沉默回退到普通生成；最终答案 `→ answer: ...` 仍然会有。

**`web-search` 报 403 / `tool_use_error`**
`web_search_20260209` 需要账户开通 `code-execution-web-tools-2026-02-09`
beta。降级方案：在源码里改用 `web_search_20250305`（旧版无 beta header 依赖），
或直接跳过该 demo。

**`files` / `skills` demo 报 404 或 `beta required`**
对应的 beta（`files-api-2025-04-14` / `skills-2025-10-02`）需要账户已开通；
SDK 已经自动发对应的 `anthropic-beta` header，无需手工配置。

**只想看 SSE 流式效果**
`cargo run -p llmsdk-anthropic --example anthropic_smoke -- stream`。
观察 token 是否一段段冒出来（而不是一次性 dump 完）。

**OAuth/Bearer 鉴权**
如果你的网关用 `Authorization: Bearer ...`（如 Anthropic OAuth / 部分代理），
设置 `ANTHROPIC_AUTH_TOKEN` 并**清空** `ANTHROPIC_API_KEY`。同时设置两个会被
builder 拒绝（互斥校验）。

## 改一改

新增 demo：在 `anthropic_smoke.rs` 里写一个 `async fn demo_xxx(provider: &Anthropic)
-> Result<(), DynErr>`，然后在 `run` 的 `match` 和 `demos` 列表里
各加一行即可。所有 helper（`system` / `user_text` / `print_text_content`
/ `header` / `env_or`）都已就位。

要切换其它 typed server tool（`computer_*` / `text_editor_*` / `code_execution_*`
/ `memory_20250818` / `tool_search_*` / `advisor_20260301` / ...），
直接从 `llmsdk_anthropic::tools::*` 选对应的 factory 函数替换 `web_search_20260209`
即可——routing / beta header 都由 messages 模块自动处理。
