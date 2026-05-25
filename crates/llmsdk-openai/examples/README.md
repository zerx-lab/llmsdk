# llmsdk-openai · examples

OpenAI provider 端到端可用性 smoke 测试。一条命令把 `llmsdk-openai`
当前实现的所有公开能力都跑一遍，用来确认本地配置 / API key / 模型
权限 / SDK 行为是否符合预期。

> **不是单元测试。** 这里调用真实 OpenAI 端点，会**消耗 API 配额**
> （chat / embedding 几乎可忽略，image / responses-web 较贵）。

## 快速开始

```bash
# 1. 在仓库根准备 .env（.gitignore 已忽略）
cp .env.example .env
$EDITOR .env     # 至少填 OPENAI_API_KEY

# 2. 跑全部 10 个 demo
cargo run -p llmsdk-openai --example openai_smoke

# 3. 或只跑单个
cargo run -p llmsdk-openai --example openai_smoke -- stream
```

启动时会打印实际加载到的 `.env` 路径，便于排查"为什么没读到"。

## 示例清单

`openai_smoke` 第一个 CLI 参数选择 demo（默认 `all`）：

| 子命令          | 接口                                  | 验证点                                                                |
| --------------- | ------------------------------------- | --------------------------------------------------------------------- |
| `chat`          | `OpenAi::chat → do_generate`          | 基础文本生成；system + temperature + max_output_tokens                |
| `stream`        | `OpenAi::chat → do_stream`            | **SSE 流式**；逐 token 打印 + 终态 `Finish` 帧聚合 usage              |
| `tools`         | `do_generate × 2`                     | 函数工具多轮：模型 ToolCall → 本地"执行" → 喂回 ToolResult → 总结    |
| `json`          | `ResponseFormat::Json`                | 带 JSON Schema 的结构化输出                                           |
| `vision`        | `UserPart::File + FileData::Url`      | 多模态：图片 URL 输入                                                 |
| `reasoning`     | `ReasoningEffort::Low`                | 推理模型；识别 `Content::Reasoning` 块（部分模型不外露 trace）        |
| `embedding`     | `OpenAi::embedding → do_embed`        | 批量嵌入 + 余弦相似度（EN ↔ ZH 同义 vs 无关）                         |
| `image`         | `OpenAi::image → do_generate`         | 文生图，写盘 + `ImageUsage`                                           |
| `responses`     | `OpenAi::responses → do_generate`     | Responses API 端点（非 Chat Completions）                             |
| `responses-web` | + `Tool::Provider openai.web_search` | provider-defined 工具路由 + `Content::Source` 引用块                  |

任意单个 demo 失败不会终止其它 demo；结束时打印 `N 通过 / M 失败`，
进程退出码反映整体结果（有失败即非零）。

## 环境变量

最少只需 `OPENAI_API_KEY`。其余按需覆盖：

| 变量                       | 默认                              | 说明                                  |
| -------------------------- | --------------------------------- | ------------------------------------- |
| `OPENAI_API_KEY`           | (必填)                            | OpenAI API key                        |
| `OPENAI_BASE_URL`          | `https://api.openai.com/v1`       | 自建网关 / Azure / 代理               |
| `OPENAI_ORG`               | -                                 | `OpenAI-Organization` header          |
| `OPENAI_PROJECT`           | -                                 | `OpenAI-Project` header               |
| `OPENAI_CHAT_MODEL`        | `gpt-4o-mini`                     | `chat` / `stream` / `tools` / `json` |
| `OPENAI_REASONING_MODEL`   | `o3-mini`                         | `reasoning`                           |
| `OPENAI_VISION_MODEL`      | `gpt-4o-mini`                     | `vision`                              |
| `OPENAI_EMBEDDING_MODEL`   | `text-embedding-3-small`          | `embedding`                           |
| `OPENAI_IMAGE_MODEL`       | `dall-e-3`                        | `image`                               |
| `OPENAI_RESPONSES_MODEL`   | `gpt-4o-mini`                     | `responses` / `responses-web`         |
| `OPENAI_VISION_IMAGE_URL`  | wiki 一张 320px JPEG              | `vision` 输入图片                     |
| `OPENAI_IMAGE_OUTPUT_PATH` | `/tmp/llmsdk_openai_demo.png`     | `image` 输出保存路径                  |

## `.env` 加载顺序

实际进程环境变量**永远最高优先**；如果未设置，按以下顺序合并 `.env`
（先到先得，已存在的 key 不会被后续覆盖）：

1. `$CWD/.env`                              — 在仓库根运行时
2. `crates/llmsdk-openai/.env`              — crate-local 覆盖
3. workspace 根 `.env`                      — 通过 `CARGO_MANIFEST_DIR/../..` 编译期路径
4. `crates/llmsdk-openai/examples/.env`     — example 同目录

加载器是零依赖手写的（约 25 行）。支持：
- `KEY=VALUE` / `export KEY=VALUE`
- `#` 行注释
- 首尾的单引号 / 双引号会被剥掉
- 不展开变量、不做转义（够用即可）

## 常见问题

**`401 Unauthorized` / "API key not loaded"**
检查启动时打印的"加载到 N 个 .env 文件"那一行；如果是 0，说明
没找到任何 `.env`。最稳的位置是仓库根 `.env`。

**`reasoning` demo 没打印任何 `(reasoning, ...)` 行**
`o3-mini` / `gpt-5*` 等 reasoning 模型默认**不外露** reasoning
trace（OpenAI 会以加密形式回传，由 SDK 折叠成 `provider_metadata`）。
最终答案行 `→ answer: ...` 仍然会有。

**`responses-web` 报 403 / 工具未生效**
`openai.web_search` 需要账户在 Responses API 上开通了该 provider
tool。降级方案：用 `chat` 端点上的 `openai.web_search_preview`
（M10 已实现路由），或直接跳过该 demo。

**`image` demo 速度慢 / 费贵**
`dall-e-3` 单张约 1-3 秒、$0.04 量级；改 `OPENAI_IMAGE_MODEL=gpt-image-1`
更便宜但要确认账户开通了。开发期间用 `cargo run ... -- chat`
（不带 `all`）省钱。

**只想看 SSE 流式效果**
`cargo run -p llmsdk-openai --example openai_smoke -- stream`。
观察 token 是否一段段冒出来（而不是一次性 dump 完）。

## 改一改

新增 demo：在 `openai_smoke.rs` 里写一个 `async fn demo_xxx(provider: &OpenAi)
-> Result<(), DynErr>`，然后在 `run` 的 `match` 和 `demos` 列表里
各加一行即可。所有 helper（`system` / `user_text` / `print_text_content`
/ `header` / `env_or`）都已就位。
