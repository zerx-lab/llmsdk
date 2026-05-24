# llmsdk

[![spec](https://img.shields.io/badge/ai--sdk%20spec-v4-blue)](https://github.com/vercel/ai/tree/main/packages/provider)
[![rust](https://img.shields.io/badge/rust-1.95%2B-orange)](#)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Rust 实现的 LLM provider SDK，对标 Vercel [`ai-sdk`](https://github.com/vercel/ai) `@ai-sdk/provider` v4。
提供统一的 `LanguageModel` / `EmbeddingModel` / `ImageModel` trait，
以及可组合的中间件栈（retry / logging / cache / 推理抽取 / 流式模拟 / ...）。

> 状态：M1–M10 已完成。Workspace 191 测试全绿；
> `cargo fmt --check` 与 `cargo clippy -- -D warnings` 通过。
> 1.0 之前 API 仍可能变动。

## Workspace 一览

| Crate | 说明 | 对应上游 |
| --- | --- | --- |
| [`llmsdk-provider`](crates/llmsdk-provider) | trait 抽象、统一错误、共享类型、中间件层 | `@ai-sdk/provider` |
| [`llmsdk-provider-utils`](crates/llmsdk-provider-utils) | HTTP / SSE / multipart / api key 加载 | `@ai-sdk/provider-utils` |
| [`llmsdk-openai`](crates/llmsdk-openai) | OpenAI Chat / Embedding / Image | `@ai-sdk/openai` |
| [`llmsdk-anthropic`](crates/llmsdk-anthropic) | Anthropic Messages API | `@ai-sdk/anthropic` |

## 能力矩阵

| Provider | Generate | Stream | Tool Use | Embedding | Image | Reasoning / Thinking |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI    | ✓ | ✓ | ✓ | ✓ (`text-embedding-3-*`) | ✓ (DALL-E 3 / gpt-image-1, generations + edits + variations) | ✓ (o1 / o3 / o4-mini / gpt-5*) |
| Anthropic | ✓ | ✓ | ✓ + 8 个 server-side 工具 | — | — | ✓ (extended thinking, visible + redacted) |

ai-sdk v4 的细粒度特性几乎全部对齐：OpenAI 端覆盖 `prediction` / `store` /
`metadata` / `service_tier` / `prompt_cache_key` / `logit_bias` / `text.verbosity` /
`top_logprobs` / `strict_json_schema` / `web_search_preview` 工具 / `url_citation`
注解 / 流式 error chunk 提取；Anthropic 端覆盖 8 种服务器工具路由、9 种 tool block
解析、5 种位置的 `cache_control`、`citations` + `title` + `context`、
`context_management` / `container` / `compaction`、`thinking` adaptive type 与
budget。详见 [`architecture/0003-m10-design.md`](architecture/0003-m10-design.md)。

未覆盖：Gemini provider、OpenAI Responses API 端点（剩余 9 个 provider-defined
工具）、Anthropic Files API 上传端点。见 [`todo.md`](todo.md)。

## 快速开始

`Cargo.toml`：

```toml
[dependencies]
llmsdk-provider = { git = "https://github.com/zerx-lab/llmsdk" }
llmsdk-openai   = { git = "https://github.com/zerx-lab/llmsdk" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

### OpenAI Chat

```rust
use llmsdk_openai::OpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};

#[tokio::main]
async fn main() -> Result<(), llmsdk_provider::ProviderError> {
    let provider = OpenAi::builder().api_key("sk-...").build()?;
    let model = provider.chat("gpt-4o-mini");

    let result = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "用一句话介绍 Rust 的所有权。".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            ..Default::default()
        })
        .await?;

    println!("{result:?}");
    Ok(())
}
```

### Anthropic Messages

```rust
use llmsdk_anthropic::Anthropic;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};

let provider = Anthropic::builder().api_key("sk-ant-...").build()?;
let model = provider.messages("claude-3-5-sonnet-latest");

let result = model
    .do_generate(CallOptions {
        prompt: vec![Message::User {
            content: vec![UserPart::Text(TextPart {
                text: "Hi".into(),
                provider_options: None,
            })],
            provider_options: None,
        }],
        max_output_tokens: Some(64),
        ..Default::default()
    })
    .await?;
```

### 流式调用

`LanguageModel::do_stream` 返回 `Pin<Box<dyn Stream<Item = Result<StreamPart, _>> + Send>>`，
直接用 `futures::StreamExt::next()` 消费。Drop stream 即取消（无需 `AbortSignal`）。

## 中间件栈

所有模型表面都能用 `wrap_*` 组合器叠加跨切面逻辑，列表头最外层执行：

```rust
use llmsdk_provider::{
    CacheMiddleware, LoggingMiddleware, RetryMiddleware, StderrLogger,
    MemoryCacheStore, wrap_language_model,
};
use std::sync::Arc;

let model = wrap_language_model(
    base_model,
    vec![
        Arc::new(LoggingMiddleware::new(StderrLogger::default())),
        Arc::new(
            RetryMiddleware::builder()
                .max_attempts(4)
                .jitter_ratio(0.2)
                .build(),
        ),
        Arc::new(CacheMiddleware::new(
            MemoryCacheStore::builder().max_entries(1024).build(),
        )),
    ],
);
```

内置中间件：

- `RetryMiddleware` — 指数退避 + jitter，仅对 `is_retryable` 错误重试
- `LoggingMiddleware` — start / end / error + 可选 per-frame stream 事件，自有 `Logger` trait 不绑定 tracing
- `CacheMiddleware` — TTL + LRU，stream 边走边收集，命中标记 `provider_metadata.llmsdk.cache = "hit"`
- `DefaultSettingsMiddleware` / `DefaultEmbeddingSettingsMiddleware` — 选项默认值注入
- `ExtractReasoningMiddleware` — tag-based reasoning 切分
- `SimulateStreamingMiddleware` — `do_generate` 结果转 stream
- `ExtractJsonMiddleware` — 剥离 markdown fence
- `AddToolInputExamplesMiddleware` — examples 拼到 tool description

设计：[`architecture/0002-middleware-design.md`](architecture/0002-middleware-design.md)。

## 工程约束

- `#![forbid(unsafe_code)]` 全 workspace
- 非测试代码禁止 `unwrap()` / `expect()`，错误一律走 `?` + `thiserror`
- 公开 API 必须有 doc comment + 至少一个 doctest 或 example
- 依赖最小化：除 `schemars` 外，jitter / LRU / multipart / base64 全部自实现
- 默认 runtime：`tokio`
- workspace lints 启用 clippy `pedantic` + 部分 restriction lint

## 构建与测试

```bash
cargo check --workspace
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace        # 推荐
# cargo test --workspace             # 备选
```

单 crate 验证：

```bash
cargo check -p llmsdk-openai --lib
cargo nextest run -p llmsdk-anthropic
```

## 文档

- [`architecture/0001-trait-design.md`](architecture/0001-trait-design.md) — provider trait ground truth
- [`architecture/0002-middleware-design.md`](architecture/0002-middleware-design.md) — middleware 层设计
- [`architecture/0003-m10-design.md`](architecture/0003-m10-design.md) — M10 全量 ai-sdk v4 对齐
- [`todo.md`](todo.md) — 跨里程碑追踪与 M11+ 候选
- [`CLAUDE.md`](CLAUDE.md) — 协作约束与里程碑边界

## 路线图

M11+ 候选（待规划，未排序）：

- Gemini provider — 验证 trait 抽象在第三家上的稳定性
- OpenAI Responses API 端点 — 解锁剩余 9 个 provider-defined tools
- Anthropic Files API 端点
- 中间件：分布式缓存参考实现（Redis）、双向 `MiddlewareContext`、tracing span 自动衔接
- 契约测试扩展：image-edit / image-variation / anthropic server tool

## License

Apache License 2.0 — 见 [LICENSE](LICENSE)。
