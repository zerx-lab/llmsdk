# llmsdk

[![rust](https://img.shields.io/badge/rust-1.95%2B-orange)](#)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

[English](README.md) · **简体中文**

一个 Rust 实现的 LLM provider SDK，使用体验与 Vercel
[`ai-sdk`](https://github.com/vercel/ai) 保持一致。用过 TypeScript 里
`@ai-sdk/openai` / `@ai-sdk/anthropic` 的话，这里的 API 形态会很熟悉 ——
一样的 provider/model 工厂模式、一样的 call options、一样的流式模型、
一样的 provider-options 透传逃生口。

```rust
let provider = OpenAi::builder().api_key("sk-...").build()?;
let model = provider.chat("gpt-4o-mini");
let result = model.do_generate(opts).await?;
```

## Workspace

| Crate | 说明 |
| --- | --- |
| [`llmsdk-provider`](crates/llmsdk-provider) | 核心 trait、错误类型、共享类型、中间件层 |
| [`llmsdk-provider-utils`](crates/llmsdk-provider-utils) | HTTP / SSE / multipart / API key 加载 |
| [`llmsdk-openai`](crates/llmsdk-openai) | OpenAI Chat / Responses / Embeddings / Images |
| [`llmsdk-anthropic`](crates/llmsdk-anthropic) | Anthropic Messages / Files / Skills / typed server tools |

## 能力矩阵

| Provider | Generate | Stream | Tools | Embedding | Image | Reasoning | Files / Skills |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI Chat       | ✓ | ✓ | ✓ | ✓ (`text-embedding-3-*`) | ✓ (DALL-E 3 / gpt-image-1：generations + edits + variations) | ✓ (o1 / o3 / o4-mini / gpt-5*) | — |
| OpenAI Responses  | ✓ | ✓ | ✓ + 11 个 provider-defined 工具 | — | — | ✓ (reasoning summary 流式) | — |
| Anthropic         | ✓ | ✓ | ✓ + 20 个 typed server 工具 | — | — | ✓ (extended thinking，visible + redacted) | ✓ |

provider 特有的细粒度选项（OpenAI 的 `prediction` / `store` / `service_tier` /
`prompt_cache_key` / `logit_bias` / `text.verbosity` / `strict_json_schema` /
…，Anthropic 的 `cache_control` / `citations` / `context_management` /
`container` / `thinking` budget / …）统一通过 `provider_options.<provider>.*`
透传 —— 与 `ai-sdk` 完全相同的形式。

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

`LanguageModel::do_stream` 返回
`Pin<Box<dyn Stream<Item = Result<StreamPart, _>> + Send>>`，
用 `futures::StreamExt::next()` 消费。Drop stream 即取消，无需 `AbortSignal`。

## 中间件栈

所有模型表面（`LanguageModel` / `EmbeddingModel` / `ImageModel`）都能用
`wrap_*` 组合器叠加跨切面逻辑，列表头最外层执行：

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
- `LoggingMiddleware` — start / end / error + 可选 per-frame stream 事件，自有 `Logger` trait，不绑 tracing
- `CacheMiddleware` — TTL + LRU，stream 边走边收集，命中标记 `provider_metadata.llmsdk.cache = "hit"`
- `DefaultSettingsMiddleware` / `DefaultEmbeddingSettingsMiddleware` — 默认调用选项注入
- `ExtractReasoningMiddleware` — 基于 tag 的 reasoning 切分
- `SimulateStreamingMiddleware` — 把 `do_generate` 结果转成 stream
- `ExtractJsonMiddleware` — 剥离 markdown fence
- `AddToolInputExamplesMiddleware` — 把 examples 拼到 tool description

## 工程约束

- 全 workspace `#![forbid(unsafe_code)]`
- 非测试代码禁止 `unwrap()` / `expect()`，错误一律 `?` + `thiserror`
- 公开 API 必须有 doc comment + 至少一个 doctest 或 example
- 依赖最小化：jitter / LRU / multipart / base64 全部自实现，仅引入 `schemars` 处理 JSON schema
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

## License

Apache License 2.0 — 见 [LICENSE](LICENSE)。
