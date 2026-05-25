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
| [`llmsdk`](crates/llmsdk) | 聚合 facade —— 根路径 re-export `llmsdk-provider`，并为每个 provider 暴露一个 feature-gated 模块 |
| [`llmsdk-provider`](crates/llmsdk-provider) | 核心 trait、错误类型、共享类型、中间件层 |
| [`llmsdk-provider-utils`](crates/llmsdk-provider-utils) | HTTP / SSE / multipart / API key 加载 |
| [`llmsdk-openai`](crates/llmsdk-openai) | OpenAI Chat / Responses / Embeddings / Images |
| [`llmsdk-anthropic`](crates/llmsdk-anthropic) | Anthropic Messages / Files / Skills / typed server tools |
| [`llmsdk-xai`](crates/llmsdk-xai) | xAI Chat / Responses / Image / Video / Files / typed server tools |
| [`llmsdk-mistral`](crates/llmsdk-mistral) | Mistral Chat + Embedding（含 magistral reasoning） |
| [`llmsdk-azure`](crates/llmsdk-azure) | Azure OpenAI Chat / Responses / Embedding / Image |
| [`llmsdk-cohere`](crates/llmsdk-cohere) | Cohere Chat + Embedding + Reranking |
| [`llmsdk-google`](crates/llmsdk-google) | Google Gemini language + Embedding + Imagen + Veo + Files + 8 个 typed tools |
| [`llmsdk-anthropic-aws`](crates/llmsdk-anthropic-aws) | Claude on AWS（Anthropic 自有 AWS 部署） |
| [`llmsdk-amazon-bedrock`](crates/llmsdk-amazon-bedrock) | Amazon Bedrock Converse + Embedding + Image + Anthropic + Reranking |
| [`llmsdk-google-vertex`](crates/llmsdk-google-vertex) | Vertex Gemini + Embedding + Image + Video + Anthropic + xAI + MaaS |

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

推荐入口是 [`llmsdk`](crates/llmsdk) 聚合 crate。通过 cargo feature 选择
要启用的 provider —— 没启用的子 crate 不会进入依赖图。

`Cargo.toml`：

```toml
[dependencies]
llmsdk = { git = "https://github.com/zerx-lab/llmsdk", features = ["openai", "anthropic"] }
tokio  = { version = "1", features = ["macros", "rt-multi-thread"] }
```

可用 feature：`openai` · `anthropic` · `xai` · `mistral` · `azure` ·
`cohere` · `google` · `anthropic-aws` · `amazon-bedrock` · `google-vertex` ·
`utils`（即 `llmsdk-provider-utils` 的 HTTP / SSE / multipart helper）·
`full`（全开）。默认全部关闭。trait crate（`llmsdk-provider`）始终
打开并在 crate 根 glob re-export，因此 `use llmsdk::LanguageModel` 直接可用。

如果想精确控制编译产物，也可以继续直接依赖各 `llmsdk-*` 子 crate ——
它们仍是版本与发布的最小单元。

### OpenAI Chat

```rust
use llmsdk::openai::OpenAi;
use llmsdk::{CallOptions, LanguageModel, Message, ProviderError, TextPart, UserPart};

#[tokio::main]
async fn main() -> Result<(), ProviderError> {
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
use llmsdk::anthropic::Anthropic;
use llmsdk::{CallOptions, LanguageModel, Message, TextPart, UserPart};

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
use llmsdk::{
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
