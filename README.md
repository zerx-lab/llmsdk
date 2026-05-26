# llmsdk

[![rust](https://img.shields.io/badge/rust-1.95%2B-orange)](#)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**English** · [简体中文](README.zh-CN.md)

A Rust SDK for talking to LLM providers with the same ergonomics as Vercel
[`ai-sdk`](https://github.com/vercel/ai). If you have used `@ai-sdk/openai` or
`@ai-sdk/anthropic` in TypeScript, the shape of the API here will feel
immediately familiar — same provider/model factory pattern, same call options,
same streaming model, same provider-options escape hatch.

```rust
let provider = OpenAi::builder().api_key("sk-...").build()?;
let model = provider.chat("gpt-4o-mini");
let result = model.do_generate(opts).await?;
```

## Workspace

| Crate | Description |
| --- | --- |
| [`llmsdk`](crates/llmsdk) | Umbrella facade — re-exports `llmsdk-provider` at the root plus one feature-gated module per concrete provider |
| [`llmsdk-provider`](crates/llmsdk-provider) | Core traits, error types, shared types, middleware layer |
| [`llmsdk-provider-utils`](crates/llmsdk-provider-utils) | HTTP / SSE / multipart / API-key loading (+ optional AWS SigV4 / EventStream helpers) |
| [`llmsdk-openai`](crates/llmsdk-openai) | OpenAI Chat / Completion / Responses / Embedding / Image / Files / Skills / Speech / Transcription |
| [`llmsdk-anthropic`](crates/llmsdk-anthropic) | Anthropic Messages + Files + Skills + 20 typed server tools |
| [`llmsdk-xai`](crates/llmsdk-xai) | xAI Chat / Responses / Image / Video / Files + 7 typed server tools |
| [`llmsdk-mistral`](crates/llmsdk-mistral) | Mistral Chat + Embedding (incl. magistral reasoning) |
| [`llmsdk-azure`](crates/llmsdk-azure) | Azure OpenAI Chat / Completion / Responses / Embedding / Image / Speech / Transcription |
| [`llmsdk-cohere`](crates/llmsdk-cohere) | Cohere Chat + Embedding + Reranking |
| [`llmsdk-google`](crates/llmsdk-google) | Google Gemini language + Embedding + Imagen + Veo + Files + 8 typed tools |
| [`llmsdk-anthropic-aws`](crates/llmsdk-anthropic-aws) | Claude on AWS (Anthropic-managed deployment, SigV4 / API-key auth) |
| [`llmsdk-amazon-bedrock`](crates/llmsdk-amazon-bedrock) | Amazon Bedrock Converse + Embedding + Image + Anthropic + Reranking |
| [`llmsdk-google-vertex`](crates/llmsdk-google-vertex) | Vertex Gemini + Embedding + Image + Video + Anthropic + xAI + MaaS |

## Capability matrix

`Tools` lists provider-defined / typed server tools (function tools are
available everywhere `Lang` is `✓`). `Reason.` is reasoning output —
either `Content::Reasoning` in the response or `reasoning_effort` /
`thinking` knobs in `provider_options.*`.

| Provider | Lang | Stream | Tools | Reason. | Embed | Image | Video | Rerank | Files | Skills | Speech / STT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI               | ✓ Chat + Completion + Responses | ✓ | 11 (Responses) | ✓ (o1 / o3 / o4-mini / gpt-5* + Responses summary) | ✓ (`text-embedding-3-*`) | ✓ (DALL-E 3 / gpt-image-1: generations + edits + variations) | — | — | ✓ | ✓ | ✓ / ✓ |
| Anthropic            | ✓ Messages | ✓ | 20 | ✓ (extended thinking, visible + redacted) | — | — | — | — | ✓ | ✓ | — / — |
| xAI                  | ✓ Chat + Responses | ✓ | 7  | ✓ (grok-3-mini / grok-4 `reasoning_effort`) | — | ✓ | ✓ (Aurora LRO) | — | ✓ | — | — / — |
| Mistral              | ✓ Chat | ✓ | — | ✓ (magistral `thinking` + `reasoning_effort`) | ✓ | — | — | — | — | — | — / — |
| Azure (OpenAI)       | ✓ Chat + Completion + Responses | ✓ | 11 | ✓ (inherits OpenAI) | ✓ | ✓ | — | — | — | — | ✓ / ✓ |
| Cohere               | ✓ Chat | ✓ | — | ✓ (`tool_plan` → reasoning) | ✓ | — | — | ✓ | — | — | — / — |
| Google Gemini        | ✓ | ✓ | 8 | ✓ (`includeThoughts` + `thinkingBudget`) | ✓ | ✓ (Imagen) | ✓ (Veo LRO) | — | ✓ | — | — / — |
| Anthropic on AWS     | ✓ Messages | ✓ | 20 (Anthropic) | ✓ | — | — | — | — | ✓ | ✓ | — / — |
| Amazon Bedrock       | ✓ Converse + Anthropic-on-Bedrock | ✓ | 19 (Anthropic-on-Bedrock only) | ✓ (reasoning metadata + Anthropic thinking) | ✓ (Titan / Cohere Embed / Nova) | ✓ (Nova, 5 task types) | — | ✓ (`amazon.rerank-v1` / `cohere.rerank-v3-5`) | — | — | — / — |
| Google Vertex        | ✓ Gemini + Anthropic + xAI + MaaS | ✓ | 8 Gemini + 20 Anthropic-on-Vertex | ✓ (inherits Gemini + Anthropic) | ✓ | ✓ (Imagen + Gemini) | ✓ (Veo) | — | — | — | — / — |

Provider-specific knobs (OpenAI `prediction` / `store` / `service_tier` /
`prompt_cache_key` / `logit_bias` / `text.verbosity` / `strict_json_schema` /
…, Anthropic `cache_control` / `citations` / `context_management` /
`container` / `thinking` budget / …) are all exposed through
`provider_options.<provider>.*` — same pattern as `ai-sdk`.

## Quick start

The recommended entry point is the [`llmsdk`](crates/llmsdk) umbrella crate.
Pick the providers you need with cargo features — everything else stays out
of your dependency graph.

`Cargo.toml`:

```toml
[dependencies]
llmsdk = { git = "https://github.com/zerx-lab/llmsdk", features = ["openai", "anthropic"] }
tokio  = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Available features: `openai` · `anthropic` · `xai` · `mistral` · `azure` ·
`cohere` · `google` · `anthropic-aws` · `amazon-bedrock` · `google-vertex` ·
`utils` (the `llmsdk-provider-utils` HTTP/SSE/multipart helpers) · `full`
(everything). All are off by default. The trait crate (`llmsdk-provider`) is
always pulled in and its contents are re-exported at the crate root, so
`use llmsdk::LanguageModel` works directly.

If you want tighter control over what compiles, you can keep depending on
the individual `llmsdk-*` crates — they remain the unit of versioning and
release.

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
                    text: "Explain Rust ownership in one sentence.".into(),
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

### Streaming

`LanguageModel::do_stream` returns
`Pin<Box<dyn Stream<Item = Result<StreamPart, _>> + Send>>`. Consume with
`futures::StreamExt::next()`. Drop the stream to cancel — no `AbortSignal`
needed.

## Middleware stack

Every model surface (`LanguageModel` / `EmbeddingModel` / `ImageModel` /
`VideoModel` / `RerankingModel`) can be wrapped with composable middleware;
`wrap_provider` lifts a `ProviderMiddlewareSet` over an entire `Provider`.
Order is outermost-first:

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

Built-in middleware:

- `RetryMiddleware` — exponential backoff + jitter; retries only `is_retryable` errors
- `LoggingMiddleware` — start / end / error + optional per-frame stream events; pluggable `Logger` trait, no `tracing` lock-in
- `CacheMiddleware` — TTL + LRU; streams are captured progressively; hits tagged with `provider_metadata.llmsdk.cache = "hit"`
- `DefaultSettingsMiddleware` / `DefaultEmbeddingSettingsMiddleware` — inject default call options
- `ExtractReasoningMiddleware` — tag-based reasoning extraction
- `SimulateStreamingMiddleware` — turn a `do_generate` result into a stream
- `ExtractJsonMiddleware` — strip markdown fences from JSON output
- `AddToolInputExamplesMiddleware` — append examples to tool descriptions

## Engineering constraints

- `#![forbid(unsafe_code)]` across the workspace
- No `unwrap()` / `expect()` outside tests — errors flow through `?` + `thiserror`
- Public API requires doc comments + at least one doctest or example
- Minimal dependencies: jitter, LRU, multipart, and base64 are all implemented in-tree. Third-party crates are only pulled in where the protocol forces it: `schemars` for JSON Schema, the `aws-sigv4` / `aws-smithy-eventstream` family for Bedrock + Anthropic-on-AWS, and `gcp_auth` for Vertex Standard Mode OAuth.
- Default runtime: `tokio`
- Workspace lints enable `clippy::pedantic` plus selected restriction lints

## Build & test

```bash
cargo check --workspace
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace        # preferred
# cargo test --workspace             # fallback
```

Single-crate verification:

```bash
cargo check -p llmsdk-openai --lib
cargo nextest run -p llmsdk-anthropic
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
