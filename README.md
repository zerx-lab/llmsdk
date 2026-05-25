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
| [`llmsdk-provider-utils`](crates/llmsdk-provider-utils) | HTTP / SSE / multipart / API-key loading |
| [`llmsdk-openai`](crates/llmsdk-openai) | OpenAI Chat, Responses, Embeddings, Images |
| [`llmsdk-anthropic`](crates/llmsdk-anthropic) | Anthropic Messages, Files, Skills, typed server tools |
| [`llmsdk-xai`](crates/llmsdk-xai) | xAI Chat, Responses, Image, Video, Files, typed server tools |
| [`llmsdk-mistral`](crates/llmsdk-mistral) | Mistral Chat + Embedding (incl. magistral reasoning) |
| [`llmsdk-azure`](crates/llmsdk-azure) | Azure OpenAI Chat / Responses / Embedding / Image |
| [`llmsdk-cohere`](crates/llmsdk-cohere) | Cohere Chat + Embedding + Reranking |
| [`llmsdk-google`](crates/llmsdk-google) | Google Gemini language + Embedding + Imagen + Veo + Files + 8 typed tools |
| [`llmsdk-anthropic-aws`](crates/llmsdk-anthropic-aws) | Claude on AWS (Anthropic-managed deployment) |
| [`llmsdk-amazon-bedrock`](crates/llmsdk-amazon-bedrock) | Amazon Bedrock Converse + Embedding + Image + Anthropic + Reranking |
| [`llmsdk-google-vertex`](crates/llmsdk-google-vertex) | Vertex Gemini + Embedding + Image + Video + Anthropic + xAI + MaaS |

## Capability matrix

| Provider | Generate | Stream | Tools | Embedding | Image | Reasoning | Files / Skills |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI Chat       | ✓ | ✓ | ✓ | ✓ (`text-embedding-3-*`) | ✓ (DALL-E 3 / gpt-image-1: generations + edits + variations) | ✓ (o1 / o3 / o4-mini / gpt-5*) | — |
| OpenAI Responses  | ✓ | ✓ | ✓ + 11 provider-defined tools | — | — | ✓ (reasoning summary streaming) | — |
| Anthropic         | ✓ | ✓ | ✓ + 20 typed server tools | — | — | ✓ (extended thinking, visible + redacted) | ✓ |

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

Every model surface (`LanguageModel` / `EmbeddingModel` / `ImageModel`) can be
wrapped with composable middleware. Order is outermost-first:

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
- Minimal dependencies: jitter, LRU, multipart, and base64 are all implemented in-tree (only `schemars` is pulled in for JSON schema)
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
