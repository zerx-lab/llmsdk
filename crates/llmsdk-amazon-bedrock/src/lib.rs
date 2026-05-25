//! Amazon Bedrock provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/amazon-bedrock`](https://github.com/vercel/ai/tree/main/packages/amazon-bedrock).
//! Surfaces all five model classes Bedrock exposes through a single provider
//! handle:
//!
//! - [`AmazonBedrockChatModel`] — Converse API (`POST /model/{id}/converse` +
//!   `:converse-stream`) for every Bedrock chat model (Anthropic, Nova,
//!   Llama, Mistral, Cohere Command, ...).
//! - [`AmazonBedrockEmbeddingModel`] — `POST /model/{id}/invoke` with
//!   family-aware request bodies (Titan / Nova / Cohere Embed).
//! - [`AmazonBedrockImageModel`] — `POST /model/{id}/invoke` for image
//!   generation (Nova Canvas, Titan Image, SDXL) covering all five Nova task
//!   types (`TEXT_IMAGE` / `IMAGE_VARIATION` / `INPAINTING` /
//!   `OUTPAINTING` / `BACKGROUND_REMOVAL`).
//! - [`AmazonBedrockAnthropicModel`] — Anthropic-on-Bedrock via the
//!   `invoke` / `invoke-with-response-stream` endpoints (re-uses
//!   [`llmsdk_anthropic::Anthropic`] under the hood with a `SigV4` hook).
//! - [`AmazonBedrockRerankingModel`] — `POST /rerank` on the
//!   `bedrock-agent-runtime` endpoint for `amazon.rerank-v1:0` and
//!   `cohere.rerank-v3-5:0`.
//!
//! Authentication is AWS `SigV4` by default (via [`llmsdk_provider_utils::aws_sigv4`])
//! or a bearer token when `AWS_BEARER_TOKEN_BEDROCK` is set.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
//! use llmsdk_amazon_bedrock::AmazonBedrock;
//! use llmsdk_provider::LanguageModel;
//! use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
//!
//! let bedrock = AmazonBedrock::builder()
//!     .region("us-east-1")
//!     .access_key_id("AKIA...")
//!     .secret_access_key("...")
//!     .build()?;
//! let model = bedrock.language_model("anthropic.claude-3-5-sonnet-20241022-v2:0");
//! let _ = model
//!     .do_generate(CallOptions {
//!         prompt: vec![Message::User {
//!             content: vec![UserPart::Text(TextPart {
//!                 text: "hi".into(),
//!                 provider_options: None,
//!             })],
//!             provider_options: None,
//!         }],
//!         ..Default::default()
//!     })
//!     .await?;
//! # Ok(()) }
//! ```
// Rust guideline compliant 2026-05-25

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Pedantic lints disabled with a documented reason:
//
// - `doc_markdown` flags `SigV4 / Converse / EventStream / InvokeModel /
//   ConverseStream` (AWS product / wire-format proper nouns) as missing
//   backticks. They are nouns, not code identifiers; backticking every
//   mention would obscure the prose without helping documentation
//   readers — the same approach the rest of the workspace takes.
// - `too_many_lines` flags four `async fn`s that are essentially long
//   match statements over Bedrock's wire surface (image task-type fan-out,
//   embedding family dispatch, Converse stream chunk handling). Splitting
//   them would only shuffle the same wire-mapping logic into helper
//   functions that would still need to be read sequentially to follow
//   the flow.
// - `struct_excessive_bools` flags the streaming state machine in
//   `chat::stream::State`; the bools (`is_mistral`,
//   `uses_json_response_tool`, `is_json_response_from_tool`,
//   `start_emitted`) mirror upstream fields one-to-one and are not
//   independent enough to warrant a typestate refactor.
// - `default_trait_access` (`Map::default()` vs `Default::default()`) is
//   stylistic; we keep `Default::default()` for symmetry with the rest
//   of the workspace.
#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::struct_excessive_bools,
    clippy::default_trait_access,
    reason = "see lib.rs preamble"
)]

mod anthropic;
mod chat;
mod config;
mod embedding;
mod image;
mod reranking;
mod sigv4_auth;

pub use anthropic::AmazonBedrockAnthropicModel;
pub use chat::AmazonBedrockChatModel;
pub use config::{AmazonBedrock, AmazonBedrockBuilder};
pub use embedding::AmazonBedrockEmbeddingModel;
pub use image::AmazonBedrockImageModel;
pub use reranking::AmazonBedrockRerankingModel;

/// Provider id reported by every Bedrock model handle.
pub const PROVIDER_ID: &str = "amazon-bedrock";

/// Environment variable for AWS region (consulted when not set explicitly).
pub const REGION_ENV_VAR: &str = "AWS_REGION";

/// Environment variable for the static access-key id.
pub const ACCESS_KEY_ID_ENV_VAR: &str = "AWS_ACCESS_KEY_ID";

/// Environment variable for the static secret access key.
pub const SECRET_ACCESS_KEY_ENV_VAR: &str = "AWS_SECRET_ACCESS_KEY";

/// Environment variable for an optional session token.
pub const SESSION_TOKEN_ENV_VAR: &str = "AWS_SESSION_TOKEN";

/// Environment variable for Bedrock bearer-token authentication.
///
/// When set, the provider authenticates with `Authorization: Bearer {token}`
/// instead of computing per-request `SigV4` signatures. Mirrors
/// `@ai-sdk/amazon-bedrock`.
pub const BEARER_TOKEN_ENV_VAR: &str = "AWS_BEARER_TOKEN_BEDROCK";
