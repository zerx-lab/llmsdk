//! `Anthropic` provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/anthropic`](https://github.com/vercel/ai/tree/main/packages/anthropic).
//! Covers the Messages API (`/v1/messages`) with both `do_generate` and
//! `do_stream`. Files, citations, cache control, server-side tools
//! (`web_search`, `code_execution`, `mcp`, ...) and `thinking` are deferred.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
//! use llmsdk_anthropic::Anthropic;
//! use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
//! use llmsdk_provider::LanguageModel;
//!
//! let provider = Anthropic::builder().api_key("sk-ant-...").build()?;
//! let model = provider.messages("claude-3-5-sonnet-latest");
//!
//! let result = model
//!     .do_generate(CallOptions {
//!         prompt: vec![Message::User {
//!             content: vec![UserPart::Text(TextPart {
//!                 text: "Hi".into(),
//!                 provider_options: None,
//!             })],
//!             provider_options: None,
//!         }],
//!         max_output_tokens: Some(64),
//!         ..Default::default()
//!     })
//!     .await?;
//! println!("{result:?}");
//! # Ok(())
//! # }
//! ```
// Rust guideline compliant 2026-02-21

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod config;
mod error;
mod messages;

pub use config::{Anthropic, AnthropicBuilder};
pub use messages::AnthropicMessagesModel;

/// Default base URL for the `Anthropic` HTTP API.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "ANTHROPIC_API_KEY";

/// Provider id reported via [`llmsdk_provider::LanguageModel::provider`].
pub const PROVIDER_ID: &str = "anthropic";

/// Default `anthropic-version` header value.
///
/// Mirrors `@ai-sdk/anthropic`'s pinned version. Override via
/// [`AnthropicBuilder::version`].
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";
