//! `OpenAI` provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/openai`](https://github.com/vercel/ai/tree/main/packages/openai).
//! Implements three model surfaces: Chat Completions
//! ([`OpenAiChatModel`]), Embeddings ([`OpenAiEmbeddingModel`]), and
//! Image Generation ([`OpenAiImageModel`]).
//!
//! # Quick start
//!
//! ```no_run
//! # async fn doc() -> Result<(), llmsdk_provider::ProviderError> {
//! use llmsdk_openai::OpenAi;
//! use llmsdk_provider::language_model::{CallOptions, Message};
//! use llmsdk_provider::LanguageModel;
//!
//! let provider = OpenAi::builder().api_key("sk-...").build()?;
//! let model = provider.chat("gpt-4o-mini");
//!
//! let result = model
//!     .do_generate(CallOptions {
//!         prompt: vec![Message::System {
//!             content: "Be concise.".into(),
//!             provider_options: None,
//!         }],
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

mod chat;
mod config;
mod embedding;
mod error;
mod image;

pub use chat::OpenAiChatModel;
pub use config::{OpenAi, OpenAiBuilder};
pub use embedding::OpenAiEmbeddingModel;
pub use image::OpenAiImageModel;

/// Default base URL for the `OpenAI` HTTP API.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";

/// Provider id reported via [`llmsdk_provider::LanguageModel::provider`].
pub const PROVIDER_ID: &str = "openai";
