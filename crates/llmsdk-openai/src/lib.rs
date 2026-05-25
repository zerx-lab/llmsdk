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
mod completion;
mod config;
mod embedding;
mod error;
mod files;
mod image;
mod responses;
mod skills;
mod speech;
mod transcription;

pub use chat::OpenAiChatModel;
pub use completion::OpenAiCompletionLanguageModel;
pub use config::{OpenAi, OpenAiBuilder};
pub use embedding::OpenAiEmbeddingModel;
pub use files::OpenAiFiles;
pub use image::OpenAiImageModel;
pub use responses::OpenAiResponsesLanguageModel;
pub use skills::OpenAiSkills;
pub use speech::OpenAiSpeechModel;
pub use transcription::OpenAiTranscriptionModel;

/// Default base URL for the `OpenAI` HTTP API.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";

/// Provider id reported via [`llmsdk_provider::LanguageModel::provider`].
pub const PROVIDER_ID: &str = "openai";

/// Internal API surface for cross-crate composition.
///
/// Re-exports the wiring primitives ([`Inner`], [`UrlStrategy`]) plus the
/// four model types' `new()` constructors so wrapping providers (Azure
/// `OpenAI` today; potentially other `OpenAI`-compatible vendors tomorrow)
/// can build models without going through the [`OpenAi`] builder.
///
/// **Stability**: this module mirrors `@ai-sdk/openai/internal` — it is
/// considered semi-public and may evolve more frequently than the
/// user-facing API. Pin a specific version if you depend on it directly.
///
/// [`Inner`]: crate::config::Inner
/// [`UrlStrategy`]: crate::config::UrlStrategy
pub mod internal {
    pub use crate::chat::OpenAiChatModel;
    pub use crate::completion::OpenAiCompletionLanguageModel;
    pub use crate::config::{Inner, RequestSigner, UrlStrategy};
    pub use crate::embedding::OpenAiEmbeddingModel;
    pub use crate::files::OpenAiFiles;
    pub use crate::image::OpenAiImageModel;
    pub use crate::responses::OpenAiResponsesLanguageModel;
    pub use crate::skills::OpenAiSkills;
    pub use crate::speech::OpenAiSpeechModel;
    pub use crate::transcription::OpenAiTranscriptionModel;

    /// Provider-defined tool argument / output types (Responses API).
    ///
    /// Re-export of [`crate::responses::tools`] so cross-crate callers
    /// (Azure `OpenAI` today) can construct typed `Tool::Provider` args
    /// without needing direct visibility into the private `responses`
    /// module tree.
    pub mod tools {
        pub use crate::responses::tools::*;
    }
}
