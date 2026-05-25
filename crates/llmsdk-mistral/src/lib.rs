//! Mistral provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/mistral`](https://github.com/vercel/ai/tree/main/packages/mistral).
//! Implements two model surfaces: Chat Completions ([`MistralChatModel`])
//! and Text Embeddings ([`MistralEmbeddingModel`]).
// Rust guideline compliant 2026-05-25

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod chat;
mod config;
mod embedding;

pub use chat::MistralChatModel;
pub use config::{Mistral, MistralBuilder};
pub use embedding::MistralEmbeddingModel;

/// Default base URL for the Mistral HTTP API.
pub const DEFAULT_BASE_URL: &str = "https://api.mistral.ai/v1";

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "MISTRAL_API_KEY";

/// Provider id reported via the `LanguageModel::provider` trait method.
pub const PROVIDER_ID: &str = "mistral";
