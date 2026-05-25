//! Cohere provider for llmsdk.
//!
//! Rust port of [`@ai-sdk/cohere`](https://github.com/vercel/ai/tree/main/packages/cohere).
//! Implements three model surfaces: Chat ([`CohereChatModel`]),
//! Embeddings ([`CohereEmbeddingModel`]), and Reranking ([`CohereRerankingModel`]).
//!
//! [`CohereRerankingModel`] is the first implementation of the
//! [`llmsdk_provider::RerankingModel`] trait in the workspace.
// Rust guideline compliant 2026-05-25

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod chat;
mod config;
mod embedding;
mod reranking;

pub use chat::CohereChatModel;
pub use config::{Cohere, CohereBuilder};
pub use embedding::CohereEmbeddingModel;
pub use reranking::CohereRerankingModel;

/// Default base URL for the Cohere v2 API.
pub const DEFAULT_BASE_URL: &str = "https://api.cohere.com/v2";

/// Environment variable consulted when no explicit API key is given.
pub const API_KEY_ENV_VAR: &str = "COHERE_API_KEY";

/// Provider id reported by [`llmsdk_provider::LanguageModel::provider`] and
/// [`llmsdk_provider::EmbeddingModel::provider`].
pub const PROVIDER_ID: &str = "cohere";
