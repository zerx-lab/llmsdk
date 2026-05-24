//! Embedding model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/embedding-model/v4/*`. Generic over the
//! embedding input type because some providers accept binary inputs
//! (image embeddings) in addition to text.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::shared::{Headers, ProviderMetadata, ProviderOptions, RequestInfo, ResponseInfo};

/// Contract every text-embedding model implements.
///
/// Mirrors `EmbeddingModelV4`. We pin the input type to `String` for now —
/// audio / image embeddings will introduce a parallel trait when needed.
#[async_trait]
pub trait EmbeddingModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"openai"`.
    fn provider(&self) -> &str;

    /// Provider-specific model id, e.g. `"text-embedding-3-small"`.
    fn model_id(&self) -> &str;

    /// Maximum inputs the provider accepts per call.
    ///
    /// `None` means "no documented limit"; callers should still batch
    /// conservatively. Defaults to `None`.
    async fn max_embeddings_per_call(&self) -> Option<u32> {
        None
    }

    /// Whether the model can handle multiple embed calls in parallel.
    async fn supports_parallel_calls(&self) -> bool {
        true
    }

    /// Embed a batch of inputs.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails or
    /// the response is malformed.
    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResult>;
}

/// Options for one [`EmbeddingModel::do_embed`] call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedOptions {
    /// Inputs to embed.
    pub values: Vec<String>,
    /// Extra HTTP headers (HTTP providers only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// One embedding vector.
pub type Embedding = Vec<f32>;

/// Result of [`EmbeddingModel::do_embed`].
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// Embeddings in input order.
    pub embeddings: Vec<Embedding>,
    /// Token usage if reported.
    pub usage: Option<EmbeddingUsage>,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Request info (telemetry).
    pub request: Option<RequestInfo>,
    /// Response info (telemetry).
    pub response: Option<ResponseInfo>,
}

/// Token usage for an embedding call.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingUsage {
    /// Tokens consumed.
    pub tokens: Option<u64>,
}
