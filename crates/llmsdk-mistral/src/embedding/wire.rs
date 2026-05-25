//! Wire-level Mistral Embeddings request / response types.
//!
//! Mirrors the embedded zod schemas in
//! `@ai-sdk/mistral/src/mistral-embedding-model.ts`.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

/// `POST /v1/embeddings` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    pub encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dtype: Option<String>,
}

/// `POST /v1/embeddings` response body.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    #[serde(default)]
    pub usage: Option<EmbeddingUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbeddingData {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbeddingUsage {
    pub prompt_tokens: u64,
}
