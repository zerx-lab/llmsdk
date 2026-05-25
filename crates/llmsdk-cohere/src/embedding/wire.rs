//! Wire types for the Cohere v2 embeddings endpoint.
//!
//! Mirrors the embedded zod schemas in `cohere-embedding-model.ts`.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

/// `POST /v2/embed` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct EmbedRequest {
    pub model: String,
    pub texts: Vec<String>,
    pub embedding_types: Vec<String>,
    pub input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimension: Option<u32>,
}

/// Successful response shape.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbedResponse {
    pub embeddings: EmbedResponseVectors,
    #[serde(default)]
    pub meta: Option<EmbedResponseMeta>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbedResponseVectors {
    #[serde(default)]
    pub float: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    pub int8: Option<Vec<Vec<i64>>>,
    #[serde(default)]
    pub uint8: Option<Vec<Vec<i64>>>,
    #[serde(default)]
    pub binary: Option<Vec<Vec<i64>>>,
    #[serde(default)]
    pub ubinary: Option<Vec<Vec<i64>>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EmbedResponseMeta {
    #[serde(default)]
    pub billed_units: Option<EmbedResponseBilledUnits>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct EmbedResponseBilledUnits {
    #[serde(default)]
    pub input_tokens: Option<u64>,
}
