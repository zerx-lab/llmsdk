//! Wire types for Gemini embedding endpoints.
// Rust guideline compliant 2026-05-25

use serde::Deserialize;

/// Response shape for `:embedContent` (single).
#[derive(Debug, Deserialize)]
pub(crate) struct SingleEmbedResponse {
    pub embedding: EmbeddingVec,
}

/// Response shape for `:batchEmbedContents`.
#[derive(Debug, Deserialize)]
pub(crate) struct BatchEmbedResponse {
    pub embeddings: Vec<EmbeddingVec>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingVec {
    pub values: Vec<f32>,
}
