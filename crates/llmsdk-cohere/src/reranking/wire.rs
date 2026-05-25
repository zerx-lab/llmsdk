//! Wire types for the Cohere v2 reranking endpoint.
//!
//! Mirrors `cohere-reranking-api.ts`.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

/// `POST /v2/rerank` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_doc: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
}

/// Successful response shape.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RerankResponse {
    #[serde(default)]
    pub id: Option<String>,
    pub results: Vec<RerankResponseResult>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct RerankResponseResult {
    pub index: u32,
    pub relevance_score: f64,
}
