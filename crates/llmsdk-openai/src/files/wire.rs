//! Wire types for `POST /v1/files`.
//!
//! Mirrors `openai-files-api.ts` `openaiFilesResponseSchema`.
// Rust guideline compliant 2026-02-21

use serde::Deserialize;

/// Successful response body from `POST /v1/files`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireFileResponse {
    /// Server-assigned file id.
    pub id: String,
    #[serde(default)]
    pub bytes: Option<u64>,
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub purpose: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub expires_at: Option<i64>,
}
