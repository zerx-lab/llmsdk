//! Wire types for `POST /v1/skills`.
//!
//! Mirrors `openai-skills-api.ts` `openaiSkillResponseSchema`.
// Rust guideline compliant 2026-02-21

use serde::Deserialize;

/// Successful response body from `POST /v1/skills`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireSkillResponse {
    /// Server-assigned skill id.
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default_version: Option<String>,
    #[serde(default)]
    pub latest_version: Option<String>,
    /// Required by the upstream schema; defaulted defensively in case the
    /// server omits it under some plan tier.
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub updated_at: Option<i64>,
}
