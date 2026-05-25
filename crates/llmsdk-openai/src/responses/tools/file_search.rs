//! `openai.file_search` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/file-search.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Args for `Tool::Provider { id: "openai.file_search", args, .. }`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    pub vector_store_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranking: Option<Ranking>,
    /// Either a `ComparisonFilter` or a `CompoundFilter` (recursive).
    /// Modeled as raw JSON to support arbitrary nesting without explosion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<JsonValue>,
}

/// Ranking options.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Ranking {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

/// One row of the `file_search_call.results[]` array (response side).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ResultRow {
    pub attributes: JsonValue,
    pub file_id: String,
    pub filename: String,
    pub score: f64,
    pub text: String,
}
