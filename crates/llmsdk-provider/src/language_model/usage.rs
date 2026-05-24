//! Token usage reported by a language model call.
//!
//! Mirrors `language-model-v4-usage.ts`. All counters are optional because
//! many providers omit subsets.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use crate::json::JsonObject;

/// Aggregated token usage for a call.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    /// Input-side counters.
    #[serde(rename = "inputTokens")]
    pub input_tokens: InputTokenUsage,
    /// Output-side counters.
    #[serde(rename = "outputTokens")]
    pub output_tokens: OutputTokenUsage,
    /// Provider-native usage object preserved verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<JsonObject>,
}

/// Input-side token counters.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputTokenUsage {
    /// Total input tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// Non-cached input tokens.
    #[serde(default, rename = "noCache", skip_serializing_if = "Option::is_none")]
    pub no_cache: Option<u64>,
    /// Cached input tokens read.
    #[serde(default, rename = "cacheRead", skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
    /// Cached input tokens written.
    #[serde(
        default,
        rename = "cacheWrite",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_write: Option<u64>,
}

/// Output-side token counters.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputTokenUsage {
    /// Total output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// Output text tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<u64>,
    /// Output reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
}
