//! Unified finish reason returned by a language model.
//!
//! Mirrors `language-model-v4-finish-reason.ts`. We pair the unified enum
//! with the provider-raw string so callers can branch on either.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Why the model stopped generating.
///
/// `unified` is normalized across providers; `raw` preserves the original
/// string for telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FinishReason {
    /// Normalized reason.
    pub unified: FinishReasonKind,
    /// Provider-reported raw value. `None` when not provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

/// Normalized reason variants.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum FinishReasonKind {
    /// Model generated a stop sequence.
    Stop,
    /// Model reached max-tokens.
    Length,
    /// Content filter triggered.
    ContentFilter,
    /// Model emitted tool calls.
    ToolCalls,
    /// Model stopped because of an error.
    Error,
    /// Anything else.
    Other,
}

impl FinishReason {
    /// Build a normalized reason without raw provider data.
    #[must_use]
    pub const fn new(kind: FinishReasonKind) -> Self {
        Self {
            unified: kind,
            raw: None,
        }
    }

    /// Build a normalized reason while preserving the provider string.
    pub fn with_raw(kind: FinishReasonKind, raw: impl Into<String>) -> Self {
        Self {
            unified: kind,
            raw: Some(raw.into()),
        }
    }
}
