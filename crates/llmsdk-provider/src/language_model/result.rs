//! Results returned by `do_generate` / `do_stream`.
//!
//! Mirrors `-generate-result.ts`, `-stream-result.ts`, `-response-metadata.ts`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::shared::{Headers, ProviderMetadata, RequestInfo, Warning};

use super::BoxStream;
use super::content::Content;
use super::finish_reason::FinishReason;
use super::stream_part::StreamPart;
use super::usage::Usage;

/// Result of [`super::LanguageModel::do_generate`].
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Ordered model output.
    pub content: Vec<Content>,
    /// Why the model stopped.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: Usage,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Request info (telemetry).
    pub request: Option<RequestInfo>,
    /// Response info (telemetry).
    pub response: Option<GenerateResponse>,
    /// Warnings, e.g. unsupported settings.
    pub warnings: Vec<Warning>,
}

/// Response-side info attached to [`GenerateResult`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GenerateResponse {
    /// Inline response metadata.
    #[serde(flatten)]
    pub metadata: ResponseMetadata,
    /// Response HTTP body for debugging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<crate::json::JsonValue>,
}

/// Response metadata reported by a provider mid-stream or post-call.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResponseMetadata {
    /// Provider-reported response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-reported timestamp (ISO-8601).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Provider-reported model id.
    #[serde(default, rename = "modelId", skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Response headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
}

/// Result of [`super::LanguageModel::do_stream`].
///
/// Owns the stream of [`StreamPart`]s. Each item is a `Result` so the
/// transport can surface partial failures; in-stream provider errors are
/// delivered as [`StreamPart::Error`] (still `Ok`).
#[expect(
    missing_debug_implementations,
    reason = "BoxStream is not Debug; trait obj has no useful repr"
)]
pub struct StreamResult {
    /// Yielded parts.
    pub stream: BoxStream<Result<StreamPart>>,
    /// Request info (telemetry).
    pub request: Option<RequestInfo>,
    /// Response headers captured at stream start.
    pub response: Option<StreamResponse>,
}

/// Headers / metadata available when the stream opens.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StreamResponse {
    /// Response headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
}

/// Native-URL support map returned by [`super::LanguageModel::supported_urls`].
///
/// Keys are media-type globs (e.g. `"*/*"`, `"image/*"`, `"application/pdf"`).
/// Values are regex pattern strings — kept as plain `String` so that
/// downstream crates pick their preferred regex engine.
pub type SupportedUrls = HashMap<String, Vec<UrlPattern>>;

/// One supported-URL regex pattern entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct UrlPattern(pub String);

impl UrlPattern {
    /// Wrap an existing pattern string.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self(pattern.into())
    }

    /// Borrow the pattern.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
