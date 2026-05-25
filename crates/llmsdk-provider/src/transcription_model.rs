//! Speech-to-text model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/transcription-model/v4/*`. Implementations
//! turn binary audio into transcribed text plus per-segment timing info.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::shared::{FileBytes, Headers, ProviderMetadata, ProviderOptions, Warning};

/// Contract every speech-to-text model implements.
///
/// Mirrors `TranscriptionModelV4`.
#[async_trait]
pub trait TranscriptionModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"openai.transcription"`.
    fn provider(&self) -> &str;

    /// Model id, e.g. `"whisper-1"` / `"gpt-4o-transcribe"`.
    fn model_id(&self) -> &str;

    /// Specification version (currently `"v4"`).
    fn specification_version(&self) -> &'static str {
        "v4"
    }

    /// Transcribe audio into text + segments.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails or
    /// the response is malformed.
    async fn do_generate(&self, options: TranscriptionOptions) -> Result<TranscriptionResult>;
}

/// Options for one [`TranscriptionModel::do_generate`] call.
///
/// Mirrors `TranscriptionModelV4CallOptions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionOptions {
    /// Raw audio bytes (or a base64 string captured as `FileBytes::Base64`).
    pub audio: FileBytes,
    /// IANA media type of the audio (e.g. `"audio/wav"`).
    #[serde(rename = "mediaType")]
    pub media_type: String,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
    /// Extra HTTP headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
}

/// Result of [`TranscriptionModel::do_generate`].
///
/// Mirrors `TranscriptionModelV4Result`.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Full transcribed text.
    pub text: String,
    /// Time-indexed segments (or words promoted to segments when only word
    /// timings are available).
    pub segments: Vec<TranscriptionSegment>,
    /// Detected ISO 639-1 language code (e.g. `"en"`).
    pub language: Option<String>,
    /// Audio duration in seconds.
    pub duration_in_seconds: Option<f64>,
    /// Warnings.
    pub warnings: Vec<Warning>,
    /// Response info (timestamp / headers / model id).
    pub response: TranscriptionResponseInfo,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
}

/// One transcription segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment text.
    pub text: String,
    /// Start time in seconds.
    #[serde(rename = "startSecond")]
    pub start_second: f64,
    /// End time in seconds.
    #[serde(rename = "endSecond")]
    pub end_second: f64,
}

/// Response metadata for transcription results.
#[derive(Debug, Clone)]
pub struct TranscriptionResponseInfo {
    /// Timestamp the call started (RFC 3339).
    pub timestamp: String,
    /// Model id used.
    pub model_id: String,
    /// Response headers.
    pub headers: Option<Headers>,
    /// Raw response body (for debugging).
    pub body: Option<serde_json::Value>,
}
