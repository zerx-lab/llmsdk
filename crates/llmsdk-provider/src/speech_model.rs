//! Text-to-speech model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/speech-model/v4/*`. Implementations turn
//! a single text input into binary audio bytes.
//!
//! Kept separate from [`crate::Provider`] for the same reason as
//! [`crate::FilesModel`]: only providers exposing a TTS endpoint implement
//! this trait.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::shared::{Headers, ProviderMetadata, ProviderOptions, Warning};

/// Contract every text-to-speech model implements.
///
/// Mirrors `SpeechModelV4`.
#[async_trait]
pub trait SpeechModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"openai.speech"`.
    fn provider(&self) -> &str;

    /// Model id, e.g. `"tts-1"` / `"gpt-4o-mini-tts"`.
    fn model_id(&self) -> &str;

    /// Specification version (currently `"v4"`).
    fn specification_version(&self) -> &'static str {
        "v4"
    }

    /// Generate audio bytes from text.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails or
    /// the response is malformed.
    async fn do_generate(&self, options: SpeechOptions) -> Result<SpeechResult>;
}

/// Options for one [`SpeechModel::do_generate`] call.
///
/// Mirrors `SpeechModelV4CallOptions`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeechOptions {
    /// Text to convert to speech.
    pub text: String,
    /// Provider-specific voice id (e.g. `"alloy"` for `OpenAI`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Desired output container / codec (e.g. `"mp3"` / `"wav"`).
    #[serde(
        default,
        rename = "outputFormat",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_format: Option<String>,
    /// Speaking-style instructions (provider-specific).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Playback speed multiplier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// ISO 639-1 language code or `"auto"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
    /// Extra HTTP headers (HTTP providers only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
}

/// Result of [`SpeechModel::do_generate`].
///
/// Mirrors `SpeechModelV4Result`.
#[derive(Debug, Clone)]
pub struct SpeechResult {
    /// Raw audio bytes — encoding matches the requested `output_format`.
    pub audio: Vec<u8>,
    /// Warnings (e.g. setting coerced away).
    pub warnings: Vec<Warning>,
    /// Wire-level request information for telemetry.
    pub request: Option<crate::shared::RequestInfo>,
    /// Response info (timestamp / headers / model id).
    pub response: SpeechResponseInfo,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
}

/// Response metadata attached to [`SpeechResult`].
#[derive(Debug, Clone)]
pub struct SpeechResponseInfo {
    /// Timestamp the call started (RFC 3339 string for cross-rt compatibility).
    pub timestamp: String,
    /// Model id used.
    pub model_id: String,
    /// Response headers.
    pub headers: Option<Headers>,
    /// Raw response body (for debugging).
    pub body: Option<crate::shared::FileBytes>,
}
