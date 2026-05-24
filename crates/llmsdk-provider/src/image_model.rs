//! Image generation model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/image-model/v4/*`.
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::language_model::FilePart;
use crate::shared::{
    Headers, ProviderMetadata, ProviderOptions, RequestInfo, ResponseInfo, Warning,
};

/// Contract every image-generation model implements.
#[async_trait]
pub trait ImageModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"openai"`.
    fn provider(&self) -> &str;

    /// Provider-specific model id, e.g. `"dall-e-3"`.
    fn model_id(&self) -> &str;

    /// Maximum images that can be requested per call.
    async fn max_images_per_call(&self) -> Option<u32> {
        None
    }

    /// Generate images.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails or
    /// the response is malformed.
    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResult>;
}

/// Options for one [`ImageModel::do_generate`] call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageOptions {
    /// Prompt describing the desired image.
    pub prompt: String,
    /// Number of images to generate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Size, formatted as `WIDTHxHEIGHT` (e.g. `"1024x1024"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Aspect ratio (e.g. `"16:9"`).
    #[serde(
        default,
        rename = "aspectRatio",
        skip_serializing_if = "Option::is_none"
    )]
    pub aspect_ratio: Option<String>,
    /// Random seed for deterministic generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Source images for editing / variation endpoints.
    ///
    /// Plain `do_generate` (text → image) ignores this field. Edit / variation
    /// endpoints take the first entry as the source; `OpenAI`'s edit endpoint
    /// accepts multiple files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<FilePart>>,
    /// Optional mask for image edits (transparent regions = areas to edit).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<FilePart>,
    /// Extra HTTP headers (HTTP providers only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// Result of [`ImageModel::do_generate`].
#[derive(Debug, Clone)]
pub struct ImageResult {
    /// Generated images.
    pub images: Vec<GeneratedImage>,
    /// Warnings for the call, e.g. unsupported settings coerced away.
    pub warnings: Vec<Warning>,
    /// Token usage if reported by the provider (e.g. `OpenAI` `gpt-image-1`).
    pub usage: Option<ImageUsage>,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Request info (telemetry).
    pub request: Option<RequestInfo>,
    /// Response info (telemetry).
    pub response: Option<ResponseInfo>,
}

/// Token usage reported by an image-generation model.
///
/// Mirrors `OpenAI`'s `gpt-image-1` response shape; other providers populate the
/// fields they support and leave the rest `None`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageUsage {
    /// Total input tokens consumed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Total output tokens emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Breakdown of input tokens by modality.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<ImageUsageInputDetails>,
}

/// Input-token breakdown by modality.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageUsageInputDetails {
    /// Text tokens in the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u64>,
    /// Image tokens (edits / variations source).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<u64>,
}

/// One image returned by the provider.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Image bytes (typically PNG / JPEG).
    pub bytes: Bytes,
    /// IANA media type.
    pub media_type: String,
}
