//! Video provider options parser.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::ProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct VideoOptionsParsed {
    #[serde(default, rename = "pollIntervalMs")]
    pub poll_interval_ms: Option<u64>,
    #[serde(default, rename = "pollTimeoutMs")]
    pub poll_timeout_ms: Option<u64>,
    #[serde(default, rename = "personGeneration")]
    pub person_generation: Option<String>,
    #[serde(default, rename = "negativePrompt")]
    pub negative_prompt: Option<String>,
    #[serde(default, rename = "referenceImages")]
    pub reference_images: Option<Vec<ReferenceImage>>,
    /// Pass-through for any other parameter known to upstream.
    #[serde(flatten)]
    pub extras: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct ReferenceImage {
    #[serde(default, rename = "bytesBase64Encoded")]
    pub bytes_base64_encoded: Option<String>,
    #[serde(default, rename = "gcsUri")]
    pub gcs_uri: Option<String>,
}

pub(crate) fn parse(
    provider_options: Option<&ProviderOptions>,
) -> Result<Option<VideoOptionsParsed>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    let Some(payload) = opts.get("google") else {
        return Ok(None);
    };
    let value = Value::Object(payload.clone());
    let parsed: VideoOptionsParsed = serde_json::from_value(value.clone()).map_err(|e| {
        ProviderError::type_validation("provider_options.google", value, e.to_string())
    })?;
    Ok(Some(parsed))
}
