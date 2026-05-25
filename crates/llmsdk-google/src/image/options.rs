//! Image provider options parser.
//!
//! Mirrors `googleImageModelOptionsSchema` in
//! `@ai-sdk/google/src/google-image-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::ProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parsed image options.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct ImageOptionsParsed {
    #[serde(default, rename = "personGeneration")]
    pub person_generation: Option<String>,
    #[serde(default, rename = "aspectRatio")]
    pub aspect_ratio: Option<String>,
    /// Pass-through for `google.tools.googleSearch` args (Gemini image only).
    #[serde(default, rename = "googleSearch")]
    pub google_search: Option<Value>,
    /// Pass-through bag for everything else (`safetyFilterLevel`,
    /// `outputMimeType`, ...).
    #[serde(flatten)]
    pub extras: serde_json::Map<String, Value>,
}

/// Parse `provider_options["google"]` into [`ImageOptionsParsed`].
pub(crate) fn parse(
    provider_options: Option<&ProviderOptions>,
) -> Result<Option<ImageOptionsParsed>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    let Some(payload) = opts.get("google") else {
        return Ok(None);
    };
    let value = Value::Object(payload.clone());
    let parsed: ImageOptionsParsed = serde_json::from_value(value.clone()).map_err(|e| {
        ProviderError::type_validation("provider_options.google", value, e.to_string())
    })?;
    Ok(Some(parsed))
}
