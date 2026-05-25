//! Typed view of `provider_options["amazonBedrock"]` for image models.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;
use serde_json::Value;

/// Image-side options.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct ImageProviderOptions {
    /// `"standard"` / `"premium"`.
    pub quality: Option<String>,
    /// CFG scale guidance strength.
    pub cfg_scale: Option<f32>,
    /// Negative-prompt text.
    pub negative_text: Option<String>,
    /// `"natural"` / `"vivid"` (Nova Canvas).
    pub style: Option<String>,
    /// Explicit task type override.
    pub task_type: Option<String>,
    /// Mask prompt (used in lieu of an image mask).
    pub mask_prompt: Option<String>,
    /// Out-painting mode (`"DEFAULT"` / `"PRECISE"`).
    pub out_painting_mode: Option<String>,
    /// Similarity strength for image variations.
    pub similarity_strength: Option<f32>,
}

/// Parse the `amazonBedrock` (preferred) or `bedrock` (legacy) slot.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> ImageProviderOptions {
    let Some(map) = options else {
        return ImageProviderOptions::default();
    };
    let raw = map.get("amazonBedrock").or_else(|| map.get("bedrock"));
    let Some(value) = raw else {
        return ImageProviderOptions::default();
    };
    serde_json::from_value::<ImageProviderOptions>(Value::Object(value.clone())).unwrap_or_default()
}
