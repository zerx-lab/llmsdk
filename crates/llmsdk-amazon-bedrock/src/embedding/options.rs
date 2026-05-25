//! Typed view of `provider_options["amazonBedrock"]` for embedding models.
//!
//! Mirrors `amazon-bedrock-embedding-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;
use serde_json::Value;

/// Embedding-side options.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct EmbeddingOptions {
    /// Titan: `256` / `512` / `1024`.
    pub dimensions: Option<u32>,
    /// Titan: normalize output embeddings (defaults to `true`).
    pub normalize: Option<bool>,
    /// Nova: dimensions (`256` / `384` / `1024` / `3072`).
    pub embedding_dimension: Option<u32>,
    /// Nova: embedding purpose enum.
    pub embedding_purpose: Option<String>,
    /// Cohere: `search_document` / `search_query` / `classification` / `clustering`.
    pub input_type: Option<String>,
    /// Cohere / Nova: truncation policy (`"NONE" / "START" / "END"`).
    pub truncate: Option<String>,
    /// Cohere v4+: output dimension.
    pub output_dimension: Option<u32>,
}

/// Parse the `amazonBedrock` (preferred) or `bedrock` (legacy) slot.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> EmbeddingOptions {
    let Some(map) = options else {
        return EmbeddingOptions::default();
    };
    let raw = map.get("amazonBedrock").or_else(|| map.get("bedrock"));
    let Some(value) = raw else {
        return EmbeddingOptions::default();
    };
    serde_json::from_value::<EmbeddingOptions>(Value::Object(value.clone())).unwrap_or_default()
}
