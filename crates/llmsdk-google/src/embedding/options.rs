//! Embedding model provider options parser.
//!
//! Mirrors `googleEmbeddingModelOptions` in
//! `@ai-sdk/google/src/google-embedding-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::ProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parsed embedding options.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct EmbeddingOptions {
    #[serde(default, rename = "outputDimensionality")]
    pub output_dimensionality: Option<u32>,
    #[serde(default, rename = "taskType")]
    pub task_type: Option<String>,
    /// Per-value multimodal extra parts; length must match `values.len()`.
    #[serde(default)]
    pub content: Option<Vec<Option<Vec<Value>>>>,
}

/// Parse `provider_options["google"]` into [`EmbeddingOptions`].
pub(crate) fn parse(
    provider_options: Option<&ProviderOptions>,
) -> Result<Option<EmbeddingOptions>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    let Some(payload) = opts.get("google") else {
        return Ok(None);
    };
    let value = Value::Object(payload.clone());
    let parsed: EmbeddingOptions = serde_json::from_value(value.clone()).map_err(|e| {
        ProviderError::type_validation("provider_options.google", value, e.to_string())
    })?;
    Ok(Some(parsed))
}
