//! Parse the `mistral` slot of [`ProviderOptions`] into typed embedding fields.
//!
//! Mirrors the `MistralEmbeddingOptions` schema implied by the upstream
//! TS interfaces (`outputDimension`, `outputDtype`).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["mistral"]` for embeddings.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct MistralEmbeddingOptions {
    /// `outputDimension`: number of dimensions to return
    /// (e.g. codestral-embed supports configurable dimensions).
    pub output_dimension: Option<u32>,
    /// `outputDtype`: requested precision (`float`, `int8`, `uint8`,
    /// `binary`, `ubinary`).
    pub output_dtype: Option<String>,
}

/// Parse the `mistral` slot of [`ProviderOptions`], or return defaults.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> MistralEmbeddingOptions {
    let Some(map) = options else {
        return MistralEmbeddingOptions::default();
    };
    let Some(slot) = map.get("mistral") else {
        return MistralEmbeddingOptions::default();
    };
    serde_json::from_value::<MistralEmbeddingOptions>(serde_json::Value::Object(slot.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with(map: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("mistral".into(), map.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn missing_returns_defaults() {
        let parsed = parse(None);
        assert!(parsed.output_dimension.is_none());
        assert!(parsed.output_dtype.is_none());
    }

    #[test]
    fn parses_dimension_and_dtype() {
        let po = opts_with(&json!({"outputDimension": 256, "outputDtype": "int8"}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.output_dimension, Some(256));
        assert_eq!(parsed.output_dtype.as_deref(), Some("int8"));
    }

    #[test]
    fn ignores_unknown_keys() {
        let po = opts_with(&json!({"unknown": true, "outputDimension": 32}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.output_dimension, Some(32));
    }
}
