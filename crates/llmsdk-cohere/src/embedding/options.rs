//! Parse the `cohere` slot of [`ProviderOptions`] for embedding calls.
//!
//! Mirrors `cohereEmbeddingModelOptions` in
//! `cohere-embedding-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["cohere"]` on the embed call.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct CohereEmbeddingOptions {
    /// `inputType`: how to interpret the inputs. Defaults to `search_query`.
    pub input_type: Option<String>,
    /// `truncate`: `NONE` / `START` / `END`.
    pub truncate: Option<String>,
    /// `outputDimension`: `256` / `512` / `1024` / `1536` (embed-v4 only).
    pub output_dimension: Option<u32>,
    /// `embeddingTypes`: which numeric encodings to request from the API.
    /// Defaults to `["float"]` when not set.
    pub embedding_types: Option<Vec<String>>,
}

/// Parse the `cohere` slot, or return defaults on missing / malformed entries.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> CohereEmbeddingOptions {
    let Some(map) = options else {
        return CohereEmbeddingOptions::default();
    };
    let Some(cohere) = map.get("cohere") else {
        return CohereEmbeddingOptions::default();
    };
    serde_json::from_value::<CohereEmbeddingOptions>(serde_json::Value::Object(cohere.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn po(v: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("cohere".into(), v.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn defaults_when_missing() {
        let parsed = parse(None);
        assert!(parsed.input_type.is_none());
        assert!(parsed.truncate.is_none());
        assert!(parsed.output_dimension.is_none());
        assert!(parsed.embedding_types.is_none());
    }

    #[test]
    fn parses_all_fields() {
        let opts = po(&json!({
            "inputType": "search_document",
            "truncate": "END",
            "outputDimension": 1024,
            "embeddingTypes": ["float", "int8"]
        }));
        let parsed = parse(Some(&opts));
        assert_eq!(parsed.input_type.as_deref(), Some("search_document"));
        assert_eq!(parsed.truncate.as_deref(), Some("END"));
        assert_eq!(parsed.output_dimension, Some(1024));
        let types = parsed.embedding_types.unwrap();
        assert_eq!(types, vec!["float", "int8"]);
    }
}
