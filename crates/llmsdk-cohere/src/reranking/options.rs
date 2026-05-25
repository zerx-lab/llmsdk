//! Parse the `cohere` slot of [`ProviderOptions`] for rerank calls.
//!
//! Mirrors `cohereRerankingModelOptionsSchema` from
//! `cohere-reranking-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["cohere"]` on the rerank call.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct CohereRerankingOptions {
    /// `maxTokensPerDoc`: long documents are truncated to this many tokens.
    pub max_tokens_per_doc: Option<u32>,
    /// `priority`: request priority hint.
    pub priority: Option<i32>,
}

/// Parse the `cohere` slot, or return defaults on missing / malformed entries.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> CohereRerankingOptions {
    let Some(map) = options else {
        return CohereRerankingOptions::default();
    };
    let Some(cohere) = map.get("cohere") else {
        return CohereRerankingOptions::default();
    };
    serde_json::from_value::<CohereRerankingOptions>(serde_json::Value::Object(cohere.clone()))
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
        assert!(parsed.max_tokens_per_doc.is_none());
        assert!(parsed.priority.is_none());
    }

    #[test]
    fn parses_camel_case_keys() {
        let opts = po(&json!({"maxTokensPerDoc": 4096, "priority": 1}));
        let parsed = parse(Some(&opts));
        assert_eq!(parsed.max_tokens_per_doc, Some(4096));
        assert_eq!(parsed.priority, Some(1));
    }
}
