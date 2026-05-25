//! Typed view of `provider_options["amazonBedrock"]` for reranking.
//!
//! Mirrors `amazon-bedrock-reranking-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;
use serde_json::Value;

/// Reranking-side options.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct RerankingProviderOptions {
    /// Pagination token returned by a previous call.
    pub next_token: Option<String>,
    /// Pass-through model-request fields (e.g. `{ "api_version": 2 }`).
    pub additional_model_request_fields: Option<Value>,
}

/// Parse the `amazonBedrock` (preferred) or `bedrock` (legacy) slot.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> RerankingProviderOptions {
    let Some(map) = options else {
        return RerankingProviderOptions::default();
    };
    let raw = map.get("amazonBedrock").or_else(|| map.get("bedrock"));
    let Some(value) = raw else {
        return RerankingProviderOptions::default();
    };
    serde_json::from_value::<RerankingProviderOptions>(Value::Object(value.clone()))
        .unwrap_or_default()
}
