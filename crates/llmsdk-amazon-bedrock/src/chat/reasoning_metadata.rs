//! Reasoning metadata round-tripped through `provider_options.amazonBedrock`.
//!
//! Mirrors `amazon-bedrock-reasoning-metadata.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Reasoning metadata round-tripped through provider options.
///
/// Exactly one of `signature` / `redacted_data` is populated.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ReasoningMetadata {
    /// Visible-reasoning signature (round-tripped to enable multi-turn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Redacted opaque blob (`reasoningContent.redactedReasoning.data`).
    #[serde(
        default,
        rename = "redactedData",
        skip_serializing_if = "Option::is_none"
    )]
    pub redacted_data: Option<String>,
}

/// Extract reasoning metadata from a part's `providerOptions` map.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> Option<ReasoningMetadata> {
    let map = options?;
    let value = map.get("amazonBedrock").or_else(|| map.get("bedrock"))?;
    serde_json::from_value::<ReasoningMetadata>(Value::Object(value.clone())).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pop(value: &serde_json::Value, key: &str) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert(key.to_owned(), value.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn parses_signature() {
        let po = pop(&json!({ "signature": "sig-abc" }), "amazonBedrock");
        let m = parse(Some(&po)).unwrap();
        assert_eq!(m.signature.as_deref(), Some("sig-abc"));
        assert!(m.redacted_data.is_none());
    }

    #[test]
    fn parses_redacted_via_legacy_key() {
        let po = pop(&json!({ "redactedData": "deadbeef" }), "bedrock");
        let m = parse(Some(&po)).unwrap();
        assert_eq!(m.redacted_data.as_deref(), Some("deadbeef"));
    }
}
