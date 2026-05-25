//! Typed view of `provider_options["amazonBedrock"]` (and the legacy
//! `provider_options["bedrock"]`).
//!
//! Mirrors `amazon-bedrock-chat-language-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;
use serde_json::Value;

/// Typed view of the `amazonBedrock` / `bedrock` provider slot for chat.
///
/// Unknown keys are tolerated so the caller can pass forward-compatible
/// payloads without forcing a Rust update.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct BedrockChatOptions {
    /// Pass-through `additionalModelRequestFields` — Bedrock forwards this
    /// verbatim to the model.
    pub additional_model_request_fields: Option<Value>,
    /// Reasoning configuration (`{ type, budgetTokens, maxReasoningEffort,
    /// display }`).
    pub reasoning_config: Option<ReasoningConfig>,
    /// Anthropic beta tokens to add to `additionalModelRequestFields.anthropic_beta`.
    pub anthropic_beta: Option<Vec<String>>,
    /// Service tier (`"reserved"` / `"priority"` / `"default"` / `"flex"`).
    pub service_tier: Option<String>,
    /// Guardrail configuration (forwarded verbatim).
    pub guardrail_config: Option<Value>,
    /// Performance configuration (`{ "latency": "standard"|"optimized" }`).
    pub performance_config: Option<Value>,
    /// Opaque `requestMetadata` map.
    pub request_metadata: Option<Value>,
    /// `promptVariables` map.
    pub prompt_variables: Option<Value>,
}

/// Reasoning configuration block.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct ReasoningConfig {
    /// `"enabled"` / `"disabled"` / `"adaptive"`.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Budget tokens for explicit reasoning.
    pub budget_tokens: Option<u32>,
    /// Max reasoning effort (`"low" | "medium" | "high" | "xhigh" | "max"`).
    pub max_reasoning_effort: Option<String>,
    /// Display mode for adaptive thinking (`"omitted"` / `"summarized"`).
    pub display: Option<String>,
}

/// Typed view of the `amazonBedrock` slot on a file part — currently only
/// `{ citations: { enabled: bool } }`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct BedrockFilePartOptions {
    /// `{ enabled: bool }` citation toggle.
    pub citations: Option<CitationToggle>,
}

/// `citations` payload.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CitationToggle {
    /// Whether to enable citations for this document.
    pub enabled: bool,
}

/// Parse the `amazonBedrock` (preferred) or `bedrock` (legacy) slot.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> BedrockChatOptions {
    let Some(map) = options else {
        return BedrockChatOptions::default();
    };
    let raw = map.get("amazonBedrock").or_else(|| map.get("bedrock"));
    let Some(value) = raw else {
        return BedrockChatOptions::default();
    };
    serde_json::from_value::<BedrockChatOptions>(Value::Object(value.clone())).unwrap_or_default()
}

/// Parse the file-part level provider options.
pub(crate) fn parse_file_part(options: Option<&ProviderOptions>) -> BedrockFilePartOptions {
    let Some(map) = options else {
        return BedrockFilePartOptions::default();
    };
    let raw = map.get("amazonBedrock").or_else(|| map.get("bedrock"));
    let Some(value) = raw else {
        return BedrockFilePartOptions::default();
    };
    serde_json::from_value::<BedrockFilePartOptions>(Value::Object(value.clone()))
        .unwrap_or_default()
}

/// Optional `cachePoint` extracted from a message / part / system entry.
///
/// Mirrors `getCachePoint` in `convert-to-amazon-bedrock-chat-messages.ts`.
pub(crate) fn parse_cache_point(
    options: Option<&ProviderOptions>,
) -> Option<(String, Option<String>)> {
    let map = options?;
    let value = map.get("amazonBedrock").or_else(|| map.get("bedrock"))?;
    let cache = value.get("cachePoint")?;
    let kind = cache
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_owned();
    let ttl = cache.get("ttl").and_then(Value::as_str).map(str::to_owned);
    Some((kind, ttl))
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
    fn missing_returns_defaults() {
        let parsed = parse(None);
        assert!(parsed.reasoning_config.is_none());
    }

    #[test]
    fn parses_reasoning_enabled() {
        let po = pop(
            &json!({ "reasoningConfig": { "type": "enabled", "budgetTokens": 2048 } }),
            "amazonBedrock",
        );
        let parsed = parse(Some(&po));
        let rc = parsed.reasoning_config.unwrap();
        assert_eq!(rc.kind.as_deref(), Some("enabled"));
        assert_eq!(rc.budget_tokens, Some(2048));
    }

    #[test]
    fn legacy_bedrock_key_is_accepted() {
        let po = pop(&json!({ "serviceTier": "flex" }), "bedrock");
        let parsed = parse(Some(&po));
        assert_eq!(parsed.service_tier.as_deref(), Some("flex"));
    }

    #[test]
    fn cache_point_extracts_type_and_ttl() {
        let po = pop(
            &json!({ "cachePoint": { "type": "default", "ttl": "1h" } }),
            "amazonBedrock",
        );
        let cp = parse_cache_point(Some(&po)).unwrap();
        assert_eq!(cp.0, "default");
        assert_eq!(cp.1.as_deref(), Some("1h"));
    }

    #[test]
    fn citations_enabled_from_file_part() {
        let po = pop(
            &json!({ "citations": { "enabled": true } }),
            "amazonBedrock",
        );
        let parsed = parse_file_part(Some(&po));
        assert!(parsed.citations.unwrap().enabled);
    }
}
