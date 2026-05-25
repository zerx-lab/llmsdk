//! Parse the `cohere` slot of [`ProviderOptions`] into typed fields.
//!
//! Mirrors `cohereLanguageModelChatOptions` and `cohereImagePartProviderOptions`
//! from `@ai-sdk/cohere/src/cohere-chat-language-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["cohere"]` on the chat call.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct CohereChatOptions {
    /// `thinking` configuration for reasoning models.
    pub thinking: Option<ThinkingConfig>,
}

/// Typed view of `provider_options["cohere"]` on a user file part (`image_url`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct CohereImagePartOptions {
    /// Detail level passed through as `image_url.detail`.
    pub detail: Option<String>,
}

/// `thinking` provider-options shape.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ThinkingConfig {
    /// `enabled` or `disabled`. Defaults to `enabled` upstream.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Maximum thinking tokens.
    pub token_budget: Option<u32>,
}

/// Parse the `cohere` slot of [`ProviderOptions`], or return defaults.
///
/// Unknown / non-object entries fall back to defaults rather than failing
/// the call — ai-sdk has the same forgiving behavior.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> CohereChatOptions {
    let Some(map) = options else {
        return CohereChatOptions::default();
    };
    let Some(cohere) = map.get("cohere") else {
        return CohereChatOptions::default();
    };
    serde_json::from_value::<CohereChatOptions>(serde_json::Value::Object(cohere.clone()))
        .unwrap_or_default()
}

/// Parse the `cohere` slot of a user-file `provider_options`.
pub(crate) fn parse_image_part(options: Option<&ProviderOptions>) -> CohereImagePartOptions {
    let Some(map) = options else {
        return CohereImagePartOptions::default();
    };
    let Some(cohere) = map.get("cohere") else {
        return CohereImagePartOptions::default();
    };
    serde_json::from_value::<CohereImagePartOptions>(serde_json::Value::Object(cohere.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with(map: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("cohere".into(), map.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn missing_options_yields_defaults() {
        assert!(parse(None).thinking.is_none());
        assert!(parse_image_part(None).detail.is_none());
    }

    #[test]
    fn parses_thinking_enabled_with_budget() {
        let po = opts_with(&json!({
            "thinking": { "type": "enabled", "tokenBudget": 2048 }
        }));
        let parsed = parse(Some(&po));
        let t = parsed.thinking.expect("thinking");
        assert_eq!(t.kind.as_deref(), Some("enabled"));
        assert_eq!(t.token_budget, Some(2048));
    }

    #[test]
    fn parses_image_detail() {
        let po = opts_with(&json!({"detail": "high"}));
        let parsed = parse_image_part(Some(&po));
        assert_eq!(parsed.detail.as_deref(), Some("high"));
    }

    #[test]
    fn unknown_keys_ignored() {
        let po = opts_with(&json!({"unknownField": 42}));
        let parsed = parse(Some(&po));
        assert!(parsed.thinking.is_none());
    }
}
