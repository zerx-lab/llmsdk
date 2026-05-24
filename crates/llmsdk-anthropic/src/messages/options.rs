//! Parse the `anthropic` slot of [`ProviderOptions`] into typed fields.
//!
//! Mirrors `anthropic-messages-language-model-options.ts`. M7 covers the
//! `thinking` knob; other Anthropic-specific options remain deferred.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["anthropic"]`.
///
/// Unknown keys are ignored so that callers can use newer ai-sdk fields
/// without forcing a Rust update.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct AnthropicChatOptions {
    /// Extended-thinking config.
    pub thinking: Option<ThinkingConfig>,
    /// Edit strategies that trim context as the conversation grows.
    ///
    /// Forwarded verbatim to the wire `context_management` field.
    pub context_management: Option<serde_json::Value>,
    /// Container (Skills framework) configuration.
    pub container: Option<serde_json::Value>,
}

/// Extended-thinking configuration mirroring Anthropic's `thinking` field.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ThinkingConfig {
    /// Extended thinking is enabled with a budget in tokens.
    Enabled {
        /// Token budget devoted to internal reasoning.
        #[serde(default, rename = "budgetTokens")]
        budget_tokens: Option<u32>,
    },
    /// Adaptive thinking: server decides whether to run thinking blocks.
    Adaptive,
    /// Extended thinking is disabled.
    Disabled,
}

/// Parse the `anthropic` slot or return defaults.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> AnthropicChatOptions {
    let Some(map) = options else {
        return AnthropicChatOptions::default();
    };
    let Some(anthropic) = map.get("anthropic") else {
        return AnthropicChatOptions::default();
    };
    serde_json::from_value::<AnthropicChatOptions>(serde_json::Value::Object(anthropic.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with(value: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("anthropic".into(), value.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn missing_returns_defaults() {
        assert!(parse(None).thinking.is_none());
    }

    #[test]
    fn enabled_with_budget() {
        let po = opts_with(&json!({"thinking": {"type": "enabled", "budgetTokens": 2048}}));
        let parsed = parse(Some(&po));
        assert_eq!(
            parsed.thinking,
            Some(ThinkingConfig::Enabled {
                budget_tokens: Some(2048)
            })
        );
    }

    #[test]
    fn enabled_without_budget_is_still_enabled() {
        let po = opts_with(&json!({"thinking": {"type": "enabled"}}));
        assert_eq!(
            parse(Some(&po)).thinking,
            Some(ThinkingConfig::Enabled {
                budget_tokens: None
            })
        );
    }

    #[test]
    fn disabled_round_trips() {
        let po = opts_with(&json!({"thinking": {"type": "disabled"}}));
        assert_eq!(parse(Some(&po)).thinking, Some(ThinkingConfig::Disabled));
    }
}
