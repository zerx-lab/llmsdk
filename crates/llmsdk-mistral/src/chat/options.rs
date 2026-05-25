//! Parse the `mistral` slot of [`ProviderOptions`] into typed fields.
//!
//! Mirrors `mistralLanguageModelChatOptions` from
//! `@ai-sdk/mistral/src/mistral-chat-language-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["mistral"]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct MistralChatOptions {
    /// `safePrompt`: inject Mistral's safety prompt before the conversation.
    pub safe_prompt: Option<bool>,
    /// `documentImageLimit`: cap document image count.
    pub document_image_limit: Option<u32>,
    /// `documentPageLimit`: cap document page count.
    pub document_page_limit: Option<u32>,
    /// `structuredOutputs`: emit `response_format=json_schema` (default true).
    pub structured_outputs: Option<bool>,
    /// `strictJsonSchema`: forward `strict` in the JSON-schema response format
    /// (default false).
    pub strict_json_schema: Option<bool>,
    /// `parallelToolCalls`: forwarded as `parallel_tool_calls` when tools
    /// are present. Default true (Mistral default).
    pub parallel_tool_calls: Option<bool>,
    /// `reasoningEffort`: `"high"` or `"none"` for adjustable-reasoning models.
    pub reasoning_effort: Option<String>,
}

/// Parse the `mistral` slot of [`ProviderOptions`], or return defaults.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> MistralChatOptions {
    let Some(map) = options else {
        return MistralChatOptions::default();
    };
    let Some(slot) = map.get("mistral") else {
        return MistralChatOptions::default();
    };
    serde_json::from_value::<MistralChatOptions>(serde_json::Value::Object(slot.clone()))
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
    fn missing_provider_options_yields_defaults() {
        let parsed = parse(None);
        assert!(parsed.reasoning_effort.is_none());
        assert!(parsed.safe_prompt.is_none());
    }

    #[test]
    fn missing_mistral_key_yields_defaults() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"reasoningEffort": "high"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        let parsed = parse(Some(&po));
        assert!(parsed.reasoning_effort.is_none());
    }

    #[test]
    fn parses_safe_prompt() {
        let po = opts_with(&json!({"safePrompt": true}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.safe_prompt, Some(true));
    }

    #[test]
    fn parses_reasoning_effort() {
        let po = opts_with(&json!({"reasoningEffort": "high"}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn parses_structured_and_strict() {
        let po = opts_with(&json!({"structuredOutputs": false, "strictJsonSchema": true}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.structured_outputs, Some(false));
        assert_eq!(parsed.strict_json_schema, Some(true));
    }

    #[test]
    fn parses_document_limits() {
        let po = opts_with(&json!({"documentImageLimit": 5, "documentPageLimit": 10}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.document_image_limit, Some(5));
        assert_eq!(parsed.document_page_limit, Some(10));
    }

    #[test]
    fn parses_parallel_tool_calls() {
        let po = opts_with(&json!({"parallelToolCalls": false}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.parallel_tool_calls, Some(false));
    }

    #[test]
    fn unknown_keys_ignored() {
        let po = opts_with(&json!({"unknownField": 42, "safePrompt": true}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.safe_prompt, Some(true));
    }
}
