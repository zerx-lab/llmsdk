//! Parse the `xai` slot of [`ProviderOptions`] into typed responses-API fields.
//!
//! Mirrors `xaiLanguageModelResponsesOptions` from
//! `@ai-sdk/xai/src/responses/xai-responses-language-model-options.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["xai"]` for the responses endpoint.
///
/// All fields are optional; unknown keys are silently ignored (matches the
/// ai-sdk `parseProviderOptions` behaviour). The endpoint-specific keys
/// recognised here are documented on the upstream schema.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct XaiResponsesOptions {
    /// `reasoningEffort`: `none` / `low` / `medium` / `high`.
    pub reasoning_effort: Option<String>,
    /// `reasoningSummary`: `auto` / `concise` / `detailed`.
    pub reasoning_summary: Option<String>,
    /// `logprobs`: enable token-level logprobs.
    pub logprobs: Option<bool>,
    /// `topLogprobs`: 0..=8.
    pub top_logprobs: Option<u32>,
    /// `store`: server-side response retention. Defaults to `true` upstream.
    pub store: Option<bool>,
    /// `previousResponseId`.
    pub previous_response_id: Option<String>,
    /// `include`: additional output data; currently only
    /// `file_search_call.results` is documented.
    pub include: Option<Vec<String>>,
}

/// Parse the `xai` slot of [`ProviderOptions`], or return defaults.
///
/// Unknown / non-object entries fall back to defaults rather than failing
/// the call — ai-sdk has the same forgiving behaviour.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> XaiResponsesOptions {
    let Some(map) = options else {
        return XaiResponsesOptions::default();
    };
    let Some(xai) = map.get("xai") else {
        return XaiResponsesOptions::default();
    };
    serde_json::from_value::<XaiResponsesOptions>(serde_json::Value::Object(xai.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with(map: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("xai".into(), map.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn missing_provider_options_yields_defaults() {
        let parsed = parse(None);
        assert!(parsed.reasoning_effort.is_none());
        assert!(parsed.store.is_none());
    }

    #[test]
    fn parses_full_options() {
        let po = opts_with(&json!({
            "reasoningEffort": "high",
            "reasoningSummary": "detailed",
            "logprobs": true,
            "topLogprobs": 5,
            "store": false,
            "previousResponseId": "resp_abc",
            "include": ["file_search_call.results"]
        }));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(parsed.reasoning_summary.as_deref(), Some("detailed"));
        assert_eq!(parsed.logprobs, Some(true));
        assert_eq!(parsed.top_logprobs, Some(5));
        assert_eq!(parsed.store, Some(false));
        assert_eq!(parsed.previous_response_id.as_deref(), Some("resp_abc"));
        assert_eq!(
            parsed.include.as_deref(),
            Some(&["file_search_call.results".to_owned()][..])
        );
    }

    #[test]
    fn unknown_keys_ignored() {
        let po = opts_with(&json!({"unknownField": 42, "reasoningEffort": "low"}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("low"));
    }
}
