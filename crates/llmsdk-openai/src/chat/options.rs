//! Parse the `openai` slot of [`ProviderOptions`] into typed fields.
//!
//! Mirrors `openaiLanguageModelChatOptions` (subset). M7 covers the
//! reasoning + logprobs fields; the rest stay deferred (see `todo.md`).
// Rust guideline compliant 2026-02-21

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

/// Typed view of `provider_options["openai"]`.
///
/// Unknown keys are ignored (forward-compatible with provider option
/// growth without breaking older callers).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct OpenAiChatOptions {
    /// Override [`CallOptions::reasoning`]'s effort value. Takes precedence
    /// over the top-level field.
    pub reasoning_effort: Option<String>,
    /// Force-treat the model as a reasoning model regardless of id.
    ///
    /// Useful when a deployment alias hides the underlying reasoning model.
    pub force_reasoning: Option<bool>,
    /// Enable / configure logprobs. `true` returns flat logprobs; an integer
    /// returns the top-N alternates per token.
    pub logprobs: Option<LogprobsOption>,
    /// `top_logprobs` field independent of [`Self::logprobs`].
    ///
    /// When set, sent as-is on the wire; otherwise the value is derived from
    /// `logprobs` (see [`LogprobsOption::top_logprobs`]).
    pub top_logprobs: Option<u32>,
    /// Strict JSON schema enforcement for the `response_format` field.
    /// Defaults to `true`.
    pub strict_json_schema: Option<bool>,
    /// Predicted output content payload (forwarded verbatim).
    pub prediction: Option<serde_json::Value>,
    /// Persist the call on `OpenAI` side.
    pub store: Option<bool>,
    /// Free-form key/value metadata.
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// Service tier: `auto` / `default` / `flex` / `priority`.
    pub service_tier: Option<String>,
    /// Safety identifier (caller-side opaque id).
    pub safety_identifier: Option<String>,
    /// Prompt cache key (shared cache prefix).
    pub prompt_cache_key: Option<String>,
    /// Allow / forbid parallel tool calls.
    pub parallel_tool_calls: Option<bool>,
    /// Per-token bias map.
    pub logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    /// Caller-supplied user id (legacy).
    pub user: Option<String>,
    /// Text-shape configuration; currently only `verbosity` is recognized.
    pub text_verbosity: Option<String>,
}

/// `logprobs` provider option — a boolean toggle or a numeric `top_logprobs`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub(crate) enum LogprobsOption {
    /// Enable / disable logprobs without a top-N alternates list.
    Flag(bool),
    /// Enable logprobs and request this many alternates per token.
    Count(u32),
}

impl LogprobsOption {
    /// Whether logprobs themselves should be requested.
    pub(crate) fn enabled(&self) -> bool {
        match self {
            Self::Flag(b) => *b,
            Self::Count(_) => true,
        }
    }

    /// The `top_logprobs` count to send, if any.
    ///
    /// Matches ai-sdk: a `true` flag implies `top_logprobs: 0`; a `false`
    /// flag returns `None`; a count returns the count.
    pub(crate) fn top_logprobs(&self) -> Option<u32> {
        match self {
            Self::Flag(true) => Some(0),
            Self::Flag(false) => None,
            Self::Count(n) => Some(*n),
        }
    }
}

/// Parse the `openai` slot of [`ProviderOptions`], or return defaults.
///
/// Unknown / non-object entries fall back to defaults rather than failing
/// the call — ai-sdk has the same forgiving behavior.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> OpenAiChatOptions {
    let Some(map) = options else {
        return OpenAiChatOptions::default();
    };
    let Some(openai) = map.get("openai") else {
        return OpenAiChatOptions::default();
    };
    serde_json::from_value::<OpenAiChatOptions>(serde_json::Value::Object(openai.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with(map: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert("openai".into(), map.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn missing_provider_options_yields_defaults() {
        let parsed = parse(None);
        assert!(parsed.reasoning_effort.is_none());
        assert!(parsed.logprobs.is_none());
    }

    #[test]
    fn missing_openai_key_yields_defaults() {
        let mut po = ProviderOptions::new();
        po.insert(
            "anthropic".into(),
            json!({"reasoningEffort": "high"})
                .as_object()
                .cloned()
                .unwrap(),
        );
        let parsed = parse(Some(&po));
        assert!(parsed.reasoning_effort.is_none());
    }

    #[test]
    fn parses_reasoning_effort_and_force() {
        let po = opts_with(&json!({"reasoningEffort": "high", "forceReasoning": true}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(parsed.force_reasoning, Some(true));
    }

    #[test]
    fn logprobs_flag_true_implies_top_zero() {
        let po = opts_with(&json!({"logprobs": true}));
        let parsed = parse(Some(&po));
        let lp = parsed.logprobs.unwrap();
        assert!(lp.enabled());
        assert_eq!(lp.top_logprobs(), Some(0));
    }

    #[test]
    fn logprobs_flag_false_disables() {
        let po = opts_with(&json!({"logprobs": false}));
        let parsed = parse(Some(&po));
        let lp = parsed.logprobs.unwrap();
        assert!(!lp.enabled());
        assert!(lp.top_logprobs().is_none());
    }

    #[test]
    fn logprobs_count_passes_through() {
        let po = opts_with(&json!({"logprobs": 5}));
        let parsed = parse(Some(&po));
        let lp = parsed.logprobs.unwrap();
        assert!(lp.enabled());
        assert_eq!(lp.top_logprobs(), Some(5));
    }

    #[test]
    fn unknown_keys_ignored() {
        let po = opts_with(&json!({"unknownField": 42, "reasoningEffort": "low"}));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("low"));
    }
}
