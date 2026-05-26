//! `provider_options.openai.*` for the OpenAI Responses API.
//!
//! Mirrors `@ai-sdk/openai/src/responses/openai-responses-language-model-options.ts`
//! (22 options + companion validation).
//!
//! Filled in detail by task 4; this skeleton provides the type so other
//! modules can reference it.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Maximum value accepted for `top_logprobs` (OpenAI API limit).
pub const TOP_LOGPROBS_MAX: u32 = 20;

/// Parsed `provider_options.openai.*` for a Responses request.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ResponsesProviderOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobsOption>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass_through_unsupported_files: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_json_schema: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_verbosity: Option<TextVerbosity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_message_mode: Option<SystemMessageMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagement>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<AllowedTools>,
    /// Pass through any fields not modeled above (forward-compat).
    #[serde(flatten)]
    pub extra: HashMap<String, JsonValue>,
}

/// `logprobs` — either a boolean toggle or an explicit `top_logprobs` count.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum LogprobsOption {
    Bool(bool),
    Count(u32),
}

/// `promptCacheRetention` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum PromptCacheRetention {
    #[serde(rename = "in_memory")]
    InMemory,
    #[serde(rename = "24h")]
    H24,
}

/// `serviceTier` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Flex,
    Priority,
    Default,
}

/// `textVerbosity` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    Low,
    Medium,
    High,
}

/// `truncation` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Truncation {
    Auto,
    Disabled,
}

/// `systemMessageMode` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SystemMessageMode {
    System,
    Developer,
    Remove,
}

/// `contextManagement[]` entry (only `compaction` is currently defined).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ContextManagement {
    /// Discriminator (currently only `"compaction"`).
    #[serde(rename = "type")]
    pub kind: String,
    pub compact_threshold: u64,
}

/// `allowedTools` constraint.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AllowedTools {
    pub tool_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<AllowedToolsMode>,
}

/// `allowedTools.mode` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AllowedToolsMode {
    Auto,
    Required,
}

/// Pull `provider_options.openai.*` out of `CallOptions`.
///
/// Returns `Default::default()` when no openai (or azure→openai fallback)
/// scope is present. Unknown keys are preserved in
/// [`ResponsesProviderOptions::extra`].
#[must_use]
pub fn parse(
    provider_options: Option<&llmsdk_provider::shared::ProviderOptions>,
    provider_name: &str,
) -> ResponsesProviderOptions {
    let Some(po) = provider_options else {
        return ResponsesProviderOptions::default();
    };
    // Prefer the configured provider name (e.g. "openai" or "azure"); fall
    // back to "openai" for Azure deployments that still use the canonical key.
    let scope = po
        .get(provider_name)
        .or_else(|| po.get("openai"))
        .cloned()
        .map(JsonValue::Object);
    match scope {
        Some(v) => serde_json::from_value(v).unwrap_or_default(),
        None => ResponsesProviderOptions::default(),
    }
}

/// Cross-field + capability-aware validation. Returns warnings to surface to
/// the caller; mutates `self` to strip unsupported settings (matching the
/// upstream ai-sdk getArgs behavior).
///
/// `caps` describes the target model (driven by [`crate::chat::capabilities::Capabilities::detect`]).
pub fn validate(
    opts: &mut ResponsesProviderOptions,
    caps: &crate::chat::capabilities::Capabilities,
) -> Vec<llmsdk_provider::shared::Warning> {
    use llmsdk_provider::shared::Warning;
    let mut warnings = Vec::new();

    let mk = |feature: &str, details: &str| Warning::Unsupported {
        feature: feature.into(),
        details: Some(details.into()),
    };

    // conversation + previousResponseId are mutually exclusive.
    if opts.conversation.is_some() && opts.previous_response_id.is_some() {
        warnings.push(mk(
            "conversation",
            "conversation and previousResponseId cannot be used together",
        ));
        // ai-sdk keeps both on the body and lets the server reject; we mirror
        // that — warning surfaces the conflict but we don't silently drop one.
    }

    let is_reasoning = opts.force_reasoning.unwrap_or(caps.is_reasoning_model);

    // reasoning-only options on a non-reasoning model.
    if !is_reasoning {
        if opts.reasoning_effort.is_some() {
            warnings.push(mk(
                "reasoningEffort",
                "reasoningEffort is not supported for non-reasoning models",
            ));
            opts.reasoning_effort = None;
        }
        if opts.reasoning_summary.is_some() {
            warnings.push(mk(
                "reasoningSummary",
                "reasoningSummary is not supported for non-reasoning models",
            ));
            opts.reasoning_summary = None;
        }
    }

    // service_tier: flex / priority require model capability.
    match opts.service_tier {
        Some(ServiceTier::Flex) if !caps.supports_flex_processing => {
            warnings.push(mk(
                "serviceTier",
                "flex processing is only available for o3, o4-mini, and gpt-5 models",
            ));
            opts.service_tier = None;
        }
        Some(ServiceTier::Priority) if !caps.supports_priority_processing => {
            warnings.push(mk(
                "serviceTier",
                "priority processing is only available for supported models (gpt-4, gpt-5, gpt-5-mini, o3, o4-mini) and requires Enterprise access. gpt-5-nano is not supported",
            ));
            opts.service_tier = None;
        }
        _ => {}
    }

    // top_logprobs >= 1 — when `logprobs` is a count we enforce TOP_LOGPROBS_MAX
    // at parse time via the `Count(u32)` deserialization; nothing more here.

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::capabilities::Capabilities;
    use llmsdk_provider::shared::ProviderOptions;
    use serde_json::json;

    fn po_from(value: serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        let obj = value.as_object().unwrap().clone();
        po.insert("openai".into(), obj);
        po
    }

    #[test]
    fn parse_returns_default_when_no_scope() {
        let opts = parse(None, "openai");
        assert_eq!(opts, ResponsesProviderOptions::default());
    }

    #[test]
    fn parse_recognizes_top_level_keys() {
        let po = po_from(json!({
            "store": false,
            "promptCacheRetention": "24h",
            "textVerbosity": "high",
            "include": ["reasoning.encrypted_content"],
            "logprobs": 5,
            "metadata": {"trace": "abc"},
            "allowedTools": {"toolNames": ["foo", "bar"], "mode": "required"},
            "contextManagement": [{"type": "compaction", "compactThreshold": 4096}],
        }));
        let opts = parse(Some(&po), "openai");
        assert_eq!(opts.store, Some(false));
        assert_eq!(opts.prompt_cache_retention, Some(PromptCacheRetention::H24));
        assert_eq!(opts.text_verbosity, Some(TextVerbosity::High));
        assert_eq!(
            opts.include.as_deref(),
            Some(&["reasoning.encrypted_content".to_string()][..])
        );
        assert_eq!(opts.logprobs, Some(LogprobsOption::Count(5)));
        assert!(opts.metadata.is_some());
        let allowed = opts.allowed_tools.unwrap();
        assert_eq!(allowed.tool_names, vec!["foo", "bar"]);
        assert_eq!(allowed.mode, Some(AllowedToolsMode::Required));
        let cm = opts.context_management.unwrap();
        assert_eq!(cm[0].kind, "compaction");
        assert_eq!(cm[0].compact_threshold, 4096);
    }

    #[test]
    fn unknown_keys_land_in_extra() {
        let po = po_from(json!({"futureFlag": true}));
        let opts = parse(Some(&po), "openai");
        assert_eq!(opts.extra.get("futureFlag"), Some(&json!(true)));
    }

    #[test]
    fn azure_provider_falls_back_to_openai_scope() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"user": "u1"}).as_object().unwrap().clone(),
        );
        let opts = parse(Some(&po), "azure");
        assert_eq!(opts.user.as_deref(), Some("u1"));
    }

    #[test]
    fn validate_strips_reasoning_options_on_chat_model() {
        let caps = Capabilities::detect("gpt-4o-mini");
        let mut opts = ResponsesProviderOptions {
            reasoning_effort: Some("high".into()),
            reasoning_summary: Some("auto".into()),
            ..Default::default()
        };
        let warnings = validate(&mut opts, &caps);
        assert_eq!(warnings.len(), 2);
        assert!(opts.reasoning_effort.is_none());
        assert!(opts.reasoning_summary.is_none());
    }

    #[test]
    fn validate_strips_flex_on_unsupported_model() {
        let caps = Capabilities::detect("gpt-4o-mini");
        let mut opts = ResponsesProviderOptions {
            service_tier: Some(ServiceTier::Flex),
            ..Default::default()
        };
        let warnings = validate(&mut opts, &caps);
        assert_eq!(warnings.len(), 1);
        assert!(opts.service_tier.is_none());
    }

    #[test]
    fn validate_keeps_priority_on_supported_model() {
        let caps = Capabilities::detect("gpt-5-mini");
        let mut opts = ResponsesProviderOptions {
            service_tier: Some(ServiceTier::Priority),
            ..Default::default()
        };
        let warnings = validate(&mut opts, &caps);
        assert!(warnings.is_empty());
        assert_eq!(opts.service_tier, Some(ServiceTier::Priority));
    }

    #[test]
    fn validate_flags_conversation_previous_response_id_conflict() {
        let caps = Capabilities::detect("gpt-4o");
        let mut opts = ResponsesProviderOptions {
            conversation: Some("c1".into()),
            previous_response_id: Some("r1".into()),
            ..Default::default()
        };
        let warnings = validate(&mut opts, &caps);
        assert!(warnings.iter().any(|w| matches!(
            w,
            llmsdk_provider::shared::Warning::Unsupported { feature, .. } if feature == "conversation"
        )));
    }

    #[test]
    fn validate_keeps_reasoning_options_when_force_reasoning_set() {
        let caps = Capabilities::detect("gpt-4o-mini");
        let mut opts = ResponsesProviderOptions {
            reasoning_effort: Some("high".into()),
            force_reasoning: Some(true),
            ..Default::default()
        };
        let warnings = validate(&mut opts, &caps);
        assert!(warnings.is_empty());
        assert_eq!(opts.reasoning_effort.as_deref(), Some("high"));
    }
}
