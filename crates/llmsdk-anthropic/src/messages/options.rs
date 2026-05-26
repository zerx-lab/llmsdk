//! Parse the `anthropic` slot of [`ProviderOptions`] into typed fields.
//!
//! Mirrors `anthropic-language-model-options.ts`.
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
    /// Whether to send `Reasoning` / `ReasoningFile` parts back to the
    /// model. Default `true`. Set `false` for models that don't support
    /// receiving reasoning input.
    pub send_reasoning: Option<bool>,
    /// Strategy for structured outputs: `outputFormat` / `jsonTool` /
    /// `auto`. `outputFormat` routes the schema through native
    /// `output_config.format`; `jsonTool` synthesizes a `name="json"`
    /// function tool and renders its call as text (see
    /// `model::build_request` jsonResponseTool path); `auto` (default)
    /// picks `outputFormat` when the model supports native structured
    /// output and falls back to `jsonTool` otherwise.
    pub structured_output_mode: Option<String>,
    /// Extended-thinking config.
    pub thinking: Option<ThinkingConfig>,
    /// Disable parallel tool calling. When `true`, sent as
    /// `tool_choice.disable_parallel_tool_use = true`.
    pub disable_parallel_tool_use: Option<bool>,
    /// Cache-control hint applied to the request body. Forwarded verbatim
    /// to the wire `cache_control` field.
    pub cache_control: Option<serde_json::Value>,
    /// Request-level metadata (`metadata.user_id` is the only field
    /// upstream defines).
    pub metadata: Option<MetadataConfig>,
    /// MCP server list.
    pub mcp_servers: Option<Vec<McpServerConfig>>,
    /// Container (Skills framework) configuration.
    pub container: Option<serde_json::Value>,
    /// Default `eager_input_streaming` for function tools. Default `true`.
    pub tool_streaming: Option<bool>,
    /// Reasoning effort for agentic flows. Forwarded to `output_config.effort`.
    pub effort: Option<String>,
    /// Task budget for agentic flows.
    pub task_budget: Option<TaskBudgetConfig>,
    /// `fast` / `standard` inference speed (Opus 4.6 only).
    pub speed: Option<String>,
    /// `us` / `global` inference geography.
    pub inference_geo: Option<String>,
    /// Extra `anthropic-beta` tokens to add to the header.
    pub anthropic_beta: Option<Vec<String>>,
    /// Edit strategies that trim context as the conversation grows.
    /// Forwarded verbatim to the wire `context_management` field.
    pub context_management: Option<serde_json::Value>,
}

/// `metadata` block. Today only `userId` is read.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub(crate) struct MetadataConfig {
    /// Caller-supplied opaque user identifier — forwarded as `metadata.user_id`.
    pub user_id: Option<String>,
}

/// One entry in `mcpServers`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct McpServerConfig {
    /// Always `"url"` upstream; we forward it verbatim.
    #[serde(rename = "type")]
    pub kind: String,
    /// Server name.
    pub name: String,
    /// Server URL.
    pub url: String,
    /// Optional authorization token forwarded as `authorization_token`.
    pub authorization_token: Option<String>,
    /// Optional tool configuration (allowed tools / enabled flag).
    pub tool_configuration: Option<McpToolConfiguration>,
}

/// `mcpServers[].toolConfiguration` block.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct McpToolConfiguration {
    /// Whether the server is enabled.
    pub enabled: Option<bool>,
    /// Optional allow-list of tool names.
    pub allowed_tools: Option<Vec<String>>,
}

/// `taskBudget` block.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TaskBudgetConfig {
    /// Always `"tokens"` upstream.
    #[serde(rename = "type")]
    pub kind: String,
    /// Total tokens budgeted for the task.
    pub total: u64,
    /// Optional remaining-budget hint.
    #[serde(default)]
    pub remaining: Option<u64>,
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
    /// `display` (Opus 4.7+) controls whether the model returns thinking
    /// content (`summarized`) or empty placeholder blocks (`omitted`, default).
    Adaptive {
        /// `"omitted"` or `"summarized"`.
        #[serde(default)]
        display: Option<String>,
    },
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
