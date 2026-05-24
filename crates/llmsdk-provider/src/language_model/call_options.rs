//! Call options passed to `do_generate` / `do_stream`.
//!
//! Mirrors `language-model-v4-call-options.ts`. We deliberately omit
//! `abortSignal`: callers cancel by dropping the returned future / stream.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use crate::json::JsonSchema;
use crate::shared::{Headers, ProviderOptions};

use super::prompt::Prompt;
use super::tool::{Tool, ToolChoice};

/// Options for one model invocation.
///
/// Built directly; only `prompt` is required.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CallOptions {
    /// Standardized prompt; not the user-facing prompt.
    pub prompt: Prompt,
    /// Hard cap on generated tokens.
    #[serde(
        default,
        rename = "maxOutputTokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Stop sequences (provider may cap the count).
    #[serde(
        default,
        rename = "stopSequences",
        skip_serializing_if = "Option::is_none"
    )]
    pub stop_sequences: Option<Vec<String>>,
    /// Nucleus sampling.
    #[serde(default, rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-K sampling.
    #[serde(default, rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Presence penalty.
    #[serde(
        default,
        rename = "presencePenalty",
        skip_serializing_if = "Option::is_none"
    )]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty.
    #[serde(
        default,
        rename = "frequencyPenalty",
        skip_serializing_if = "Option::is_none"
    )]
    pub frequency_penalty: Option<f32>,
    /// Desired response format. `None` = provider default (text).
    #[serde(
        default,
        rename = "responseFormat",
        skip_serializing_if = "Option::is_none"
    )]
    pub response_format: Option<ResponseFormat>,
    /// Sampling seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Available tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool-selection policy.
    #[serde(
        default,
        rename = "toolChoice",
        skip_serializing_if = "Option::is_none"
    )]
    pub tool_choice: Option<ToolChoice>,
    /// Include raw chunks in the stream (`do_stream` only).
    #[serde(
        default,
        rename = "includeRawChunks",
        skip_serializing_if = "Option::is_none"
    )]
    pub include_raw_chunks: Option<bool>,
    /// Extra HTTP headers (HTTP providers only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    /// Reasoning effort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningEffort>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// Response format directive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ResponseFormat {
    /// Free-form text.
    Text,
    /// JSON with an optional schema and naming hint.
    Json {
        /// Optional JSON schema constraining the output.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<JsonSchema>,
        /// Logical name of the output structure.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Human-readable description of the output structure.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

/// Reasoning effort level.
///
/// Mirrors ai-sdk's `reasoning` enum: `provider-default` means "do not
/// override the provider's default".
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningEffort {
    /// Defer to the provider.
    ProviderDefault,
    /// Disable reasoning.
    None,
    /// Minimal reasoning.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort.
    Xhigh,
}
