//! Wire-level `OpenAI` Chat Completions request / response types.
//!
//! Mirrors `openai-chat-api.ts` (limited subset — see module-level docs).
//! Only fields used by M3 are present; new fields are added when a
//! capability needs them.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// `stream_options` envelope.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// `POST /chat/completions` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// `OpenAI` reasoning-model alias for `max_tokens`; set when the model
    /// is a reasoning model and the caller supplied `max_output_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    /// Effort hint for reasoning models (`minimal` / `low` / `medium` /
    /// `high` / `xhigh` / `none`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Request token-level log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of top-N alternates per token (1-20). Requires `logprobs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// Predicted output content (faster responses for known prefixes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<serde_json::Value>,
    /// Persist the call on `OpenAI` side (`true` for replay / fine-tune).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Free-form key/value metadata stored alongside the call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// Service tier: `auto` / `default` / `flex` / `priority`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Caller-supplied identifier used in abuse / safety reporting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Prefix used to share the prompt cache with related calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Allow / forbid parallel tool calls (gpt-4o family).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Per-token bias applied during sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    /// Caller-supplied user identifier (legacy field).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Text-shape configuration (e.g. verbosity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextOptions>,
}

/// `text` configuration in the request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct TextOptions {
    /// Output verbosity: `"low"` / `"medium"` / `"high"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

/// One outgoing message.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub(crate) enum WireMessage {
    System {
        content: String,
    },
    /// `developer` role used by reasoning models in place of `system`.
    Developer {
        content: String,
    },
    User {
        content: WireUserContent,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<WireToolCall>>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// User-message content can be a single string or a list of parts.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum WireUserContent {
    Text(String),
    Parts(Vec<WireUserPart>),
}

/// One user-message content part.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireUserPart {
    Text { text: String },
    ImageUrl { image_url: WireImageUrl },
}

/// `image_url` payload.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireImageUrl {
    pub url: String,
}

/// Assistant `tool_calls` entry.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: WireToolCallKind,
    pub function: WireFunctionCall,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WireToolCallKind {
    Function,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireFunctionCall {
    pub name: String,
    /// Stringified JSON arguments — `OpenAI`'s expected shape.
    pub arguments: String,
}

/// `response_format`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    JsonObject,
    JsonSchema { json_schema: WireJsonSchema },
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireJsonSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: serde_json::Value,
    pub strict: bool,
}

/// `tools` entry.
///
/// Function tools follow the standard nested `{type:"function", function:{...}}`
/// envelope; provider-defined tools (currently only `web_search_preview` on
/// Chat Completions) are flat `{type: "<id>", ...args}` objects.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum WireTool {
    Function {
        #[serde(rename = "type")]
        kind: WireFunctionKind,
        function: WireFunctionDef,
    },
    /// `web_search_preview` (and future Chat-API provider-defined tools).
    /// Serialized as `{type: "<name>", ...args}` — args are flattened so
    /// callers control the exact wire shape via `Tool::Provider::args`.
    Provider {
        #[serde(rename = "type")]
        kind: String,
        #[serde(flatten)]
        args: serde_json::Map<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WireFunctionKind {
    Function,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireFunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// `tool_choice`.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum WireToolChoice {
    Simple(WireToolChoiceSimple),
    Tool {
        #[serde(rename = "type")]
        kind: WireToolCallKind,
        function: WireToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WireToolChoiceSimple {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireToolChoiceFunction {
    pub name: String,
}

// ---- response ---------------------------------------------------------

/// `POST /chat/completions` response body.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ChatResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct ChatChoice {
    pub message: ChatChoiceMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Per-choice token-level log probabilities (when requested).
    #[serde(default)]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct ChatChoiceMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
    /// URL citations attached to the assistant message (web-search models).
    #[serde(default)]
    pub annotations: Option<Vec<Annotation>>,
}

/// One annotation on an assistant message.
///
/// Today only `url_citation` is surfaced; unknown variants deserialize to
/// [`Self::Other`] and are dropped.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Annotation {
    UrlCitation {
        url_citation: UrlCitation,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct UrlCitation {
    pub url: String,
    #[serde(default)]
    pub title: Option<String>,
}

/// Per-choice logprobs payload.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ChoiceLogprobs {
    /// Logprobs for the streamed content tokens.
    #[serde(default)]
    pub content: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ResponseToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub _kind: Option<String>,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct WireUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<WirePromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<WireCompletionTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct WirePromptTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(
    clippy::struct_field_names,
    reason = "field names mirror OpenAI's wire schema verbatim"
)]
pub(crate) struct WireCompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Accepted prediction tokens (when `prediction` was provided).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u64>,
    /// Rejected prediction tokens (when `prediction` was provided).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u64>,
}
