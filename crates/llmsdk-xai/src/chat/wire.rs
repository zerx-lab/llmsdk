//! Wire-level xAI Chat Completions request / response types.
//!
//! Mirrors the embedded zod schemas in
//! `@ai-sdk/xai/src/xai-chat-language-model.ts`. Only fields actually used
//! by xAI's chat endpoint are surfaced; unknown fields deserialize away.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

/// `stream_options` envelope.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_function_calling: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_parameters: Option<WireSearchParameters>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
}

/// One outgoing message.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub(crate) enum WireMessage {
    System {
        content: String,
    },
    User {
        content: WireUserContent,
    },
    Assistant {
        #[serde(default, skip_serializing_if = "is_empty_string")]
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<WireToolCall>>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

fn is_empty_string(s: &str) -> bool {
    s.is_empty()
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
    File { file: WireFileRef },
}

/// `image_url` payload.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireImageUrl {
    pub url: String,
}

/// `file` payload (xAI uses `{file_id: "..."}` for uploaded files).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireFileRef {
    pub file_id: String,
}

/// Assistant `tool_calls` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: WireToolCallKind,
    pub function: WireFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WireToolCallKind {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// `response_format` wire shape.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseFormat {
    JsonObject,
    JsonSchema { json_schema: WireJsonSchema },
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireJsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

/// `tools` entry.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireTool {
    #[serde(rename = "type")]
    pub kind: WireFunctionKind,
    pub function: WireFunctionDef,
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

/// `search_parameters` wire shape (xAI Live Search).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireSearchParameters {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_citations: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_search_results: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<WireSearchSource>>,
}

/// One source entry inside `search_parameters.sources[]`.
///
/// `type` discriminates web / x / news / rss with `snake_case` wire field names.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum WireSearchSource {
    Web {
        #[serde(skip_serializing_if = "Option::is_none")]
        country: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        excluded_websites: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        allowed_websites: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        safe_search: Option<bool>,
    },
    X {
        #[serde(skip_serializing_if = "Option::is_none")]
        excluded_x_handles: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        included_x_handles: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        post_favorite_count: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        post_view_count: Option<u32>,
    },
    News {
        #[serde(skip_serializing_if = "Option::is_none")]
        country: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        excluded_websites: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        safe_search: Option<bool>,
    },
    Rss {
        links: Vec<String>,
    },
}

// ---- response ---------------------------------------------------------

/// `POST /v1/chat/completions` non-streaming response body.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
    /// xAI Live Search citation URLs (when `search_parameters` is active).
    #[serde(default)]
    pub citations: Option<Vec<String>>,
    /// `code` field on JSON error responses returned with 200 status.
    #[serde(default)]
    pub code: Option<String>,
    /// `error` field on JSON error responses returned with 200 status.
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChoice {
    pub message: ChatChoiceMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Choice index — captured for telemetry parity with upstream but not
    /// surfaced through the trait (xAI's chat completions only emit a single
    /// choice per call).
    #[serde(default)]
    pub _index: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChoiceMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCall>>,
}

// ---- streaming response ----------------------------------------------

/// One SSE chunk frame on the streaming endpoint.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
    #[serde(default)]
    pub citations: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChunkChoice {
    #[serde(default)]
    pub delta: Option<ChatChunkDelta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub index: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChunkDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCall>>,
}

/// Stream-only error body returned as `application/json` instead of SSE.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamErrorBody {
    pub code: String,
    pub error: String,
}

// ---- usage -----------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(
    clippy::struct_field_names,
    reason = "field names mirror xAI's wire schema verbatim"
)]
pub(crate) struct WirePromptTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(
    clippy::struct_field_names,
    reason = "field names mirror xAI's wire schema verbatim"
)]
pub(crate) struct WireCompletionTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u64>,
}
