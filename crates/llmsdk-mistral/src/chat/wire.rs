//! Wire-level Mistral Chat Completions request / response types.
//!
//! Mirrors the embedded zod schemas in
//! `@ai-sdk/mistral/src/mistral-chat-language-model.ts`.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

/// `POST /v1/chat/completions` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Mistral-specific: inject Mistral's safety prompt before the conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Mistral spells this `random_seed`, not `seed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Mistral-specific: limit document image pages considered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_image_limit: Option<u32>,
    /// Mistral-specific: limit document pages considered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_page_limit: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// One outgoing message.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub(crate) enum WireMessage {
    System {
        content: String,
    },
    User {
        content: Vec<WireUserPart>,
    },
    Assistant {
        content: String,
        /// Mistral-specific: marks the assistant message as a continuation
        /// prefix the model must complete.
        #[serde(skip_serializing_if = "Option::is_none")]
        prefix: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<WireToolCall>>,
    },
    Tool {
        name: String,
        content: String,
        tool_call_id: String,
    },
}

/// One user-message content part.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireUserPart {
    Text { text: String },
    ImageUrl { image_url: String },
    DocumentUrl { document_url: String },
}

/// Assistant `tool_calls` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireToolCall {
    pub id: String,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<WireToolCallKind>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
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

/// `tool_choice`: Mistral exposes only the simple string form
/// (`"auto"` / `"none"` / `"any"`). To force a specific tool the upstream
/// filters the tool list and emits `"any"` — there is no first-class
/// function form on Mistral.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum WireToolChoice {
    Simple(WireToolChoiceSimple),
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WireToolChoiceSimple {
    Auto,
    None,
    /// Mistral spells `"required"` as `"any"`.
    Any,
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
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChoice {
    pub message: ChatChoiceMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub _index: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChoiceMessage {
    #[serde(default)]
    pub content: Option<MistralContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCall>>,
}

/// Mistral message content can be a plain string or a list of typed parts
/// (text / thinking / `image_url` / reference).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum MistralContent {
    Text(String),
    Parts(Vec<MistralContentPart>),
}

#[allow(
    dead_code,
    reason = "image_url / reference variants are required for the discriminated-union deserializer even though their payloads are dropped"
)]
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MistralContentPart {
    Text {
        text: String,
    },
    Thinking {
        thinking: Vec<MistralThinkingChunk>,
    },
    /// Image-url part can be either a bare string or an object — ignored on
    /// output.
    ImageUrl {
        #[serde(default)]
        image_url: serde_json::Value,
    },
    Reference {
        #[serde(default)]
        reference_ids: serde_json::Value,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MistralThinkingChunk {
    Text { text: String },
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
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChunkChoice {
    #[serde(default)]
    pub delta: Option<ChatChunkDelta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Choice index — captured for parity with upstream but not surfaced.
    #[serde(default)]
    #[allow(
        dead_code,
        reason = "index parsed for parity with upstream wire schema; not surfaced through the trait"
    )]
    pub index: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChunkDelta {
    #[serde(default)]
    pub content: Option<MistralContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCall>>,
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
    /// Mistral exposes cached tokens in three different shapes depending on
    /// the API version; we keep all three as optional fields and read whichever
    /// is populated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<WireCachedTokens>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_token_details: Option<WireCachedTokens>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct WireCachedTokens {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}
