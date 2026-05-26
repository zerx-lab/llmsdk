//! Wire-level Cohere v2 chat request / response types.
//!
//! Mirrors the embedded zod schemas in
//! `@ai-sdk/cohere/src/cohere-chat-language-model.ts` and the message types
//! in `cohere-chat-prompt.ts`. Cohere v2 chat is NOT OpenAI-compatible: the
//! request carries `messages[]` (with `tool_plan` on assistant turns) and
//! optional `documents[]` for RAG.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};

/// `POST /v2/chat` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Cohere uses `p` for top-p (`nucleus`) sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<f32>,
    /// Cohere uses `k` for top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<WireResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,

    /// RAG documents (passed through as `{data: {text, title?}}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documents: Option<Vec<WireDocument>>,

    /// Reasoning / thinking configuration (Cohere `command-a-reasoning`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<WireThinking>,
}

/// `response_format` wire shape — Cohere uses `json_object` with optional
/// `json_schema` payload.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireResponseFormat {
    JsonObject {
        #[serde(skip_serializing_if = "Option::is_none")]
        json_schema: Option<serde_json::Value>,
    },
}

/// `thinking` wire shape.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireThinking {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_budget: Option<u32>,
}

/// `documents[]` entry.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireDocument {
    pub data: WireDocumentData,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireDocumentData {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_plan: Option<String>,
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

/// `image_url` payload — Cohere accepts the same shape as `OpenAI`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<super::options::CohereImageDetail>,
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

/// `tools[]` entry — Cohere only accepts `type: function`.
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
}

/// `tool_choice` is `NONE` / `REQUIRED` (upper-case) per Cohere docs.
///
/// `auto` becomes "absent" upstream-style.
#[derive(Debug, Clone, Serialize)]
pub(crate) enum WireToolChoice {
    #[serde(rename = "NONE")]
    None,
    #[serde(rename = "REQUIRED")]
    Required,
}

// ---- response ---------------------------------------------------------

/// Non-streaming response body.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatResponse {
    #[serde(default)]
    pub generation_id: Option<String>,
    #[serde(default)]
    pub message: ChatResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(
    dead_code,
    reason = "`role` is captured for telemetry parity with ai-sdk but not surfaced through the trait"
)]
pub(crate) struct ChatResponseMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<ChatResponseContent>>,
    #[serde(default)]
    pub tool_plan: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(default)]
    pub citations: Option<Vec<ChatResponseCitation>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum ChatResponseContent {
    Text { text: String },
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatResponseCitation {
    pub start: u32,
    pub end: u32,
    pub text: String,
    pub sources: Vec<ChatResponseCitationSource>,
    #[serde(default)]
    #[serde(rename = "type")]
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatResponseCitationSource {
    #[serde(default)]
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    pub document: ChatResponseCitationDocument,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatResponseCitationDocument {
    #[serde(default)]
    pub id: Option<String>,
    pub text: String,
    pub title: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct WireUsage {
    #[serde(default)]
    pub billed_units: Option<WireUsageTokens>,
    #[serde(default)]
    pub tokens: Option<WireUsageTokens>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct WireUsageTokens {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
}

// ---- streaming response ----------------------------------------------

/// One SSE chunk frame on the streaming endpoint.
///
/// Cohere streams a discriminated union; we capture only the events used by
/// the implementation. Unknown variants deserialize to [`ChatChunk::Other`]
/// and are silently dropped.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(crate) enum ChatChunk {
    #[serde(rename = "message-start")]
    MessageStart {
        #[serde(default)]
        id: Option<String>,
    },
    #[serde(rename = "content-start")]
    ContentStart {
        index: u32,
        delta: ContentStartDelta,
    },
    #[serde(rename = "content-delta")]
    ContentDelta {
        index: u32,
        delta: ContentDeltaWrapper,
    },
    #[serde(rename = "content-end")]
    ContentEnd { index: u32 },
    #[serde(rename = "tool-plan-delta")]
    ToolPlanDelta { delta: ToolPlanDeltaWrapper },
    #[serde(rename = "tool-call-start")]
    ToolCallStart { delta: ToolCallStartDelta },
    #[serde(rename = "tool-call-delta")]
    ToolCallDelta { delta: ToolCallDeltaWrapper },
    #[serde(rename = "tool-call-end")]
    ToolCallEnd,
    #[serde(rename = "citation-start")]
    CitationStart {
        #[serde(default)]
        delta: Option<CitationStartDelta>,
    },
    #[serde(rename = "citation-end")]
    CitationEnd,
    #[serde(rename = "message-end")]
    MessageEnd { delta: MessageEndDelta },
    /// Catch-all for events we don't care about.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentStartDelta {
    pub message: ContentStartDeltaMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentStartDeltaMessage {
    pub content: ContentStartDeltaContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[allow(
    dead_code,
    reason = "initial text / thinking payload is captured to validate the wire shape; deltas arrive via ContentDelta"
)]
pub(crate) enum ContentStartDeltaContent {
    Text { text: String },
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentDeltaWrapper {
    pub message: ContentDeltaMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ContentDeltaMessage {
    pub content: ContentDeltaContent,
}

/// Cohere doesn't tag content-delta with `type`; presence of `thinking`
/// vs `text` discriminates.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ContentDeltaContent {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolPlanDeltaWrapper {
    pub message: ToolPlanDeltaMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolPlanDeltaMessage {
    pub tool_plan: String,
}

impl ToolPlanDeltaWrapper {
    /// Extract the tool-plan text fragment from this delta.
    pub(crate) fn plan_text(&self) -> &str {
        &self.message.tool_plan
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallStartDelta {
    pub message: ToolCallStartDeltaMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallStartDeltaMessage {
    pub tool_calls: WireToolCall,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallDeltaWrapper {
    pub message: ToolCallDeltaMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallDeltaMessage {
    pub tool_calls: ToolCallDeltaCall,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallDeltaCall {
    pub function: ToolCallDeltaFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallDeltaFunction {
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CitationStartDelta {
    #[serde(default)]
    pub message: Option<CitationStartDeltaMessage>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CitationStartDeltaMessage {
    #[serde(default)]
    pub citations: Option<ChatResponseCitation>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessageEndDelta {
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}
