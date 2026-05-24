//! Wire types for the streaming chat-completions endpoint.
//!
//! Mirrors `openaiChatChunkSchema` (the union over a normal chunk and the
//! error envelope) in `openai-chat-api.ts`. We model the union as
//! `untagged` and dispatch by structural shape.
// Rust guideline compliant 2026-02-21

use serde::Deserialize;

use super::wire::{Annotation, ChoiceLogprobs, WireUsage};

/// One SSE-decoded chunk.
///
/// The stream is heterogeneous — a chunk is either a normal delta payload,
/// or an OpenAI-style error envelope when the server reports a streaming
/// failure mid-flight.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum ChatChunk {
    /// Error variant must come first: its required `error` field
    /// disambiguates against the all-optional [`ChatDeltaChunk`].
    Error(ChatErrorChunk),
    Delta(ChatDeltaChunk),
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatDeltaChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatChoiceDelta>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChatChoiceDelta {
    #[serde(default)]
    pub delta: Option<ChunkDelta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Per-choice logprobs payload (when requested).
    #[serde(default)]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ChunkDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// URL citations streamed by web-search models.
    #[serde(default)]
    pub annotations: Option<Vec<Annotation>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallDelta {
    /// Index keyed by `OpenAI` to identify which streaming tool call this
    /// delta belongs to. Required.
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ToolCallFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Error chunk emitted mid-stream by `OpenAI`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatErrorChunk {
    pub error: ChatErrorChunkInner,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ChatErrorChunkInner {
    pub message: String,
    /// Error type (`"server_error"` / `"invalid_request_error"` / ...).
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// Provider-specific error code (string or object depending on endpoint).
    #[serde(default)]
    pub code: Option<serde_json::Value>,
}
