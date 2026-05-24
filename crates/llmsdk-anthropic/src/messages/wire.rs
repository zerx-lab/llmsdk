//! Wire-level `Anthropic` Messages API request and response types.
//!
//! Mirrors a *subset* of `anthropic-api.ts`. Only fields used by M6 are
//! present.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// `POST /v1/messages` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct MessagesRequest {
    pub model: String,
    /// `Anthropic` requires `max_tokens`. We always send a value (caller
    /// default falls back to [`crate::messages::model::DEFAULT_MAX_TOKENS`]).
    pub max_tokens: u32,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    /// Extended-thinking config (omitted when not requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<WireThinking>,
}

/// `thinking` request field.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireThinking {
    Enabled {
        #[serde(skip_serializing_if = "Option::is_none")]
        budget_tokens: Option<u32>,
    },
    Disabled,
}

/// One message in `messages[]`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub(crate) enum WireMessage {
    User { content: Vec<WireUserPart> },
    Assistant { content: Vec<WireAssistantPart> },
}

/// User-message content part.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireUserPart {
    Text {
        text: String,
    },
    Image {
        source: WireImageSource,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Image source: URL or base64.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireImageSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

/// Assistant-message content part.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireAssistantPart {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: JsonValue,
    },
    /// Visible thinking block (signature required for cache replay).
    Thinking {
        thinking: String,
        /// Opaque signature returned by the server; required when replaying
        /// thinking blocks back to `Anthropic`.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Redacted thinking block — opaque to clients.
    RedactedThinking {
        data: String,
    },
}

/// `tools[]` entry — function tool only in M6.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: JsonValue,
}

/// `tool_choice`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

// ---- response ---------------------------------------------------------

/// `POST /v1/messages` response body.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessagesResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub content: Vec<ResponseContent>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    pub usage: ResponseUsage,
}

/// Response content part (only the variants we surface today).
///
/// Unknown variants deserialize to [`Self::Other`] so future server-tool
/// types don't break us.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: JsonValue,
    },
    /// Visible reasoning trace from extended thinking.
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    /// Server-redacted thinking block — only the opaque payload survives.
    RedactedThinking {
        data: String,
    },
    #[serde(other)]
    Other,
}

/// Response `usage` object.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[expect(
    clippy::struct_field_names,
    reason = "field names match Anthropic JSON wire format and must not be renamed"
)]
pub(crate) struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}
