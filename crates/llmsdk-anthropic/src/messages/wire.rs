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
    /// `context_management` edit strategies (provider-option pass-through).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<serde_json::Value>,
    /// `container` Skills configuration (provider-option pass-through).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<serde_json::Value>,
}

/// `thinking` request field.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireThinking {
    Enabled {
        #[serde(skip_serializing_if = "Option::is_none")]
        budget_tokens: Option<u32>,
    },
    Adaptive,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: WireImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: WireDocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Image source: URL or base64.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireImageSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

/// Document source: URL, base64, or inline text.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireDocumentSource {
    Url { url: String, media_type: String },
    Base64 { media_type: String, data: String },
    Text { media_type: String, data: String },
}

/// `cache_control` standard block.
///
/// Anthropic supports `{"type": "ephemeral", "ttl"?: "5m" | "1h"}`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

/// `citations` config block.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CitationsConfig {
    pub enabled: bool,
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

/// `tools[]` entry.
///
/// Anthropic accepts two top-level shapes:
///
/// - Function tools: `{name, description?, input_schema}`
/// - Server tools: `{type: "<tool_type_with_version>", name?, ...args}`
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum WireTool {
    Function(WireFunctionTool),
    Server(WireServerTool),
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireFunctionTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: JsonValue,
}

/// Server-tool wire entry: `{type, name?, ...args}`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WireServerTool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub args: serde_json::Map<String, serde_json::Value>,
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
        /// Optional `citations[]` attached to the text block by server-side
        /// tools (`web_search` / `web_fetch`). Captured as-is.
        #[serde(default)]
        citations: Option<JsonValue>,
    },
    ToolUse {
        id: String,
        name: String,
        input: JsonValue,
        /// Tool-call metadata (`caller`, `dynamic`, `programmatic-tool-call`).
        #[serde(default)]
        caller: Option<JsonValue>,
        #[serde(default)]
        dynamic: Option<bool>,
    },
    /// Server-emitted compaction block (response trimming notice).
    Compaction(JsonValue),
    /// Visible reasoning trace from extended thinking.
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    /// Server-redacted thinking block — only the opaque payload survives.
    RedactedThinking { data: String },
    /// Server-side tool invocation reflected back to the client.
    ServerToolUse {
        id: String,
        name: String,
        input: JsonValue,
    },
    /// Server-side tool result block. Wire `type` ends in `_tool_result`;
    /// the inner shape varies per tool, so we capture the whole object.
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult(JsonValue),
    #[serde(rename = "web_fetch_tool_result")]
    WebFetchToolResult(JsonValue),
    #[serde(rename = "code_execution_tool_result")]
    CodeExecutionToolResult(JsonValue),
    #[serde(rename = "bash_code_execution_tool_result")]
    BashCodeExecutionToolResult(JsonValue),
    #[serde(rename = "text_editor_code_execution_tool_result")]
    TextEditorCodeExecutionToolResult(JsonValue),
    #[serde(rename = "mcp_tool_use")]
    McpToolUse(JsonValue),
    #[serde(rename = "mcp_tool_result")]
    McpToolResult(JsonValue),
    #[serde(rename = "tool_search_tool_result")]
    ToolSearchToolResult(JsonValue),
    #[serde(rename = "advisor_tool_result")]
    AdvisorToolResult(JsonValue),
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
