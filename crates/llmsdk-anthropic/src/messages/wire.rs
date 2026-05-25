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
    /// `output_config` block carrying `effort` / `task_budget` / `format`
    /// (structured-output schema). Built up from provider options +
    /// `response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<serde_json::Value>,
    /// Inference speed (`"fast"` / `"standard"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
    /// Inference geography (`"us"` / `"global"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// Request-level `cache_control` hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
    /// Request `metadata` (today only `user_id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// `mcp_servers` list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<serde_json::Value>,
}

/// `thinking` request field.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireThinking {
    Enabled {
        #[serde(skip_serializing_if = "Option::is_none")]
        budget_tokens: Option<u32>,
    },
    Adaptive {
        /// `"omitted"` (Opus 4.7+ default — blocks present but empty)
        /// or `"summarized"` (reasoning content returned).
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<String>,
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
        content: WireToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// `tool_result.content` payload.
///
/// Anthropic accepts either a string or an array of nested content parts
/// (text / image / document / `tool_reference`). Mirrors the union type in
/// `anthropic-api.ts` `AnthropicToolResultContent.content`.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum WireToolResultContent {
    Text(String),
    Parts(Vec<WireNestedToolResultContent>),
}

/// One nested part inside [`WireToolResultContent::Parts`].
///
/// Mirrors `AnthropicNestedTextContent` / `AnthropicNestedImageContent` /
/// `AnthropicNestedDocumentContent` / `AnthropicToolReferenceContent` in
/// upstream. Nested document parts intentionally omit `cache_control`
/// (upstream `Omit<AnthropicDocumentContent, 'cache_control'>`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireNestedToolResultContent {
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
    },
    ToolReference {
        tool_name: String,
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
    /// `eager_input_streaming` hint. Per-tool override of the
    /// model-level `toolStreaming` default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eager_input_streaming: Option<bool>,
    /// Defer tool loading until the model asks for it
    /// (`provider_options.anthropic.deferLoading`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    /// Subset of callers allowed to invoke this tool — currently:
    /// `direct` / `code_execution_20250825` / `code_execution_20260120`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<String>>,
    /// Example inputs forwarded from `FunctionTool::input_examples`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_examples: Option<Vec<serde_json::Map<String, JsonValue>>>,
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
///
/// `disable_parallel_tool_use` is supported on every variant
/// (Anthropic accepts it on `auto`, `any`, and `tool`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireToolChoice {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
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
    /// Server-allocated Skills container metadata.
    #[serde(default)]
    pub container: Option<WireContainerMetadata>,
    /// Applied `context_management` edits (compaction / clear / etc.).
    #[serde(default)]
    pub context_management: Option<WireContextManagement>,
}

/// `container` block on the response.
///
/// Mirrors `AnthropicMessageMetadata.container`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireContainerMetadata {
    pub expires_at: String,
    pub id: String,
    #[serde(default)]
    pub skills: Option<Vec<WireContainerSkill>>,
}

/// One skill entry under `container.skills`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireContainerSkill {
    #[serde(rename = "type")]
    pub kind: String,
    pub skill_id: String,
    #[serde(default)]
    pub version: Option<String>,
}

/// `context_management.applied_edits` envelope.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WireContextManagement {
    pub applied_edits: Vec<WireAppliedEdit>,
}

/// One applied context edit. Three known variants, plus a catch-all so future
/// edit types parse without erroring.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum WireAppliedEdit {
    #[serde(rename = "clear_tool_uses_20250919")]
    ClearToolUses {
        #[serde(default)]
        cleared_tool_uses: Option<u32>,
        #[serde(default)]
        cleared_input_tokens: Option<u64>,
    },
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking {
        #[serde(default)]
        cleared_thinking_turns: Option<u32>,
        #[serde(default)]
        cleared_input_tokens: Option<u64>,
    },
    #[serde(rename = "compact_20260112")]
    Compact {
        #[serde(default)]
        cleared_input_tokens: Option<u64>,
    },
    /// Unknown applied-edit variant; surfaced as raw JSON.
    #[serde(other)]
    Other,
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
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[allow(
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
    /// Per-iteration sub-usage (advisor / compaction breakdown).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<Vec<WireUsageIteration>>,
}

/// One entry in `usage.iterations[]`.
///
/// Mirrors `AnthropicUsageIteration` in `convert-anthropic-usage.ts`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireUsageIteration {
    /// Sub-iteration of a compaction step.
    Compaction {
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_creation_input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_read_input_tokens: Option<u64>,
    },
    /// Generic message-level iteration.
    Message {
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_creation_input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_read_input_tokens: Option<u64>,
    },
    /// Advisor sub-inference (carries the advisor model id).
    AdvisorMessage {
        model: String,
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_creation_input_tokens: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_read_input_tokens: Option<u64>,
    },
}
