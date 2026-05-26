//! Wire-level xAI Responses API request / response / SSE chunk types.
//!
//! Mirrors the embedded zod schemas in
//! `@ai-sdk/xai/src/responses/xai-responses-api.ts`. Only fields actually
//! used by xAI's responses endpoint are surfaced; unknown fields deserialize
//! away via `serde(default)` and `#[serde(other)]`.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---- request ---------------------------------------------------------

/// `POST /v1/responses` request body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ResponsesRequest {
    pub model: String,
    pub input: Vec<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

// ---- response (non-streaming) -----------------------------------------

/// Non-streaming `POST /v1/responses` response.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ResponsesResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub created_at: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
}

/// `incomplete_details` envelope reported on `response.incomplete` /
/// `response.failed`.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

/// One element in `response.output[]`.
///
/// Tagged by `type`. We catch the eight server-side tool call types plus
/// the three primary kinds (`message`, `reasoning`, `function_call`). Any
/// other `type` falls into [`OutputItem::Other`] and is ignored downstream.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum OutputItem {
    #[serde(rename = "message")]
    Message(MessageItem),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningItem),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionCallItem),
    #[serde(rename = "custom_tool_call")]
    CustomToolCall(ToolCallItem),
    #[serde(rename = "web_search_call")]
    WebSearchCall(ToolCallItem),
    #[serde(rename = "x_search_call")]
    XSearchCall(ToolCallItem),
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall(ToolCallItem),
    #[serde(rename = "code_execution_call")]
    CodeExecutionCall(ToolCallItem),
    #[serde(rename = "view_image_call")]
    ViewImageCall(ToolCallItem),
    #[serde(rename = "view_x_video_call")]
    ViewXVideoCall(ToolCallItem),
    #[serde(rename = "file_search_call")]
    FileSearchCall(FileSearchCallItem),
    #[serde(rename = "mcp_call")]
    McpCall(McpCallItem),
    #[serde(other)]
    Other,
}

/// `type: "message"`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MessageItem {
    pub id: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub content: Vec<MessageContentPart>,
}

/// One entry inside `message.content[]`.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct MessageContentPart {
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub annotations: Option<Vec<Annotation>>,
}

/// One annotation on a message content part.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum Annotation {
    UrlCitation {
        #[serde(rename = "type")]
        kind: String,
        url: String,
        #[serde(default)]
        title: Option<String>,
    },
    Other(serde_json::Map<String, Value>),
}

impl Annotation {
    /// Extract `(url, title)` if this is a `url_citation` annotation.
    pub fn as_url_citation(&self) -> Option<(&str, Option<&str>)> {
        match self {
            Self::UrlCitation { kind, url, title } if kind == "url_citation" => {
                Some((url.as_str(), title.as_deref()))
            }
            _ => None,
        }
    }
}

/// `type: "reasoning"`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ReasoningItem {
    pub id: String,
    #[serde(default)]
    pub summary: Vec<ReasoningSummaryPart>,
    #[serde(default)]
    pub content: Option<Vec<ReasoningTextPart>>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
}

/// One entry in `reasoning.summary[]`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ReasoningSummaryPart {
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    pub text: String,
}

/// One entry in `reasoning.content[]`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ReasoningTextPart {
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    pub text: String,
}

/// `type: "function_call"`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FunctionCallItem {
    pub name: String,
    pub arguments: String,
    pub call_id: String,
    pub id: String,
}

/// Shared shape for server-side tool calls (`web_search_call` /
/// `x_search_call` / `code_interpreter_call` / `code_execution_call` /
/// `view_image_call` / `view_x_video_call` / `custom_tool_call`).
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ToolCallItem {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub action: Option<Value>,
}

/// `type: "mcp_call"`.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct McpCallItem {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub server_label: Option<String>,
}

/// `type: "file_search_call"`.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FileSearchCallItem {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub queries: Option<Vec<String>>,
    #[serde(default)]
    pub results: Option<Vec<FileSearchResult>>,
}

/// One entry inside `file_search_call.results[]`.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FileSearchResult {
    pub file_id: String,
    pub filename: String,
    pub score: f64,
    pub text: String,
}

// ---- usage -----------------------------------------------------------

/// Raw `response.usage` object.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct WireUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<WireInputTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<WireOutputTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_sources_used: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_server_side_tools_used: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_in_usd_ticks: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct WireInputTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct WireOutputTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

// ---- streaming SSE chunks --------------------------------------------

/// One decoded SSE event on `POST /v1/responses` (`stream=true`).
///
/// Tagged by `type`. Any unknown event type falls into
/// [`ResponsesChunk::Other`] and is ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ResponsesChunk {
    #[serde(rename = "response.created")]
    ResponseCreated { response: ResponsesResponse },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress { response: ResponsesResponse },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { item: OutputItem, output_index: u32 },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { item: OutputItem, output_index: u32 },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        #[serde(default)]
        output_index: u32,
        #[serde(default)]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: String,
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        annotations: Option<Vec<Annotation>>,
    },
    #[serde(rename = "response.output_text.annotation.added")]
    OutputTextAnnotationAdded {
        item_id: String,
        #[serde(default)]
        annotation_index: u32,
        annotation: Annotation,
    },
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        item_id: String,
        #[serde(default)]
        summary_index: u32,
        part: ReasoningSummaryPart,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        item_id: String,
        #[serde(default)]
        summary_index: u32,
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        item_id: String,
        #[serde(default)]
        summary_index: u32,
        #[serde(default)]
        text: Option<String>,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        item_id: String,
        #[serde(default)]
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        item_id: String,
        #[serde(default)]
        text: Option<String>,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        #[serde(default)]
        item_id: String,
        output_index: u32,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        #[serde(default)]
        item_id: String,
        output_index: u32,
        #[serde(default)]
        arguments: Option<String>,
    },
    #[serde(rename = "response.custom_tool_call_input.delta")]
    CustomToolCallInputDelta { item_id: String, delta: String },
    #[serde(rename = "response.custom_tool_call_input.done")]
    CustomToolCallInputDone {
        item_id: String,
        #[serde(default)]
        input: Option<String>,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
        #[serde(default)]
        content_index: Option<u32>,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
        #[serde(default)]
        content_index: Option<u32>,
    },
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        summary_index: Option<u32>,
    },
    // ----- Tool-call lifecycle events ---------------------------------
    //
    // ai-sdk's xai-responses-language-model.ts does not consume these
    // status frames directly (they are surfaced via the matching
    // `response.output_item.added` / `.done` envelope). They are listed
    // explicitly so the wire schema is 1:1 with `xai-responses-api.ts`
    // and future status events can be migrated without re-parsing the
    // catch-all variant.
    #[serde(rename = "response.web_search_call.in_progress")]
    WebSearchCallInProgress {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.web_search_call.searching")]
    WebSearchCallSearching {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.web_search_call.completed")]
    WebSearchCallCompleted {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.x_search_call.in_progress")]
    XSearchCallInProgress {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.x_search_call.searching")]
    XSearchCallSearching {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.x_search_call.completed")]
    XSearchCallCompleted {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.file_search_call.in_progress")]
    FileSearchCallInProgress {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.file_search_call.searching")]
    FileSearchCallSearching {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.file_search_call.completed")]
    FileSearchCallCompleted {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_execution_call.in_progress")]
    CodeExecutionCallInProgress {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_execution_call.executing")]
    CodeExecutionCallExecuting {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_execution_call.completed")]
    CodeExecutionCallCompleted {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_interpreter_call.in_progress")]
    CodeInterpreterCallInProgress {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_interpreter_call.executing")]
    CodeInterpreterCallExecuting {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_interpreter_call.interpreting")]
    CodeInterpreterCallInterpreting {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_interpreter_call.completed")]
    CodeInterpreterCallCompleted {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.code_interpreter_call_code.delta")]
    CodeInterpreterCallCodeDelta {
        #[serde(default)]
        item_id: Option<String>,
        delta: String,
    },
    #[serde(rename = "response.code_interpreter_call_code.done")]
    CodeInterpreterCallCodeDone {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        code: Option<String>,
    },
    #[serde(rename = "response.mcp_call.in_progress")]
    McpCallInProgress {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.mcp_call.executing")]
    McpCallExecuting {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.mcp_call.completed")]
    McpCallCompleted {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.mcp_call.failed")]
    McpCallFailed {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output_index: Option<u32>,
    },
    #[serde(rename = "response.mcp_call_arguments.delta")]
    McpCallArgumentsDelta {
        #[serde(default)]
        item_id: Option<String>,
        delta: String,
    },
    #[serde(rename = "response.mcp_call_arguments.done")]
    McpCallArgumentsDone {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        arguments: Option<String>,
    },
    #[serde(rename = "response.mcp_call_output.delta")]
    McpCallOutputDelta {
        #[serde(default)]
        item_id: Option<String>,
        delta: String,
    },
    #[serde(rename = "response.mcp_call_output.done")]
    McpCallOutputDone {
        #[serde(default)]
        item_id: Option<String>,
        #[serde(default)]
        output: Option<String>,
    },
    #[serde(rename = "response.done")]
    ResponseDone { response: ResponsesResponse },
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: ResponsesResponse },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete { response: ResponsesResponse },
    #[serde(rename = "response.failed")]
    ResponseFailed { response: ResponsesResponse },
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        code: Option<String>,
        message: String,
        #[serde(default)]
        param: Option<String>,
    },
    #[serde(other)]
    Other,
}
