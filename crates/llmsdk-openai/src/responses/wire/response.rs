//! Non-streaming response shape for `POST /v1/responses`.
//!
//! Mirrors `openaiResponsesResponseSchema` in
//! `@ai-sdk/openai/src/responses/openai-responses-api.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::responses::tools;
use crate::responses::usage::ResponsesUsage;

/// Top-level `/v1/responses` JSON body (non-streaming).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResponsesResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub created_at: Option<f64>,
    #[serde(default)]
    pub error: Option<ResponsesErrorBody>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub output: Option<Vec<OutputItem>>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
}

/// Top-level `error` body when the call failed.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub param: Option<String>,
    pub code: String,
}

/// `incomplete_details` block.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

/// One entry in `output[]`. Mirrors the discriminated union in upstream.
///
/// Variants are listed in the same order as upstream
/// `openaiResponsesResponseSchema` for review parity.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message(MessageItem),
    Reasoning(ReasoningItem),
    FunctionCall(FunctionCallItem),
    CustomToolCall(CustomToolCallItem),
    WebSearchCall(WebSearchCallItem),
    FileSearchCall(FileSearchCallItem),
    CodeInterpreterCall(CodeInterpreterCallItem),
    ImageGenerationCall(ImageGenerationCallItem),
    LocalShellCall(LocalShellCallItem),
    ComputerCall(ComputerCallItem),
    McpCall(McpCallItem),
    McpListTools(McpListToolsItem),
    McpApprovalRequest(McpApprovalRequestItem),
    ApplyPatchCall(ApplyPatchCallItem),
    ShellCall(ShellCallItem),
    Compaction(CompactionItem),
    ShellCallOutput(ShellCallOutputItem),
    ToolSearchCall(ToolSearchCallItem),
    ToolSearchOutput(ToolSearchOutputItem),
    /// Catch-all for unmodeled item types (forward-compat).
    #[serde(other)]
    Unknown,
}

/// `{ type: "message", ... }` assistant message item.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageItem {
    pub id: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub phase: Option<MessagePhase>,
    pub content: Vec<MessageContentPart>,
}

/// `phase` discriminator on message items.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessagePhase {
    Commentary,
    FinalAnswer,
}

/// One content part inside a `message` item.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContentPart {
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Annotation>,
        #[serde(default)]
        logprobs: Option<Vec<LogprobEntry>>,
    },
}

/// One annotation on an `output_text` part.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    UrlCitation {
        start_index: u64,
        end_index: u64,
        url: String,
        title: String,
    },
    FileCitation {
        file_id: String,
        filename: String,
        index: u64,
    },
    ContainerFileCitation {
        container_id: String,
        file_id: String,
        filename: String,
        start_index: u64,
        end_index: u64,
    },
    FilePath {
        file_id: String,
        index: u64,
    },
}

/// One logprob entry.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f64,
    #[serde(default)]
    pub top_logprobs: Vec<LogprobAlternative>,
}

/// Alternative token in `top_logprobs[]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogprobAlternative {
    pub token: String,
    pub logprob: f64,
}

/// `{ type: "reasoning", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReasoningItem {
    pub id: String,
    #[serde(default)]
    pub encrypted_content: Option<String>,
    #[serde(default)]
    pub summary: Vec<ReasoningSummary>,
}

/// One entry in `reasoning.summary[]`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningSummary {
    SummaryText { text: String },
}

/// `{ type: "function_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallItem {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub namespace: Option<String>,
}

/// `{ type: "custom_tool_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CustomToolCallItem {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub input: String,
}

/// `{ type: "web_search_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WebSearchCallItem {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub action: Option<tools::web_search::Action>,
}

/// `{ type: "file_search_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FileSearchCallItem {
    pub id: String,
    pub queries: Vec<String>,
    #[serde(default)]
    pub results: Option<Vec<tools::file_search::ResultRow>>,
}

/// `{ type: "code_interpreter_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodeInterpreterCallItem {
    pub id: String,
    pub container_id: String,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub outputs: Option<Vec<tools::code_interpreter::Output>>,
    #[serde(default)]
    pub status: Option<String>,
}

/// `{ type: "image_generation_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageGenerationCallItem {
    pub id: String,
    pub result: String,
}

/// `{ type: "local_shell_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalShellCallItem {
    pub id: String,
    pub call_id: String,
    pub action: tools::local_shell::Action,
}

/// `{ type: "computer_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ComputerCallItem {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
}

/// `{ type: "mcp_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpCallItem {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
    pub arguments: String,
    pub name: String,
    pub server_label: String,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub error: Option<JsonValue>,
    #[serde(default)]
    pub approval_request_id: Option<String>,
}

/// `{ type: "mcp_list_tools", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpListToolsItem {
    pub id: String,
    pub server_label: String,
    #[serde(default)]
    pub tools: Vec<JsonValue>,
    #[serde(default)]
    pub error: Option<JsonValue>,
}

/// `{ type: "mcp_approval_request", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpApprovalRequestItem {
    pub id: String,
    pub server_label: String,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub approval_request_id: Option<String>,
}

/// `{ type: "apply_patch_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApplyPatchCallItem {
    pub id: String,
    pub call_id: String,
    pub status: ApplyPatchCallStatus,
    pub operation: tools::apply_patch::Operation,
}

/// `apply_patch_call.status` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApplyPatchCallStatus {
    InProgress,
    Completed,
}

/// `{ type: "shell_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShellCallItem {
    pub id: String,
    pub call_id: String,
    pub status: ShellCallStatus,
    pub action: ShellCallAction,
}

/// Slim action inside `shell_call`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ShellCallAction {
    pub commands: Vec<String>,
}

/// `shell_call.status` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShellCallStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// `{ type: "shell_call_output", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShellCallOutputItem {
    pub id: String,
    pub call_id: String,
    pub status: ShellCallStatus,
    pub output: Vec<tools::shell::OutputRow>,
}

/// `{ type: "compaction", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompactionItem {
    pub id: String,
    pub encrypted_content: String,
}

/// `{ type: "tool_search_call", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolSearchCallItem {
    pub id: String,
    pub execution: tools::tool_search::Execution,
    pub call_id: Option<String>,
    pub status: ShellCallStatus,
    pub arguments: JsonValue,
}

/// `{ type: "tool_search_output", ... }`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolSearchOutputItem {
    pub id: String,
    pub execution: tools::tool_search::Execution,
    pub call_id: Option<String>,
    pub status: ShellCallStatus,
    pub tools: Vec<JsonValue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_message_item_with_url_citation() {
        let body = serde_json::json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "hello",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "start_index": 0,
                            "end_index": 5,
                            "url": "https://example.com",
                            "title": "Example"
                        }
                    ]
                }
            ]
        });
        let item: OutputItem = serde_json::from_value(body).unwrap();
        let OutputItem::Message(m) = item else {
            panic!("expected message variant");
        };
        assert_eq!(m.id, "msg_1");
        assert_eq!(m.phase, Some(MessagePhase::FinalAnswer));
        assert_eq!(m.content.len(), 1);
        let MessageContentPart::OutputText { annotations, .. } = &m.content[0];
        assert_eq!(annotations.len(), 1);
        assert!(matches!(annotations[0], Annotation::UrlCitation { .. }));
    }

    #[test]
    fn parses_reasoning_with_summary_and_encrypted() {
        let body = serde_json::json!({
            "type": "reasoning",
            "id": "rsn_1",
            "encrypted_content": "abc",
            "summary": [{ "type": "summary_text", "text": "thought" }]
        });
        let item: OutputItem = serde_json::from_value(body).unwrap();
        let OutputItem::Reasoning(r) = item else {
            panic!("expected reasoning");
        };
        assert_eq!(r.encrypted_content.as_deref(), Some("abc"));
        assert_eq!(r.summary.len(), 1);
    }

    #[test]
    fn parses_apply_patch_call() {
        let body = serde_json::json!({
            "type": "apply_patch_call",
            "id": "ap_1",
            "call_id": "call_1",
            "status": "completed",
            "operation": {
                "type": "update_file",
                "path": "src/main.rs",
                "diff": "@@ -1 +1 @@\n-old\n+new"
            }
        });
        let item: OutputItem = serde_json::from_value(body).unwrap();
        let OutputItem::ApplyPatchCall(p) = item else {
            panic!("expected apply_patch_call");
        };
        assert_eq!(p.status, ApplyPatchCallStatus::Completed);
    }

    #[test]
    fn unknown_type_lands_in_unknown_variant() {
        let body = serde_json::json!({ "type": "future_widget", "id": "x" });
        let item: OutputItem = serde_json::from_value(body).unwrap();
        assert!(matches!(item, OutputItem::Unknown));
    }
}
