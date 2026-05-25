//! SSE event types for `POST /v1/responses` (`stream: true`).
//!
//! Mirrors the `openaiResponsesChunkSchema` union in
//! `@ai-sdk/openai/src/responses/openai-responses-api.ts` (~30 variants).
// Rust guideline compliant 2026-02-21

use serde::Deserialize;
use serde_json::Value as JsonValue;

use super::response::{Annotation, LogprobEntry};
use crate::responses::tools;
use crate::responses::usage::ResponsesUsage;

/// One SSE chunk delivered by `POST /v1/responses` (`stream: true`).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesChunk {
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        delta: String,
        #[serde(default)]
        logprobs: Option<Vec<LogprobEntry>>,
    },

    #[serde(rename = "response.created")]
    Created { response: CreatedSnapshot },

    #[serde(rename = "response.completed")]
    Completed { response: FinishedSnapshot },

    #[serde(rename = "response.incomplete")]
    Incomplete { response: FinishedSnapshot },

    #[serde(rename = "response.failed")]
    Failed { response: FailedSnapshot },

    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { output_index: u32, item: AddedItem },

    #[serde(rename = "response.output_item.done")]
    OutputItemDone { output_index: u32, item: DoneItem },

    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: u32,
        delta: String,
    },

    #[serde(rename = "response.custom_tool_call_input.delta")]
    CustomToolCallInputDelta {
        item_id: String,
        output_index: u32,
        delta: String,
    },

    #[serde(rename = "response.image_generation_call.partial_image")]
    ImageGenerationPartialImage {
        item_id: String,
        output_index: u32,
        partial_image_b64: String,
    },

    #[serde(rename = "response.code_interpreter_call_code.delta")]
    CodeInterpreterCodeDelta {
        item_id: String,
        output_index: u32,
        delta: String,
    },

    #[serde(rename = "response.code_interpreter_call_code.done")]
    CodeInterpreterCodeDone {
        item_id: String,
        output_index: u32,
        code: String,
    },

    #[serde(rename = "response.output_text.annotation.added")]
    AnnotationAdded { annotation: Annotation },

    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded { item_id: String, summary_index: u32 },

    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        item_id: String,
        summary_index: u32,
        delta: String,
    },

    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone { item_id: String, summary_index: u32 },

    #[serde(rename = "response.apply_patch_call_operation_diff.delta")]
    ApplyPatchOpDiffDelta {
        item_id: String,
        output_index: u32,
        delta: String,
        #[serde(default)]
        obfuscation: Option<String>,
    },

    #[serde(rename = "response.apply_patch_call_operation_diff.done")]
    ApplyPatchOpDiffDone {
        item_id: String,
        output_index: u32,
        diff: String,
    },

    #[serde(rename = "error")]
    Error {
        sequence_number: u64,
        error: ErrorChunk,
    },

    /// Catch-all for chunks not modeled above (forward-compat).
    #[serde(other)]
    Unknown,
}

/// `error` chunk inner shape.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorChunk {
    #[serde(rename = "type")]
    pub kind: String,
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub param: Option<String>,
}

/// `response.created` snapshot (lean shape).
#[derive(Debug, Clone, Deserialize)]
pub struct CreatedSnapshot {
    pub id: String,
    pub created_at: f64,
    pub model: String,
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// `response.completed` / `response.incomplete` snapshot.
#[derive(Debug, Clone, Deserialize)]
pub struct FinishedSnapshot {
    #[serde(default)]
    pub incomplete_details: Option<super::response::IncompleteDetails>,
    pub usage: ResponsesUsage,
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// `response.failed` snapshot (error + optional usage).
#[derive(Debug, Clone, Deserialize)]
pub struct FailedSnapshot {
    #[serde(default)]
    pub error: Option<super::response::ResponsesErrorBody>,
    #[serde(default)]
    pub incomplete_details: Option<super::response::IncompleteDetails>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// `response.output_item.added`'s item — a partial discriminated union covering
/// just the variants ai-sdk relies on for start-of-item bookkeeping.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AddedItem {
    Message {
        id: String,
        #[serde(default)]
        phase: Option<super::response::MessagePhase>,
    },
    Reasoning {
        id: String,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default)]
        namespace: Option<String>,
    },
    WebSearchCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
    },
    ComputerCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
    },
    FileSearchCall {
        id: String,
    },
    ImageGenerationCall {
        id: String,
    },
    CodeInterpreterCall {
        id: String,
        container_id: String,
        #[serde(default)]
        code: Option<String>,
        #[serde(default)]
        outputs: Option<Vec<tools::code_interpreter::Output>>,
        #[serde(default)]
        status: Option<String>,
    },
    McpCall {
        id: String,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        approval_request_id: Option<String>,
    },
    McpListTools {
        id: String,
    },
    McpApprovalRequest {
        id: String,
    },
    ApplyPatchCall {
        id: String,
        call_id: String,
        status: super::response::ApplyPatchCallStatus,
        operation: tools::apply_patch::Operation,
    },
    CustomToolCall {
        id: String,
        call_id: String,
        name: String,
        input: String,
    },
    ShellCall {
        id: String,
        call_id: String,
        status: super::response::ShellCallStatus,
        action: super::response::ShellCallAction,
    },
    Compaction {
        id: String,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    ShellCallOutput {
        id: String,
        call_id: String,
        status: super::response::ShellCallStatus,
        output: Vec<tools::shell::OutputRow>,
    },
    ToolSearchCall {
        id: String,
        execution: tools::tool_search::Execution,
        call_id: Option<String>,
        status: super::response::ShellCallStatus,
        arguments: JsonValue,
    },
    ToolSearchOutput {
        id: String,
        execution: tools::tool_search::Execution,
        call_id: Option<String>,
        status: super::response::ShellCallStatus,
        tools: Vec<JsonValue>,
    },
    /// Catch-all for unmodeled added-item types.
    #[serde(other)]
    Unknown,
}

/// `response.output_item.done`'s item — same shape family, terminal version.
///
/// Many fields (`status`, etc.) become required at done time; we model them
/// `Option<>` to stay forgiving of upstream tweaks.
pub type DoneItem = super::response::OutputItem;

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: serde_json::Value) -> ResponsesChunk {
        serde_json::from_value(value).expect("decode chunk")
    }

    #[test]
    fn output_text_delta_round_trip() {
        let c = parse(serde_json::json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "hi"
        }));
        assert!(matches!(c, ResponsesChunk::OutputTextDelta { .. }));
    }

    #[test]
    fn created_carries_service_tier() {
        let c = parse(serde_json::json!({
            "type": "response.created",
            "response": {
                "id": "resp_1",
                "created_at": 1700000000.0,
                "model": "gpt-5",
                "service_tier": "flex"
            }
        }));
        let ResponsesChunk::Created { response } = c else {
            panic!("created");
        };
        assert_eq!(response.service_tier.as_deref(), Some("flex"));
    }

    #[test]
    fn completed_has_usage() {
        let c = parse(serde_json::json!({
            "type": "response.completed",
            "response": {
                "usage": { "input_tokens": 10, "output_tokens": 5 }
            }
        }));
        let ResponsesChunk::Completed { response } = c else {
            panic!("completed");
        };
        assert_eq!(response.usage.input_tokens, 10);
    }

    #[test]
    fn apply_patch_delta_with_obfuscation() {
        let c = parse(serde_json::json!({
            "type": "response.apply_patch_call_operation_diff.delta",
            "item_id": "ap_1",
            "output_index": 2,
            "delta": "@@ -1 +1 @@\n-old\n+new",
            "obfuscation": null
        }));
        assert!(matches!(c, ResponsesChunk::ApplyPatchOpDiffDelta { .. }));
    }

    #[test]
    fn output_item_added_function_call() {
        let c = parse(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_x",
                "name": "weather",
                "arguments": ""
            }
        }));
        let ResponsesChunk::OutputItemAdded { item, .. } = c else {
            panic!();
        };
        assert!(matches!(item, AddedItem::FunctionCall { .. }));
    }

    #[test]
    fn error_chunk_extracts_code_and_message() {
        let c = parse(serde_json::json!({
            "type": "error",
            "sequence_number": 7,
            "error": {
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "message": "slow down"
            }
        }));
        let ResponsesChunk::Error { error, .. } = c else {
            panic!("error");
        };
        assert_eq!(error.code, "rate_limit_exceeded");
        assert_eq!(error.message, "slow down");
    }

    #[test]
    fn annotation_added_url_citation() {
        let c = parse(serde_json::json!({
            "type": "response.output_text.annotation.added",
            "annotation": {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 1,
                "url": "https://example.com",
                "title": "x"
            }
        }));
        assert!(matches!(c, ResponsesChunk::AnnotationAdded { .. }));
    }

    #[test]
    fn unknown_chunk_falls_through() {
        let c = parse(serde_json::json!({ "type": "response.future_event", "foo": 1 }));
        assert!(matches!(c, ResponsesChunk::Unknown));
    }
}
