//! `POST /v1/responses` request body + input items + tool wire shape.
//!
//! Mirrors `@ai-sdk/openai/src/responses/openai-responses-api.ts`
//! (`OpenAIResponsesInput*` + `OpenAIResponsesTool`).
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::responses::tools;

/// Top-level request body sent to `/v1/responses`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<InputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<JsonValue>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Catch-all for fields not modeled above (extra provider option keys).
    #[serde(flatten)]
    pub extra: HashMap<String, JsonValue>,
}

/// One entry in `input[]`. Mirrors `OpenAIResponsesInputItem`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum InputItem {
    /// `role: system | developer | user | assistant` chat message.
    Message(InputMessage),
    /// Anything keyed by a `type` discriminator (function_call, etc.).
    Typed(TypedInputItem),
}

/// `{ role: ..., content: ... }` input message item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum InputMessage {
    SystemOrDeveloper {
        role: SystemRole,
        content: String,
    },
    User {
        /// Fixed as `"user"`.
        role: UserRole,
        content: Vec<UserContentPart>,
    },
    Assistant {
        /// Fixed as `"assistant"`.
        role: AssistantRole,
        content: Vec<AssistantContentPart>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<super::response::MessagePhase>,
    },
}

/// `"system" | "developer"`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SystemRole {
    System,
    Developer,
}

/// `"user"`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    User,
}

/// `"assistant"`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AssistantRole {
    Assistant,
}

/// One content part inside a user `content` array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContentPart {
    InputText { text: String },
    InputImage(InputImage),
    InputFile(InputFile),
}

/// `input_image` payload (one of three shapes — modeled untagged).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum InputImage {
    Url { image_url: String },
    Reference { file_id: String },
}

/// `input_file` payload (one of three shapes).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum InputFile {
    Url { file_url: String },
    Data { filename: String, file_data: String },
    Reference { file_id: String },
}

/// One content part inside an assistant `content` array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContentPart {
    OutputText { text: String },
}

/// Non-message input items, distinguished by their `type` tag.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedInputItem {
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    FunctionCallOutput {
        call_id: String,
        output: FunctionCallOutputBody,
    },
    CustomToolCall {
        call_id: String,
        name: String,
        input: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    CustomToolCallOutput {
        call_id: String,
        output: FunctionCallOutputBody,
    },
    McpApprovalResponse {
        approval_request_id: String,
        approve: bool,
    },
    ComputerCall {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    LocalShellCall {
        id: String,
        call_id: String,
        action: tools::local_shell::Action,
    },
    LocalShellCallOutput {
        call_id: String,
        output: String,
    },
    ShellCall {
        id: String,
        call_id: String,
        status: super::response::ShellCallStatus,
        action: super::response::ShellCallAction,
    },
    ShellCallOutput {
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<super::response::ShellCallStatus>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_output_length: Option<u32>,
        output: Vec<tools::shell::OutputRow>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    ApplyPatchCall {
        call_id: String,
        status: super::response::ApplyPatchCallStatus,
        operation: tools::apply_patch::Operation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    ApplyPatchCallOutput {
        call_id: String,
        status: tools::apply_patch::Status,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
    },
    ToolSearchCall {
        id: String,
        execution: tools::tool_search::Execution,
        call_id: Option<String>,
        status: super::response::ShellCallStatus,
        arguments: JsonValue,
    },
    ToolSearchOutput {
        execution: tools::tool_search::Execution,
        call_id: Option<String>,
        status: super::response::ShellCallStatus,
        tools: Vec<JsonValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        summary: Vec<super::response::ReasoningSummary>,
    },
    ItemReference {
        id: String,
    },
    Compaction {
        id: String,
        encrypted_content: String,
    },
}

/// `function_call_output.output` — string or rich array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FunctionCallOutputBody {
    Text(String),
    Parts(Vec<UserContentPart>),
}

/// One entry in `tools[]` — modeled as `JsonValue` because the union is
/// huge (11 variants + future additions) and is composed by
/// [`super::super::prepare_tools`] using `serde_json::to_value`.
pub type WireTool = JsonValue;

/// `tool_choice` shape (string mode or `{ type, ... }` selector).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum WireToolChoice {
    Mode(ToolChoiceMode),
    Selector(JsonValue),
}

/// `"auto" | "none" | "required"` simple modes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_message_serializes() {
        let m = InputItem::Message(InputMessage::SystemOrDeveloper {
            role: SystemRole::Developer,
            content: "Be concise.".into(),
        });
        let s = serde_json::to_value(&m).unwrap();
        assert_eq!(s["role"], "developer");
        assert_eq!(s["content"], "Be concise.");
    }

    #[test]
    fn user_with_image_url() {
        let m = InputItem::Message(InputMessage::User {
            role: UserRole::User,
            content: vec![
                UserContentPart::InputText { text: "hi".into() },
                UserContentPart::InputImage(InputImage::Url {
                    image_url: "https://x".into(),
                }),
            ],
        });
        let s = serde_json::to_value(&m).unwrap();
        assert_eq!(s["role"], "user");
        assert_eq!(s["content"][1]["type"], "input_image");
        assert_eq!(s["content"][1]["image_url"], "https://x");
    }

    #[test]
    fn function_call_output_round_trip() {
        let item = InputItem::Typed(TypedInputItem::FunctionCallOutput {
            call_id: "c1".into(),
            output: FunctionCallOutputBody::Text("ok".into()),
        });
        let s = serde_json::to_value(&item).unwrap();
        assert_eq!(s["type"], "function_call_output");
        assert_eq!(s["call_id"], "c1");
        assert_eq!(s["output"], "ok");
    }

    #[test]
    fn mcp_approval_response_serializes() {
        let item = InputItem::Typed(TypedInputItem::McpApprovalResponse {
            approval_request_id: "appr_1".into(),
            approve: true,
        });
        let s = serde_json::to_value(&item).unwrap();
        assert_eq!(s["type"], "mcp_approval_response");
        assert_eq!(s["approve"], true);
    }

    #[test]
    fn reasoning_with_encrypted_content_serializes() {
        let item = InputItem::Typed(TypedInputItem::Reasoning {
            id: Some("r_1".into()),
            encrypted_content: Some("enc".into()),
            summary: vec![super::super::response::ReasoningSummary::SummaryText {
                text: "thinking".into(),
            }],
        });
        let s = serde_json::to_value(&item).unwrap();
        assert_eq!(s["type"], "reasoning");
        assert_eq!(s["encrypted_content"], "enc");
        assert_eq!(s["summary"][0]["type"], "summary_text");
    }
}
