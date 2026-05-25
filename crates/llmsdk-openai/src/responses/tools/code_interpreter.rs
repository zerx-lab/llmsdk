//! `openai.code_interpreter` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/code-interpreter.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Args for `Tool::Provider { id: "openai.code_interpreter", args, .. }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    /// `null` → auto with no file IDs; string → container id; object → auto with file IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerArg>,
}

/// Container spec input shape.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ContainerArg {
    Id(String),
    Auto(ContainerAuto),
}

/// `{ fileIds: [...] }` shape.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ContainerAuto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_ids: Option<Vec<String>>,
}

/// Output items inside `code_interpreter_call.outputs[]` (response side).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Output {
    Logs { logs: String },
    Image { url: String },
}
