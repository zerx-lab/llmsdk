//! `openai.apply_patch` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/apply-patch.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Args struct (no fields; the tool takes no configuration).
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Args;

/// Input emitted as part of `apply_patch_call` items.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Input {
    pub call_id: String,
    pub operation: Operation,
}

/// Discriminated `operation`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Operation {
    CreateFile { path: String, diff: String },
    DeleteFile { path: String },
    UpdateFile { path: String, diff: String },
}

/// Output for `apply_patch_call_output`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Output {
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// `status` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Completed,
    Failed,
}
