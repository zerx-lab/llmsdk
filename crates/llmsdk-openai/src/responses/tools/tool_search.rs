//! `openai.tool_search` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/tool-search.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Args for `Tool::Provider { id: "openai.tool_search", args, .. }`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Args {
    /// `"server"` (default; hosted) or `"client"` (your app executes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<Execution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonValue>,
}

/// `execution` enum.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Execution {
    Server,
    Client,
}

/// Input embedded in `tool_search_call`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Input {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
}

/// Output for `tool_search_output`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Output {
    pub tools: Vec<JsonValue>,
}
