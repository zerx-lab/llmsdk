//! `openai.mcp` provider-defined tool.
//!
//! Mirrors `@ai-sdk/openai/src/tool/mcp.ts`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Args for `Tool::Provider { id: "openai.mcp", args, .. }`.
///
/// Either `serverUrl` or `connectorId` must be provided; this is validated
/// at request-build time (mirrors the zod `.refine` in the upstream schema).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    pub server_label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<AllowedTools>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<RequireApproval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
}

/// Either a plain array of names or a filter object.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum AllowedTools {
    List(Vec<String>),
    Filter {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        read_only: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_names: Option<Vec<String>>,
    },
}

/// Either an `"always" | "never"` string, or a structured filter.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum RequireApproval {
    Mode(ApprovalMode),
    Filter {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        never: Option<NeverFilter>,
    },
}

/// Plain `"always" | "never"` mode.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    Always,
    Never,
}

/// Filter for the `never`-arm of `requireApproval`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NeverFilter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,
}

/// Output emitted in `mcp_call` items.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Output {
    /// Always `"call"`.
    #[serde(rename = "type")]
    pub kind: String,
    pub server_label: String,
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonValue>,
}
