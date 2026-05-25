//! Provider-defined tool args + output structs for the OpenAI Responses API.
//!
//! Mirrors `@ai-sdk/openai/src/tool/*.ts`. Each module exports an `Args`
//! struct (input-side, camelCase via serde rename) for parsing the
//! `Tool::Provider.args` JSON, and an `Output` struct (response-side) used
//! when reconstructing a `ToolResult` from the Responses API output items.
//!
//! Tool routing lives in [`super::prepare_tools`]; wire types in
//! [`super::wire::request::WireTool`].
// Rust guideline compliant 2026-02-21

pub mod apply_patch;
pub mod code_interpreter;
pub mod custom;
pub mod file_search;
pub mod image_generation;
pub mod local_shell;
pub mod mcp;
pub mod shell;
pub mod tool_search;
pub mod web_search;
pub mod web_search_preview;

/// Provider-defined tool ids recognized by the Responses model.
///
/// Mirrors the `'openai.X'` discriminator strings used in
/// [`llmsdk_provider::language_model::Tool::Provider.id`].
pub mod ids {
    /// `openai.apply_patch`
    pub const APPLY_PATCH: &str = "openai.apply_patch";
    /// `openai.code_interpreter`
    pub const CODE_INTERPRETER: &str = "openai.code_interpreter";
    /// `openai.custom`
    pub const CUSTOM: &str = "openai.custom";
    /// `openai.file_search`
    pub const FILE_SEARCH: &str = "openai.file_search";
    /// `openai.image_generation`
    pub const IMAGE_GENERATION: &str = "openai.image_generation";
    /// `openai.local_shell`
    pub const LOCAL_SHELL: &str = "openai.local_shell";
    /// `openai.mcp`
    pub const MCP: &str = "openai.mcp";
    /// `openai.shell`
    pub const SHELL: &str = "openai.shell";
    /// `openai.tool_search`
    pub const TOOL_SEARCH: &str = "openai.tool_search";
    /// `openai.web_search`
    pub const WEB_SEARCH: &str = "openai.web_search";
    /// `openai.web_search_preview`
    pub const WEB_SEARCH_PREVIEW: &str = "openai.web_search_preview";
}

/// Shared `userLocation` block reused by `web_search` and `web_search_preview`.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UserLocation {
    /// Discriminator, only `"approximate"` is currently defined.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}
