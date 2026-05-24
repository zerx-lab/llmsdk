//! Tool definitions passed in [`super::CallOptions::tools`].
//!
//! Mirrors `language-model-v4-function-tool.ts`, `-provider-tool.ts`, and
//! `-tool-choice.ts`.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use crate::json::{JsonObject, JsonSchema};
use crate::shared::ProviderOptions;

/// Either a client-defined function tool or a provider-defined tool.
///
/// Wire `type` matches ai-sdk v4 verbatim (`"function"` / `"provider"`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Tool {
    /// Client-defined function tool.
    #[serde(rename = "function")]
    Function(FunctionTool),
    /// Provider-defined tool, e.g. `OpenAI`'s `web_search_preview`.
    #[serde(rename = "provider")]
    Provider(ProviderTool),
}

/// Client-defined function tool with a JSON schema input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionTool {
    /// Unique tool name within the call.
    pub name: String,
    /// Optional natural-language description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema for the input arguments.
    #[serde(rename = "inputSchema")]
    pub input_schema: JsonSchema,
    /// Example inputs to guide the model.
    #[serde(
        default,
        rename = "inputExamples",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_examples: Option<Vec<ToolInputExample>>,
    /// Request strict schema enforcement when the provider supports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// One input example for [`FunctionTool::input_examples`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolInputExample {
    /// Example input matching the schema.
    pub input: JsonObject,
}

/// Provider-defined tool referenced by id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderTool {
    /// Provider-defined tool id, e.g. `"openai.web_search_preview"`.
    pub id: String,
    /// Logical name within the call.
    pub name: String,
    /// Provider-defined arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<JsonObject>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

/// How the model should choose among the configured tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolChoice {
    /// Automatic selection (may pick no tool).
    #[default]
    Auto,
    /// Model must not call any tool.
    None,
    /// Model must call exactly one tool.
    Required,
    /// Model must call this specific tool.
    Tool {
        /// Tool name to force.
        #[serde(rename = "toolName")]
        tool_name: String,
    },
}
