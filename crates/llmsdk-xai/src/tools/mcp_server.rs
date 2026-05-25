//! `xai.mcp` typed factory (Model Context Protocol relay).
//!
//! Mirrors `@ai-sdk/xai/src/tool/mcp-server.ts`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;

use llmsdk_provider::language_model::{ProviderTool, Tool};
use serde::Serialize;

/// Required + optional knobs for [`mcp_server`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerOptions {
    /// URL of the MCP server.
    pub server_url: String,
    /// Human-readable label for the MCP server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_label: Option<String>,
    /// Description of the MCP server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_description: Option<String>,
    /// Allowlist of tool names to expose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// Extra HTTP headers to send to the MCP server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Bearer-style `Authorization` header value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,
}

/// Build a `xai.mcp` provider tool.
///
/// # Examples
///
/// ```
/// use llmsdk_xai::tools::{mcp_server, McpServerOptions};
/// let tool = mcp_server(&McpServerOptions {
///     server_url: "https://mcp.example.com".into(),
///     server_label: Some("my-mcp".into()),
///     server_description: None,
///     allowed_tools: None,
///     headers: None,
///     authorization: None,
/// });
/// let _ = tool;
/// ```
#[must_use]
pub fn mcp_server(opts: &McpServerOptions) -> Tool {
    let args = serde_json::to_value(opts)
        .ok()
        .and_then(|v| v.as_object().cloned());
    Tool::Provider(ProviderTool {
        id: "xai.mcp".into(),
        name: "mcp".into(),
        args,
        provider_options: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_url_required_field_emitted() {
        let Tool::Provider(p) = mcp_server(&McpServerOptions {
            server_url: "https://x".into(),
            server_label: Some("lbl".into()),
            server_description: None,
            allowed_tools: Some(vec!["t".into()]),
            headers: None,
            authorization: None,
        }) else {
            panic!("expected provider tool");
        };
        let args = p.args.unwrap();
        assert_eq!(args["serverUrl"], "https://x");
        assert_eq!(args["serverLabel"], "lbl");
        assert_eq!(args["allowedTools"][0], "t");
    }
}
