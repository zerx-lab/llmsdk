//! `xai.web_search` typed factory.
//!
//! Mirrors `@ai-sdk/xai/src/tool/web-search.ts`. Three optional knobs:
//! domain allowlist, denylist, and image-understanding toggle.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{ProviderTool, Tool};
use serde::Serialize;

/// Optional knobs for [`web_search`].
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSearchOptions {
    /// Restrict searches to the given domains (max 5 upstream).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    /// Exclude the given domains (max 5 upstream).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excluded_domains: Option<Vec<String>>,
    /// Enable image understanding for fetched pages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_image_understanding: Option<bool>,
}

/// Build a `xai.web_search` provider tool.
///
/// Pass [`WebSearchOptions::default`] (or `WebSearchOptions { .. }`) to skip
/// every optional field.
///
/// # Examples
///
/// ```
/// use llmsdk_xai::tools::{web_search, WebSearchOptions};
/// let tool = web_search(&WebSearchOptions {
///     allowed_domains: Some(vec!["example.com".into()]),
///     ..Default::default()
/// });
/// let _ = tool;
/// ```
#[must_use]
pub fn web_search(opts: &WebSearchOptions) -> Tool {
    let args = serde_json::to_value(opts)
        .ok()
        .and_then(|v| v.as_object().cloned());
    Tool::Provider(ProviderTool {
        id: "xai.web_search".into(),
        name: "web_search".into(),
        args,
        provider_options: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_args_serialize_to_empty_object() {
        let Tool::Provider(p) = web_search(&WebSearchOptions::default()) else {
            panic!("expected provider tool");
        };
        assert!(p.args.is_some_and(|m| m.is_empty()));
    }

    #[test]
    fn allowed_domains_serialize_with_snake_case_wire_key() {
        let Tool::Provider(p) = web_search(&WebSearchOptions {
            allowed_domains: Some(vec!["a.com".into()]),
            ..Default::default()
        }) else {
            panic!("expected provider tool");
        };
        let args = p.args.unwrap();
        assert_eq!(args["allowedDomains"][0], "a.com");
    }
}
