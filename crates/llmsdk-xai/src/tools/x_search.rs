//! `xai.x_search` typed factory.
//!
//! Mirrors `@ai-sdk/xai/src/tool/x-search.ts`.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::language_model::{ProviderTool, Tool};
use serde::Serialize;

/// Optional knobs for [`x_search`].
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct XSearchOptions {
    /// Restrict results to the given X handles (max 10 upstream).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_x_handles: Option<Vec<String>>,
    /// Exclude the given X handles (max 10 upstream).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excluded_x_handles: Option<Vec<String>>,
    /// Lower bound on result date (ISO-8601 string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_date: Option<String>,
    /// Upper bound on result date (ISO-8601 string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_date: Option<String>,
    /// Enable image understanding for attached media.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_image_understanding: Option<bool>,
    /// Enable video understanding for attached media.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_video_understanding: Option<bool>,
}

/// Build a `xai.x_search` provider tool.
///
/// # Examples
///
/// ```
/// use llmsdk_xai::tools::{x_search, XSearchOptions};
/// let tool = x_search(&XSearchOptions {
///     from_date: Some("2020-01-01".into()),
///     ..Default::default()
/// });
/// let _ = tool;
/// ```
#[must_use]
pub fn x_search(opts: &XSearchOptions) -> Tool {
    let args = serde_json::to_value(opts)
        .ok()
        .and_then(|v| v.as_object().cloned());
    Tool::Provider(ProviderTool {
        id: "xai.x_search".into(),
        name: "x_search".into(),
        args,
        provider_options: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_date_emits_camel_case() {
        let Tool::Provider(p) = x_search(&XSearchOptions {
            from_date: Some("2020-01-01".into()),
            enable_video_understanding: Some(true),
            ..Default::default()
        }) else {
            panic!("expected provider tool");
        };
        let args = p.args.unwrap();
        assert_eq!(args["fromDate"], "2020-01-01");
        assert_eq!(args["enableVideoUnderstanding"], true);
    }
}
