//! `web_fetch_*` — server-side document fetch tools.
//!
//! Mirrors `tool/web-fetch-20250910.ts` / `web-fetch-20260209.ts`.
//! Beta headers:
//! - `20250910` → `web-fetch-2025-09-10`
//! - `20260209` → `code-execution-web-tools-2026-02-09`
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::Tool;
use serde::Serialize;

use super::common::{CitationsConfig, build};

/// Args for [`web_fetch_20250910`] / [`web_fetch_20260209`].
#[derive(Debug, Clone, Serialize, Default)]
pub struct WebFetchArgs {
    /// Maximum fetches per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    /// Allow-list (whitelist).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    /// Block-list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
    /// Enable per-passage citations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
    /// Content token cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<u32>,
}

/// `web_fetch_20250910` — initial web-fetch tool.
#[must_use]
pub fn web_fetch_20250910(args: WebFetchArgs) -> Tool {
    build("anthropic.web_fetch_20250910", "web_fetch", args)
}

/// `web_fetch_20260209` — latest web-fetch tool.
#[must_use]
pub fn web_fetch_20260209(args: WebFetchArgs) -> Tool {
    build("anthropic.web_fetch_20260209", "web_fetch", args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_args_omits_field() {
        let t = web_fetch_20260209(WebFetchArgs::default());
        match t {
            Tool::Provider(p) => assert!(p.args.is_none()),
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }

    #[test]
    fn fully_populated_args_use_snake_case() {
        let t = web_fetch_20260209(WebFetchArgs {
            max_uses: Some(3),
            allowed_domains: Some(vec!["example.com".into()]),
            blocked_domains: None,
            citations: Some(CitationsConfig { enabled: true }),
            max_content_tokens: Some(1000),
        });
        match t {
            Tool::Provider(p) => {
                let args = p.args.unwrap();
                assert_eq!(args.get("max_uses").unwrap(), &serde_json::json!(3));
                assert_eq!(
                    args.get("allowed_domains").unwrap(),
                    &serde_json::json!(["example.com"])
                );
                assert_eq!(args.get("citations").unwrap()["enabled"], true);
                assert_eq!(
                    args.get("max_content_tokens").unwrap(),
                    &serde_json::json!(1000)
                );
                assert!(args.get("blocked_domains").is_none());
            }
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }
}
