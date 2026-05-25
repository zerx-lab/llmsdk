//! `web_search_*` — server-side web search tools.
//!
//! Mirrors `tool/web-search_20250305.ts` / `web-search_20260209.ts`.
//! `20260209` requires beta header `code-execution-web-tools-2026-02-09`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::Tool;
use serde::Serialize;

use super::common::{UserLocation, build};

/// Args for [`web_search_20250305`] / [`web_search_20260209`].
#[derive(Debug, Clone, Serialize, Default)]
pub struct WebSearchArgs {
    /// Maximum searches per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    /// Allow-list (whitelist).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    /// Block-list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
    /// Approximate user location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
}

/// `web_search_20250305` — initial web-search tool.
#[must_use]
pub fn web_search_20250305(args: WebSearchArgs) -> Tool {
    build("anthropic.web_search_20250305", "web_search", args)
}

/// `web_search_20260209` — latest web-search tool.
#[must_use]
pub fn web_search_20260209(args: WebSearchArgs) -> Tool {
    build("anthropic.web_search_20260209", "web_search", args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::UserLocationKind;

    #[test]
    fn empty_args_omits_field() {
        let t = web_search_20260209(WebSearchArgs::default());
        match t {
            Tool::Provider(p) => assert!(p.args.is_none()),
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }

    #[test]
    fn user_location_with_city_emits_approximate() {
        let t = web_search_20260209(WebSearchArgs {
            max_uses: Some(5),
            allowed_domains: None,
            blocked_domains: None,
            user_location: Some(UserLocation {
                kind: UserLocationKind::Approximate,
                city: Some("Paris".into()),
                region: Some("Île-de-France".into()),
                country: Some("FR".into()),
                timezone: Some("Europe/Paris".into()),
            }),
        });
        match t {
            Tool::Provider(p) => {
                let args = p.args.unwrap();
                assert_eq!(args.get("max_uses").unwrap(), &serde_json::json!(5));
                let loc = args.get("user_location").unwrap();
                assert_eq!(loc["type"], "approximate");
                assert_eq!(loc["city"], "Paris");
                assert_eq!(loc["timezone"], "Europe/Paris");
            }
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }
}
