//! `advisor_20260301` — executor + advisor model pairing.
//!
//! Mirrors `@ai-sdk/anthropic/src/tool/advisor_20260301.ts`. Beta header
//! `advisor-tool-2026-03-01`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::language_model::Tool;
use serde::Serialize;

use super::common::{EphemeralCache, build};

/// Construction-time args for [`advisor_20260301`].
#[derive(Debug, Clone, Serialize, Default)]
pub struct AdvisorArgs {
    /// Advisor model id (required), e.g. `"claude-opus-4-7"`.
    pub model: String,
    /// Maximum advisor calls per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    /// Optional ephemeral cache config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caching: Option<EphemeralCache>,
}

/// `advisor_20260301` — pairs a faster executor with a higher-intelligence
/// advisor model.
///
/// `args.model` is required: an invalid pair returns
/// `400 invalid_request_error` from the API.
#[must_use]
pub fn advisor_20260301(args: AdvisorArgs) -> Tool {
    build("anthropic.advisor_20260301", "advisor", args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::{EphemeralCacheKind, EphemeralCacheTtl};

    fn args_of(tool: &Tool) -> serde_json::Value {
        match tool {
            Tool::Provider(p) => serde_json::Value::Object(p.args.clone().unwrap_or_default()),
            Tool::Function(_) => panic!("expected provider tool"),
        }
    }

    #[test]
    fn advisor_minimal_args() {
        let t = advisor_20260301(AdvisorArgs {
            model: "claude-opus-4-7".into(),
            ..Default::default()
        });
        let args = args_of(&t);
        assert_eq!(args["model"], "claude-opus-4-7");
        assert!(args.get("max_uses").is_none());
        assert!(args.get("caching").is_none());
    }

    #[test]
    fn advisor_with_caching() {
        let t = advisor_20260301(AdvisorArgs {
            model: "claude-opus-4-7".into(),
            max_uses: Some(5),
            caching: Some(EphemeralCache {
                kind: EphemeralCacheKind::Ephemeral,
                ttl: EphemeralCacheTtl::OneHour,
            }),
        });
        let args = args_of(&t);
        assert_eq!(args["max_uses"], 5);
        assert_eq!(args["caching"]["type"], "ephemeral");
        assert_eq!(args["caching"]["ttl"], "1h");
    }
}
