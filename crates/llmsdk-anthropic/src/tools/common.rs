//! Shared building blocks for Anthropic provider-defined tool args.
//!
//! Mirrors fragments reused across `web_fetch_*`, `web_search_*`,
//! `advisor_*`, etc. in `@ai-sdk/anthropic/src/tool/`.
// Rust guideline compliant 2026-02-21

use llmsdk_provider::json::JsonObject;
use llmsdk_provider::language_model::{ProviderTool, Tool};
use serde::Serialize;

/// Build a [`Tool::Provider`] from an Anthropic tool id, on-wire name, and
/// typed args. `args` are serialized to a JSON object; an empty struct
/// becomes `None` so the wire payload omits the `args` field.
pub(crate) fn build<A>(id: &'static str, name: &'static str, args: A) -> Tool
where
    A: Serialize,
{
    let value = serde_json::to_value(args).expect("Anthropic tool args serialize");
    let args = match value {
        serde_json::Value::Object(obj) if obj.is_empty() => None,
        serde_json::Value::Object(obj) => Some(obj),
        // Args structs are always objects by construction; this branch is
        // defensive — fall back to wrapping the value under "args".
        other => Some({
            let mut o = JsonObject::new();
            o.insert("args".to_owned(), other);
            o
        }),
    };

    Tool::Provider(ProviderTool {
        id: id.to_owned(),
        name: name.to_owned(),
        args,
        provider_options: None,
    })
}

/// `{ enabled: bool }` shared by `web_fetch_*` and document blocks.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CitationsConfig {
    /// Enable citations on fetched documents.
    pub enabled: bool,
}

/// Approximate user location for `web_search_*`.
#[derive(Debug, Clone, Serialize)]
pub struct UserLocation {
    /// Always `"approximate"`.
    #[serde(rename = "type")]
    pub kind: UserLocationKind,
    /// City name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// Region or state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Country.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// IANA timezone id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

/// Allowed [`UserLocation::kind`] values.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UserLocationKind {
    /// Approximate location (only currently-supported variant).
    Approximate,
}

impl Default for UserLocation {
    fn default() -> Self {
        Self {
            kind: UserLocationKind::Approximate,
            city: None,
            region: None,
            country: None,
            timezone: None,
        }
    }
}

/// `{ type: "ephemeral", ttl: "5m" | "1h" }` for `advisor_20260301.caching`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EphemeralCache {
    /// Always `"ephemeral"`.
    #[serde(rename = "type")]
    pub kind: EphemeralCacheKind,
    /// Cache TTL.
    pub ttl: EphemeralCacheTtl,
}

/// Cache kind discriminator.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EphemeralCacheKind {
    /// Always `"ephemeral"`.
    Ephemeral,
}

/// Supported ephemeral cache TTL values.
#[derive(Debug, Clone, Copy, Serialize)]
pub enum EphemeralCacheTtl {
    /// 5-minute TTL.
    #[serde(rename = "5m")]
    FiveMinutes,
    /// 1-hour TTL.
    #[serde(rename = "1h")]
    OneHour,
}
