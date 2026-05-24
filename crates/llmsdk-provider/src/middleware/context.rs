//! Shared context that travels with a call across the middleware chain.
//!
//! Middleware stages can pass structured metadata (request id, trace span id,
//! parent operation id, ...) to later stages and to the provider impl by
//! stashing it in `CallOptions.provider_options["llmsdk"]`. The `"llmsdk"`
//! bucket is reserved for this purpose — provider crates ignore it on the
//! wire.
//!
//! Why not extend the trait? Adding a `&mut Context` argument to every method
//! would be a viral breaking change. Reusing the existing
//! [`crate::shared::ProviderOptions`] surface keeps the trait stable and gives
//! callers a place to drop their own fields too.
// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::language_model::CallOptions;
use crate::shared::ProviderOptions;

/// Reserved provider id for cross-middleware metadata.
pub const LLMSDK_OPTIONS_KEY: &str = "llmsdk";

/// Structured fields carried across the middleware chain.
///
/// Round-trips through `serde_json` so the bag stays JSON-compatible with the
/// rest of `provider_options`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MiddlewareContext {
    /// Unique id for this end-to-end request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Trace id for distributed tracing (W3C trace-context style).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Parent span id within the trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Logical operation name (e.g. `"chat.completion"`, `"embed.query"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
}

impl MiddlewareContext {
    /// Build with a fresh request id.
    #[must_use]
    pub fn with_request_id(id: impl Into<String>) -> Self {
        Self {
            request_id: Some(id.into()),
            ..Default::default()
        }
    }

    /// Read the context (if any) from a [`CallOptions`].
    ///
    /// Returns `None` when no `llmsdk` bucket exists or when its contents
    /// don't deserialize as [`MiddlewareContext`].
    #[must_use]
    pub fn read(options: &CallOptions) -> Option<Self> {
        Self::read_from(options.provider_options.as_ref()?)
    }

    /// Read from a raw `ProviderOptions` map.
    #[must_use]
    pub fn read_from(options: &ProviderOptions) -> Option<Self> {
        let bucket = options.get(LLMSDK_OPTIONS_KEY)?;
        serde_json::from_value::<Self>(Value::Object(bucket.clone())).ok()
    }

    /// Write `self` into a [`CallOptions`], merging onto any existing
    /// `llmsdk` bucket (caller fields win).
    pub fn write(&self, options: &mut CallOptions) {
        let bucket = options
            .provider_options
            .get_or_insert_with(ProviderOptions::default)
            .entry(LLMSDK_OPTIONS_KEY.to_owned())
            .or_default();
        let value = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(map) = value {
            for (k, v) in map {
                bucket.insert(k, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_call_options() {
        let ctx = MiddlewareContext {
            request_id: Some("req-123".into()),
            trace_id: Some("trace-abc".into()),
            parent_span_id: None,
            operation: Some("chat.completion".into()),
        };
        let mut opts = CallOptions::default();
        ctx.write(&mut opts);

        let read = MiddlewareContext::read(&opts).expect("present");
        assert_eq!(read, ctx);
    }

    #[test]
    fn write_preserves_existing_llmsdk_bucket_fields() {
        let mut opts = CallOptions::default();
        let mut po = ProviderOptions::default();
        let mut bucket = serde_json::Map::new();
        bucket.insert("custom".into(), Value::String("value".into()));
        po.insert(LLMSDK_OPTIONS_KEY.into(), bucket);
        opts.provider_options = Some(po);

        MiddlewareContext::with_request_id("req-1").write(&mut opts);

        let bucket = opts
            .provider_options
            .as_ref()
            .unwrap()
            .get(LLMSDK_OPTIONS_KEY)
            .unwrap();
        assert_eq!(bucket.get("custom"), Some(&Value::String("value".into())));
        assert_eq!(
            bucket.get("request_id"),
            Some(&Value::String("req-1".into()))
        );
    }

    #[test]
    fn read_returns_none_when_no_bucket() {
        let opts = CallOptions::default();
        assert!(MiddlewareContext::read(&opts).is_none());
    }
}
