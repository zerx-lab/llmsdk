//! Parse the `xai` slot of [`ProviderOptions`] for image-generation calls.
//!
//! Mirrors `xaiImageModelOptions` from
//! `@ai-sdk/xai/src/xai-image-model-options.ts` — the Zod schema accepts
//! six optional fields and is permissive about unknown keys, so this parser
//! follows the same forgiving shape (silently drops the slot on type
//! mismatch instead of failing the call).
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;

use crate::PROVIDER_ID;

/// Typed view of `provider_options["xai"]` for image generation.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct XaiImageOptions {
    /// Aspect ratio, e.g. `"16:9"`. Overrides the top-level
    /// [`ImageOptions::aspect_ratio`] **only** when the latter is absent
    /// (top-level wins to mirror upstream).
    ///
    /// [`ImageOptions::aspect_ratio`]: llmsdk_provider::image_model::ImageOptions::aspect_ratio
    pub aspect_ratio: Option<String>,
    /// Output container — e.g. `"png"` / `"jpeg"`.
    pub output_format: Option<String>,
    /// Whether to wait synchronously for generation.
    pub sync_mode: Option<bool>,
    /// `"1k"` or `"2k"` resolution preset.
    pub resolution: Option<String>,
    /// `"low"` / `"medium"` / `"high"` quality preset.
    pub quality: Option<String>,
    /// End-user identifier for upstream telemetry.
    pub user: Option<String>,
}

/// Parse the `xai` slot of [`ProviderOptions`], or return defaults.
///
/// Unknown / non-object entries fall back to defaults rather than failing
/// the call — ai-sdk has the same forgiving behavior.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> XaiImageOptions {
    let Some(map) = options else {
        return XaiImageOptions::default();
    };
    let Some(xai) = map.get(PROVIDER_ID) else {
        return XaiImageOptions::default();
    };
    serde_json::from_value::<XaiImageOptions>(serde_json::Value::Object(xai.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts_with(map: &serde_json::Value) -> ProviderOptions {
        let mut po = ProviderOptions::new();
        po.insert(PROVIDER_ID.into(), map.as_object().cloned().unwrap());
        po
    }

    #[test]
    fn missing_provider_options_yields_defaults() {
        let parsed = parse(None);
        assert!(parsed.aspect_ratio.is_none());
        assert!(parsed.output_format.is_none());
        assert!(parsed.sync_mode.is_none());
        assert!(parsed.resolution.is_none());
        assert!(parsed.quality.is_none());
        assert!(parsed.user.is_none());
    }

    #[test]
    fn missing_xai_key_yields_defaults() {
        let mut po = ProviderOptions::new();
        po.insert(
            "openai".into(),
            json!({"quality": "high"}).as_object().cloned().unwrap(),
        );
        let parsed = parse(Some(&po));
        assert!(parsed.quality.is_none());
    }

    #[test]
    fn parses_all_six_known_fields() {
        let po = opts_with(&json!({
            "aspect_ratio": "16:9",
            "output_format": "png",
            "sync_mode": true,
            "resolution": "2k",
            "quality": "high",
            "user": "alice@example.com"
        }));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(parsed.output_format.as_deref(), Some("png"));
        assert_eq!(parsed.sync_mode, Some(true));
        assert_eq!(parsed.resolution.as_deref(), Some("2k"));
        assert_eq!(parsed.quality.as_deref(), Some("high"));
        assert_eq!(parsed.user.as_deref(), Some("alice@example.com"));
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let po = opts_with(&json!({
            "unknownField": 42,
            "quality": "low"
        }));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.quality.as_deref(), Some("low"));
    }

    #[test]
    fn malformed_xai_slot_falls_back_to_defaults() {
        // sync_mode wants a bool — string should not derail the rest.
        // serde_json refuses the whole object on a single mismatch, so
        // this exercises the `unwrap_or_default()` branch.
        let po = opts_with(&json!({
            "sync_mode": "yes please",
            "quality": "medium"
        }));
        let parsed = parse(Some(&po));
        assert!(parsed.sync_mode.is_none());
        assert!(parsed.quality.is_none());
    }
}
