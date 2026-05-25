//! Parse the `xai` slot of [`ProviderOptions`] for video-generation calls.
//!
//! Mirrors `xaiVideoModelOptionsSchema` from
//! `@ai-sdk/xai/src/xai-video-model-options.ts`. Unlike chat / image, the
//! Zod schema is **discriminated by `mode`** with `passthrough()` allowing
//! arbitrary extra `xai.*` keys to flow onto the wire — this parser
//! mirrors that behavior:
//!
//! 1. Known options (`mode`, `pollIntervalMs`, `pollTimeoutMs`, `resolution`,
//!    `videoUrl`, `referenceImageUrls`) are pulled into [`XaiVideoOptions`].
//! 2. The original JSON object is returned alongside so the caller can spread
//!    the unknown keys onto the request body.
//!
//! Malformed types do **not** fail the call — we mirror upstream's forgiving
//! `lazySchema` behavior by falling back to defaults instead.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::shared::ProviderOptions;
use serde::Deserialize;
use serde_json::{Map, Value as JsonValue};

use crate::PROVIDER_ID;

/// Logical mode declared in `provider_options.xai.mode`.
///
/// Auto-detected when absent — see [`super::model::resolve_mode`]. Variant
/// names intentionally keep the `Video` suffix to match the upstream
/// `'edit-video' | 'extend-video' | 'reference-to-video'` string literals
/// 1:1; `allow(enum_variant_names)` silences the cosmetic clippy hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[allow(
    clippy::enum_variant_names,
    reason = "variants mirror upstream literal strings 1:1"
)]
pub(crate) enum XaiVideoMode {
    EditVideo,
    ExtendVideo,
    ReferenceToVideo,
}

/// Typed view of the `xai` slot for video generation.
///
/// `extras` captures any **unknown** keys (i.e. keys other than the six
/// known names) so they can be spread onto the request body verbatim,
/// matching upstream's last-loop pass-through.
#[derive(Debug, Clone, Default)]
pub(crate) struct XaiVideoOptions {
    pub(crate) mode: Option<XaiVideoMode>,
    pub(crate) poll_interval_ms: Option<u64>,
    pub(crate) poll_timeout_ms: Option<u64>,
    pub(crate) resolution: Option<String>,
    pub(crate) video_url: Option<String>,
    pub(crate) reference_image_urls: Option<Vec<String>>,
    /// Pass-through keys not recognized above.
    pub(crate) extras: Map<String, JsonValue>,
}

/// Names of the keys consumed by [`XaiVideoOptions`].
///
/// Anything **not** in this list ends up in `extras` (mirrors upstream's
/// `if (![...known].includes(key)) body[key] = value`). Kept under `cfg(test)`
/// because the runtime parser uses an explicit `match` instead of a lookup.
#[cfg(test)]
const KNOWN_KEYS: &[&str] = &[
    "mode",
    "pollIntervalMs",
    "pollTimeoutMs",
    "resolution",
    "videoUrl",
    "referenceImageUrls",
];

/// Parse `provider_options["xai"]`, or return defaults when missing.
pub(crate) fn parse(options: Option<&ProviderOptions>) -> XaiVideoOptions {
    let Some(map) = options else {
        return XaiVideoOptions::default();
    };
    let Some(xai) = map.get(PROVIDER_ID) else {
        return XaiVideoOptions::default();
    };

    let mut parsed = XaiVideoOptions::default();
    for (key, value) in xai {
        match key.as_str() {
            "mode" => {
                parsed.mode = serde_json::from_value(value.clone()).ok();
            }
            "pollIntervalMs" => {
                parsed.poll_interval_ms = parse_positive_u64(value);
            }
            "pollTimeoutMs" => {
                parsed.poll_timeout_ms = parse_positive_u64(value);
            }
            "resolution" => {
                parsed.resolution = parse_resolution(value);
            }
            "videoUrl" => {
                parsed.video_url = value.as_str().filter(|s| !s.is_empty()).map(str::to_owned);
            }
            "referenceImageUrls" => {
                parsed.reference_image_urls = parse_reference_image_urls(value);
            }
            _ => {
                parsed.extras.insert(key.clone(), value.clone());
            }
        }
    }
    parsed
}

/// Accept only the two upstream values `"480p"` and `"720p"`. Anything else
/// is silently dropped (matches `z.enum(['480p', '720p']).nullish()`).
fn parse_resolution(value: &JsonValue) -> Option<String> {
    let s = value.as_str()?;
    if matches!(s, "480p" | "720p") {
        Some(s.to_owned())
    } else {
        None
    }
}

/// Accept any positive integer that fits in u64. `0`, negatives, fractions
/// and non-numeric input all drop to `None`, matching upstream's
/// `z.number().positive().nullish()`.
fn parse_positive_u64(value: &JsonValue) -> Option<u64> {
    let n = value.as_u64()?;
    if n == 0 { None } else { Some(n) }
}

/// Accept `Vec<String>` between 1 and 7 entries with all entries non-empty,
/// matching `z.array(nonEmptyStringSchema).min(1).max(7)`. Any violation
/// drops the whole field.
fn parse_reference_image_urls(value: &JsonValue) -> Option<Vec<String>> {
    let arr = value.as_array()?;
    if arr.is_empty() || arr.len() > 7 {
        return None;
    }
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let s = item.as_str()?;
        if s.is_empty() {
            return None;
        }
        out.push(s.to_owned());
    }
    Some(out)
}

/// True when this key is one of [`KNOWN_KEYS`]. Exposed so the test module
/// can pin the contract instead of duplicating the list.
#[cfg(test)]
pub(crate) fn is_known_key(key: &str) -> bool {
    KNOWN_KEYS.contains(&key)
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
        assert!(parsed.mode.is_none());
        assert!(parsed.video_url.is_none());
        assert!(parsed.reference_image_urls.is_none());
        assert!(parsed.extras.is_empty());
    }

    #[test]
    fn parses_all_six_known_keys() {
        let po = opts_with(&json!({
            "mode": "edit-video",
            "pollIntervalMs": 1500,
            "pollTimeoutMs": 30000,
            "resolution": "720p",
            "videoUrl": "https://x.ai/in.mp4",
            "referenceImageUrls": ["https://x.ai/a.png"]
        }));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.mode, Some(XaiVideoMode::EditVideo));
        assert_eq!(parsed.poll_interval_ms, Some(1500));
        assert_eq!(parsed.poll_timeout_ms, Some(30_000));
        assert_eq!(parsed.resolution.as_deref(), Some("720p"));
        assert_eq!(parsed.video_url.as_deref(), Some("https://x.ai/in.mp4"));
        assert_eq!(parsed.reference_image_urls.as_ref().unwrap().len(), 1);
        assert!(parsed.extras.is_empty());
    }

    #[test]
    fn unknown_keys_flow_into_extras() {
        let po = opts_with(&json!({
            "mode": "extend-video",
            "videoUrl": "https://x.ai/in.mp4",
            "watermark": "off",
            "loops": 2
        }));
        let parsed = parse(Some(&po));
        assert_eq!(parsed.mode, Some(XaiVideoMode::ExtendVideo));
        assert_eq!(parsed.extras.len(), 2);
        assert_eq!(parsed.extras["watermark"], "off");
        assert_eq!(parsed.extras["loops"], 2);
    }

    #[test]
    fn invalid_resolution_drops_to_none() {
        let po = opts_with(&json!({ "resolution": "1080p" }));
        assert!(parse(Some(&po)).resolution.is_none());
    }

    #[test]
    fn invalid_poll_interval_zero_drops_to_none() {
        let po = opts_with(&json!({ "pollIntervalMs": 0 }));
        assert!(parse(Some(&po)).poll_interval_ms.is_none());
    }

    #[test]
    fn too_many_reference_images_drops_to_none() {
        let urls: Vec<&str> = (0..8).map(|_| "https://x.ai/a.png").collect();
        let po = opts_with(&json!({ "referenceImageUrls": urls }));
        assert!(parse(Some(&po)).reference_image_urls.is_none());
    }

    #[test]
    fn empty_reference_images_drops_to_none() {
        let po = opts_with(&json!({ "referenceImageUrls": [] }));
        assert!(parse(Some(&po)).reference_image_urls.is_none());
    }

    #[test]
    fn known_keys_list_is_complete() {
        for key in [
            "mode",
            "pollIntervalMs",
            "pollTimeoutMs",
            "resolution",
            "videoUrl",
            "referenceImageUrls",
        ] {
            assert!(is_known_key(key), "{key} should be known");
        }
        assert!(!is_known_key("unknown"));
    }
}
