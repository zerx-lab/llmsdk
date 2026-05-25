//! xAI video-generation wire types.
//!
//! Mirrors the request/response shapes embedded inside
//! `xai-video-model.ts`. The wire shape is shared by all three POST
//! endpoints (`/videos/generations`, `/videos/edits`, `/videos/extensions`)
//! and by the polling `GET /videos/{request_id}` endpoint.
// Rust guideline compliant 2026-05-25

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as JsonValue};

/// Request body for the three POST endpoints.
///
/// The body is intentionally untyped (a free-form JSON object) because:
///
/// - Only `model` and `prompt` are required across every mode.
/// - `aspect_ratio` / `resolution` / `duration` / `image` / `video` /
///   `reference_images` are gated by mode (see [`build_body`]).
/// - Upstream additionally spreads unknown `xai.*` provider options onto
///   the wire verbatim (mirrored here by the `extras` map).
///
/// [`build_body`]: super::model::build_body
#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct VideoRequest {
    pub(crate) model: String,
    pub(crate) prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) image: Option<VideoSourceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) video: Option<VideoSourceRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reference_images: Option<Vec<VideoSourceRef>>,
    /// Pass-through `xai.*` provider options the parser did not consume.
    ///
    /// Flattened on the wire so each key lives at the request root, matching
    /// upstream's `body[key] = value` loop.
    #[serde(flatten)]
    pub(crate) extras: Map<String, JsonValue>,
}

/// Wire shape for a video / image source: `{ "url": "..." }`.
///
/// `url` may be either an absolute HTTP(S) URL or a `data:<media>;base64,...`
/// URI (image inputs only — videos are always referenced by URL upstream).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct VideoSourceRef {
    pub(crate) url: String,
}

/// Response body of the three POST endpoints — just a job acknowledgement.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CreateVideoResponse {
    /// Job id used for polling. Nullable upstream — we surface that here so
    /// the model can raise a clear `XAI_VIDEO_GENERATION_ERROR` instead of
    /// crashing on deserialization.
    #[serde(default)]
    pub(crate) request_id: Option<String>,
}

/// Response body of `GET /videos/{request_id}` — the polling endpoint.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct VideoStatusResponse {
    /// `"pending"` / `"done"` / `"expired"` / `"failed"`. Nullable upstream:
    /// some early responses omit the field but still include `video.url`,
    /// which the model treats as `done`.
    #[serde(default)]
    pub(crate) status: Option<String>,
    #[serde(default)]
    pub(crate) video: Option<VideoPayload>,
    #[serde(default)]
    pub(crate) usage: Option<VideoUsage>,
    /// Optional 0-100 progress indicator surfaced into `provider_metadata`.
    #[serde(default)]
    pub(crate) progress: Option<f64>,
}

/// The `video` sub-object on a successful poll response.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VideoPayload {
    /// Absolute URL of the generated video. Required when status is `done`.
    pub(crate) url: String,
    /// Duration in seconds, when xAI reports it.
    #[serde(default)]
    pub(crate) duration: Option<f64>,
    /// `false` means the content was blocked by moderation; we promote that
    /// to a `XAI_VIDEO_MODERATION_ERROR`.
    #[serde(default)]
    pub(crate) respect_moderation: Option<bool>,
}

/// `usage` block — xAI-specific cost counter.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VideoUsage {
    #[serde(default)]
    pub(crate) cost_in_usd_ticks: Option<u64>,
}

/// Minimal RFC 4648 base64 **encoder** (alphabet `A-Za-z0-9+/`, `=` padding).
///
/// Inline copy of the same routine in `crate::image::wire::base64_encode` —
/// re-implemented here to keep the `image` module sealed (its `wire` is
/// `mod wire;`, not `pub(crate)`).
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut chunks = input.chunks_exact(3);
    for chunk in chunks.by_ref() {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(n & 0x3F) as usize] as char);
    }
    let rem = chunks.remainder();
    match rem.len() {
        0 => {}
        1 => {
            let n = u32::from(rem[0]) << 16;
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = (u32::from(rem[0]) << 16) | (u32::from(rem[1]) << 8);
            out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        _ => unreachable!("chunks_exact remainder is always < 3"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_body_skips_unset_optional_fields_and_flattens_extras() {
        let mut extras = Map::new();
        extras.insert("custom_key".into(), JsonValue::String("v".into()));
        let req = VideoRequest {
            model: "grok-imagine-video".into(),
            prompt: "a cat".into(),
            extras,
            ..Default::default()
        };
        let value = serde_json::to_value(&req).unwrap();
        let obj = value.as_object().unwrap();
        assert!(obj.contains_key("model"));
        assert!(obj.contains_key("prompt"));
        assert!(!obj.contains_key("duration"));
        assert!(!obj.contains_key("aspect_ratio"));
        assert!(!obj.contains_key("image"));
        assert!(!obj.contains_key("video"));
        assert!(!obj.contains_key("reference_images"));
        // Flattened — `extras` keys live at the root, not under a nested obj.
        assert_eq!(value["custom_key"], "v");
    }

    #[test]
    fn request_body_serializes_nested_references() {
        let req = VideoRequest {
            model: "grok-imagine-video".into(),
            prompt: "edit me".into(),
            video: Some(VideoSourceRef {
                url: "https://x.ai/in.mp4".into(),
            }),
            reference_images: Some(vec![
                VideoSourceRef {
                    url: "https://x.ai/a.png".into(),
                },
                VideoSourceRef {
                    url: "https://x.ai/b.png".into(),
                },
            ]),
            ..Default::default()
        };
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["video"]["url"], "https://x.ai/in.mp4");
        let imgs = value["reference_images"].as_array().unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0]["url"], "https://x.ai/a.png");
    }

    #[test]
    fn create_response_accepts_missing_request_id() {
        let parsed: CreateVideoResponse = serde_json::from_value(json!({})).unwrap();
        assert!(parsed.request_id.is_none());
        let parsed: CreateVideoResponse = serde_json::from_value(json!({
            "request_id": "req-123"
        }))
        .unwrap();
        assert_eq!(parsed.request_id.as_deref(), Some("req-123"));
    }

    #[test]
    fn status_response_parses_done_payload_with_video_url_duration_and_usage() {
        let parsed: VideoStatusResponse = serde_json::from_value(json!({
            "status": "done",
            "video": {
                "url": "https://cdn.x.ai/v/1.mp4",
                "duration": 6.0,
                "respect_moderation": true
            },
            "usage": { "cost_in_usd_ticks": 42 },
            "progress": 100.0
        }))
        .unwrap();
        assert_eq!(parsed.status.as_deref(), Some("done"));
        let video = parsed.video.as_ref().unwrap();
        assert_eq!(video.url, "https://cdn.x.ai/v/1.mp4");
        assert_eq!(video.duration, Some(6.0));
        assert_eq!(video.respect_moderation, Some(true));
        assert_eq!(parsed.usage.unwrap().cost_in_usd_ticks, Some(42));
        assert_eq!(parsed.progress, Some(100.0));
    }

    #[test]
    fn base64_encode_matches_rfc4648_vectors() {
        let cases: &[(&[u8], &str)] = &[
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
        ];
        for (raw, encoded) in cases {
            assert_eq!(base64_encode(raw).as_str(), *encoded, "vector {encoded}");
        }
    }

    #[test]
    fn status_response_accepts_pending_with_no_video() {
        let parsed: VideoStatusResponse = serde_json::from_value(json!({
            "status": "pending"
        }))
        .unwrap();
        assert_eq!(parsed.status.as_deref(), Some("pending"));
        assert!(parsed.video.is_none());
    }
}
