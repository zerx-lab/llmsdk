//! Video generation model trait and supporting types.
//!
//! Mirrors `@ai-sdk/provider/src/video-model/v4/*` (upstream marks the v4
//! trait as `Experimental_VideoModelV4`; we drop the `Experimental_` prefix
//! to keep the Rust trait surface uniform with the other 5 model traits).
// Rust guideline compliant 2026-02-21

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::shared::{FileBytes, Headers, ProviderMetadata, ProviderOptions, Warning};

/// Contract every video-generation model implements.
#[async_trait]
pub trait VideoModel: Send + Sync + std::fmt::Debug {
    /// Provider id, e.g. `"xai"`.
    fn provider(&self) -> &str;

    /// Provider-specific model id, e.g. `"grok-2-video"`.
    fn model_id(&self) -> &str;

    /// Specification version (currently `"v4"`).
    fn specification_version(&self) -> &'static str {
        "v4"
    }

    /// Maximum videos that can be requested per call.
    ///
    /// Most video models only support `n=1` due to computational cost; the
    /// default returns `Some(1)`.
    async fn max_videos_per_call(&self) -> Option<u32> {
        Some(1)
    }

    /// Generate videos.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::ProviderError`] when the upstream call fails, the
    /// generation job times out, or the response is malformed.
    async fn do_generate(&self, options: VideoOptions) -> Result<VideoResult>;
}

/// Options for one [`VideoModel::do_generate`] call.
///
/// Mirrors `VideoModelV4CallOptions`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoOptions {
    /// Text prompt for the video generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Number of videos to generate. Default: 1.
    #[serde(default = "default_n")]
    pub n: u32,
    /// Aspect ratio, formatted as `WIDTH:HEIGHT` (e.g. `"16:9"`).
    #[serde(
        default,
        rename = "aspectRatio",
        skip_serializing_if = "Option::is_none"
    )]
    pub aspect_ratio: Option<String>,
    /// Resolution, formatted as `WIDTHxHEIGHT` (e.g. `"1280x720"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    /// Duration of the video in seconds.
    ///
    /// Serialized as `duration` to match upstream
    /// `video-model-v4-call-options.ts:36` (`duration: number | undefined`).
    /// The Rust field keeps the `_seconds` suffix for clarity at the call
    /// site; only the JSON key differs.
    #[serde(default, rename = "duration", skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Frames per second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Seed for deterministic generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Source image or video for image-to-video / editing endpoints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<VideoFile>,
    /// Extra HTTP headers (HTTP providers only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    /// Provider-specific options.
    #[serde(
        default,
        rename = "providerOptions",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_options: Option<ProviderOptions>,
}

fn default_n() -> u32 {
    1
}

/// Input image or video for image-to-video / video editing endpoints.
///
/// Mirrors `VideoModelV4File`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum VideoFile {
    /// Inline file bytes.
    File {
        /// IANA media type (e.g. `"video/mp4"`, `"image/png"`).
        #[serde(rename = "mediaType")]
        media_type: String,
        /// File bytes (raw or base64-encoded).
        data: FileBytes,
        /// Provider-specific options.
        #[serde(
            default,
            rename = "providerOptions",
            skip_serializing_if = "Option::is_none"
        )]
        provider_options: Option<ProviderOptions>,
    },
    /// URL pointing to the file.
    Url {
        /// Absolute URL.
        url: String,
        /// Provider-specific options.
        #[serde(
            default,
            rename = "providerOptions",
            skip_serializing_if = "Option::is_none"
        )]
        provider_options: Option<ProviderOptions>,
    },
}

/// Result of [`VideoModel::do_generate`].
///
/// Mirrors `VideoModelV4Result`.
#[derive(Debug, Clone)]
pub struct VideoResult {
    /// Generated videos.
    pub videos: Vec<VideoData>,
    /// Warnings for the call.
    pub warnings: Vec<Warning>,
    /// Provider-specific metadata.
    pub provider_metadata: Option<ProviderMetadata>,
    /// Response info (telemetry).
    ///
    /// Unlike `RequestInfo` / `ResponseInfo` reused elsewhere, this struct
    /// pins `timestamp` and `model_id` as required fields to match the
    /// upstream `VideoModelV4Result.response` contract (both are required
    /// in TS).
    pub response: VideoResponseInfo,
}

/// Response metadata for [`VideoModel::do_generate`].
///
/// Mirrors `VideoModelV4Result.response`. Unlike [`crate::shared::ResponseInfo`]
/// the `timestamp` and `model_id` fields are required.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoResponseInfo {
    /// Timestamp for the start of the generated response (ISO-8601 string).
    pub timestamp: String,
    /// Model id reported by the provider.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// Response headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
}

/// One video returned by the provider.
///
/// Mirrors `VideoModelV4VideoData` (tagged union over URL / base64 / binary).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum VideoData {
    /// Video available at a URL (most common for video providers).
    Url {
        /// Absolute URL to the video file.
        url: String,
        /// IANA media type (e.g. `"video/mp4"`).
        #[serde(rename = "mediaType")]
        media_type: String,
    },
    /// Video as a base64-encoded string.
    Base64 {
        /// Base64-encoded payload.
        data: String,
        /// IANA media type.
        #[serde(rename = "mediaType")]
        media_type: String,
    },
    /// Video as raw binary bytes.
    Binary {
        /// Raw bytes.
        #[serde(with = "binary_serde")]
        data: Bytes,
        /// IANA media type.
        #[serde(rename = "mediaType")]
        media_type: String,
    },
}

mod binary_serde {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(b: &Bytes, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(b)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Bytes, D::Error> {
        let v: Vec<u8> = Vec::deserialize(d)?;
        Ok(Bytes::from(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn options_default_n_is_one() {
        let v: VideoOptions = serde_json::from_value(json!({})).unwrap();
        assert_eq!(v.n, 1);
    }

    #[test]
    fn options_roundtrip_camelcase() {
        let v = VideoOptions {
            prompt: Some("a cat".into()),
            n: 2,
            aspect_ratio: Some("16:9".into()),
            resolution: Some("1920x1080".into()),
            duration_seconds: Some(5.0),
            fps: Some(30),
            seed: Some(42),
            image: None,
            headers: None,
            provider_options: None,
        };
        let j = serde_json::to_value(&v).unwrap();
        assert_eq!(j["aspectRatio"], "16:9");
        // Mirrors upstream `video-model-v4-call-options.ts:36` — the wire key
        // is `duration` (no `Seconds` suffix), even though the Rust field is
        // named `duration_seconds` for call-site clarity.
        assert_eq!(j["duration"], 5.0);
        let back: VideoOptions = serde_json::from_value(j).unwrap();
        assert_eq!(back.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(back.fps, Some(30));
    }

    #[test]
    fn file_tagged_correctly() {
        let f = VideoFile::Url {
            url: "https://example.com/start.png".into(),
            provider_options: None,
        };
        let j = serde_json::to_value(&f).unwrap();
        assert_eq!(j["type"], "url");
    }

    #[test]
    fn data_tagged_correctly() {
        let d = VideoData::Url {
            url: "https://example.com/x.mp4".into(),
            media_type: "video/mp4".into(),
        };
        let j = serde_json::to_value(&d).unwrap();
        assert_eq!(j["type"], "url");
        assert_eq!(j["mediaType"], "video/mp4");
    }
}
