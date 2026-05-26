//! Veo video model implementation with LRO polling.
//!
//! Mirrors `GoogleVideoModel` from
//! `@ai-sdk/google/src/google-video-model.ts`. Steps:
//!
//! 1. `POST /models/{model}:predictLongRunning` with the wire body.
//! 2. Loop: `GET /<operation.name>` every `pollIntervalMs` until
//!    `done: true`, an `error`, or the deadline (`pollTimeoutMs`).
//! 3. Append the provider's `x-goog-api-key` header value as a `?key=`
//!    query string to each returned video URL so the caller can download
//!    it without re-authenticating.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::{FileBytes, Headers, ProviderMetadata};
use llmsdk_provider::video_model::{
    VideoData, VideoFile, VideoModel, VideoOptions, VideoResponseInfo, VideoResult,
};
use llmsdk_provider_utils::http::{JsonRequest, get_json, post_json};
use serde_json::{Map, Value};

use crate::PROVIDER_ID;
use crate::base64::encode_bytes as base64_encode;
use crate::config::Inner;
use crate::error::rewrite_google_error;

use super::options::parse as parse_options;
use super::wire::OperationResponse;

const DEFAULT_POLL_INTERVAL_MS: u64 = 10_000;
const DEFAULT_POLL_TIMEOUT_MS: u64 = 600_000;

/// Gemini video-model handle.
#[derive(Debug, Clone)]
pub struct GoogleVideoModel {
    inner: Arc<Inner>,
    model_id: String,
}

impl GoogleVideoModel {
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn merged_headers(&self, extra: Option<&Headers>) -> HashMap<String, Option<String>> {
        let mut h = self.inner.headers.clone();
        if let Some(extra) = extra {
            for (k, v) in extra {
                h.insert(k.clone(), v.clone());
            }
        }
        h
    }
}

#[async_trait]
impl VideoModel for GoogleVideoModel {
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_videos_per_call(&self) -> Option<u32> {
        Some(4)
    }

    async fn do_generate(&self, options: VideoOptions) -> Result<VideoResult, ProviderError> {
        let google_options = parse_options(options.provider_options.as_ref())?.unwrap_or_default();
        let mut warnings = Vec::new();

        let mut instance = Map::new();
        if let Some(prompt) = &options.prompt {
            instance.insert("prompt".into(), Value::String(prompt.clone()));
        }

        if let Some(image) = &options.image {
            match image {
                VideoFile::Url { .. } => {
                    warnings.push(llmsdk_provider::shared::Warning::Unsupported {
                        feature: "URL-based image input".into(),
                        details: Some(
                            "Google Generative AI video models require base64-encoded images. URL will be ignored.".into(),
                        ),
                    });
                }
                VideoFile::File {
                    media_type, data, ..
                } => {
                    let b64 = match data {
                        FileBytes::Base64(s) => s.clone(),
                        FileBytes::Bytes(b) => base64_encode(b),
                    };
                    let mut inline = Map::new();
                    inline.insert(
                        "mimeType".into(),
                        Value::String(if media_type.is_empty() {
                            "image/png".into()
                        } else {
                            media_type.clone()
                        }),
                    );
                    inline.insert("data".into(), Value::String(b64));
                    let mut img = Map::new();
                    img.insert("inlineData".into(), Value::Object(inline));
                    instance.insert("image".into(), Value::Object(img));
                }
            }
        }

        if let Some(refs) = &google_options.reference_images {
            let mut arr: Vec<Value> = Vec::new();
            for r in refs {
                if let Some(b) = &r.bytes_base64_encoded {
                    let mut inline = Map::new();
                    inline.insert("mimeType".into(), Value::String("image/png".into()));
                    inline.insert("data".into(), Value::String(b.clone()));
                    let mut wrap = Map::new();
                    wrap.insert("inlineData".into(), Value::Object(inline));
                    arr.push(Value::Object(wrap));
                } else if let Some(gcs) = &r.gcs_uri {
                    let mut wrap = Map::new();
                    wrap.insert("gcsUri".into(), Value::String(gcs.clone()));
                    arr.push(Value::Object(wrap));
                }
            }
            instance.insert("referenceImages".into(), Value::Array(arr));
        }

        let instances = Value::Array(vec![Value::Object(instance)]);

        let mut params = Map::new();
        params.insert("sampleCount".into(), Value::from(options.n));
        if let Some(ar) = &options.aspect_ratio {
            params.insert("aspectRatio".into(), Value::String(ar.clone()));
        }
        if let Some(res) = &options.resolution {
            let mapped = match res.as_str() {
                "1280x720" => "720p",
                "1920x1080" => "1080p",
                "3840x2160" => "4k",
                other => other,
            };
            params.insert("resolution".into(), Value::String(mapped.into()));
        }
        if let Some(d) = options.duration_seconds {
            params.insert("durationSeconds".into(), Value::from(d));
        }
        if let Some(seed) = options.seed {
            params.insert("seed".into(), Value::from(seed));
        }
        if let Some(pg) = &google_options.person_generation {
            params.insert("personGeneration".into(), Value::String(pg.clone()));
        }
        if let Some(np) = &google_options.negative_prompt {
            params.insert("negativePrompt".into(), Value::String(np.clone()));
        }
        for (k, v) in &google_options.extras {
            params.insert(k.clone(), v.clone());
        }

        let mut body = Map::new();
        body.insert("instances".into(), instances);
        body.insert("parameters".into(), Value::Object(params));

        let url = format!(
            "{}/models/{}:predictLongRunning",
            self.inner.base_url, self.model_id
        );
        let headers = self.merged_headers(options.headers.as_ref());
        let mut req = JsonRequest::new(url, Value::Object(body));
        req.headers = headers.clone();

        let envelope = match post_json::<_, OperationResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(e) => return Err(rewrite_google_error(e)),
        };

        let operation = envelope.value;
        let operation_name = operation.name.clone().ok_or_else(|| {
            ProviderError::api_call_builder(
                format!(
                    "{}/models/{}:predictLongRunning",
                    self.inner.base_url, self.model_id
                ),
                "No operation name returned from API",
            )
            .build()
        })?;

        let poll_interval = Duration::from_millis(
            google_options
                .poll_interval_ms
                .unwrap_or(DEFAULT_POLL_INTERVAL_MS),
        );
        let poll_timeout = Duration::from_millis(
            google_options
                .poll_timeout_ms
                .unwrap_or(DEFAULT_POLL_TIMEOUT_MS),
        );
        let deadline = Instant::now() + poll_timeout;

        let mut final_op = operation;
        let mut last_headers: Option<HashMap<String, String>> = Some(envelope.headers);
        while !final_op.done.unwrap_or(false) {
            if Instant::now() > deadline {
                return Err(ProviderError::api_call_builder(
                    format!("{}/{}", self.inner.base_url, operation_name),
                    format!(
                        "Video generation timed out after {} ms",
                        poll_timeout.as_millis()
                    ),
                )
                .build());
            }
            tokio::time::sleep(poll_interval).await;
            let poll_url = format!("{}/{}", self.inner.base_url, operation_name);
            let polled =
                match get_json::<OperationResponse, _>(&self.inner.http, &poll_url, &headers).await
                {
                    Ok(r) => r,
                    Err(e) => return Err(rewrite_google_error(e)),
                };
            last_headers = Some(polled.headers);
            final_op = polled.value;
        }

        if let Some(err) = &final_op.error {
            return Err(ProviderError::api_call_builder(
                format!("{}/{}", self.inner.base_url, operation_name),
                format!("Video generation failed: {}", err.message),
            )
            .build());
        }

        let samples = final_op
            .response
            .as_ref()
            .and_then(|r| r.generate_video_response.as_ref())
            .and_then(|g| g.generated_samples.as_ref())
            .cloned()
            .unwrap_or_default();
        if samples.is_empty() {
            return Err(ProviderError::api_call_builder(
                format!("{}/{}", self.inner.base_url, operation_name),
                "No videos in response".to_owned(),
            )
            .build());
        }

        let api_key = self
            .inner
            .headers
            .get("x-goog-api-key")
            .and_then(|v| v.clone());

        let mut videos: Vec<VideoData> = Vec::new();
        let mut video_meta: Vec<Value> = Vec::new();
        for sample in samples {
            if let Some(uri) = sample.video.as_ref().and_then(|v| v.uri.clone()) {
                let url = if let Some(k) = &api_key {
                    if uri.contains('?') {
                        format!("{uri}&key={k}")
                    } else {
                        format!("{uri}?key={k}")
                    }
                } else {
                    uri.clone()
                };
                videos.push(VideoData::Url {
                    url,
                    media_type: "video/mp4".into(),
                });
                let mut o = Map::new();
                o.insert("uri".into(), Value::String(uri));
                video_meta.push(Value::Object(o));
            }
        }

        let mut g_meta = Map::new();
        g_meta.insert("videos".into(), Value::Array(video_meta));
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert(PROVIDER_ID.into(), g_meta);

        Ok(VideoResult {
            videos,
            warnings,
            provider_metadata: Some(provider_metadata),
            response: VideoResponseInfo {
                timestamp: now_iso8601(),
                model_id: self.model_id.clone(),
                headers: last_headers.map(|h| h.into_iter().map(|(k, v)| (k, Some(v))).collect()),
            },
        })
    }
}

fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("{secs}")
}
