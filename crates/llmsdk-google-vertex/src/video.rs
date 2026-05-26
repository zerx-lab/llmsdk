//! Vertex AI Veo video generation (LRO polling).
//!
//! Mirrors `google-vertex-video-model.ts`. Steps:
//! 1. `POST {publishers/google}/models/{id}:predictLongRunning` with
//!    Vertex's `instances[]` + `parameters` wire.
//! 2. Loop: `POST {publishers/google}/models/{id}:fetchPredictOperation`
//!    with `{ operationName }` every `pollIntervalMs` until `done: true`
//!    or the `pollTimeoutMs` deadline trips.
//! 3. Surface base64 / GCS-URI videos plus per-video metadata under
//!    `provider_metadata.googleVertex.videos[]`.
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::{FileBytes, Headers, ProviderMetadata, Warning};
use llmsdk_provider::video_model::{
    VideoData, VideoFile, VideoModel, VideoOptions, VideoResponseInfo, VideoResult,
};
use llmsdk_provider_utils::http::{JsonRequest, post_json};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::PROVIDER_ID_VIDEO;
use crate::auth::cloud_platform_token;
use crate::config::{VertexAuthMode, VertexInner};
use crate::image::encode_base64_for_video;

const DEFAULT_POLL_INTERVAL_MS: u64 = 10_000;
const DEFAULT_POLL_TIMEOUT_MS: u64 = 600_000;

/// Vertex Veo video model handle.
#[derive(Debug, Clone)]
pub struct GoogleVertexVideoModel {
    inner: Arc<VertexInner>,
    model_id: String,
}

impl GoogleVertexVideoModel {
    pub(crate) fn new(inner: Arc<VertexInner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    async fn merged_headers(
        &self,
        per_call: Option<&Headers>,
    ) -> Result<HashMap<String, Option<String>>, ProviderError> {
        let mut headers = self.inner.extra_headers.clone();
        match &self.inner.auth {
            VertexAuthMode::Express { api_key } => {
                headers.insert("x-goog-api-key".into(), Some(api_key.clone()));
            }
            VertexAuthMode::OAuth { token_provider, .. } => {
                let token = cloud_platform_token(token_provider.as_ref()).await?;
                headers.insert("Authorization".into(), Some(format!("Bearer {token}")));
            }
        }
        if let Some(h) = per_call {
            for (k, v) in h {
                headers.insert(k.clone(), v.clone());
            }
        }
        Ok(headers)
    }
}

#[async_trait]
impl VideoModel for GoogleVertexVideoModel {
    fn provider(&self) -> &str {
        PROVIDER_ID_VIDEO
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_videos_per_call(&self) -> Option<u32> {
        Some(4)
    }

    async fn do_generate(&self, options: VideoOptions) -> Result<VideoResult, ProviderError> {
        let vertex_opts =
            parse_video_options(options.provider_options.as_ref())?.unwrap_or_default();
        let mut warnings: Vec<Warning> = Vec::new();

        let mut instance = Map::new();
        if let Some(prompt) = &options.prompt {
            instance.insert("prompt".into(), Value::String(prompt.clone()));
        }

        if let Some(image) = &options.image {
            match image {
                VideoFile::Url { .. } => {
                    warnings.push(Warning::Unsupported {
                        feature: "URL-based image input".into(),
                        details: Some(
                            "Vertex AI video models require base64-encoded images or GCS URIs. URL will be ignored."
                                .into(),
                        ),
                    });
                }
                VideoFile::File {
                    media_type, data, ..
                } => {
                    let bytes_b64 = match data {
                        FileBytes::Base64(s) => s.clone(),
                        FileBytes::Bytes(b) => encode_base64_for_video(b),
                    };
                    let mut img = Map::new();
                    img.insert("bytesBase64Encoded".into(), Value::String(bytes_b64));
                    img.insert(
                        "mimeType".into(),
                        Value::String(if media_type.is_empty() {
                            "image/png".into()
                        } else {
                            media_type.clone()
                        }),
                    );
                    instance.insert("image".into(), Value::Object(img));
                }
            }
        }

        if let Some(refs) = &vertex_opts.reference_images {
            let mut arr: Vec<Value> = Vec::new();
            for r in refs {
                let mut entry = Map::new();
                if let Some(b) = &r.bytes_base64_encoded {
                    entry.insert("bytesBase64Encoded".into(), Value::String(b.clone()));
                }
                if let Some(g) = &r.gcs_uri {
                    entry.insert("gcsUri".into(), Value::String(g.clone()));
                }
                arr.push(Value::Object(entry));
            }
            instance.insert("referenceImages".into(), Value::Array(arr));
        }

        let mut parameters = Map::new();
        parameters.insert("sampleCount".into(), Value::from(options.n));
        if let Some(ar) = &options.aspect_ratio {
            parameters.insert("aspectRatio".into(), Value::String(ar.clone()));
        }
        if let Some(res) = &options.resolution {
            let mapped = match res.as_str() {
                "1280x720" => "720p",
                "1920x1080" => "1080p",
                "3840x2160" => "4k",
                other => other,
            };
            parameters.insert("resolution".into(), Value::String(mapped.into()));
        }
        if let Some(d) = options.duration_seconds {
            parameters.insert("durationSeconds".into(), Value::from(d));
        }
        if let Some(seed) = options.seed {
            parameters.insert("seed".into(), Value::from(seed));
        }
        if let Some(pg) = &vertex_opts.person_generation {
            parameters.insert("personGeneration".into(), Value::String(pg.clone()));
        }
        if let Some(np) = &vertex_opts.negative_prompt {
            parameters.insert("negativePrompt".into(), Value::String(np.clone()));
        }
        if let Some(ga) = vertex_opts.generate_audio {
            parameters.insert("generateAudio".into(), Value::Bool(ga));
        }
        if let Some(gcs) = &vertex_opts.gcs_output_directory {
            parameters.insert("gcsOutputDirectory".into(), Value::String(gcs.clone()));
        }
        for (k, v) in &vertex_opts.extras {
            parameters.insert(k.clone(), v.clone());
        }

        let mut body = Map::new();
        body.insert(
            "instances".into(),
            Value::Array(vec![Value::Object(instance)]),
        );
        body.insert("parameters".into(), Value::Object(parameters));

        let base = self.inner.publishers_google_base();
        let predict_url = format!("{base}/models/{}:predictLongRunning", self.model_id);
        let poll_url = format!("{base}/models/{}:fetchPredictOperation", self.model_id);
        let headers = self.merged_headers(options.headers.as_ref()).await?;

        let mut req = JsonRequest::new(predict_url.clone(), Value::Object(body));
        req.headers = headers.clone();

        let envelope = post_json::<_, OperationResponse>(&self.inner.http, req).await?;

        let operation_name = envelope.value.name.clone().ok_or_else(|| {
            ProviderError::api_call_builder(
                predict_url.clone(),
                "No operation name returned from API",
            )
            .build()
        })?;

        let poll_interval = Duration::from_millis(
            vertex_opts
                .poll_interval_ms
                .unwrap_or(DEFAULT_POLL_INTERVAL_MS),
        );
        let poll_timeout = Duration::from_millis(
            vertex_opts
                .poll_timeout_ms
                .unwrap_or(DEFAULT_POLL_TIMEOUT_MS),
        );
        let deadline = Instant::now() + poll_timeout;

        let mut final_op = envelope.value;
        let mut last_headers: HashMap<String, String> = envelope.headers;

        while !final_op.done.unwrap_or(false) {
            if Instant::now() > deadline {
                return Err(ProviderError::api_call_builder(
                    poll_url.clone(),
                    format!(
                        "Video generation timed out after {} ms",
                        poll_timeout.as_millis()
                    ),
                )
                .build());
            }
            tokio::time::sleep(poll_interval).await;

            let mut poll_body = Map::new();
            poll_body.insert(
                "operationName".into(),
                Value::String(operation_name.clone()),
            );
            let mut poll_req = JsonRequest::new(poll_url.clone(), Value::Object(poll_body));
            poll_req.headers = self.merged_headers(options.headers.as_ref()).await?;
            let polled = post_json::<_, OperationResponse>(&self.inner.http, poll_req).await?;
            last_headers = polled.headers;
            final_op = polled.value;
        }

        if let Some(err) = &final_op.error {
            return Err(ProviderError::api_call_builder(
                poll_url,
                format!("Video generation failed: {}", err.message),
            )
            .build());
        }

        let videos_payload = final_op
            .response
            .as_ref()
            .and_then(|r| r.videos.clone())
            .unwrap_or_default();
        if videos_payload.is_empty() {
            return Err(ProviderError::api_call_builder(
                poll_url,
                format!("No videos in response. Response: {final_op:?}"),
            )
            .build());
        }

        let mut videos: Vec<VideoData> = Vec::new();
        let mut video_meta: Vec<Value> = Vec::new();
        for v in videos_payload {
            if let Some(b64) = v.bytes_base64_encoded {
                videos.push(VideoData::Base64 {
                    data: b64,
                    media_type: v.mime_type.clone().unwrap_or_else(|| "video/mp4".into()),
                });
                let mut o = Map::new();
                if let Some(m) = &v.mime_type {
                    o.insert("mimeType".into(), Value::String(m.clone()));
                }
                video_meta.push(Value::Object(o));
            } else if let Some(gcs) = v.gcs_uri {
                videos.push(VideoData::Url {
                    url: gcs.clone(),
                    media_type: v.mime_type.clone().unwrap_or_else(|| "video/mp4".into()),
                });
                let mut o = Map::new();
                o.insert("gcsUri".into(), Value::String(gcs));
                if let Some(m) = &v.mime_type {
                    o.insert("mimeType".into(), Value::String(m.clone()));
                }
                video_meta.push(Value::Object(o));
            }
        }
        if videos.is_empty() {
            return Err(ProviderError::api_call_builder(
                predict_url,
                "No valid videos in response",
            )
            .build());
        }

        let mut payload = Map::new();
        payload.insert("videos".into(), Value::Array(video_meta));
        let mut provider_metadata = ProviderMetadata::new();
        provider_metadata.insert("googleVertex".into(), payload.clone());
        provider_metadata.insert("google-vertex".into(), payload.clone());
        provider_metadata.insert("vertex".into(), payload);

        Ok(VideoResult {
            videos,
            warnings,
            provider_metadata: Some(provider_metadata),
            response: VideoResponseInfo {
                timestamp: timestamp_iso(),
                model_id: self.model_id.clone(),
                headers: Some(
                    last_headers
                        .into_iter()
                        .map(|(k, v)| (k, Some(v)))
                        .collect(),
                ),
            },
        })
    }
}

fn timestamp_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("{secs}")
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct VertexVideoOptions {
    #[serde(default, rename = "pollIntervalMs")]
    poll_interval_ms: Option<u64>,
    #[serde(default, rename = "pollTimeoutMs")]
    poll_timeout_ms: Option<u64>,
    #[serde(default, rename = "personGeneration")]
    person_generation: Option<String>,
    #[serde(default, rename = "negativePrompt")]
    negative_prompt: Option<String>,
    #[serde(default, rename = "generateAudio")]
    generate_audio: Option<bool>,
    #[serde(default, rename = "gcsOutputDirectory")]
    gcs_output_directory: Option<String>,
    #[serde(default, rename = "referenceImages")]
    reference_images: Option<Vec<VertexReferenceImage>>,
    /// Passthrough for any other parameter known to upstream.
    #[serde(flatten)]
    extras: Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct VertexReferenceImage {
    #[serde(default, rename = "bytesBase64Encoded")]
    bytes_base64_encoded: Option<String>,
    #[serde(default, rename = "gcsUri")]
    gcs_uri: Option<String>,
}

fn parse_video_options(
    provider_options: Option<&llmsdk_provider::shared::ProviderOptions>,
) -> Result<Option<VertexVideoOptions>, ProviderError> {
    let Some(opts) = provider_options else {
        return Ok(None);
    };
    for key in ["googleVertex", "vertex"] {
        if let Some(payload) = opts.get(key) {
            let value = Value::Object(payload.clone());
            let parsed: VertexVideoOptions =
                serde_json::from_value(value.clone()).map_err(|e| {
                    ProviderError::type_validation(
                        format!("provider_options.{key}"),
                        value,
                        e.to_string(),
                    )
                })?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

#[derive(Debug, Clone, Deserialize)]
struct OperationResponse {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    error: Option<OperationError>,
    #[serde(default)]
    response: Option<OperationPayload>,
}

#[derive(Debug, Clone, Deserialize)]
struct OperationError {
    message: String,
    #[serde(default)]
    #[allow(dead_code, reason = "captured for diagnostics; not surfaced today")]
    code: Option<i64>,
    #[serde(default)]
    #[allow(dead_code, reason = "captured for diagnostics; not surfaced today")]
    status: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OperationPayload {
    #[serde(default)]
    videos: Option<Vec<OperationVideo>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OperationVideo {
    #[serde(default, rename = "bytesBase64Encoded")]
    bytes_base64_encoded: Option<String>,
    #[serde(default, rename = "gcsUri")]
    gcs_uri: Option<String>,
    #[serde(default, rename = "mimeType")]
    mime_type: Option<String>,
}
