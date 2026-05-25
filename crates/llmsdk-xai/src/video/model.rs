//! [`VideoModel`] implementation for xAI video generation.
//!
//! Mirrors `XaiVideoModel` from `@ai-sdk/xai/src/xai-video-model.ts`. Entry:
//! [`XaiVideoModel::new`] via [`crate::Xai::video`].
//!
//! Implementation note: the four supported modes
//! (`text-to-video` / `edit-video` / `extend-video` / `reference-to-video`)
//! all share the same long-running-operation polling loop — they only differ
//! in (a) the POST endpoint and (b) the body fields permitted. The polling
//! loop sleeps with [`tokio::time::sleep`] and is bounded by a wall-clock
//! deadline computed from [`std::time::Instant::now`].
// Rust guideline compliant 2026-05-25

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use llmsdk_provider::ProviderError;
use llmsdk_provider::shared::{Headers, ProviderMetadata};
use llmsdk_provider::video_model::{
    VideoData, VideoModel, VideoOptions, VideoResponseInfo, VideoResult,
};
use llmsdk_provider_utils::http::{JsonRequest, get_json, post_json};
use serde_json::{Map, Value as JsonValue};

use crate::PROVIDER_ID;
use crate::config::Inner;

use super::build::{build_body, resolve_mode};
use super::options::{XaiVideoMode, parse as parse_xai_options};
use super::timestamp::now_iso8601;
use super::wire::{CreateVideoResponse, VideoStatusResponse, VideoUsage};

/// xAI video-generation model handle.
///
/// Cheap to clone — shares the provider's HTTP client and auth state via
/// [`Xai`](crate::Xai)'s `Arc`. Supports `grok-imagine-video` and any future
/// `grok-*-video` id passed verbatim.
#[derive(Debug, Clone)]
pub struct XaiVideoModel {
    inner: Arc<Inner>,
    model_id: String,
}

/// Default poll interval (ms). Mirrors upstream `5000`.
const DEFAULT_POLL_INTERVAL_MS: u64 = 5_000;

/// Default total poll timeout (ms). Mirrors upstream `600000` (10 minutes).
const DEFAULT_POLL_TIMEOUT_MS: u64 = 600_000;

impl XaiVideoModel {
    /// Construct from shared provider state and a model id.
    pub(crate) fn new(inner: Arc<Inner>, model_id: String) -> Self {
        Self { inner, model_id }
    }

    fn endpoint_for(&self, mode: Option<XaiVideoMode>) -> String {
        let base = &self.inner.base_url;
        match mode {
            Some(XaiVideoMode::EditVideo) => format!("{base}/videos/edits"),
            Some(XaiVideoMode::ExtendVideo) => format!("{base}/videos/extensions"),
            _ => format!("{base}/videos/generations"),
        }
    }

    fn poll_endpoint(&self, request_id: &str) -> String {
        format!("{}/videos/{}", self.inner.base_url, request_id)
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
impl VideoModel for XaiVideoModel {
    fn provider(&self) -> &str {
        PROVIDER_ID
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn max_videos_per_call(&self) -> Option<u32> {
        Some(1)
    }

    async fn do_generate(&self, options: VideoOptions) -> Result<VideoResult, ProviderError> {
        let xai = parse_xai_options(options.provider_options.as_ref());
        let mode = resolve_mode(&xai);

        let (body, warnings) = build_body(&self.model_id, &options, &xai, mode)?;
        let endpoint = self.endpoint_for(mode);
        let headers = self.merged_headers(options.headers.as_ref());

        // Step 1 — submit the job.
        let mut req = JsonRequest::new(endpoint.clone(), body);
        req.headers = headers.clone();
        let resp = post_json::<_, CreateVideoResponse>(&self.inner.http, req).await?;
        let request_id = resp.value.request_id.ok_or_else(|| {
            ProviderError::api_call_builder(&endpoint, "xAI did not return a request_id")
                .response_body("missing request_id in /videos/* create response")
                .build()
        })?;

        // Step 2 — poll until done / failed / expired / deadline exceeded.
        let interval =
            Duration::from_millis(xai.poll_interval_ms.unwrap_or(DEFAULT_POLL_INTERVAL_MS));
        let timeout = Duration::from_millis(xai.poll_timeout_ms.unwrap_or(DEFAULT_POLL_TIMEOUT_MS));
        let deadline = Instant::now() + timeout;
        let poll_url = self.poll_endpoint(&request_id);

        loop {
            tokio::time::sleep(interval).await;
            if Instant::now() > deadline {
                return Err(timeout_error(&poll_url, timeout));
            }

            let polled =
                get_json::<VideoStatusResponse, _>(&self.inner.http, &poll_url, &headers).await?;
            let status = polled.value;
            let response_headers = polled.headers;

            match classify_status(&status) {
                StatusOutcome::Done => {
                    let video = status.video.as_ref().ok_or_else(|| {
                        ProviderError::api_call_builder(
                            &poll_url,
                            "xAI reported `done` without a video payload",
                        )
                        .response_body("missing `video` field in status response")
                        .build()
                    })?;

                    if video.respect_moderation == Some(false) {
                        return Err(ProviderError::api_call_builder(
                            &poll_url,
                            "video generation blocked by content moderation",
                        )
                        .response_body("respect_moderation=false")
                        .build());
                    }

                    return Ok(VideoResult {
                        videos: vec![VideoData::Url {
                            url: video.url.clone(),
                            media_type: "video/mp4".into(),
                        }],
                        warnings,
                        provider_metadata: Some(build_provider_metadata(
                            &request_id,
                            video.url.as_str(),
                            video.duration,
                            status.usage.as_ref(),
                            status.progress,
                        )),
                        response: VideoResponseInfo {
                            timestamp: now_iso8601(),
                            model_id: self.model_id.clone(),
                            headers: Some(headers_to_provider(response_headers)),
                        },
                    });
                }
                StatusOutcome::Expired => {
                    return Err(ProviderError::api_call_builder(
                        &poll_url,
                        "video generation request expired",
                    )
                    .response_body("status=expired")
                    .build());
                }
                StatusOutcome::Failed => {
                    return Err(ProviderError::api_call_builder(
                        &poll_url,
                        "video generation failed",
                    )
                    .response_body("status=failed")
                    .build());
                }
                StatusOutcome::Pending => {}
            }
        }
    }
}

/// Logical interpretation of the `status` field on a poll response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatusOutcome {
    Done,
    Pending,
    Expired,
    Failed,
}

/// Mirror upstream's status disambiguation:
///
/// - `status === 'done'` → done.
/// - `status` absent **and** `video.url` present → done (early API).
/// - `status === 'expired'` / `'failed'` → terminal failure.
/// - Anything else (including `'pending'`) → keep polling.
fn classify_status(s: &VideoStatusResponse) -> StatusOutcome {
    match s.status.as_deref() {
        Some("done") => StatusOutcome::Done,
        Some("expired") => StatusOutcome::Expired,
        Some("failed") => StatusOutcome::Failed,
        None if s.video.as_ref().is_some_and(|v| !v.url.is_empty()) => StatusOutcome::Done,
        _ => StatusOutcome::Pending,
    }
}

fn timeout_error(url: &str, timeout: Duration) -> ProviderError {
    ProviderError::api_call_builder(
        url,
        format!(
            "video generation polling timed out after {} ms",
            timeout.as_millis()
        ),
    )
    .response_body("client-side polling deadline exceeded")
    .build()
}

/// Build the `provider_metadata.xai` payload for a successful job.
fn build_provider_metadata(
    request_id: &str,
    video_url: &str,
    duration: Option<f64>,
    usage: Option<&VideoUsage>,
    progress: Option<f64>,
) -> ProviderMetadata {
    let mut xai = Map::new();
    xai.insert("requestId".into(), JsonValue::String(request_id.to_owned()));
    xai.insert("videoUrl".into(), JsonValue::String(video_url.to_owned()));
    if let Some(d) = duration {
        xai.insert("duration".into(), JsonValue::from(d));
    }
    if let Some(usage) = usage
        && let Some(ticks) = usage.cost_in_usd_ticks
    {
        xai.insert("costInUsdTicks".into(), JsonValue::from(ticks));
    }
    if let Some(p) = progress {
        xai.insert("progress".into(), JsonValue::from(p));
    }
    let mut pm = ProviderMetadata::new();
    pm.insert(PROVIDER_ID.into(), xai);
    pm
}

fn headers_to_provider(raw: HashMap<String, String>) -> Headers {
    raw.into_iter().map(|(k, v)| (k, Some(v))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_status_handles_all_documented_states() {
        let s = VideoStatusResponse {
            status: Some("done".into()),
            video: Some(super::super::wire::VideoPayload {
                url: "https://x.ai/v.mp4".into(),
                duration: None,
                respect_moderation: Some(true),
            }),
            ..Default::default()
        };
        assert_eq!(classify_status(&s), StatusOutcome::Done);

        let s2 = VideoStatusResponse {
            status: Some("pending".into()),
            ..Default::default()
        };
        assert_eq!(classify_status(&s2), StatusOutcome::Pending);

        let s3 = VideoStatusResponse {
            status: Some("expired".into()),
            ..Default::default()
        };
        assert_eq!(classify_status(&s3), StatusOutcome::Expired);

        let s4 = VideoStatusResponse {
            status: Some("failed".into()),
            ..Default::default()
        };
        assert_eq!(classify_status(&s4), StatusOutcome::Failed);

        // legacy: status absent + video.url present ⇒ done.
        let s5 = VideoStatusResponse {
            video: Some(super::super::wire::VideoPayload {
                url: "https://x.ai/v.mp4".into(),
                duration: None,
                respect_moderation: None,
            }),
            ..Default::default()
        };
        assert_eq!(classify_status(&s5), StatusOutcome::Done);
    }

    #[test]
    fn build_metadata_includes_all_optional_fields_when_present() {
        let usage = VideoUsage {
            cost_in_usd_ticks: Some(99),
        };
        let pm = build_provider_metadata(
            "req-1",
            "https://cdn.x.ai/v.mp4",
            Some(6.0),
            Some(&usage),
            Some(75.0),
        );
        let xai = pm.get(PROVIDER_ID).expect("xai slot");
        assert_eq!(xai["requestId"], "req-1");
        assert_eq!(xai["videoUrl"], "https://cdn.x.ai/v.mp4");
        assert_eq!(xai["duration"], 6.0);
        assert_eq!(xai["costInUsdTicks"], 99);
        assert_eq!(xai["progress"], 75.0);
    }
}
