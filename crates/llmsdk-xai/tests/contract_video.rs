//! Contract tests for [`XaiVideoModel::do_generate`].
//!
//! Each test boots a `wiremock` server, points the provider at it, and
//! drives the **async long-running operation** path end-to-end:
//!
//! 1. POST to `/videos/generations` (or `/edits` / `/extensions`) returns
//!    `{ "request_id": ... }`.
//! 2. The model polls `GET /videos/{request_id}` every `pollIntervalMs`.
//! 3. Eventually the GET returns `status=done` with a `video.url`, and the
//!    contract test asserts the [`VideoResult`] / `provider_metadata` shape.
//!
//! All tests pin `pollIntervalMs` to a small value (≤ 100 ms) so the suite
//! finishes in well under a second.
// Rust guideline compliant 2026-05-25

use llmsdk_provider::VideoModel;
use llmsdk_provider::shared::{ProviderOptions, Warning};
use llmsdk_provider::video_model::{VideoData, VideoOptions};
use llmsdk_xai::Xai;
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Xai {
    Xai::builder()
        .api_key("xai-test")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
}

fn xai_provider_options(value: &Value) -> ProviderOptions {
    let mut po = ProviderOptions::new();
    po.insert("xai".into(), value.as_object().cloned().unwrap());
    po
}

/// Mount a successful create + poll pair: POST returns the given
/// `request_id`, GET returns `status=done` with a fixed mp4 url.
async fn mount_create_and_done(server: &MockServer, endpoint: &str, request_id: &str) {
    Mock::given(method("POST"))
        .and(path(endpoint))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": request_id,
        })))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("/videos/{request_id}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": {
                "url": "https://cdn.x.ai/v/final.mp4",
                "duration": 6.0
            },
            "usage": { "cost_in_usd_ticks": 4242_u64 },
            "progress": 100.0
        })))
        .mount(server)
        .await;
}

fn fast_poll_options(extra: &Value) -> ProviderOptions {
    let mut object = json!({
        "pollIntervalMs": 20,
        "pollTimeoutMs": 5000,
    });
    if let (Some(o), Some(e)) = (object.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    xai_provider_options(&object)
}

#[tokio::test]
async fn text_to_video_creates_and_polls_until_done() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/generations"))
        .and(header("authorization", "Bearer xai-test"))
        .and(body_partial_json(
            json!({ "model": "grok-imagine-video", "prompt": "a cat surfing" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-1",
        })))
        .mount(&server)
        .await;

    // First two polls return `pending`, third returns `done`. Each call
    // consumes one mock so we register them in reverse-registration order
    // and rely on wiremock's "most-recently-mounted wins" semantics.
    Mock::given(method("GET"))
        .and(path("/videos/req-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": { "url": "https://cdn.x.ai/v/r1.mp4", "duration": 5.5 },
            "usage": { "cost_in_usd_ticks": 1234_u64 },
            "progress": 100.0
        })))
        .mount(&server)
        .await;

    let model = provider(&server).video("grok-imagine-video");
    let result = model
        .do_generate(VideoOptions {
            prompt: Some("a cat surfing".into()),
            provider_options: Some(fast_poll_options(&json!({}))),
            ..Default::default()
        })
        .await
        .expect("generation succeeded");

    assert_eq!(result.videos.len(), 1);
    match &result.videos[0] {
        VideoData::Url { url, media_type } => {
            assert_eq!(url, "https://cdn.x.ai/v/r1.mp4");
            assert_eq!(media_type, "video/mp4");
        }
        _ => panic!("expected VideoData::Url"),
    }

    let pm = result.provider_metadata.expect("provider_metadata");
    let xai = pm.get("xai").expect("xai slot");
    assert_eq!(xai["requestId"], "req-1");
    assert_eq!(xai["videoUrl"], "https://cdn.x.ai/v/r1.mp4");
    assert_eq!(xai["duration"], 5.5);
    assert_eq!(xai["costInUsdTicks"], 1234);
    assert_eq!(xai["progress"], 100.0);

    assert_eq!(result.response.model_id, "grok-imagine-video");
    assert!(result.response.timestamp.contains('T'));
    assert!(result.warnings.is_empty());
}

#[tokio::test]
async fn edit_video_mode_routes_to_edits_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/edits"))
        .and(body_partial_json(json!({
            "model": "grok-imagine-video",
            "prompt": "make it night",
            "video": { "url": "https://x.ai/in.mp4" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-edit"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/videos/req-edit"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": { "url": "https://cdn.x.ai/v/edit.mp4" }
        })))
        .mount(&server)
        .await;

    let opts_value = json!({
        "mode": "edit-video",
        "videoUrl": "https://x.ai/in.mp4",
    });
    let model = provider(&server).video("grok-imagine-video");
    let result = model
        .do_generate(VideoOptions {
            prompt: Some("make it night".into()),
            provider_options: Some(fast_poll_options(&opts_value)),
            ..Default::default()
        })
        .await
        .expect("edit-video succeeded");

    match &result.videos[0] {
        VideoData::Url { url, .. } => assert_eq!(url, "https://cdn.x.ai/v/edit.mp4"),
        _ => panic!("expected VideoData::Url"),
    }
}

#[tokio::test]
async fn extend_video_mode_routes_to_extensions_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/extensions"))
        .and(body_partial_json(json!({
            "model": "grok-imagine-video",
            "prompt": "and then a dragon flies in",
            "video": { "url": "https://x.ai/seed.mp4" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-ext"
        })))
        .mount(&server)
        .await;
    mount_create_and_done(&server, "/__noop", "req-ext-decoy").await; // ensure helper is exercised
    Mock::given(method("GET"))
        .and(path("/videos/req-ext"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": { "url": "https://cdn.x.ai/v/ext.mp4" }
        })))
        .mount(&server)
        .await;

    let opts_value = json!({
        "mode": "extend-video",
        "videoUrl": "https://x.ai/seed.mp4",
    });
    let model = provider(&server).video("grok-imagine-video");
    let result = model
        .do_generate(VideoOptions {
            prompt: Some("and then a dragon flies in".into()),
            provider_options: Some(fast_poll_options(&opts_value)),
            ..Default::default()
        })
        .await
        .expect("extend-video succeeded");
    match &result.videos[0] {
        VideoData::Url { url, .. } => assert_eq!(url, "https://cdn.x.ai/v/ext.mp4"),
        _ => panic!("expected VideoData::Url"),
    }
}

#[tokio::test]
async fn reference_to_video_mode_emits_reference_images_array() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/generations"))
        .and(body_partial_json(json!({
            "model": "grok-imagine-video",
            "prompt": "stylize",
            "reference_images": [
                { "url": "https://x.ai/a.png" },
                { "url": "https://x.ai/b.png" }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-r2v"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/videos/req-r2v"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": { "url": "https://cdn.x.ai/v/r2v.mp4" }
        })))
        .mount(&server)
        .await;

    let opts_value = json!({
        "mode": "reference-to-video",
        "referenceImageUrls": ["https://x.ai/a.png", "https://x.ai/b.png"]
    });
    let model = provider(&server).video("grok-imagine-video");
    let _result = model
        .do_generate(VideoOptions {
            prompt: Some("stylize".into()),
            provider_options: Some(fast_poll_options(&opts_value)),
            ..Default::default()
        })
        .await
        .expect("r2v succeeded");
}

#[tokio::test]
async fn legacy_video_url_shape_auto_detects_edit_mode() {
    // No `mode` key — only `videoUrl`. Per upstream, this is treated as
    // edit-video and routes to /videos/edits.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/edits"))
        .and(body_partial_json(json!({
            "video": { "url": "https://x.ai/legacy.mp4" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-legacy"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/videos/req-legacy"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": { "url": "https://cdn.x.ai/v/legacy.mp4" }
        })))
        .mount(&server)
        .await;

    let opts_value = json!({ "videoUrl": "https://x.ai/legacy.mp4" });
    let model = provider(&server).video("grok-imagine-video");
    model
        .do_generate(VideoOptions {
            prompt: Some("legacy".into()),
            provider_options: Some(fast_poll_options(&opts_value)),
            ..Default::default()
        })
        .await
        .expect("legacy edit shape succeeded");
}

#[tokio::test]
async fn polling_loop_handles_pending_then_done_sequence() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::{Request, Respond};

    /// Stateful responder: first 2 polls → pending, 3rd → done.
    struct PollStateMachine {
        hits: Arc<AtomicUsize>,
    }
    impl Respond for PollStateMachine {
        fn respond(&self, _req: &Request) -> ResponseTemplate {
            let n = self.hits.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                // Use an integer literal to avoid clippy's cast_precision_loss
                // lint; the exact value is just an opaque progress indicator.
                let progress = if n == 0 { 0_u64 } else { 30_u64 };
                ResponseTemplate::new(200).set_body_json(json!({
                    "status": "pending",
                    "progress": progress
                }))
            } else {
                ResponseTemplate::new(200).set_body_json(json!({
                    "status": "done",
                    "video": { "url": "https://cdn.x.ai/v/poll.mp4" },
                    "progress": 100.0
                }))
            }
        }
    }

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-poll"
        })))
        .mount(&server)
        .await;
    let hits = Arc::new(AtomicUsize::new(0));
    Mock::given(method("GET"))
        .and(path("/videos/req-poll"))
        .respond_with(PollStateMachine {
            hits: Arc::clone(&hits),
        })
        .mount(&server)
        .await;

    let model = provider(&server).video("grok-imagine-video");
    let result = model
        .do_generate(VideoOptions {
            prompt: Some("poll me".into()),
            provider_options: Some(fast_poll_options(&json!({}))),
            ..Default::default()
        })
        .await
        .expect("polling resolved");
    match &result.videos[0] {
        VideoData::Url { url, .. } => assert_eq!(url, "https://cdn.x.ai/v/poll.mp4"),
        _ => panic!("expected VideoData::Url"),
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        3,
        "expected 3 GETs (pending, pending, done)"
    );
}

#[tokio::test]
async fn failed_status_surfaces_as_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-fail"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path_regex("^/videos/req-fail$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "failed"
        })))
        .mount(&server)
        .await;

    let model = provider(&server).video("grok-imagine-video");
    let err = model
        .do_generate(VideoOptions {
            prompt: Some("doomed".into()),
            provider_options: Some(fast_poll_options(&json!({}))),
            ..Default::default()
        })
        .await
        .expect_err("should fail");
    assert!(
        err.to_string().to_lowercase().contains("failed"),
        "got: {err}"
    );
}

#[tokio::test]
async fn moderation_blocks_promote_to_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-mod"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/videos/req-mod"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": {
                "url": "https://cdn.x.ai/v/x.mp4",
                "respect_moderation": false
            }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).video("grok-imagine-video");
    let err = model
        .do_generate(VideoOptions {
            prompt: Some("spicy".into()),
            provider_options: Some(fast_poll_options(&json!({}))),
            ..Default::default()
        })
        .await
        .expect_err("moderation should block");
    assert!(
        err.to_string().to_lowercase().contains("moderation"),
        "got: {err}"
    );
}

#[tokio::test]
async fn empty_video_url_surfaces_as_provider_error() {
    // Mirrors upstream `xai-video-model.ts:324-330`: when the status payload
    // is `done` but `video.url` is the empty string, throw rather than
    // returning a bogus URL.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "request_id": "req-empty-url"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/videos/req-empty-url"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "status": "done",
            "video": { "url": "" }
        })))
        .mount(&server)
        .await;

    let model = provider(&server).video("grok-imagine-video");
    let err = model
        .do_generate(VideoOptions {
            prompt: Some("hello".into()),
            provider_options: Some(fast_poll_options(&json!({}))),
            ..Default::default()
        })
        .await
        .expect_err("empty url should fail");
    assert!(
        err.to_string().to_lowercase().contains("no video url"),
        "got: {err}"
    );
}

#[tokio::test]
async fn missing_request_id_in_post_response_is_a_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/videos/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;

    let model = provider(&server).video("grok-imagine-video");
    let err = model
        .do_generate(VideoOptions {
            prompt: Some("hi".into()),
            provider_options: Some(fast_poll_options(&json!({}))),
            ..Default::default()
        })
        .await
        .expect_err("missing request_id should fail");
    assert!(
        err.to_string().to_lowercase().contains("request_id"),
        "got: {err}"
    );
}

#[tokio::test]
async fn fps_seed_and_n_emit_unsupported_warnings_but_call_succeeds() {
    let server = MockServer::start().await;
    mount_create_and_done(&server, "/videos/generations", "req-warn").await;

    let model = provider(&server).video("grok-imagine-video");
    let result = model
        .do_generate(VideoOptions {
            prompt: Some("hi".into()),
            n: 3,
            fps: Some(30),
            seed: Some(99),
            provider_options: Some(fast_poll_options(&json!({}))),
            ..Default::default()
        })
        .await
        .expect("ok");
    let kinds: Vec<&str> = result
        .warnings
        .iter()
        .filter_map(|w| match w {
            Warning::Unsupported { feature, .. } => Some(feature.as_str()),
            _ => None,
        })
        .collect();
    assert!(kinds.contains(&"fps"));
    assert!(kinds.contains(&"seed"));
    assert!(kinds.contains(&"n"));
}

#[tokio::test]
async fn max_videos_per_call_reports_one() {
    let server = MockServer::start().await;
    let model = provider(&server).video("grok-imagine-video");
    assert_eq!(model.max_videos_per_call().await, Some(1));
    assert_eq!(model.provider(), "xai");
    assert_eq!(model.model_id(), "grok-imagine-video");
}
