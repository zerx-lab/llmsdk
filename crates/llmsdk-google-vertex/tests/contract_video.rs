//! Vertex Veo video contract tests (LRO polling).
// Rust guideline compliant 2026-05-25

use llmsdk_google_vertex::GoogleVertex;
use llmsdk_provider::VideoModel;
use llmsdk_provider::shared::ProviderOptions;
use llmsdk_provider::video_model::{VideoData, VideoOptions};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn provider(server: &MockServer) -> GoogleVertex {
    GoogleVertex::builder()
        .api_key("k")
        .language_base_url(server.uri())
        .build()
        .await
        .expect("ok")
}

#[tokio::test]
async fn predict_long_running_polls_until_done_and_surfaces_videos() {
    let server = MockServer::start().await;

    // 1st call: kick off the LRO.
    Mock::given(method("POST"))
        .and(path("/models/veo-3.1-fast-generate-001:predictLongRunning"))
        .and(body_partial_json(json!({
            "instances": [{"prompt": "a clip"}],
            "parameters": {"sampleCount": 1, "aspectRatio": "16:9"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "operations/abc",
            "done": false
        })))
        .mount(&server)
        .await;

    // 2nd call: poll → done with one base64 video.
    Mock::given(method("POST"))
        .and(path(
            "/models/veo-3.1-fast-generate-001:fetchPredictOperation",
        ))
        .and(body_partial_json(json!({
            "operationName": "operations/abc"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "operations/abc",
            "done": true,
            "response": {
                "videos": [{"bytesBase64Encoded": "QkFTRTY0", "mimeType": "video/mp4"}]
            }
        })))
        .mount(&server)
        .await;

    let p = provider(&server).await;
    let m = p.video("veo-3.1-fast-generate-001");
    let mut opts = ProviderOptions::new();
    opts.insert(
        "googleVertex".into(),
        json!({"pollIntervalMs": 1}).as_object().unwrap().clone(),
    );
    let r = m
        .do_generate(VideoOptions {
            prompt: Some("a clip".into()),
            n: 1,
            aspect_ratio: Some("16:9".into()),
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.videos.len(), 1);
    match &r.videos[0] {
        VideoData::Base64 { media_type, .. } => assert_eq!(media_type, "video/mp4"),
        other => panic!("expected base64 video, got {other:?}"),
    }
}

#[tokio::test]
async fn lro_error_message_surfaces() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/veo-3.0-generate-001:predictLongRunning"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "operations/x",
            "done": true,
            "error": {"message": "Quota exceeded"}
        })))
        .mount(&server)
        .await;

    let p = provider(&server).await;
    let m = p.video("veo-3.0-generate-001");
    let err = m
        .do_generate(VideoOptions {
            prompt: Some("x".into()),
            n: 1,
            ..Default::default()
        })
        .await
        .expect_err("lro error");
    assert!(format!("{err}").contains("Quota exceeded"));
}
