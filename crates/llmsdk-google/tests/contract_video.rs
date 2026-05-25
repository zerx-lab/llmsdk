//! Contract tests for [`llmsdk_google::GoogleVideoModel`] (Veo LRO).
// Rust guideline compliant 2026-05-25

use llmsdk_google::Google;
use llmsdk_provider::VideoModel;
use llmsdk_provider::shared::ProviderOptions;
use llmsdk_provider::video_model::{VideoData, VideoOptions};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Google {
    Google::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .expect("ok")
}

#[tokio::test]
async fn happy_path_done_on_first_poll() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/veo-3.0-generate-001:predictLongRunning"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"name": "operations/abc"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/operations/abc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "operations/abc",
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {"video": {"uri": "https://example.com/v.mp4"}}
                    ]
                }
            }
        })))
        .mount(&server)
        .await;
    let mut opts = ProviderOptions::new();
    // Make polling fast.
    opts.insert(
        "google".into(),
        json!({"pollIntervalMs": 10, "pollTimeoutMs": 5000})
            .as_object()
            .unwrap()
            .clone(),
    );

    let model = provider(&server).video("veo-3.0-generate-001");
    let r = model
        .do_generate(VideoOptions {
            prompt: Some("a cat".into()),
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert_eq!(r.videos.len(), 1);
    let VideoData::Url { url, .. } = &r.videos[0] else {
        panic!("expected url");
    };
    assert!(url.contains("example.com/v.mp4"));
    // API key appended:
    assert!(url.contains("key=test-key"));
}

#[tokio::test]
async fn failed_operation_returns_err() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/veo-3.0-generate-001:predictLongRunning"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"name": "operations/x"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/operations/x"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "operations/x",
            "done": true,
            "error": { "code": 500, "message": "internal" }
        })))
        .mount(&server)
        .await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "google".into(),
        json!({"pollIntervalMs": 10, "pollTimeoutMs": 5000})
            .as_object()
            .unwrap()
            .clone(),
    );

    let model = provider(&server).video("veo-3.0-generate-001");
    let err = model
        .do_generate(VideoOptions {
            prompt: Some("a cat".into()),
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect_err("should err");
    assert!(format!("{err}").contains("internal"));
}
