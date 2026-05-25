//! Vertex Gemini language model contract tests.
//!
//! Exercises the bridge between `GoogleVertexLanguageModel` and
//! `llmsdk-google::internal::GoogleLanguageModel`. The wire format is
//! identical to Public-API Gemini; what changes here is the URL
//! (`publishers/google/models/...`) and the auth header (`x-goog-api-key`
//! in Express mode).
// Rust guideline compliant 2026-05-25

use llmsdk_google_vertex::GoogleVertex;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Content, Message, TextPart, UserPart};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{header, method, path};
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
async fn do_generate_routes_to_publishers_google_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(header("x-goog-api-key", "k"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hi back"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.language_model("gemini-2.5-flash");
    let result = m
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(matches!(result.content.first(), Some(Content::Text(_))));
    if let Some(Content::Text(t)) = result.content.first() {
        assert_eq!(t.text, "hi back");
    }
}

#[tokio::test]
async fn provider_string_routes_to_vertex_chat() {
    let server = MockServer::start().await;
    let p = provider(&server).await;
    let m = p.chat("gemini-2.5-flash");
    assert_eq!(m.provider(), "google.vertex.chat");
}

#[tokio::test]
async fn shared_request_type_sent_as_vertex_paygo_header() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "googleVertex".into(),
        json!({"sharedRequestType": "flex"})
            .as_object()
            .unwrap()
            .clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(header("X-Vertex-AI-LLM-Shared-Request-Type", "flex"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.language_model("gemini-2.5-flash");
    let _ = m
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn request_type_sent_as_vertex_paygo_header() {
    let server = MockServer::start().await;
    let mut opts = ProviderOptions::new();
    opts.insert(
        "googleVertex".into(),
        json!({"sharedRequestType": "priority", "requestType": "shared"})
            .as_object()
            .unwrap()
            .clone(),
    );
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash:generateContent"))
        .and(header("X-Vertex-AI-LLM-Shared-Request-Type", "priority"))
        .and(header("X-Vertex-AI-LLM-Request-Type", "shared"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP"
            }]
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.language_model("gemini-2.5-flash");
    let _ = m
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            provider_options: Some(opts),
            ..Default::default()
        })
        .await
        .expect("ok");
}
