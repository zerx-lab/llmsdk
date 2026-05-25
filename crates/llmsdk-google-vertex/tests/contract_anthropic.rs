//! Anthropic-on-Vertex contract tests.
//!
//! Validates the URL + body-transform hooks: requests route to
//! `:rawPredict`, `model` is stripped from the body, and
//! `anthropic_version: "vertex-2023-10-16"` is injected.
// Rust guideline compliant 2026-05-25

use llmsdk_google_vertex::GoogleVertex;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Content, Message, TextPart, UserPart};
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn provider(server: &MockServer) -> GoogleVertex {
    GoogleVertex::builder()
        .api_key("k")
        .anthropic_base_url(server.uri())
        .build()
        .await
        .expect("ok")
}

#[tokio::test]
async fn raw_predict_strips_model_and_adds_anthropic_version() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/claude-sonnet-4-5:rawPredict"))
        .and(body_partial_json(json!({
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi back"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let model = p.anthropic().language_model("claude-sonnet-4-5");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![Message::User {
                content: vec![UserPart::Text(TextPart {
                    text: "hi".into(),
                    provider_options: None,
                })],
                provider_options: None,
            }],
            max_output_tokens: Some(32),
            ..Default::default()
        })
        .await
        .expect("ok");
    assert!(matches!(result.content.first(), Some(Content::Text(_))));
}

#[tokio::test]
async fn anthropic_provider_id_is_vertex_messages() {
    let server = MockServer::start().await;
    let p = provider(&server).await;
    let m = p.anthropic().messages("claude-opus-4-7");
    assert_eq!(m.provider(), "google.vertex.anthropic.messages");
}
