//! xAI / `MaaS` on Vertex contract tests (`OpenAI`-compatible wire at
//! `endpoints/openapi/chat/completions`).
// Rust guideline compliant 2026-05-25

use llmsdk_google_vertex::GoogleVertex;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Content, Message, TextPart, UserPart};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn provider(server: &MockServer) -> GoogleVertex {
    GoogleVertex::builder()
        .api_key("k")
        .sub_provider_base_url(server.uri())
        .build()
        .await
        .expect("ok")
}

fn chat_response_body() -> serde_json::Value {
    json!({
        "id": "cc_1",
        "object": "chat.completion",
        "created": 0,
        "model": "ignored",
        "choices": [
            {"index": 0, "finish_reason": "stop",
             "message": {"role": "assistant", "content": "hi"}}
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    })
}

#[tokio::test]
async fn xai_on_vertex_routes_to_chat_completions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("x-goog-api-key", "k"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_body()))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.xai().chat("xai/grok-4.20-reasoning");
    let r = m
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
    assert!(matches!(r.content.first(), Some(Content::Text(_))));
}

#[tokio::test]
async fn maas_on_vertex_routes_to_chat_completions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("x-goog-api-key", "k"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_body()))
        .mount(&server)
        .await;
    let p = provider(&server).await;
    let m = p.maas().chat("deepseek-ai/deepseek-v3.2-maas");
    let r = m
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
    assert!(matches!(r.content.first(), Some(Content::Text(_))));
}
