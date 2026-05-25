//! URL-shape contract tests for Azure `OpenAI`.
//!
//! Asserts the v1 vs legacy deployment URL layouts and `api-version`
//! query handling against a `wiremock` server.
// Rust guideline compliant 2026-02-21

use llmsdk_azure::AzureOpenAi;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{CallOptions, Message, TextPart, UserPart};
use serde_json::json;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn happy_chat_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-1",
        "created": 1_700_000_000_u64,
        "model": "gpt-4o-mini-deployment",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hi" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

fn user_text(text: &str) -> Message {
    Message::User {
        content: vec![UserPart::Text(TextPart {
            text: text.into(),
            provider_options: None,
        })],
        provider_options: None,
    }
}

#[tokio::test]
async fn v1_mode_routes_to_v1_path_with_api_version() {
    let server = MockServer::start().await;

    // v1 mode: POST {base}/v1/chat/completions?api-version=v1
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(query_param("api-version", "v1"))
        .and(header("api-key", "az-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_chat_response()))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-test-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds");

    let model = provider.chat("gpt-4o-mini-deployment");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(model.provider(), "azure.chat");
    assert!(!result.content.is_empty());
}

#[tokio::test]
async fn legacy_mode_includes_deployment_id_in_path() {
    let server = MockServer::start().await;

    // legacy mode: POST {base}/deployments/{id}/chat/completions?api-version=...
    Mock::given(method("POST"))
        .and(path("/deployments/gpt-4o-mini-deployment/chat/completions"))
        .and(query_param("api-version", "2024-08-01-preview"))
        .and(header("api-key", "az-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_chat_response()))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-test-key")
        .base_url(server.uri())
        .api_version("2024-08-01-preview")
        .use_deployment_based_urls(true)
        .build()
        .expect("provider builds");

    let model = provider.chat("gpt-4o-mini-deployment");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert!(!result.content.is_empty());
}

#[tokio::test]
async fn responses_endpoint_v1_mode() {
    let server = MockServer::start().await;

    let resp = json!({
        "id": "resp_1",
        "object": "response",
        "created_at": 1_700_000_000_u64,
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    { "type": "output_text", "text": "ok" }
                ]
            }
        ],
        "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(query_param("api-version", "preview"))
        .and(header("api-key", "az-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(resp))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-test-key")
        .base_url(server.uri())
        .api_version("preview")
        .build()
        .expect("provider builds");

    let model = provider.responses("gpt-4o-mini");
    let result = model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");

    assert_eq!(model.provider(), "azure.responses");
    assert!(!result.content.is_empty());
}

#[tokio::test]
async fn extra_header_merges_with_api_key() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(query_param("api-version", "v1"))
        .and(header("api-key", "az-test-key"))
        .and(header("x-correlation-id", "req-42"))
        .respond_with(ResponseTemplate::new(200).set_body_json(happy_chat_response()))
        .mount(&server)
        .await;

    let provider = AzureOpenAi::builder()
        .api_key("az-test-key")
        .base_url(server.uri())
        .header("x-correlation-id", Some("req-42".into()))
        .build()
        .expect("provider builds");

    let model = provider.chat("dep-1");
    model
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            ..Default::default()
        })
        .await
        .expect("ok");
}
