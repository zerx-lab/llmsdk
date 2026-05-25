//! Contract tests for Mistral `provider_options` + standardized options.
// Rust guideline compliant 2026-05-25

use llmsdk_mistral::Mistral;
use llmsdk_provider::LanguageModel;
use llmsdk_provider::language_model::{
    CallOptions, Message, ReasoningEffort, ResponseFormat, TextPart, UserPart,
};
use llmsdk_provider::shared::ProviderOptions;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(server: &MockServer) -> Mistral {
    Mistral::builder()
        .api_key("test-api-key")
        .base_url(server.uri())
        .build()
        .expect("provider builds")
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

fn empty_response() -> serde_json::Value {
    json!({
        "id": "r1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

#[tokio::test]
async fn seed_serializes_as_random_seed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "random_seed": 7 })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            seed: Some(7),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn stop_sequences_serialize_as_stop() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "stop": ["END"] })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            stop_sequences: Some(vec!["END".into()]),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn safe_prompt_provider_option_forwarded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "safe_prompt": true })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_response()))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "mistral".into(),
        json!({"safePrompt": true}).as_object().cloned().unwrap(),
    );
    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn document_limits_provider_option_forwarded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "document_image_limit": 5,
            "document_page_limit": 10
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_response()))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "mistral".into(),
        json!({"documentImageLimit": 5, "documentPageLimit": 10})
            .as_object()
            .cloned()
            .unwrap(),
    );
    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn reasoning_effort_serializes_for_supported_model() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({ "reasoning_effort": "high" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            reasoning: Some(ReasoningEffort::High),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn json_object_response_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "response_format": { "type": "json_object" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_response()))
        .mount(&server)
        .await;

    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            response_format: Some(ResponseFormat::Json {
                schema: None,
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("ok");
}

#[tokio::test]
async fn json_schema_response_format_with_strict() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": { "name": "MySchema", "strict": true }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(empty_response()))
        .mount(&server)
        .await;

    let mut po = ProviderOptions::new();
    po.insert(
        "mistral".into(),
        json!({"strictJsonSchema": true})
            .as_object()
            .cloned()
            .unwrap(),
    );
    let provider = provider(&server);
    let _ = provider
        .chat("mistral-small-latest")
        .do_generate(CallOptions {
            prompt: vec![user_text("hi")],
            response_format: Some(ResponseFormat::Json {
                schema: Some(serde_json::from_value(json!({"type":"object"})).unwrap()),
                name: Some("MySchema".into()),
                description: None,
            }),
            provider_options: Some(po),
            ..Default::default()
        })
        .await
        .expect("ok");
}
